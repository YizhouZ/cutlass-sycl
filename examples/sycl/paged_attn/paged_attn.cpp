#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "flash_attention_v2/kernel/xe_paged.hpp"
#include "flash_attention_v2/collective/xe_paged_mma.hpp"
#include "flash_attention_v2/collective/paged_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"

#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/sycl_event_manager.hpp"

#include <cute/tensor.hpp>
#include <random>

#include "helper.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include <torch/torch.h>

using namespace torch::indexing;
using namespace at;
using namespace cute;

static std::string shape_to_string(const c10::IntArrayRef& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (i + 1 < shape.size()) oss << ", ";
    }
    oss << "]";
    return oss.str();
}

void assert_allclose(const torch::Tensor &a, const torch::Tensor &b, float rtol = 1e-2, float atol = 1e-2) {
    if (!a.sizes().equals(b.sizes())) {
        throw std::runtime_error("Tensor sizes do not match: " +
                                 shape_to_string(a.sizes()) + " vs " + shape_to_string(b.sizes()));
    }

    auto diff = torch::abs(a - b);
    auto tol = atol + rtol * torch::abs(b);

    auto mask = diff > tol;  // Boolean mask where tensors differ more than allowed
    
    if (mask.any().item<bool>()) {
        auto indices = mask.nonzero();
        std::cout << "Mismatches found (" << indices.size(0) << " positions):\n";

        for (int i = 0; i < indices.size(0); ++i) {
            auto idx = indices[i];
            std::vector<int64_t> idx_vec(idx.size(0));
            for (int j = 0; j < idx.size(0); ++j) {
                idx_vec[j] = idx[j].item<int64_t>();
            }

            std::vector<torch::indexing::TensorIndex> ti;
            ti.reserve(idx_vec.size());
            for (auto v : idx_vec) {
                ti.push_back(v);
            }

            float a_val = a.index(ti).item<float>();
            float b_val = b.index(ti).item<float>();
            float d_val = diff.index(ti).item<float>();

            std::cout << "  idx=[";
            for (size_t j = 0; j < idx_vec.size(); ++j) {
                std::cout << idx_vec[j] << (j + 1 < idx_vec.size() ? ", " : "");
            }
            std::cout << "] diff=" << d_val << " (a=" << a_val << ", b=" << b_val << ")\n";
        }
    } else {
        std::cout << "Tensors are allclose.\n";
    }
}

int num_heads_q = 8;
int num_heads_kv = 1;
int seq_len_q = 1;
int num_block = 8;
int block_size = 64;
int head_size = 128;
int seq_len_kv = num_block * block_size;
int max_blocks_per_seq = num_block;
int group_heads = num_heads_q / num_heads_kv;
torch::Tensor block_tables, query, key_cache, value_cache, scores;

void init_values() {
  query = torch::randn({seq_len_q, num_heads_q, head_size}, torch::kBFloat16)
          .to(torch::kXPU);
  key_cache = torch::randn({num_block, block_size, num_heads_kv, head_size}, torch::kBFloat16)
          .to(torch::kXPU);
  // for (int i = 0; i < num_block; ++i) {
  //   key_cache[i].fill_(i + 1);
  // }
  value_cache = torch::randn({num_block, block_size, num_heads_kv, head_size}, torch::kBFloat16)
          .to(torch::kXPU);
  block_tables = torch::ones({seq_len_q, max_blocks_per_seq}, torch::kInt).to(torch::kXPU);
  scores = torch::zeros({seq_len_q, num_heads_kv, group_heads, seq_len_kv}, torch::kFloat32).to(torch::kXPU);

  block_tables[0] =
      torch::arange(0, max_blocks_per_seq, torch::kInt).to(torch::kXPU);
}

torch::Tensor ref_compute_score(torch::Tensor &query, torch::Tensor &key_cache,
                                torch::Tensor &block_tables,
                                torch::Tensor &context_lens) {
  auto num_seqs = query.size(0);
  auto num_heads = query.size(1);
  auto head_size = query.size(2);

  auto num_blocks = key_cache.size(0);
  auto block_size = key_cache.size(1);
  auto num_kv_heads = key_cache.size(2);

  auto seq_len = context_lens[0].item<int>();

  auto query_group_size = num_heads / num_kv_heads;
  auto tem_query =
      query.view({num_seqs, num_kv_heads, query_group_size, head_size});

  uint32_t useful_blocks = (seq_len + block_size - 1) / block_size;
  torch::Tensor scores_ref = torch::zeros({num_seqs, num_kv_heads, useful_blocks,
                                       query_group_size, block_size},
                                      torch::kFloat32)
                             .to(query.device());

  for (int i = 0; i < num_seqs; ++i) {
    auto start_block = block_tables[i];
    for (int j = 0; j < num_kv_heads; ++j) {
      auto query_slice = tem_query[i][j];
      for (int k = 0; k < useful_blocks; ++k) {
        auto curr_key_block = key_cache[start_block[k]];
        auto key_slice = curr_key_block.index({Slice(), j, Slice()});

        scores_ref[i][j][k] = torch::matmul(query_slice, key_slice.transpose(0, 1));
      }
    }
  }
  return scores_ref.transpose(2, 3).contiguous().view(
      {num_seqs, num_kv_heads, query_group_size, useful_blocks * block_size});
}


template <int KVTile, int NumSG>
void run_gemm() {
  using TileShapeQK = Shape<_8, Int<KVTile>, _64>;
  using TileShapeOut = Shape<_8, Int<KVTile>, _64>;
  using SubgroupLayout = cute::Layout<Shape<Int<NumSG>, _1, _1>, cute::Stride<_1, _1, _1>>;

  const int PipelineStages = 2;
  using ElementInputQ = bfloat16_t;
  using ElementInputKV = bfloat16_t;
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using GmemTiledCopyQ = XE_2D_U16x8x16_LD_N;
  using GmemTiledCopyK = XE_2D_U16x16x16_LD_T; // transposed due to col major
  using GmemTiledCopyV = XE_2D_U16x32x32_LD_V;
  using ElementAccumulator = float;
  using ElementOutput = float;
  using GmemTiledCopyStore = XE_2D_U32x8x16_ST_N;

  using LayoutQ = cutlass::layout::RowMajor;
  using LayoutK = cutlass::layout::ColumnMajor;
  using LayoutV = cutlass::layout::RowMajor;
  using LayoutO = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  cutlass::KernelHardwareInfo hw_info;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  using CollectiveEpilogue = cutlass::flash_attention::collective::FlashPagedEpilogue<
        EpilogueDispatchPolicy, MMAOperation, TileShapeOut, SubgroupLayout, ElementAccumulator, ElementOutput, cutlass::gemm::TagToStrideC_t<LayoutO>,
        GmemTiledCopyStore>;

  using ProblemShapeType = cute::tuple<int, int, int, int, int, int>;
  using namespace cutlass::fmha::collective;

  // Mainloop
  using CollectiveMainloop = cutlass::flash_attention::collective::PagedMma<
      GEMMDispatchPolicy, ProblemShapeType,
      ElementInputQ, cutlass::gemm::TagToStrideA_t<LayoutQ>,
      ElementInputKV, cutlass::gemm::TagToStrideB_t<LayoutK>,
      MMAOperation,
      TileShapeQK, SubgroupLayout,
      GmemTiledCopyQ/* Q */, GmemTiledCopyK/* K */>;

  using Scheduler = cutlass::flash_attention::PagedIndividualScheduler;
  using FMHAKernel = cutlass::flash_attention::kernel::FMHAPaged<ProblemShapeType,
                                                                 CollectiveMainloop,
                                                                 CollectiveEpilogue,
                                                                 Scheduler>;
  
  ProblemShapeType problem_shape = cute::make_tuple(num_heads_q, num_heads_kv, seq_len_q, num_block, block_size, head_size);
  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideS = typename FMHAKernel::StrideS;

  StrideQ stride_Q;
  StrideK stride_K;
  StrideS stride_S;
  stride_Q = cutlass::make_cute_packed_stride(StrideQ{}, cute::make_shape(num_heads_q, head_size, seq_len_q));
  stride_K = cutlass::make_cute_packed_stride(StrideK{}, cute::make_shape(seq_len_kv, head_size, num_heads_kv));
  stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(group_heads, seq_len_kv, seq_len_q * num_heads_kv));

  typename FMHAKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_shape,
        {
          reinterpret_cast<ElementInputQ*>(query.data_ptr()), stride_Q,
          reinterpret_cast<ElementInputKV*>(key_cache.data_ptr()), stride_K,
          reinterpret_cast<int*>(block_tables.data_ptr()),
          block_size,
          max_blocks_per_seq
        },
        {reinterpret_cast<ElementOutput*>(scores.data_ptr()), stride_S},
        hw_info};

  size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  
  // init workspace
  CUTLASS_CHECK(FMHAKernel::initialize_workspace(arguments, workspace.get()));
  auto params = FMHAKernel::to_underlying_arguments(arguments, workspace.get());

  // run kernel
  dim3 const block = FMHAKernel::get_block_shape();
  dim3 const grid = FMHAKernel::get_grid_shape(params);
  int smem_size = FMHAKernel::SharedStorageSize;
  const auto sycl_block = syclcompat::dim3(block.x, block.y, block.z);
  const auto sycl_grid = syclcompat::dim3(grid.x, grid.y, grid.z);

#if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
  using namespace syclcompat::experimental;
  auto event = launch<cutlass::device_kernel<FMHAKernel>>(
      launch_policy{sycl_grid, sycl_block, local_mem_size{static_cast<std::size_t>(smem_size)},
                    kernel_properties{sycl_exp::sub_group_size<FMHAKernel::DispatchPolicy::SubgroupSize>}},
      params);
#else
  syclcompat::experimental::launch_properties launch_props {
    sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
  };
  syclcompat::experimental::kernel_properties kernel_props{
    sycl::ext::oneapi::experimental::sub_group_size<FMHAKernel::DispatchPolicy::SubgroupSize>
  };
  syclcompat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
  auto event = syclcompat::experimental::launch<cutlass::device_kernel<FMHAKernel>>(policy, params);
#endif
  EventManager::getInstance().addEvent(event);

  syclcompat::wait();
}

int main() {
  init_values();
  // std::cout << "block_tables: " << block_tables << std::endl;
  run_gemm<512, 8>();
  // for (int i = 0; i < num_block; ++i) {
  //   int row_start = i * block_size;
  //   int row_end = (i + 1) * block_size;
  //   std::cout << "block_id: " << i << std::endl;
  //   std::cout << scores[0][0][0].slice(0, row_start, row_end) << std::endl;
  // }
  // std::cout << scores[0] << std::endl;
  
  torch::Tensor context_lens = torch::ones({seq_len_q}, torch::kInt).to(torch::kXPU);
  context_lens[0] = seq_len_kv;
  auto ref_scores = ref_compute_score(query, key_cache, block_tables, context_lens);
  
  assert_allclose(ref_scores.to(torch::kFloat32), scores.to(torch::kFloat32));
  return 0;
}