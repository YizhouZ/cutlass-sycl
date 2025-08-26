#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "flash_attention_v2/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "flash_attention_v2/kernel/xe_paged.hpp"
#include "flash_attention_v2/collective/xe_paged_mma.hpp"
#include "flash_attention_v2/collective/paged_epilogue.hpp"
#include "flash_attention_v2/collective/xe_paged_softmax_epilogue.hpp"
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

void assert_allclose(const torch::Tensor &a, const torch::Tensor &b, float rtol = 1e-2, float atol = 1e-1) {
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

torch::Tensor ref_compute_out(torch::Tensor &scores, torch::Tensor &value_cache,
                              torch::Tensor &block_tables,
                              uint32_t partition_size = 512, 
                              bool use_partition = false) {
  // scores: [num_seqs, num_heads, seq_len]
  // now we need seq_len % partition_size == 0

  std::cout << "scores shape: " << scores.sizes() << std::endl;
  std::cout << "value_cache.dtype: " << value_cache.dtype() << std::endl;

  auto num_seqs = scores.size(0);
  auto num_heads = scores.size(1);
  auto seq_len = scores.size(2);
  auto num_partitions = (seq_len + partition_size - 1) / partition_size;

  auto num_blocks = value_cache.size(0);
  auto block_size = value_cache.size(1);
  auto num_kv_heads = value_cache.size(2);
  auto head_size = value_cache.size(3);
  
  auto query_group_size = num_heads / num_kv_heads;

  uint32_t useful_blocks = (seq_len + block_size - 1) / block_size;

  torch::Tensor tem_output = torch::zeros({num_seqs, num_kv_heads, num_partitions,
                                       query_group_size, head_size},
                                      torch::kBFloat16)
                             .to(scores.device());
  torch::Tensor ultimate_output = torch::zeros(
      {num_seqs, num_kv_heads, query_group_size, head_size}, torch::kBFloat16)
      .to(scores.device());
  for (int j = 0; j < num_kv_heads; ++j) {
    for (int i = 0; i < num_seqs; ++i) {
      auto start_block = block_tables[i];
      std::vector<torch::Tensor> value_blocks;
      for (int u = 0; u < useful_blocks; ++u) {
        auto curr_value_block = value_cache[start_block[u]];
        auto value_slice = curr_value_block.index({Slice(), j, Slice()});
        value_blocks.push_back(value_slice);
      }
      auto value_tensor = torch::cat(value_blocks, /*dim=*/0);
      std::cout << "value_tensor shape: " << value_tensor.sizes() << std::endl;
      if (use_partition) {
        for (int k = 0; k < num_partitions; ++k) {
          auto scores_view = scores
                                 .view({num_seqs, num_kv_heads, query_group_size,
                                        num_partitions, partition_size})
                                 .transpose(2, 3)
                                 .contiguous();
          auto score_slice = scores_view[i][j][k];
          auto value_partition =
              value_tensor.slice(0, k * partition_size, (k + 1) * partition_size)
                  .contiguous();

          std::cout << "score_slice shape: " << score_slice.sizes() << std::endl;
          tem_output[i][j][k] = torch::matmul(score_slice, value_partition);
        }
      } else {
        // no partition, just do it directly
        auto scores_view = scores
                               .view({num_seqs, num_kv_heads, query_group_size,
                                      seq_len})
                               .contiguous();
        auto score_slice = scores_view[i][j];
        auto value_partition = value_tensor.contiguous();
        std::cout << "score_slice shape: " << score_slice.sizes() << std::endl;
        // std::cout << "score_slice: " << score_slice << std::endl;
        // std::cout << "value_partition: " << value_partition << std::endl;
        ultimate_output[i][j] = torch::matmul(score_slice, value_partition);
      }
    }
  }
  if (use_partition) {
    return tem_output.transpose(2, 3).contiguous().view(
        {num_seqs, num_heads, num_partitions, head_size});
  } else {
    return ultimate_output.contiguous().view(
        {num_seqs, num_heads, head_size});
  }
}

int num_heads_q = 16;
int num_heads_kv = 2;
int seq_len_q = 1;
int num_block = 8;
int block_size = 64;
int head_size = 128;
int seq_len_kv = num_block * block_size;
int max_blocks_per_seq = num_block;
int group_heads = num_heads_q / num_heads_kv;
torch::Tensor block_tables, query, key_cache, value_cache, scores, out;

void init_values() {
  query = torch::rand({seq_len_q, num_heads_q, head_size}, torch::kBFloat16)
          .to(torch::kXPU);
  key_cache = torch::rand({num_block, block_size, num_heads_kv, head_size}, torch::kBFloat16)
          .to(torch::kXPU);
  // for (int i = 0; i < num_block; ++i) {
  //   key_cache[i].fill_(i + 1);
  // }
  // std::cout << "key_cache: " << key_cache[0] << std::endl;
  value_cache = torch::rand({num_block, block_size, num_heads_kv, head_size}, torch::kBFloat16)
          .to(torch::kXPU);
  // for (int i = 0; i < num_block; ++i) {
  //   value_cache[i].fill_(i + 1);
  // }
  // for (int i = 0; i < num_block; ++i) {
  //   for (int j = 0; j < block_size; ++j) {
  //     for (int k = 0; k < num_heads_kv; ++k) {
  //       for (int l = 0; l < head_size; ++l) {
  //         value_cache[i][j][k][l] = l + 1;
  //       }
  //     }
  //   }
  // }
  block_tables = torch::ones({seq_len_q, max_blocks_per_seq}, torch::kInt).to(torch::kXPU);
  scores = torch::zeros({seq_len_q, num_heads_kv, group_heads, seq_len_kv}, torch::kFloat32).to(torch::kXPU);
  out = torch::zeros_like(query).to(torch::kFloat32);

  // block_tables[0] =
  //     torch::arange(0, max_blocks_per_seq, torch::kInt).to(torch::kXPU);
  // block_tables[0][1] = 0;
  block_tables[0] = torch::randint(0, max_blocks_per_seq, {max_blocks_per_seq});
  std::cout << "block_tables: " << block_tables[0] << std::endl;
}

auto ref_softmax(torch::Tensor &scores, uint32_t partition_size = 512, bool use_partition = false) {
  // scores: [num_seqs, num_heads, seq_len]
  // now we need seq_len % partition_size == 0
  auto num_seqs = scores.size(0);
  auto num_heads = scores.size(1);
  auto seq_len = scores.size(2);
  auto num_partitions = (seq_len + partition_size - 1) / partition_size;

  torch::Tensor ref_max_logits =
      torch::empty({scores.size(0), scores.size(1), num_partitions},
                   torch::kFloat32)
          .to(scores.device());
  torch::Tensor ref_exp_sums =
      torch::empty({scores.size(0), scores.size(1), num_partitions},
                   torch::kFloat32)
          .to(scores.device());
  auto scores_view = use_partition ? 
                     scores.view({num_seqs, num_heads, num_partitions, partition_size}) : 
                     scores.view({num_seqs, num_heads, seq_len});
  float scale = 1 / sqrt(static_cast<float>(head_size));

  for (int i = 0; i < num_seqs; ++i)
    for (int j = 0; j < num_heads; ++j)
      for (int k = 0; k < seq_len; ++k)
        scores_view[i][j][k] *= scale;

  for (int i = 0; i < num_seqs; ++i) {
    for (int j = 0; j < num_heads; ++j) {
      if (use_partition) {
        for (int k = 0; k < num_partitions; ++k) {
          torch::Tensor ref_max_slice;
          torch::Tensor max_indices;
          std::tie(ref_max_slice, max_indices) =
              torch::max(scores_view[i][j][k], /*dim=*/0, /*keepdim=*/false);
          ref_max_logits[i][j][k] = ref_max_slice.item<float>();
          scores_view[i][j][k] = torch::exp(scores_view[i][j][k] - ref_max_slice);
          ref_exp_sums[i][j][k] = torch::sum(scores_view[i][j][k]);
          scores_view[i][j][k] = scores_view[i][j][k] / ref_exp_sums[i][j][k];
        }
      } else {
        // no partition, just do it directly
        torch::Tensor ref_max_slice;
        torch::Tensor max_indices;
        std::tie(ref_max_slice, max_indices) =
            torch::max(scores_view[i][j], /*dim=*/0, /*keepdim=*/false);
        ref_max_logits[i][j][0] = ref_max_slice.item<float>();
        scores_view[i][j] = torch::exp(scores_view[i][j] - ref_max_slice);
        ref_exp_sums[i][j][0] = torch::sum(scores_view[i][j]);
        scores_view[i][j] = scores_view[i][j] / ref_exp_sums[i][j][0];
      }
    }
  }
  scores = scores_view.view({num_seqs, num_heads, seq_len});
  return std::make_tuple(ref_max_logits, ref_exp_sums);
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
      {num_seqs, num_kv_heads * query_group_size, useful_blocks * block_size});
}


template <int KVTile, int NumSG>
void run_gemm() {
  using TileShapeQK = Shape<_8, Int<KVTile>, _64>;
  using TileShapePV = Shape<_8, _16, _64>; // block_size as last dim
  using TileShapeOut = Shape<_8, _128, Int<KVTile>>;
  using TileShapeDebug = Shape<_8, Int<KVTile>, _64>;
  using SubgroupLayout = cute::Layout<Shape<Int<NumSG>, _1, _1>, cute::Stride<_1, _1, _1>>;

  const int PipelineStages = 2;
  using ElementInputQ = bfloat16_t;
  using ElementInputKV = bfloat16_t;
  using MMAOperation = XE_8x16x16_F32BF16BF16F32_TT;
  using GmemTiledCopyQ = XE_2D_U16x8x16_LD_N;
  using GmemTiledCopyK = XE_2D_U16x16x16_LD_T; // transposed due to col major
  using GmemTiledCopyV = XE_2D_U16x32x16_LD_V; // TODO: tune perf for XE_2D_U16x32x16_LD_V and XE_2D_U16x16x16_LD_V
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

  using CollectiveSoftmaxEpilogue = cutlass::flash_attention::collective::FlashPagedSoftmaxEpilogue<EpilogueDispatchPolicy, ElementAccumulator>;
  using CollectiveEpilogue = cutlass::flash_attention::collective::FlashPagedEpilogue<
        EpilogueDispatchPolicy, MMAOperation, TileShapeOut, TileShapeDebug, SubgroupLayout, ElementAccumulator, ElementOutput, cutlass::gemm::TagToStrideC_t<LayoutO>,
        GmemTiledCopyStore>;

  using ProblemShapeType = cute::tuple<int, int, int, int, int, int, int>;
  using namespace cutlass::fmha::collective;

  // Mainloop
  using CollectiveMainloop = cutlass::flash_attention::collective::PagedMma<
      GEMMDispatchPolicy, ProblemShapeType,
      ElementInputQ, cutlass::gemm::TagToStrideA_t<LayoutQ>,
      ElementInputKV, cutlass::gemm::TagToStrideB_t<LayoutK>,
      ElementInputKV, cutlass::gemm::TagToStrideB_t<LayoutV>,
      MMAOperation,
      TileShapeQK, TileShapePV, SubgroupLayout,
      GmemTiledCopyQ/* Q */, GmemTiledCopyK/* K */,  GmemTiledCopyV/* V */>;

  using Scheduler = cutlass::flash_attention::PagedIndividualScheduler;
  using FMHAKernel = cutlass::flash_attention::kernel::FMHAPaged<ProblemShapeType,
                                                                 CollectiveMainloop,
                                                                 CollectiveSoftmaxEpilogue,
                                                                 CollectiveEpilogue,
                                                                 Scheduler>;
  
  ProblemShapeType problem_shape = cute::make_tuple(num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, num_block, block_size, head_size);
  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideV = typename FMHAKernel::StrideV;
  using StrideO = typename FMHAKernel::StrideO;
  using StrideS = typename FMHAKernel::StrideS;

  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  StrideS stride_S;
  // print("strdeQ: "); print(StrideQ{}); print("\n");
  stride_Q = cutlass::make_cute_packed_stride(StrideQ{}, cute::make_shape(num_heads_q, head_size, seq_len_q));
  stride_K = cutlass::make_cute_packed_stride(StrideK{}, cute::make_shape(seq_len_kv * num_heads_kv, head_size, 1));
  stride_V = cutlass::make_cute_packed_stride(StrideV{}, cute::make_shape(head_size, seq_len_kv * num_heads_kv, 1));
  stride_O = cutlass::make_cute_packed_stride(StrideO{}, cute::make_shape(num_heads_q, head_size, seq_len_q));
  stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(group_heads, seq_len_kv, seq_len_q * num_heads_kv));

  typename FMHAKernel::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_shape,
        {
          reinterpret_cast<ElementInputQ*>(query.data_ptr()), stride_Q,
          reinterpret_cast<ElementInputKV*>(key_cache.data_ptr()), stride_K,
          reinterpret_cast<ElementInputKV*>(value_cache.data_ptr()), stride_V,
          reinterpret_cast<int*>(block_tables.data_ptr()),
          block_size,
          max_blocks_per_seq
        },
        {1 / sqrt(static_cast<float>(head_size))},
        // {1.0f},
        {reinterpret_cast<ElementOutput*>(out.data_ptr()), stride_O, reinterpret_cast<ElementOutput*>(scores.data_ptr()), stride_S},
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
  // std::cout << scores[0][0] << std::endl;
  // std::cout << "max: " << scores[0][0].max(-1) << std::endl;
  // std::cout << "sum: " << scores[0][0].sum(-1) << std::endl;
  // std::cout << "key_cache: " << key_cache << std::endl;
  // std::cout << out << std::endl;
  
  torch::Tensor context_lens = torch::ones({seq_len_q}, torch::kInt).to(torch::kXPU);
  context_lens[0] = seq_len_kv;
  auto ref_scores = ref_compute_score(query, key_cache, block_tables, context_lens);
  auto [ref_max_logits, ref_exp_sums] = ref_softmax(ref_scores);
  auto ref_scores_view = ref_scores.view({seq_len_q, num_heads_kv, group_heads, num_block * block_size});
  assert_allclose(ref_scores_view.to(torch::kFloat32), scores.to(torch::kFloat32));

  ref_scores = ref_scores.to(value_cache.dtype());
  auto scores_dtype = scores.to(value_cache.dtype()).view_as(ref_scores);
  auto ref_out = ref_compute_out(scores_dtype, value_cache, block_tables);
  assert_allclose(ref_out.to(torch::kFloat32), out.to(torch::kFloat32));
  return 0;
}