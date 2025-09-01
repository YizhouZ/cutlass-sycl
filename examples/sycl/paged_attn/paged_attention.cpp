#include <iostream>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include <torch/torch.h>
#include <torch/extension.h>

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

using namespace torch::indexing;
using namespace at;
using namespace cute;

inline float get_exe_time(const sycl::event &e) {
  return (e.template get_profiling_info<
              sycl::info::event_profiling::command_end>() -
          e.template get_profiling_info<
              sycl::info::event_profiling::command_start>()) /
         1000.0f; // us
}

template <int KVTile, int NumSG>
float run_gemm(
    torch::Tensor& max_logits,
    torch::Tensor& exp_sums,
    torch::Tensor& temp_output,
    torch::Tensor& output,
    torch::Tensor& scores,
    const torch::Tensor& query,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& block_tables,
    const float sm_scale,
    const int num_heads_q,
    const int num_heads_kv,
    const int seqlen_q,
    const int seqlen_kv,
    const int num_blocks,
    const int block_size,
    const int max_blocks_per_seq,
    const int head_size,
    const int iterations) {
  // context
  int max_kv_tiles = (seqlen_kv + KVTile - 1) / KVTile;
  int group_heads = num_heads_q / num_heads_kv;

  // gemm descriptor
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
  using GmemTiledCopySoftmaxStore = XE_2D_U32x1x16_ST_N;

  using LayoutQ = cutlass::layout::RowMajor;
  using LayoutK = cutlass::layout::ColumnMajor;
  using LayoutV = cutlass::layout::RowMajor;
  using LayoutO = cutlass::layout::RowMajor;
  using LayoutS = cutlass::layout::RowMajor;

  cutlass::KernelHardwareInfo hw_info;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
  using EpilogueDispatchPolicy = cutlass::epilogue::IntelXeXMX16;

  using CollectiveSoftmaxEpilogue = cutlass::flash_attention::collective::FlashPagedSoftmaxEpilogue<
        EpilogueDispatchPolicy, ElementAccumulator,
        cutlass::gemm::TagToStrideC_t<LayoutS>,
        GmemTiledCopySoftmaxStore>;
  using CollectiveEpilogue = cutlass::flash_attention::collective::FlashPagedEpilogue<
        EpilogueDispatchPolicy, MMAOperation, TileShapeOut, TileShapeDebug, SubgroupLayout, ElementAccumulator, ElementOutput, cutlass::gemm::TagToStrideC_t<LayoutO>,
        GmemTiledCopyStore>;

  using ProblemShapeType = cute::tuple<int, int, int, int, int, int, int, int, int>;
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
  
  ProblemShapeType problem_shape = cute::make_tuple(
      num_heads_q, num_heads_kv,
      seqlen_q, seqlen_kv,
      num_blocks, block_size, head_size,
      max_kv_tiles, group_heads);
  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideV = typename FMHAKernel::StrideV;
  using StrideO = typename FMHAKernel::StrideO;
  using StrideS = typename FMHAKernel::StrideS;
  using StrideE = typename FMHAKernel::StrideE;

  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  StrideS stride_S;
  StrideE stride_E;

  stride_Q = cutlass::make_cute_packed_stride(StrideQ{}, cute::make_shape(num_heads_q, head_size, seqlen_q));
  stride_K = cutlass::make_cute_packed_stride(StrideK{}, cute::make_shape(seqlen_kv * num_heads_kv, head_size, 1));
  stride_V = cutlass::make_cute_packed_stride(StrideV{}, cute::make_shape(head_size, seqlen_kv * num_heads_kv, 1));
  stride_O = cutlass::make_cute_packed_stride(StrideO{}, cute::make_shape(num_heads_q, max_kv_tiles * head_size, seqlen_q));
  stride_S = cutlass::make_cute_packed_stride(StrideS{}, cute::make_shape(group_heads, seqlen_kv, seqlen_q * num_heads_kv));
  stride_E = cutlass::make_cute_packed_stride(StrideE{}, cute::make_shape(num_heads_q, max_kv_tiles, seqlen_q));

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
        {
          sm_scale,
          reinterpret_cast<ElementOutput*>(max_logits.data_ptr()),
          reinterpret_cast<ElementOutput*>(exp_sums.data_ptr()),
          stride_E
        },
        {
          reinterpret_cast<ElementOutput*>(temp_output.data_ptr()), stride_O,
          reinterpret_cast<ElementOutput*>(scores.data_ptr()), stride_S
        },
        hw_info};

  size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  
  // init workspace
  CUTLASS_CHECK(FMHAKernel::initialize_workspace(arguments, workspace.get()));
  auto params = FMHAKernel::to_underlying_arguments(arguments, workspace.get());

  std::cout << "iterations: " << iterations << std::endl;
  GPU_Clock timer;
  timer.start();
  float exec_time = 0;

  auto propList =
      sycl::property_list{sycl::property::queue::enable_profiling()};
  sycl::queue q{sycl::gpu_selector_v, propList};

  for (int i = 0; i < iterations; ++i) {
    // run kernel
    dim3 const block = FMHAKernel::get_block_shape();
    dim3 const grid = FMHAKernel::get_grid_shape(params);
    int smem_size = FMHAKernel::SharedStorageSize;
    const auto sycl_block = syclcompat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = syclcompat::dim3(grid.x, grid.y, grid.z);

// #if !defined(SYCL_EXT_ONEAPI_WORK_GROUP_SCRATCH_MEMORY)
//     using namespace syclcompat::experimental;
//     auto event = launch<cutlass::device_kernel<FMHAKernel>>(
//         launch_policy{sycl_grid, sycl_block, local_mem_size{static_cast<std::size_t>(smem_size)},
//                       kernel_properties{sycl_exp::sub_group_size<FMHAKernel::DispatchPolicy::SubgroupSize>}},
//         params);
// #else
    syclcompat::experimental::launch_properties launch_props {
      sycl::ext::oneapi::experimental::work_group_scratch_size(smem_size),
    };
    syclcompat::experimental::kernel_properties kernel_props{
      sycl::ext::oneapi::experimental::sub_group_size<FMHAKernel::DispatchPolicy::SubgroupSize>
    };
    syclcompat::experimental::launch_policy policy{sycl_grid, sycl_block, launch_props, kernel_props};
    auto event = syclcompat::experimental::launch<cutlass::device_kernel<FMHAKernel>>(policy, q, params);
// #endif
    EventManager::getInstance().addEvent(event);
    event.wait();
    exec_time += get_exe_time(event);
    std::cout << "get_exe_time(event): " << get_exe_time(event) << std::endl;
  }

  syclcompat::wait();
  float exec_time_2 = timer.milliseconds();

  // std::vector<int> num_pages_per_seq{0};
  // std::vector<int> cumulative_seqlen_kv_cache{0, 512, 1024};
  // int page_size = 64;
  // int num_pages = 0;
  // for(int b = 0; b < 2; b++) {
  //   int seq_len_cache = cumulative_seqlen_kv_cache[b + 1] - cumulative_seqlen_kv_cache[b];
  //   int pages_per_seq = ceil_div(seq_len_cache, page_size);
  //   num_pages_per_seq.push_back(num_pages_per_seq.back() + pages_per_seq);
  //   num_pages += pages_per_seq;
  // }

  // // initialize block table with random mapping for non-contiguous layout
  // std::vector<int> page_mapping(num_pages);
  // for (int b = 0; b < 2; ++b) {
  //   std::vector<int> physical_pages(num_pages_per_seq[b + 1] - num_pages_per_seq[b]);
  //   std::iota(physical_pages.begin(), physical_pages.end(), 0);
  //   // shuffle physical pages
  //   std::shuffle(physical_pages.begin(), physical_pages.end(), std::mt19937{ std::random_device{}() });
  //   for (int blk = 0; blk < physical_pages.size(); ++blk) {
  //     int logical_idx = num_pages_per_seq[b] + blk;
  //     page_mapping[logical_idx] = physical_pages[blk];
  //   }

  //   std::cout << "page_mapping: " << page_mapping << std::endl;
  //   std::cout << "physical_pages: " << physical_pages << std::endl;
  // }

  return exec_time;
}

std::vector<float> paged_attention(
    torch::Tensor& max_logits,
    torch::Tensor& exp_sums,
    torch::Tensor& temp_output, // same as below
    torch::Tensor& output, // currently float, TODO: change to same dtype as query
    torch::Tensor& scores,
    const torch::Tensor& query,
    const torch::Tensor& key_cache, // [num_blocks, block_size, num_heads_kv, head_size]
    const torch::Tensor& value_cache,
    const torch::Tensor& block_tables,
    const double sm_scale,
    const int block_size,
    const int max_context_len,
    const int iterations) {

  int seqlen_q = query.size(0);
  int num_heads_q = query.size(1);
  int head_size = query.size(2);
  int num_blocks = key_cache.size(0);
  int num_heads_kv = key_cache.size(2);
  int max_num_partitions = max_logits.size(2);
  int max_blocks_per_seq = block_tables.size(1);
    
  float exec_time = run_gemm<512, 8>(
      max_logits, exp_sums, temp_output,
      output, scores,
      query, key_cache, value_cache, block_tables,
      sm_scale,
      num_heads_q, num_heads_kv, seqlen_q, max_context_len,
      num_blocks, block_size, max_blocks_per_seq,
      head_size, iterations);
  return {exec_time};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &paged_attention, "Paged Attention");
}