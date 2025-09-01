/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"

#include "flash_attention_v2/collective/xe_paged_mma.hpp"

namespace cutlass::flash_attention::kernel {

template <class ProblemShape,
  class CollectiveMainloop,
  class CollectiveSoftmaxEpilogue,
  class CollectiveEpilogue,
  class TileScheduler_ = void>
class FMHAPaged;

///////////////////////////////////////////////////////////////////////////////

template <class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveSoftmaxEpilogue_,
  class CollectiveEpilogue_,
  class TileScheduler_>
class FMHAPaged {

public:
  using ProblemShape = ProblemShape_;
  
  static_assert(rank(ProblemShape{}) == 9, "ProblemShape{} should be <num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, num_block, block_size, head_size, max_kv_tiles, group_heads>");

  // For Mainloop Gemm
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK; // <8, 512, 64>
  using TileShapePV = typename CollectiveMainloop::TileShapePV; // <8, 128, 64>
  using TiledMmaQK = typename CollectiveMainloop::TiledMmaQK;
  using TiledMmaPV = typename CollectiveMainloop::TiledMmaPV;
  using ArchTag = typename CollectiveMainloop::ArchTag;
  using ElementQ = typename CollectiveMainloop::ElementQ;
  using StrideQ = typename CollectiveMainloop::StrideQ;
  using ElementK = typename CollectiveMainloop::ElementK;
  using StrideK = typename CollectiveMainloop::StrideK;
  using ElementV = typename CollectiveMainloop::ElementV;
  using StrideV = typename CollectiveMainloop::StrideV;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using CollectiveSoftmaxEpilogue = CollectiveSoftmaxEpilogue_;
  using SoftmaxArguments = typename CollectiveSoftmaxEpilogue::Arguments;
  using SoftmaxParams = typename CollectiveSoftmaxEpilogue::Params;
  using StrideE = typename CollectiveSoftmaxEpilogue::Stride;

  static_assert(cute::is_void_v<TileScheduler_> or cute::is_same_v<TileScheduler_, PagedIndividualScheduler>,
                "Unsupported TileScheduler for Intel PVC.");
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler =
      typename detail::TileSchedulerSelector<TileScheduler_, ArchTag>::Scheduler;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  // Debug: Temp Stride for scores
  using StrideS = typename CollectiveEpilogue::StrideO;

  using ElementO = typename CollectiveEpilogue::ElementO;
  using StrideO = typename CollectiveEpilogue::StrideO;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  using TiledMmaOutput = typename CollectiveEpilogue::TiledMmaOutput;
  static_assert(cute::is_same_v<ElementAccumulator, typename CollectiveEpilogue::ElementAccumulator>,
                "Mainloop and epilogue do not agree on accumulator value type.");

  static constexpr int SharedStorageSize = 0;

  static constexpr int SubgroupSize = CollectiveMainloop::SubgroupSize; // sub_group size
  static constexpr uint32_t MaxThreadsPerBlock = CollectiveMainloop::MaxThreadsPerBlock;
  using MmaAtomShape = typename CollectiveMainloop::MmaAtomShape;           // 8x16x16

  static constexpr int QK_BLK_M = CollectiveMainloop::QK_BLK_M; // 8
  static constexpr int QK_BLK_N = CollectiveMainloop::QK_BLK_N; // 512
  static constexpr int QK_BLK_K = CollectiveMainloop::QK_BLK_K; // 64

  static constexpr int PV_BLK_M = CollectiveMainloop::PV_BLK_M; // 8
  static constexpr int PV_BLK_N = CollectiveMainloop::PV_BLK_N; // 128
  static constexpr int PV_BLK_K = CollectiveMainloop::PV_BLK_K; // 512

  using SubgroupTileShapeQK = typename CollectiveMainloop::SubgroupTileShapeQK;
  using SubgroupTileShapePV = typename CollectiveMainloop::SubgroupTileShapePV;
  static constexpr int QK_SG_M = CollectiveMainloop::QK_SG_M; // 8
  static constexpr int QK_SG_N = CollectiveMainloop::QK_SG_N; // 64

  static constexpr int ATOM_M = CollectiveMainloop::ATOM_M; // 8
  static constexpr int ATOM_N = CollectiveMainloop::ATOM_N; // 1
  static constexpr int ATOM_K = CollectiveMainloop::ATOM_K; // 1

  static constexpr auto Num_SGs = ATOM_M * ATOM_N * ATOM_K; // 8
  static constexpr int Vec = CollectiveMainloop::Vec; // 8
  static constexpr int FragsM_S = CollectiveMainloop::FragsM_S;  // 1
  static constexpr int FragsN_S = CollectiveMainloop::FragsN_S; // 4
  static constexpr int FragsM_O = CollectiveMainloop::FragsM_O;  // 1
  static constexpr int FragsN_O = CollectiveMainloop::FragsN_O; // 1
  static constexpr int FragsM_P = CollectiveMainloop::FragsM_P;  // 1
  static constexpr int FragsN_P = CollectiveMainloop::FragsN_P; // 4

  using tScoreShape = decltype(make_shape(Int<Vec>{}, Int<FragsM_S>{}, Int<FragsN_S>{}));
  using tOutShape = decltype(make_shape(Int<Vec>{}, Int<FragsM_O>{}, Int<FragsN_O>{}));
  using tProbShape = decltype(make_shape(Int<Vec>{}, Int<FragsM_P>{}, Int<FragsN_P>{}));

  static_assert(FragsM_S == 1, "Limit the seq_len_qo to 1 MMA Atom worth of data per work-group.");
  static_assert(Vec == 8, "one thread in sub-group must compute 8x1x4 items");
  static_assert(FragsN_O == 1, "FragsN_O");
  static_assert(FragsN_P == 4, "FragsN_P");

  struct SharedStorage {
    using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;
    EpilogueTensorStorage epilogue;
  };

  // Device side arguments
  struct Arguments {
    gemm::GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    SoftmaxArguments softmax{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
  };

  // Kernel entry point API
  struct Params {
    gemm::GemmUniversalMode mode;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    SoftmaxParams softmax;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static Params to_underlying_arguments(Arguments const &args, void *workspace) {
    (void)workspace;
    return {args.mode, args.problem_shape,
            CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
            CollectiveSoftmaxEpilogue::to_underlying_arguments(args.problem_shape, args.softmax),
            CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace),
            TileScheduler::to_underlying_arguments(args.problem_shape, args.hw_info)};
  }

  static bool can_implement(Arguments const &args) {
    bool mode_implementable = args.mode == gemm::GemmUniversalMode::kGemm;
    bool valid_block_size = args.mainloop.block_size >= QK_SG_N && args.mainloop.block_size % QK_SG_N == 0;
    return mode_implementable && valid_block_size;
  }

  static int get_workspace_size(Arguments const &args) { return 0; }

  static cutlass::Status initialize_workspace(Arguments const &args, void *workspace = nullptr,
                                              cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const &params) {
    return TileScheduler::template get_grid_shape<Num_SGs>(params.scheduler);
  }

  static dim3 get_block_shape() { return dim3(MaxThreadsPerBlock, 1, 1); }

  CUTLASS_DEVICE
  void operator()(Params const &params, char *smem_buf) {
    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShapeQK>::value);
    CUTE_STATIC_ASSERT(is_static<TileShapePV>::value);
    // static_assert(cute::rank(StrideQ{}) == 3, "StrideQ must be rank-3: [seq_len_qo, head_size_qk, batch * num_heads_q].");
    // static_assert(cute::rank(StrideK{}) == 3, "StrideK must be rank-3: [head_size_qk, seq_len_kv, batch * num_heads_kv].");
    // static_assert(cute::rank(StrideV{}) == 3, "StrideV must be rank-3: [seq_len_kv, head_size_vo, batch * num_heads_kv].");

    SharedStorage &shared_storage = *reinterpret_cast<SharedStorage *>(smem_buf);
    // Separate out problem shape for convenience
    auto [num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, num_block, block_size, head_size, max_kv_tiles, group_heads] = params.problem_shape;

    // context
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto wg = syclcompat::get_nd_item<1>().get_group();
    const int sg_local_id = sg.get_local_id()[0];
    int thread_idx = int(ThreadIdxX());
    int sg_id = thread_idx / SubgroupSize;

    const int num_blocks_per_wg = ceil_div(QK_BLK_N, block_size); // 8

    TileScheduler tile_scheduler{params.scheduler};

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      /* scheduler: <seq(bs), 1, kv_heads> */
      auto blk_coord = tile_scheduler.get_block_coord();
      auto seq_coord = get<0>(blk_coord);
      auto context_coord = get<1>(blk_coord);
      auto heads_kv_coord = get<2>(blk_coord);

      /* tiling for current block */
      Tensor mQ_mkl = cute::get_xe_tensor(make_shape(num_heads_q, head_size, seq_len_q));
      Tensor mK_nkl = cute::get_xe_tensor(make_shape(QK_BLK_N, head_size, 1));
      Tensor mV_nkl = cute::get_xe_tensor(make_shape(head_size, QK_BLK_N, 1));

      Tensor mQ_mk = mQ_mkl(_, _, seq_coord);
      Tensor mK_nk = mK_nkl(_, _, 0);
      Tensor mV_nk = mV_nkl(_, _, 0); // 128x512

      auto gQ = local_tile(mQ_mk, TileShapeQK{}, make_coord(heads_kv_coord, _, _), Step<_1, X, _1>{});
      auto gK = local_tile(mK_nk, TileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
      auto gV = local_tile(mV_nk, TileShapePV{}, make_coord(_, _, _), Step<X, _1, _1>{}); // <16, 64, 8, 8>

      auto mainloop_params = params.mainloop;
      auto softmax_params = params.softmax;

      /* prefetch config */
      // auto gK_prefetch = local_tile(mK_nk, SubgroupTileShapeQK{}, make_coord(_, _, _), Step<X, _1, _1>{});
      // auto tiled_prefetch_q = cute::prefetch_selector<Shape<Int<QK_BLK_M * ATOM_M>, Int<QK_BLK_K>>, Num_SGs>(mainloop_params.gmem_tiled_copy_q);
      // auto tiled_prefetch_k = cute::prefetch_selector<decltype(take<1,3>(SubgroupTileShapeQK{})), Num_SGs>(mainloop_params.gmem_tiled_copy_k);

      // auto thr_prefetch_Q = tiled_prefetch_q.get_slice(thread_idx);
      // auto thr_prefetch_K = tiled_prefetch_k.get_slice(thread_idx);
      // auto pQgQ = thr_prefetch_Q.partition_S(gQ);
      // auto pKgK = thr_prefetch_K.partition_S(gK_prefetch);

      /* kv context */
      const int kv_splits = ceil_div(seq_len_kv, QK_BLK_N);
      auto current_block_table = mainloop_params.block_table + seq_coord * mainloop_params.max_blocks_per_seq;
      int kv_start_id = sg_id;
      int v_tile_count = ceil_div(seq_len_kv, PV_BLK_K);

      // CUTLASS_PRAGMA_UNROLL
      // for (int i = 0; i < size<3>(pQgQ); i++) {
      //   prefetch(tiled_prefetch_q, pQgQ(_, _, _, i));
      // }

      // CUTLASS_PRAGMA_UNROLL
      // for (int j = 0; j < size<4>(pKgK); j++) {
      //   prefetch(tiled_prefetch_k, pKgK(_, _, _, current_block_table[kv_start_id], j));
      // }

      CollectiveMainloop collective_mma;
      CollectiveSoftmaxEpilogue softmax;
      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};

      auto smem = syclcompat::local_mem<ElementAccumulator[((Int<size(tScoreShape{})>{}) * Num_SGs * SubgroupSize)]>();
      auto s_shape = make_shape(QK_BLK_M, QK_BLK_N);
      Tensor score_blk = make_tensor(make_smem_ptr(smem), s_shape); // 8x512
      Tensor store_score_warp = local_tile(score_blk, select<0, 1>(SubgroupTileShapeQK{}), make_coord(_, sg_id)); // 8x64
      TiledMmaQK tiled_mma_QK;
      auto thread_mma_QK = tiled_mma_QK.get_slice(thread_idx);
      auto t_store_score = thread_mma_QK.partition_C(score_blk); // 8x4

      Tensor load_score_warp = local_tile(score_blk, make_shape(1, QK_BLK_N), make_coord(sg_id, 0)); // 1x512
      Tensor t_load_score = local_tile(load_score_warp, make_shape(1, ceil_div(QK_BLK_N, SubgroupSize)), make_coord(_, sg_local_id)); // 1x32

      ElementAccumulator max_reg = ElementAccumulator{-INFINITY};
      auto sum_reg = ElementAccumulator{0};

      int kv_id = kv_start_id + context_coord * num_blocks_per_wg;
      int block_id_k = current_block_table[kv_id];

      mainloop_params = CollectiveMainloop::get_updated_copies_K(params.mainloop, params.problem_shape, heads_kv_coord, block_id_k);

      // auto softmax_coord = make_coord(seq_coord, heads_kv_coord, context_coord);
      // softmax_params =  CollectiveSoftmaxEpilogue::get_updated_copies(params.softmax, params.problem_shape, softmax_coord);
      // auto store_max = make_tensor(make_gmem_ptr(softmax_params.ptr_max + softmax_params.offset), 
      //                               make_layout(make_shape(num_heads_q, max_kv_tiles, seq_len_q),
      //                               softmax_params.dS));
      // auto store_sum = make_tensor(make_gmem_ptr(softmax_params.ptr_sum + softmax_params.offset), 
      //                               make_layout(make_shape(num_heads_q, max_kv_tiles, seq_len_q),
      //                               softmax_params.dS));

      /* thread acc register*/
      Tensor tSr = make_tensor<ElementAccumulator>(tScoreShape{});
      clear(tSr);

      // Perform GEMM S = Q*K
      collective_mma.mmaQK(tSr, gQ, gK(_, _, kv_id / ATOM_M, _), tSr, ceil_div(head_size, QK_BLK_K), mainloop_params);

      // Debug: Store S out
      // auto blk_coord_debug = make_coord(0, split, _, seq_coord * num_heads_kv + heads_kv_coord); // <group_heads, kv_lens, q_lens * heads_kv>
      // epilogue.debug_store_S(params.problem_shape, blk_coord_debug, tSr);

      // softmax
      // softmax.template operator()<Num_SGs>(softmax_params, tSr, max_reg, sum_reg, t_store_score, t_load_score, store_max, store_sum);
      

      // Perform GEMM O = P*V
      Tensor tPr = make_tensor<ElementAccumulator>(tProbShape{});
      Tensor tOr = make_tensor<ElementAccumulator>(tOutShape{});
      clear(tOr);

      // Tensor gPs = local_tile(score_blk, select<0, 2>(TileShapePV{}), make_coord(0, _)); // 8x64
      // CUTLASS_PRAGMA_UNROLL
      // for (int v = 0; v < v_tile_count; v++) {
      //   int block_id_v = current_block_table[v];
      //   mainloop_params = CollectiveMainloop::get_updated_copies_V(params.mainloop, params.problem_shape, heads_kv_coord, block_id_v);
      //   collective_mma.mmaPV(tOr, gPs(_, _, v), tPr, gV(_, _, sg_id, 0), tOr, mainloop_params, v);
      // }

      // if (cute::thread(0, 0)) {
      //   print_tensor(tOr);
      //   print_tensor(load_score_warp);
      // }

      // Store out
      // CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
      // auto blk_coord_O = make_coord(heads_kv_coord, context_coord, seq_coord); // <num_heads_q, head_size, q_lens>
      // epilogue.store_O(params.problem_shape, blk_coord_O, tOr);

      // if(cute::thread(0, 0)) {
      //   print("tScoreShape: "); print(Int<size(tScoreShape{})>{}); print("\n");
      //   print_tensor(store_score_warp);
      //   print_tensor(t_store_score);
      //   print_tensor(tSr);
      //   print_tensor(t_load_score);
      //   print("load_score_warp: "); print(load_score_warp); print("\n");
      //   print_tensor(load_score_warp);
      // }
    }
  }
};

}