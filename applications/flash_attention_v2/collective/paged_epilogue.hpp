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
/*! \file
  \brief Functor performing elementwise operations used by epilogues.
*/

#pragma once

#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/detail/layout.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace flash_attention {
namespace collective {

template <class DispatchPolicy, class MMAOperation_, class TileShapeOutput_, class TileShapeDebug_, class SubgroupLayout_, class... Args> class FlashPagedEpilogue {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy>, "Could not find an epilogue specialization.");
};

template <class MMAOperation_, class TileShapeOutput_, class TileShapeDebug_, class SubgroupLayout_,
          class ElementCompute_, class ElementO_, class StrideO_,
          // class ElementLSE_,
          class CopyOpO_>
class FlashPagedEpilogue<epilogue::IntelXeXMX16,
                          MMAOperation_, TileShapeOutput_, TileShapeDebug_,
                          SubgroupLayout_,
                          ElementCompute_, ElementO_, StrideO_,
                          //ElementLSE_,
                          CopyOpO_> {
public:
  using DispatchPolicy = epilogue::IntelXeXMX16;
  using ElementO = ElementO_;
  using StrideO = StrideO_;
  // using ElementLSE = ElementLSE_;
  using CopyOpO = CopyOpO_;
  using SubgroupLayout = SubgroupLayout_;
  using TileShapeOutput = TileShapeOutput_;
  using TileShapeDebug = TileShapeDebug_;
  using TiledMmaOutput = typename TiledMMAHelper<MMA_Atom<MMAOperation_>, Layout<TileShapeOutput>, SubgroupLayout>::TiledMMA;
  using GmemTiledCopyO = CopyOpO;
  using ElementOutput = ElementO_;
  using ElementCompute = ElementCompute_;
  using ElementAccumulator = ElementCompute_;

  static constexpr int ATOM_M = get<1>(typename TiledMmaOutput::ThrLayoutVMNK{}.shape());
  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  using SubgroupTileShape = decltype(make_shape(get<0>(TileShapeOutput{}), Int<get<1>(TileShapeOutput{}) / ATOM_M>{}, get<2>(TileShapeOutput{})));
  using SubgroupTileShapeDebug = decltype(make_shape(get<0>(TileShapeDebug{}), Int<get<1>(TileShapeDebug{}) / ATOM_M>{}, get<2>(TileShapeDebug{})));
  // using SubgroupTileShape = decltype(cute::shape_div(TileShapeOutput{}, (SubgroupLayout{}.shape())));

  static_assert(cute::rank(TileShapeOutput{}) == 3, "TileShapeOutput must be rank-3: [CTA_M_QO, CTA_N_VO, CTA_K_PV]");
  static_assert(cute::rank(StrideO{}) == 3, "StrideO must be rank-3: [seq_len_qo, head_size_vo, batch * num_heads]");

  using CopyThreadShape = Shape<_1, Int<SubgroupSize>>;

  using traits_store_O = Copy_Traits<GmemTiledCopyO, StrideO>;
  using atom_load_O = Copy_Atom<traits_store_O, ElementO>;
  using val_layout_load_O = decltype(make_layout(shape_div(typename traits_store_O::BlockShape{}, CopyThreadShape{})));
  using XE_Copy_O = decltype(make_tiled_copy(atom_load_O{}, Layout<CopyThreadShape>{}, val_layout_load_O{}));

private:
  constexpr static bool is_destination_supported = not cute::is_void_v<ElementO>;

public:
  using EmptyType = cute::tuple<>;

  struct TensorStorageImpl : cute::tuple<EmptyType, EmptyType> {};

  struct SharedStorage {
    using TensorStorage = TensorStorageImpl;

    TensorStorage tensors;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;

  // Host side epilogue arguments
  struct Arguments {
    ElementO const *ptr_O;
    StrideO dO;
    ElementO const *ptr_S;
    StrideO dS;
  };

  // Device side epilogue params
  struct Params {
    XE_Copy_O xe_store_o;
    XE_Copy_O xe_store_s;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const &problem_shape, Arguments const &args,
                                                  [[maybe_unused]] void *workspace) {
    auto [num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, num_block, block_size, head_size, max_kv_tiles, group_heads] = problem_shape;

    auto tensorO = make_tensor(make_gmem_ptr(static_cast<ElementO const*>(args.ptr_O)), 
                                             make_layout(make_shape(num_heads_q, max_kv_tiles * head_size, seq_len_qo),
                                             args.dO));
    auto tensorS = make_tensor(make_gmem_ptr(static_cast<ElementO const*>(args.ptr_S)), 
                                             make_layout(make_shape(group_heads, seq_len_kv, seq_len_qo * num_heads_kv),
                                             args.dS));
    XE_Copy_O xe_store_o{XE_Copy_O{}.with(tensorO)};
    XE_Copy_O xe_store_s{XE_Copy_O{}.with(tensorS)};
    return {
        xe_store_o, xe_store_s
    };
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const &problem_shape, Arguments const &args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const &problem_shape, Arguments const &args, void *workspace,
                                              cudaStream_t stream, CudaHostAdapter *cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool can_implement(ProblemShape const &problem_shape,
                                                [[maybe_unused]] Arguments const &args) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  FlashPagedEpilogue(Params const &params_, TensorStorage const &) : params(params_) {}

  template <class ProblemShape, class TileCoord, class FragOut>
  CUTLASS_DEVICE void store_O(ProblemShape problem_shape, TileCoord tile_coord, FragOut &out) {

    using namespace cute;
    using FragOutLayout = typename FragOut::layout_type;
    constexpr int Vec    = shape<0>(FragOutLayout{});
    constexpr int FragsM = shape<1>(FragOutLayout{});
    constexpr int FragsN = shape<2>(FragOutLayout{});

    auto out_reg = make_tensor(static_cast<decltype(out) &&>(out).data() , Shape<Int<Vec>, Int<FragsM>, Int<FragsN>>{});

    // tile the output ptr
    auto [m_coord, n_coord, l_coord] = tile_coord;
    auto [num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, num_block, block_size, head_size, max_kv_tiles, group_heads] = problem_shape;
    // tile for wg
    Tensor mO_mnl = cute::get_xe_tensor(make_shape(num_heads_q, max_kv_tiles * head_size, seq_len_q));
    Tensor mO_mn = mO_mnl(_, _, l_coord);
    Tensor g_wg_O = local_tile(mO_mn, select<0, 1>(TileShapeOutput{}), make_coord(m_coord, n_coord));
    // tile for sg
    const int m_sg = 0;
    const int n_sg = get_sub_group_id();
    // Tile the output tensor per SG
    Tensor gO = local_tile(g_wg_O, SubgroupTileShape{}, make_coord(m_sg, n_sg, _), Step<_1,_1, X>{});
    
    auto thread_xe_store_o = params.xe_store_o.get_thread_slice(ThreadIdxX());
    Tensor tOgO = thread_xe_store_o.partition_D(gO);

    copy(params.xe_store_o, out_reg, tOgO);
// #define PRINT(x) print(#x ": "); print(x); print("\n");
//   if (cute::thread(0, 0)) {
//     print("======================= S: \n");
//     // PRINT(out_reg);
//     PRINT(mO_mn);
//     PRINT(g_wg_O);
//     PRINT(gO);
//     PRINT(tOgO);
//     PRINT(thread_xe_store_o);
//     PRINT(l_coord);
//     print_tensor(out_reg);
//   }
// #undef PRINT
  }

  template <class ProblemShape, class TileCoord, class FragOut>
  CUTLASS_DEVICE void debug_store_S(ProblemShape problem_shape, TileCoord tile_coord, FragOut &score) {

    using namespace cute;
    using FragOutLayout = typename FragOut::layout_type;
    constexpr int Vec    = shape<0>(FragOutLayout{});
    constexpr int FragsM = shape<1>(FragOutLayout{});
    constexpr int FragsN = shape<2>(FragOutLayout{});

    auto out_reg = make_tensor(static_cast<decltype(score) &&>(score).data() , Shape<Int<Vec>, Int<FragsM>, Int<FragsN>>{});

    // tile the output ptr
    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord;
    auto [num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, num_block, block_size, head_size, max_kv_tiles, group_heads] = problem_shape;
    // tile for wg
    Tensor mS_mnl = cute::get_xe_tensor(make_shape(group_heads, seq_len_kv, seq_len_q * num_heads_kv));
    Tensor mS_mn = mS_mnl(_, _, l_coord);
    Tensor g_wg_S = local_tile(mS_mn, select<0, 1>(TileShapeDebug{}), make_coord(m_coord, n_coord));
    // tile for sg
    const int m_sg = 0;
    const int n_sg = get_sub_group_id();
    // Tile the output tensor per SG
    Tensor gS = local_tile(g_wg_S, SubgroupTileShapeDebug{}, make_coord(m_sg, n_sg, _), Step<_1,_1, X>{});
    // if(cute::thread(16, 0)) {
    //   print("n_sg: "); print(n_sg); print("\n");
    // }
    
    auto thread_xe_store_s = params.xe_store_s.get_thread_slice(ThreadIdxX());
    Tensor tSgS = thread_xe_store_s.partition_D(gS);
    
    copy(params.xe_store_s, out_reg, tSgS);
    // copy(xe_store_smem, out_reg, tSgS);
// #define PRINT(x) print(#x ": "); print(x); print("\n");
//   if (cute::thread(0, 1)) {
//     print("======================= S: \n");
//     PRINT(out_reg);
//     PRINT(mS_mn);
//     PRINT(g_wg_S);
//     PRINT(gS);
//     PRINT(tSgS);
//     PRINT(thread_xe_store_o);
//     PRINT(l_coord);
//   }
// #undef PRINT
  }

private:
  Params const &params;
};

}
}
}