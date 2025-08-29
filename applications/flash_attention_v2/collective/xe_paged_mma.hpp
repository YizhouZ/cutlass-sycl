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

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/tensor_predicate.hpp"
#include "cutlass/fp8_to_fp16.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::flash_attention::collective {
using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
CUTLASS_DEVICE auto convert_type(Tensor<Engine, Layout> const &tensor) {
  using From_type = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class DispatchPolicy, class ProblemShapeType_,
          class ElementQ_, class StrideQ_,
          class ElementK_, class StrideK_,
          class ElementV_, class StrideV_,
          class MMAOp_,
          class TileShapeQK_, class TileShapePV_,
          class SubgroupLayout_,
          class GmemTiledCopyQ_, class GmemTiledCopyK_, class GmemTiledCopyV_>
struct PagedMma {
  static_assert(cutlass::detail::dependent_false<ElementQ_>, "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <int Stages, class ProblemShapeType_,
          class ElementQ_, class StrideQ_,
          class ElementK_, class StrideK_,
          class ElementV_, class StrideV_,
          class MMAOp_,
          class TileShapeQK_, class TileShapePV_,
          class SubgroupLayout_,
          class GmemTiledCopyQ_, class GmemTiledCopyK_, class GmemTiledCopyV_>
struct PagedMma<gemm::MainloopIntelXeXMX16<Stages>, ProblemShapeType_,
                ElementQ_, StrideQ_,
                ElementK_, StrideK_,
                ElementV_, StrideV_,
                MMAOp_, TileShapeQK_, TileShapePV_,
                SubgroupLayout_,
                GmemTiledCopyQ_, GmemTiledCopyK_, GmemTiledCopyV_> {
  //
  // Type Aliases
  //
  using DispatchPolicy = gemm::MainloopIntelXeXMX16<Stages>;
  using TileShapeQK = TileShapeQK_; // <8, 512, 64>
  using TileShapePV = TileShapePV_; // <8, 128, 64>
  using SubgroupLayout = SubgroupLayout_;
  using ProblemShapeType = ProblemShapeType_;
  using ElementQ = ElementQ_;
  using StrideQ = StrideQ_;
  using ElementK = ElementK_;
  using StrideK = StrideK_;
  using ElementV = ElementV_;
  using StrideV = StrideV_;
  using GmemTiledCopyQ = GmemTiledCopyQ_;
  using GmemTiledCopyK = GmemTiledCopyK_;
  using GmemTiledCopyV = GmemTiledCopyV_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  using MmaAtom = MMA_Atom<MMAOp_>;
  static constexpr auto ATOM_M = decltype(get<0>(SubgroupLayout{}.shape()))::value; //8
  static constexpr auto ATOM_N = decltype(get<1>(SubgroupLayout{}.shape()))::value; //1
  static constexpr auto ATOM_K = decltype(get<2>(SubgroupLayout{}.shape()))::value; //1

  using QK_Subgroup_Layout = Layout<Shape<_1, Int<ATOM_M>, _1>, Stride<_1, Int<ATOM_N>, _1>>;
  using PV_Subgroup_Layout = Layout<Shape<_1, Int<ATOM_M>, _1>, Stride<_1, Int<ATOM_N>, _1>>;

  using TiledMmaQK = typename TiledMMAHelper<MmaAtom, Layout<TileShapeQK>, QK_Subgroup_Layout>::TiledMMA;
  using TiledMmaPV = typename TiledMMAHelper<MmaAtom, Layout<TileShapePV>, PV_Subgroup_Layout>::TiledMMA;
  using ElementAccumulator = typename TiledMmaQK::ValTypeC;
  using ValTypeC = typename TiledMmaQK::ValTypeC;

  using MmaAtomShape = typename MmaAtom::Shape_MNK;

  static constexpr auto QK_BLK_M = get<0>(TileShapeQK{}); // 8
  static constexpr auto QK_BLK_N = get<1>(TileShapeQK{}); // 512
  static constexpr auto QK_BLK_K = get<2>(TileShapeQK{}); // 64

  static constexpr auto PV_BLK_M = get<0>(TileShapePV{}); // 8
  static constexpr auto PV_BLK_N = get<1>(TileShapePV{}); // 128
  static constexpr auto PV_BLK_K = get<2>(TileShapePV{}); // 64

  // (8, 64, 64)
  using SubgroupTileShapeQK = decltype(make_shape(get<0>(TileShapeQK{}), Int<get<1>(TileShapeQK{}) / ATOM_M>{}, get<2>(TileShapeQK{})));
  using FragsShapeS = decltype(cute::shape_div(take<0, 2>(SubgroupTileShapeQK{}), take<0, 2>(MmaAtomShape())));
  static constexpr int Vec = (get<0>(MmaAtomShape()) * get<1>(MmaAtomShape())) / SubgroupSize; // 8
  static constexpr int FragsM_S = get<0>(FragsShapeS{}); // 1
  static constexpr int FragsN_S = get<1>(FragsShapeS{}); // 4

  static constexpr auto QK_SG_M = get<0>(SubgroupTileShapeQK{}); // 8
  static constexpr auto QK_SG_N = get<1>(SubgroupTileShapeQK{}); // 64
  static constexpr auto QK_SG_K = get<2>(SubgroupTileShapeQK{}); // 64

  // (8, 16, 64)
  using SubgroupTileShapePV = decltype(make_shape(get<0>(TileShapePV{}), Int<get<1>(TileShapePV{}) / ATOM_M>{}, get<2>(TileShapePV{})));
  using FragsShapeO = decltype(cute::shape_div(take<0, 2>(SubgroupTileShapePV{}), take<0, 2>(MmaAtomShape())));
  static constexpr int FragsM_O = get<0>(FragsShapeO{}); // 1
  static constexpr int FragsN_O = get<1>(FragsShapeO{}); // 1

  static constexpr auto PV_SG_M = get<0>(SubgroupTileShapePV{}); // 8
  static constexpr auto PV_SG_N = get<1>(SubgroupTileShapePV{}); // 16
  static constexpr auto PV_SG_K = get<2>(SubgroupTileShapePV{}); // 64
  
  // (8, 64) / (8, 16)
  using FragsShapeP = decltype(cute::shape_div(select<0, 2>(SubgroupTileShapePV{}), select<0, 2>(MmaAtomShape())));
  static constexpr int FragsM_P = get<0>(FragsShapeP{}); // 1
  static constexpr int FragsN_P = get<1>(FragsShapeP{}); // 4
  

  static constexpr uint32_t MaxThreadsPerBlock = size(SubgroupLayout{}) * SubgroupSize;
  using CopyThreadShape = Shape<_1, Int<SubgroupSize>>;

  using traits_load_Q = Copy_Traits<GmemTiledCopyQ, StrideQ>;
  using atom_load_Q = Copy_Atom<traits_load_Q, ElementQ>;
  using val_layout_load_Q = decltype(make_layout(shape_div(typename traits_load_Q::BlockShape{}, CopyThreadShape{})));
  using XE_Copy_Q = decltype(make_tiled_copy(atom_load_Q{}, Layout<CopyThreadShape>{}, val_layout_load_Q{}));

  using traits_load_K = Copy_Traits<GmemTiledCopyK, StrideK>;
  using atom_load_K = Copy_Atom<traits_load_K, ElementK>;
  using val_layout_load_K = decltype(make_layout(shape_div(typename traits_load_K::BlockShape{}, CopyThreadShape{})));
  using XE_Copy_K = decltype(make_tiled_copy(atom_load_K{}, Layout<CopyThreadShape>{}, val_layout_load_K{}));

  using traits_load_V = Copy_Traits<GmemTiledCopyV, StrideV>;
  using atom_load_V = Copy_Atom<traits_load_V, ElementV>;
  using val_layout_load_V = decltype(make_layout(shape_div(typename traits_load_V::BlockShape{}, CopyThreadShape{})));
  using XE_Copy_V = decltype(make_tiled_copy(atom_load_V{}, Layout<CopyThreadShape>{}, val_layout_load_V{}));

  // Host side kernel arguments
  struct Arguments {
    ElementQ const *ptr_Q;
    StrideQ strideQ;
    ElementK const *ptr_K;
    StrideK strideK;
    ElementV const *ptr_V;
    StrideV strideV;

    // Block table
    int const* block_table; // [num_seq_q, max_blocks_per_seq]
    int block_size;
    int max_blocks_per_seq;
  };

  struct Params {
    XE_Copy_Q gmem_tiled_copy_q;
    XE_Copy_K gmem_tiled_copy_k;
    XE_Copy_V gmem_tiled_copy_v;
  
    // Block table
    int const* block_table; // [num_seq_q, max_blocks_per_seq]
    int block_size;
    int max_blocks_per_seq;
  };

  //
  // Methods
  //

  PagedMma() = default;

  static constexpr Params to_underlying_arguments(ProblemShapeType const &problem_shape, Arguments const &args,
                                                  void *workspace) {
    (void)workspace;

    auto [num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, num_block, block_size, head_size, max_kv_tiles, group_heads] = problem_shape;

    auto tensorQ = make_tensor(make_gmem_ptr(args.ptr_Q), make_layout(make_shape(num_heads_q, head_size, seq_len_q), args.strideQ));
    auto tensorK = make_tensor(make_gmem_ptr(args.ptr_K), make_layout(make_shape(seq_len_kv, num_heads_kv * head_size, 1), args.strideK));
    auto tensorV = make_tensor(make_gmem_ptr(args.ptr_V), make_layout(make_shape(num_heads_kv * head_size, seq_len_kv, 1), args.strideV));

    XE_Copy_Q copyQ{XE_Copy_Q{}.with(tensorQ)};
    XE_Copy_K copyK{XE_Copy_K{}.with(tensorK)};
    XE_Copy_V copyV{XE_Copy_V{}.with(tensorV)};
  
    return Params{copyQ, copyK, copyV, args.block_table, args.block_size, args.max_blocks_per_seq};
  }

  template <class FragAccum, class TensorQ, class TensorK, class FragSrc>
  CUTLASS_DEVICE void mmaQK(FragAccum &accum, TensorQ gQ, TensorK gK, FragSrc const &frag_src,
                            int const &k_tile_count, Params const &params) {
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    const int sg_local_id = sg.get_local_id()[0];

    int thread_idx = static_cast<int>(ThreadIdxX());
    auto thr_copy_Q = params.gmem_tiled_copy_q.get_slice(thread_idx);
    auto thr_copy_K = params.gmem_tiled_copy_k.get_slice(sg_local_id);
    // Instantiate the MMA object
    TiledMmaQK tiled_mma;
    auto first_thread_in_sg_idx = sg.get_group_id()[0] * SubgroupSize;

    auto thread_mma_k = tiled_mma.get_slice(0);
    auto thread_mma_q = tiled_mma.get_slice(first_thread_in_sg_idx);

    // Partition
    Tensor tCgQ = thread_mma_q.partition_A(gQ);
    Tensor tCgK = thread_mma_k.partition_B(gK);

    // Create fragments
    Tensor tCrQ = make_tensor<ElementQ>(make_fragment_layout(params.gmem_tiled_copy_q, take<0,3>(tCgQ.shape())));
    Tensor tCrK = make_tensor<ElementK>(make_fragment_layout(params.gmem_tiled_copy_k, take<0,3>(tCgK.shape())));

    // Retile registers for copies
    Tensor tQrQ = thr_copy_Q.retile_D(tCrQ);
    Tensor tKrK = thr_copy_K.retile_D(tCrK);

    // Retile global tile for copies
    Tensor tQgQ = thr_copy_Q.retile_S(tCgQ);
    Tensor tKgK = thr_copy_K.retile_S(tCgK);

// #define PRINT(x) print(#x ": "); print(x); print("\n");
//   if (cute::thread(0, 0)) {
//     print("======================= Q: \n");
//     PRINT(gQ);
//     PRINT(tCrQ);
//     PRINT(tCgQ);
//     PRINT(tQrQ);
//     PRINT(tQgQ);
//     PRINT(params.gmem_tiled_copy_q);

//     print("=====================  K :\n");
//     PRINT(gK);
//     PRINT(tCrK);
//     PRINT(tCgK);
//     PRINT(tKrK);
//     PRINT(tKgK);

//     print("=====================  Config: \n");
//     PRINT(MaxThreadsPerBlock);
//     PRINT(SubgroupTileShapeQK{});
//     PRINT(accum);
//     PRINT(sg.get_group_id()[0]);
//     PRINT(thread_mma_q);
//     PRINT(thread_mma_k);
//   }
// #undef PRINT

    //
    // Mainloop
    //

    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
      copy(params.gmem_tiled_copy_q, tQgQ(_,_,_,k_tile), tQrQ);
      copy(params.gmem_tiled_copy_k, tKgK(_,_,_,k_tile), tKrK);
      // if (k_tile == 0 && cute::thread(1, 0)) {
      //   print_tensor(tQrQ);
      //   print_tensor(tKrK);
      //   print(params.gmem_tiled_copy_q);
      //   print(params.gmem_tiled_copy_k);
      //   print("kv_tile_idx: "); print(kv_tile_idx); print("\n");
      // }
      cute::gemm(tiled_mma, accum, tCrQ, tCrK, frag_src);
    }
  }

  template <class FragAccum, class STensorProb, class FragP, class TensorV, class FragSrc>
  CUTLASS_DEVICE void mmaPV(FragAccum &accum, STensorProb const &gPs, FragP &tPr, TensorV gV,
                            FragSrc const &frag_src, Params const &params, const int& v_idx) {
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    const int sg_local_id = sg.get_local_id()[0];

    auto gmem_thr_copy_V = params.gmem_tiled_copy_v.get_slice(sg_local_id);

    // Instantiate the MMA object
    TiledMmaPV tiled_mma;
    auto first_thread_in_sg_idx = sg.get_group_id()[0] * SubgroupSize;
    auto thread_mma_q = tiled_mma.get_slice(first_thread_in_sg_idx);
    auto thread_mma_v = tiled_mma.get_slice(0);

    auto tPs = thread_mma_q.partition_A(gPs);
    Tensor tCgV = thread_mma_v.partition_B(gV);

    Tensor tCrV = make_tensor<ElementV>(make_fragment_layout(params.gmem_tiled_copy_v, take<0, 3>(tCgV.shape())));

    Tensor tVrV = gmem_thr_copy_V.retile_D(tCrV);
    Tensor tVgV = gmem_thr_copy_V.retile_S(tCgV);

    // Convert S to P (FP32 -> BF16)
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Int<size(FragP{})>{}; i++) {
      tPr(i) = tPs(i);
    }
    Tensor tPr_dtype = convert_type<typename TiledMmaPV::ValTypeA>(tPr);

    copy(params.gmem_tiled_copy_v, tVgV, tVrV);
    cute::gemm(tiled_mma, accum, tPr_dtype, tCrV, frag_src);

    // if (cute::thread(0, 0)) {
    //   // print("gV: "); print(gV); print("\n");
    //   print("tCgV: "); print(tCgV); print("\n");
    //   // print("params.gmem_tiled_copy_v: "); print(params.gmem_tiled_copy_v); print("\n");
    //   // print("gPs: "); print(gPs); print("\n");
    //   // print("tPs: "); print(tPs); print("\n");
    //   print_tensor(tCrV);
    //   for (int i = 0; i < Int<size(FragP{})>{}; ++i) {
    //     print("FragP: "); print(tPr(i)); print("\n");
    //   }
    // }
  }

  template <class ProblemShape>
  CUTLASS_DEVICE static constexpr Params get_updated_copies_K(Params const& params, ProblemShape const& problem_shape, int const& head_id, const int& block_id) {
    auto [num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, num_block, block_size, head_size, max_kv_tiles, group_heads] = problem_shape;

    // key: [block_num, block_size, num_heads, head_size]
    // stride head_num: head_size
    // stride block_id: num_heads * head_size
    int64_t offset_k = head_id * head_size + block_id * block_size * (num_heads_kv * head_size);

    auto k_traits =
        static_cast<traits_load_K const &>(params.gmem_tiled_copy_k);
    const ElementK *k_ptr = (const ElementK *)k_traits.base_ptr;

    auto shape_k = make_shape(static_cast<int>(block_size), num_heads_kv * head_size, 1);
    
    StrideK stride_k = cutlass::make_cute_packed_stride(StrideK{}, shape_k);
    auto tensorK = make_tensor(make_gmem_ptr(k_ptr + offset_k), make_layout(shape_k, stride_k));
    
    XE_Copy_K copyK{XE_Copy_K{}.with(tensorK)};
    
    return Params{params.gmem_tiled_copy_q, copyK, params.gmem_tiled_copy_v, params.block_table, params.block_size, params.max_blocks_per_seq};
  }

  template <class ProblemShape>
  CUTLASS_DEVICE static constexpr Params get_updated_copies_V(Params const& params, ProblemShape const& problem_shape, int const& head_id, int const& block_id) {
    auto [num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, num_block, block_size, head_size, max_kv_tiles, group_heads] = problem_shape;

    // value: [block_num, block_size, num_heads, head_size]
    // stride head_num: head_size
    // stride block_id: num_heads * head_size
    const int sg_id = get_sub_group_id();
    int64_t offset_v = sg_id * PV_SG_N + head_id * head_size + block_id * block_size * (num_heads_kv * head_size);

    auto v_traits =
        static_cast<traits_load_V const &>(params.gmem_tiled_copy_v);
    const ElementV *v_ptr = (const ElementV *)v_traits.base_ptr;

    auto shape_v_stride = make_shape(num_heads_kv * head_size, block_size, 1);
    auto shape_v = make_shape(PV_SG_N, block_size, 1);
    
    StrideV stride_v = cutlass::make_cute_packed_stride(StrideV{}, shape_v_stride);
    auto tensorV = make_tensor(make_gmem_ptr(v_ptr + offset_v), make_layout(shape_v_stride, stride_v));

    XE_Copy_V copyV{XE_Copy_V{}.with(tensorV)};
    
    return Params{params.gmem_tiled_copy_q, params.gmem_tiled_copy_k, copyV, params.block_table, params.block_size, params.max_blocks_per_seq};
  }
};

} // namespace cutlass::flash_attention::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
