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
  \brief Functor performing online softmax.
*/

#pragma once

#include <sycl/sycl.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/detail/layout.hpp"

namespace cutlass {
namespace flash_attention {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <class DispatchPolicy, class... Args> class FlashPagedSoftmaxEpilogue {
  static_assert(cutlass::detail::dependent_false<DispatchPolicy>, "Could not find an epilogue specialization.");
};


template <class Element_, class Stride_, class CopyOp_>
class FlashPagedSoftmaxEpilogue<epilogue::IntelXeXMX16, Element_, Stride_, CopyOp_> {
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = epilogue::IntelXeXMX16;
  using Element = Element_;
  using Stride = Stride_;

  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;
  using CopyThreadShape = Shape<Int<SubgroupSize>, _1>;

  // using GmemTiledCopy = CopyOp_;
  // using traits_store = Copy_Traits<GmemTiledCopy, Stride>;
  // using atom_load = Copy_Atom<traits_store, Element>;
  // using val_layout_load = decltype(make_layout(shape_div(typename traits_store::BlockShape{}, CopyThreadShape{})));
  // using XE_Copy = decltype(make_tiled_copy(atom_load{}, Layout<CopyThreadShape>{}, val_layout_load{}));
  using GmemTiledCopy = XE_1D_STORE_GLOBAL<Element, Element>;
  using traits_store = Copy_Traits<GmemTiledCopy>;
  using atom_store = Copy_Atom<traits_store, Element>;
  using val_layout_store = decltype(make_layout(
      make_shape(_1{}, _1{}), make_stride(_1{}, _1{})));
  using XE_Copy = decltype(make_tiled_copy(atom_store{}, Layout<CopyThreadShape>{}, val_layout_store{}));

  // Host side epilogue arguments
  struct Arguments {
    Element const scale;

    Element *ptr_max;
    Element *ptr_sum;
    Stride dS;
  };

  // Device side epilogue params
  struct Params {
    Element scale;

    XE_Copy xe_store;
    Element *ptr_max;
    Element *ptr_sum;
    Stride dS;
    int64_t offset;
  };

  //
  // Methods
  //

  FlashPagedSoftmaxEpilogue() = default;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const &problem_shape, Arguments const &args) {
    constexpr double kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E
    Element val = args.scale * static_cast<Element>(kLog2e);

    auto [num_heads_q, num_heads_kv, seq_len_q, seq_len_kv, num_block, block_size, head_size, max_kv_tiles, group_heads] = problem_shape;

    XE_Copy xe_store{XE_Copy{}};

    return Params{val, xe_store, args.ptr_max, args.ptr_sum, args.dS, 0};
  }

  template <int Num_SGs, class FragAcc, class STensorStore, class STensorLoad, class TensorM, class TensorS>
  CUTLASS_DEVICE void operator()(Params const &params, FragAcc &frag_s, Element& max_reg, Element& sum_reg, STensorStore& smem_store,
                                  STensorLoad& smem_load, TensorM &tensor_max, TensorS &tensor_sum) {
    //context
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto wg = syclcompat::get_nd_item<1>().get_group();
    const int sg_local_id = sg.get_local_id()[0];

    auto thr_store_max = params.xe_store.get_slice(0);
    auto thr_store_sum = params.xe_store.get_slice(0);

    // store S into shared_mem
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Int<size(FragAcc{})>{}; ++i) {
      smem_store(i) = frag_s(i) * params.scale;
    }
    sycl::group_barrier(wg);

    // softmax
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Int<size(FragAcc{})>{}; ++i) {
      max_reg = sycl::max(max_reg, smem_load(i));
    }
    max_reg = reduce_over_group(sg, max_reg, sycl::maximum<>());
    // TODO: save max_reg
    if (sg_local_id == 0) {
      Tensor gD = tensor_max;
      Tensor tD = thr_store_max.partition_D(gD);
      // if (cute::thread(0, 0)) {
      //   print("params.xe_store: "); print(params.xe_store); print("\n");
      //   print("thr_store_max: "); print(thr_store_max); print("\n");
      //   print("gD: "); print(gD); print("\n");
      //   print("tD:"); print(tD); print("\n");
      //   print("max_reg: "); print(max_reg); print("\n");
      // }
      Tensor t_max_reg = make_tensor(&max_reg, Shape<_1, _1, _1>{});
      Tensor t_max_reg_store = thr_store_max.partition_S(t_max_reg);

      copy(params.xe_store, t_max_reg_store, tD);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Int<size(FragAcc{})>{}; ++i) {
      Element eq = smem_load(i) - max_reg;
      smem_load(i) = sycl::native::exp2(eq);
      sum_reg += smem_load(i);
    }
    sum_reg = reduce_over_group(sg, sum_reg, sycl::plus<>());
    // TODO: save sum_reg
    if (sg_local_id == 0) {
      Tensor gD = tensor_sum;
      Tensor tD = thr_store_sum.partition_D(gD);
      Tensor t_sum_reg = make_tensor(&sum_reg, Shape<_1, _1, _1>{});
      Tensor t_sum_reg_store = thr_store_sum.partition_S(t_sum_reg);

      copy(params.xe_store, t_sum_reg_store, tD);
    }
    sum_reg = sum_reg == 0.0f ? 1.0f : sycl::native::recip(sum_reg);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Int<size(FragAcc{})>{}; ++i) {
      smem_load(i) = smem_load(i) * sum_reg;
    }
    sycl::group_barrier(wg);

    // debug: store back
    // CUTLASS_PRAGMA_UNROLL
    // for (int i = 0; i < Int<size(FragAcc{})>{}; ++i) {
    //   frag_s(i) = smem_store(i);
    // }

    // if (cute::thread(0, 0)) {
    //   print("max_reg: "); print(max_reg); print("\n");
    //   print("sum_reg: "); print(sum_reg); print("\n");
    //   print_tensor(smem_load);
    // }
  }

  template <class ProblemShape, class TileCoord>
  CUTLASS_DEVICE static constexpr Params get_updated_copies(Params const& params, ProblemShape const& problem_shape, TileCoord tile_coord) {
    auto [num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, num_block, block_size, head_size, max_kv_tiles, group_heads] = problem_shape;
    auto [seq_q_coord, heads_coord, seq_kv_coord] = tile_coord;

    // seqlen_q, num_heads_q, max_kv_tiles
    const int sg_id = get_sub_group_id();
    int64_t offset = seq_q_coord * num_heads_kv * max_kv_tiles +
                      (heads_coord * group_heads + sg_id) * max_kv_tiles +
                      seq_kv_coord;

    return Params{params.scale, params.xe_store, params.ptr_max, params.ptr_sum, params.dS, offset};
  }
};

}
}
}