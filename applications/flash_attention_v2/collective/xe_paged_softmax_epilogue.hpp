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


template <class Element_>
class FlashPagedSoftmaxEpilogue<epilogue::IntelXeXMX16, Element_> {
public:
  //
  // Type Aliases
  //
  using DispatchPolicy = epilogue::IntelXeXMX16;
  using Element = Element_;

  using GmemTiledCopyOut = void;

  // Host side epilogue arguments
  struct Arguments {
    Element const scale;
  };

  // Device side epilogue params
  using Params = Arguments;

  //
  // Methods
  //

  static constexpr Params to_underlying_arguments(Arguments const &args) {
    constexpr double kLog2e = 1.4426950408889634074; // log_2(e) = M_LOG2E
    Element val = args.scale * static_cast<Element>(kLog2e);
    return Params{val};
  }

  template <class ProblemShape>
  static size_t get_workspace_size() {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace() {
    return Status::kSuccess;
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool can_implement() {
    return true;
  }

  CUTLASS_HOST_DEVICE
  FlashPagedSoftmaxEpilogue(Params const &params_) : params(params_) {}

  template <int Num_SGs, class FragAcc, class STensorStore, class STensorLoad>
  CUTLASS_DEVICE void operator()(FragAcc &frag_s, Element& max_reg, Element& sum_reg, STensorStore& smem_store, STensorLoad& smem_load) {
    //context
    auto sg = syclcompat::get_nd_item<1>().get_sub_group();
    auto wg = syclcompat::get_nd_item<1>().get_group();
    // store S into shared_mem
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Int<size(FragAcc{})>{}; ++i) {
      smem_store(i) = frag_s(i) * params.scale;
      // TODO: * params.scale;
    }
    sycl::group_barrier(wg);

    // softmax
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Int<size(FragAcc{})>{}; ++i) {
      max_reg = sycl::max(max_reg, smem_load(i));
    }
    max_reg = reduce_over_group(sg, max_reg, sycl::maximum<>());

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Int<size(FragAcc{})>{}; ++i) {
      Element eq = smem_load(i) - max_reg;
      smem_load(i) = sycl::native::exp2(eq);
      sum_reg += smem_load(i);
    }
    sum_reg = reduce_over_group(sg, sum_reg, sycl::plus<>());
    sum_reg = sum_reg == 0.0f ? 1.0f : sycl::native::recip(sum_reg);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Int<size(FragAcc{})>{}; ++i) {
      smem_load(i) = smem_load(i) * sum_reg;
    }
    sycl::group_barrier(wg);

    // debug: store back
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < Int<size(FragAcc{})>{}; ++i) {
      frag_s(i) = smem_store(i);
    }

    if (cute::thread(0, 0)) {
      print("max_reg: "); print(max_reg); print("\n");
      print("sum_reg: "); print(sum_reg); print("\n");
      print_tensor(smem_load);
    }
  }

  Params params;
};

}
}
}