//===- scale.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>


// Vectorized scale template (general case)
// Assume N is multiple of 16
template <typename T>
void scale_vectorized(T *a, T *c, int32_t factor, const int32_t N) {
  event0();
  constexpr int vec_factor = 32;
  T *__restrict pA1 = a;
  T *__restrict pC1 = c;
  const int F = N / vec_factor;
  T fac = factor;
  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining     // compiler pragma
    chess_loop_range(16, )           // compiler pragma
    {
      aie::vector<T, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
      pA1 += vec_factor;
      aie::accum<acc32, vec_factor> cout = aie::mul(A0, fac);
      aie::store_v(pC1, cout.template to_vector<T>(0));
      pC1 += vec_factor;
    }
  event1();
}


extern "C" {

void vector_scalar_mul_int16_vector(int16_t a_in[1024], int32_t factor[1], int16_t c_out[1024]) {
  scale_vectorized<int16_t>(a_in, c_out, factor[0], 1024);
}

} // extern "C"