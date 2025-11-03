//===- mv.cc ----------------------------------------------000---*- C++ -*-===//
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

// Simple matrix-vector multiplication for row-major matrix
template <typename T_in, typename T_out, int M, int K>
void matvec_simple(T_in *a, T_in *b, T_out *c) {
  event0();
  for (int row = 0; row < M; row++) {
    T_out sum = 0;
    for (int col = 0; col < K; col++) {
      sum += a[row * K + col] * b[col];
    }
    c[row] = sum;
  }
  event1();
}

// Vectorized matrix-vector multiplication for row-major matrix
template <typename T_in, typename T_out, int M, int K>
void matvec_vectorized_simple(T_in *a, T_in *b, T_out *c) {
  event0();
  constexpr int vec_factor = 16;
  
  for (int row = 0; row < M; row++) {
    T_out sum = 0;
    T_in *row_ptr = a + row * K;
    
    // Vectorized part
    const int F = K / vec_factor;
    aie::accum<acc32, vec_factor> acc;
    acc = aie::zeros<acc32, vec_factor>();
    
    for (int i = 0; i < F; i++) {
      aie::vector<T_in, vec_factor> A_vec = aie::load_v<vec_factor>(row_ptr + i * vec_factor);
      aie::vector<T_in, vec_factor> B_vec = aie::load_v<vec_factor>(b + i * vec_factor);
      acc = aie::mac(acc, A_vec, B_vec);
    }
    
    // Reduce accumulator
    aie::vector<T_out, vec_factor> result = acc.template to_vector<T_out>();
    for (int i = 0; i < vec_factor; i++) {
      sum += result[i];
    }
    
    // Handle remaining elements
    for (int i = F * vec_factor; i < K; i++) {
      sum += row_ptr[i] * b[i];
    }
    
    c[row] = sum;
  }
  event1();
}

extern "C" {

// Matrix-vector multiplication: A(M x K) * b(K) = c(M)
// Using flattened row-major matrix input
void matvec_vectorized_i16_i32(int16_t a_in[1024], int16_t b_in[32], int32_t c_out[32]) {
  matvec_vectorized_simple<int16_t, int32_t, 32, 32>(a_in, b_in, c_out);
}

} // extern "C"
