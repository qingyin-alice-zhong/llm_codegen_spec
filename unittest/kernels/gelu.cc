//===- gelu.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <cmath>

#include <aie_api/aie.hpp>

// Define vectorization factor for float32 on AMD Ryzen AI NPU
// Using 16 for float32 vectors (4x768 = 3072 elements, divisible by 16)
constexpr int VEC_SIZE_FLOAT32 = 16;

// GELU approximation constants
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
constexpr float GELU_SQRT_2_PI = 0.7978845608028654f;  // sqrt(2/π)
constexpr float GELU_COEF = 0.044715f;                 // coefficient for x^3 term
constexpr float GELU_HALF = 0.5f;                      // 0.5 multiplier

extern "C" {

void gelu_float32(float input[4][768], float output[4][768]) {
    event0(); // Start performance event for profiling
    
    // Flatten arrays for easier vectorized access
    float *in = reinterpret_cast<float*>(input);
    float *out = reinterpret_cast<float*>(output);
    
    constexpr int TOTAL_ELEMENTS = 4 * 768; // 512 elements
    
    // Process the input in vector chunks
    for (int i = 0; i < TOTAL_ELEMENTS; i += VEC_SIZE_FLOAT32) {
        chess_prepare_for_pipelining
        chess_loop_range(192, ) // TOTAL_ELEMENTS / VEC_SIZE_FLOAT32 = 3072 / 16 = 192
        {
            // Load input vector
            aie::vector<float, VEC_SIZE_FLOAT32> input_vec = aie::load_v<VEC_SIZE_FLOAT32>(in + i);
            
            // Compute GELU for each element in the vector
            aie::vector<float, VEC_SIZE_FLOAT32> output_vec;
            
            for (int j = 0; j < VEC_SIZE_FLOAT32; ++j) {
                float x = input_vec[j];
                
                // Compute x^3
                float x_cubed = x * x * x;
                
                // Compute inner term: sqrt(2/π) * (x + 0.044715 * x^3)
                float inner_term = GELU_SQRT_2_PI * (x + GELU_COEF * x_cubed);
                
                // Compute tanh(inner_term)
                float tanh_term = std::tanh(inner_term);
                
                // Compute final GELU: 0.5 * x * (1 + tanh_term)
                float gelu_result = GELU_HALF * x * (1.0f + tanh_term);
                
                output_vec[j] = gelu_result;
            }
            
            // Store output vector
            aie::store_v(out + i, output_vec);
        }
    }
    
    event1(); // End performance event
}

} // extern "C" 