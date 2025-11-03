//===- masked_softmax.cc -------------------------------------------------*- C++ -*-===//
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
#include <algorithm>

#include <aie_api/aie.hpp>

// Define vectorization factor for float32 on AMD Ryzen AI NPU
// Using 16 for float32 vectors (64 elements per row, divisible by 16)
constexpr int VEC_SIZE_FLOAT32 = 16;
constexpr int TILE_ROWS = 16;
constexpr int SEQ_LEN = 64;

extern "C" {

void masked_softmax_float32(float attention_score[16][64], int tile_row_start[1], float attn_weights[16][64]) {
    event0(); // Start performance event for profiling
    
    // Extract tile_row_start value
    int tile_start = tile_row_start[0];
    
    // Process each row in the tile
    for (int row = 0; row < TILE_ROWS; row++) {
        chess_prepare_for_pipelining
        chess_loop_range(16, )
        {
            // Calculate global row index for this tile row
            int global_row_idx = tile_start + row;
            
            // Step 1: Apply causal mask and find maximum value for numerical stability
            float max_val = -std::numeric_limits<float>::infinity();
            
            // Process the row in vector chunks
            for (int col = 0; col < SEQ_LEN; col += VEC_SIZE_FLOAT32) {
                // Load input vector
                aie::vector<float, VEC_SIZE_FLOAT32> input_vec = aie::load_v<VEC_SIZE_FLOAT32>(&attention_score[row][col]);
                
                // Apply causal mask: set future positions to -inf
                aie::vector<float, VEC_SIZE_FLOAT32> masked_vec;
                for (int k = 0; k < VEC_SIZE_FLOAT32; ++k) {
                    int global_col_idx = col + k;
                    if (global_col_idx > global_row_idx) {
                        // Future position - mask with -inf
                        masked_vec[k] = -std::numeric_limits<float>::infinity();
                    } else {
                        // Current or past position - keep original value
                        masked_vec[k] = input_vec[k];
                    }
                }
                
                // Find maximum in this vector chunk
                for (int k = 0; k < VEC_SIZE_FLOAT32; ++k) {
                    max_val = std::max(max_val, masked_vec[k]);
                }
                
                // Store masked values back to input buffer for reuse
                aie::store_v(&attention_score[row][col], masked_vec);
            }
            
            // Step 2: Compute exp(x - max) and sum
            float sum_exp = 0.0f;
            
            for (int col = 0; col < SEQ_LEN; col += VEC_SIZE_FLOAT32) {
                // Load masked values
                aie::vector<float, VEC_SIZE_FLOAT32> masked_vec = aie::load_v<VEC_SIZE_FLOAT32>(&attention_score[row][col]);
                
                // Compute exp(x - max) for each element
                aie::vector<float, VEC_SIZE_FLOAT32> exp_vec;
                for (int k = 0; k < VEC_SIZE_FLOAT32; ++k) {
                    float shifted_val = masked_vec[k] - max_val;
                    exp_vec[k] = std::exp(shifted_val);
                }
                
                // Accumulate sum
                for (int k = 0; k < VEC_SIZE_FLOAT32; ++k) {
                    sum_exp += exp_vec[k];
                }
                
                // Store exp values back to input buffer for reuse
                aie::store_v(&attention_score[row][col], exp_vec);
            }
            
            // Step 3: Normalize by dividing by sum
            float reciprocal_sum = 1.0f / sum_exp;
            
            for (int col = 0; col < SEQ_LEN; col += VEC_SIZE_FLOAT32) {
                // Load exp values
                aie::vector<float, VEC_SIZE_FLOAT32> exp_vec = aie::load_v<VEC_SIZE_FLOAT32>(&attention_score[row][col]);
                
                // Normalize by multiplying with reciprocal of sum
                aie::vector<float, VEC_SIZE_FLOAT32> output_vec;
                for (int k = 0; k < VEC_SIZE_FLOAT32; ++k) {
                    output_vec[k] = exp_vec[k] * reciprocal_sum;
                }
                
                // Store final softmax weights
                aie::store_v(&attn_weights[row][col], output_vec);
            }
        }
    }
    
    event1(); // End performance event
}

} // extern "C" 