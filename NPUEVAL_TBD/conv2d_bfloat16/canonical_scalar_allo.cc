// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void conv2d_bfloat16(bfloat16 input_matrix[256], bfloat16 output_matrix[224], std::int32_t param[5]) {
    event0();
    
    // Unpack parameters: kernel (2x2=4 values) and stride (all stored as float32 in int32)
    bfloat16 kernel[4];
    kernel[0] = (bfloat16)(*(float*)&param[0]);
    kernel[1] = (bfloat16)(*(float*)&param[1]);
    kernel[2] = (bfloat16)(*(float*)&param[2]);
    kernel[3] = (bfloat16)(*(float*)&param[3]);
    std::int32_t stride = (std::int32_t)(*(float*)&param[4]);
    
    constexpr std::int32_t input_rows = 16;
    constexpr std::int32_t input_cols = 16;
    constexpr std::int32_t kernel_size = 2;
    std::int32_t output_rows = 14;  // Fixed for 4-byte alignment (14x16=224)
    std::int32_t output_cols = 16;

    for (std::int32_t i = 0; i < output_rows; i++) {

        for (std::int32_t j = 0; j < output_cols; j++) {
            float conv_sum = 0.0f;
            for (std::int32_t ki = 0; ki < kernel_size; ki++) {
                for (std::int32_t kj = 0; kj < kernel_size; kj++) {

                    float inp = (float)input_matrix[(i * stride + ki) * input_cols + (j * stride + kj)];
                    float ker = (float)kernel[ki * kernel_size + kj];
                    conv_sum += inp * ker;
                }
            }
            output_matrix[i * output_cols + j] = (bfloat16)conv_sum;
        }
    }
    
    event1();
}

} // extern "C"