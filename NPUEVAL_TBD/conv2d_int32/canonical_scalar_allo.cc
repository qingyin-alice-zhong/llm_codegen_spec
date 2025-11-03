// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void conv2d_int32(std::int32_t input_matrix[256], std::int32_t output_matrix[224], std::int32_t param[5]) {
    event0();
    
    // Unpack parameters: kernel (2x2=4 values) and stride (all stored as int32)
    std::int32_t kernel[4];
    kernel[0] = param[0];
    kernel[1] = param[1];
    kernel[2] = param[2];
    kernel[3] = param[3];
    std::int32_t stride = param[4];
    
    constexpr std::int32_t input_rows = 16;
    constexpr std::int32_t input_cols = 16;
    constexpr std::int32_t kernel_size = 2;
    std::int32_t output_rows = 14;  // Fixed for 4-byte alignment (14x16=224)
    std::int32_t output_cols = 16;

    for (std::int32_t i = 0; i < output_rows; i++) {
        for (std::int32_t j = 0; j < output_cols; j++) {
            std::int32_t conv_sum = 0;
            for (std::int32_t ki = 0; ki < kernel_size; ki++) {
                for (std::int32_t kj = 0; kj < kernel_size; kj++) {
                    conv_sum += input_matrix[(i * stride + ki) * input_cols + (j * stride + kj)] * kernel[ki * kernel_size + kj];
                }
            }
            output_matrix[i * output_cols + j] = conv_sum;
        }
    }
    
    event1();
}

} // extern "C"