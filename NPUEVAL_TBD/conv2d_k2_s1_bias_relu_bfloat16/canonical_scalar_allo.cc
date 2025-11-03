// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void conv2d_k2_s1_bias_relu_bfloat16(bfloat16 input[], bfloat16 kernel[], bfloat16 output[], bfloat16 bias) {
    constexpr std::int32_t IN_ROWS = 16;
    constexpr std::int32_t IN_COLS = 16;
    constexpr std::int32_t KERNEL_SIZE = 2;
    constexpr std::int32_t STRIDE = 1;

    std::uint32_t OUT_ROWS = (IN_ROWS - KERNEL_SIZE) / STRIDE + 1;
    std::uint32_t OUT_COLS = (IN_COLS - KERNEL_SIZE) / STRIDE + 1;

    for (uint32_t i = 0; i < OUT_ROWS; i++) {
    event0();
        for (uint32_t j = 0; j < OUT_COLS; j++) {
    event0();
            float acc = (float)bias;
            for (uint32_t ki = 0; ki < KERNEL_SIZE; ki++) {
    event0();
                for (uint32_t kj = 0; kj < KERNEL_SIZE; kj++) {
    event0();
                    std::uint32_t in_idx = (i * STRIDE + ki) * IN_COLS + (j * STRIDE + kj);
                    std::uint32_t k_idx = ki * KERNEL_SIZE + kj;
                    acc += (float)input[in_idx] * (float)kernel[k_idx];
                }
            }
            float relu = acc > 0.0f ? acc : 0.0f;
            output[i * OUT_COLS + j] = (bfloat16)relu;
        }
    event1();
    }
}

} // extern "C"