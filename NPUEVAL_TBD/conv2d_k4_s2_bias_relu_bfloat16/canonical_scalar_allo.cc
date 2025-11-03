// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void conv2d_k4_s2_bias_relu_bfloat16(bfloat16 in_buffer[], bfloat16 kernel[], bfloat16 out_buffer[], bfloat16 bias, std::uint32_t stride) {
    constexpr std::int32_t IN_ROWS = 16;
    constexpr std::int32_t IN_COLS = 16;
    constexpr std::int32_t KERNEL_SIZE = 4;

    std::uint32_t out_rows = (IN_ROWS - KERNEL_SIZE) / stride + 1;
    std::uint32_t out_cols = (IN_COLS - KERNEL_SIZE) / stride + 1;

    for (uint32_t i = 0; i < out_rows; i++) {
    event0();
        for (uint32_t j = 0; j < out_cols; j++) {
    event0();
            bfloat16 acc = 0;
            for (uint32_t ki = 0; ki < KERNEL_SIZE; ki++) {
    event0();
                for (uint32_t kj = 0; kj < KERNEL_SIZE; kj++) {
    event0();
                    acc += in_buffer[(i * stride + ki) * IN_COLS + (j * stride + kj)] * kernel[ki * KERNEL_SIZE + kj];
                }
            }
            acc += bias;
            bfloat16 relu = acc > 0.0f ? acc : 0.0f;
            out_buffer[i * out_cols + j] = relu;
        }
    event1();
    }
}

} // extern "C"