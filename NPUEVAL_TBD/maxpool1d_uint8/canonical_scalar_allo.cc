// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void maxpool1d_uint8(std::uint8_t in_buffer[], std::uint8_t out_buffer[], std::uint32_t window_size, std::uint32_t stride) {
    constexpr std::uint32_t nbytes = 1024;
    std::uint32_t num_windows = (nbytes - window_size) / stride + 1;

    for (uint32_t i = 0; i < num_windows; i++) {
    event0();
        std::uint8_t max_val = 0;
        for (uint32_t j = 0; j < window_size; j++) {
    event0();
            std::uint8_t current_val = *(in_buffer + i * stride + j);
            if (current_val > max_val) {
    event0();
                max_val = current_val;
            }
        }
        out_buffer[i] = max_val;
    event1();
    }
}

} // extern "C"