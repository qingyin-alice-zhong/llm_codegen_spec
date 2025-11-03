// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void inverse_uint8(std::uint8_t in_buffer[], std::uint8_t out_buffer[]) {
    constexpr std::uint32_t nbytes = 1024;
    constexpr std::uint32_t num_elements = nbytes / sizeof(std::uint8_t);

    for (uint32_t i = 0; i < num_elements; i++) {
    event0();
        out_buffer[i] = 255 - in_buffer[i];
    event1();
    }
}

} // extern "C"