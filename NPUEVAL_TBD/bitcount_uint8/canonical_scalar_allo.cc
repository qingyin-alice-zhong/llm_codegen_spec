// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void bitcount_uint8(std::uint8_t input_vector[], std::uint8_t output_vector[]) {
    constexpr std::int32_t vector_size = 256;
    for (int i = 0; i < vector_size; i++) {
    event0();
        std::uint8_t x = input_vector[i];
        std::uint8_t count = 0;
        for (int b = 0; b < 8; b++) {
    event0();
            count += (x >> b) & 1;
        }
        output_vector[i] = count;
    event1();
    }
}

} // extern "C"