// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void negate_bfloat16(bfloat16 input_vector[], bfloat16 output_vector[]) {
    constexpr std::uint32_t vector_size = 256;
    for (uint32_t i = 0; i < vector_size; i++) {
    event0();
        output_vector[i] = (bfloat16)(-((float)input_vector[i]));
    event1();
    }
}

} // extern "C"