// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void reciprocal_bfloat16(bfloat16 input_vector[], bfloat16 output_vector[]) {
    constexpr std::int32_t vector_size = 256;
    for (uint32_t i = 0; i < vector_size; i++) {
    event0();
        float x = (float)input_vector[i];
        output_vector[i] = (bfloat16)(1.0f / x);
    event1();
    }
}

} // extern "C"