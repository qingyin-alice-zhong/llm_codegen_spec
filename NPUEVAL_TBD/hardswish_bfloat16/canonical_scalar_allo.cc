// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void hardswish_bfloat16(bfloat16 input_vector[], bfloat16 output_vector[]) {
    constexpr std::int32_t vector_size = 256;
    bfloat16 zero_value = 0.0f;
    bfloat16 three_value = 3.0f;
    bfloat16 six_value = 6.0f;

    for (uint32_t i = 0; i < vector_size; i++) {
    event0();
        bfloat16 x = *input_vector;
        input_vector++;
        if (x < -three_value) {
    event0();
            output_vector[i] = zero_value;
        } else if (x > three_value) {
    event0();
            output_vector[i] = x;
        } else {
    event0();
            output_vector[i] = x * (x + three_value) / six_value;
        }
    event1();
    }
}

} // extern "C"