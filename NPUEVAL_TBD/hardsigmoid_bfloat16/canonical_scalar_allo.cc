// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void hardsigmoid_bfloat16(bfloat16 in_buffer[], bfloat16 out_buffer[]) {
    constexpr std::int32_t num_elements = 256;
    bfloat16 zero_value = 0.0f;
    bfloat16 one_value = 1.0f;
    bfloat16 slope_value = 0.2f;
    bfloat16 offset_value = 0.5f;

    for (uint32_t i = 0; i < num_elements; i++) {
    event0();
        bfloat16 input_value = *in_buffer;
        in_buffer++;

        bfloat16 result = input_value * slope_value + offset_value;

        if (result > one_value) {
    event0();
            result = one_value;
        }
        if (result < zero_value) {
    event0();
            result = zero_value;
        }

        out_buffer[i] = result;
    event1();
    }
}

} // extern "C"