// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

#include <math.h>

void cos_bfloat16(bfloat16 input_vector[], bfloat16 output_vector[]) {
    constexpr std::int32_t vector_size = 256;
    constexpr float PI = 3.14159265359f;
    
    for (uint32_t i = 0; i < vector_size; i++) {
    event0();
        float x = (float)input_vector[i];
        while (x > PI) x -= 2.0f * PI;
        while (x < -PI) x += 2.0f * PI;
        float x2 = x * x;
        float x4 = x2 * x2;
        float x6 = x4 * x2;
        float x8 = x6 * x2;
        float x10 = x8 * x2;
        float result = 1.0f
            - x2 * 0.5f
            + x4 * 0.04166666666667f
            - x6 * 0.00138888888889f
            + x8 * 0.00002480158730f
            - x10 * 0.0000002755731922f;
        output_vector[i] = (bfloat16)result;
    event1();
    }
}

} // extern "C"