// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void euclidean_dist_bfloat16(bfloat16 input_a[], bfloat16 input_b[], bfloat16 output[]) {
    constexpr std::int32_t N = 256;

    float acc = 0.0f;
    for (int i = 0; i < N; ++i) {
    event0();
        float da = float(input_a[i]);
        float db = float(input_b[i]);
        float diff = da - db;
        acc += diff * diff;
    event1();
    }

    float y;
    if (acc <= 0.0f) {
        y = 0.0f;
    } else {
        union { uint32_t u; float f; } fu { .f = acc };
        union { uint32_t u; float f; } gu { .u = (fu.u >> 1) + 0x1F800000u };
        y = gu.f;
        y = 0.5f * (y + acc / y);
        y = 0.5f * (y + acc / y);
    }

    *output = bfloat16(y);
}

} // extern "C"