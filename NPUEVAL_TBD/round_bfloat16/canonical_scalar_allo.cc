// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

#include <aie_api/aie.hpp>
#include <cstdint>

void round_bfloat16(bfloat16 input[], bfloat16 output[]) {
    constexpr int N = 256;
    for (int i = 0; i < N; ++i) {
    event0();
        float x  = float(input[i]);
        float t  = x + 0.5f;
        int   xi = int(t);
        if (t < 0.0f && float(xi) != t) {
    event0();
            xi -= 1;
        }
        output[i] = bfloat16(float(xi));
    event1();
    }
}

} // extern "C"