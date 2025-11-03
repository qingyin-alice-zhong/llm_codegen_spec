// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void max_abs_bfloat16(bfloat16 input[], bfloat16 output[]) {
    constexpr int N = 256;
    float max_abs = 0.0f;
    for (int i = 0; i < N; ++i) {
    event0();
        float val = (float)input[i];
        float abs_val = val < 0.0f ? -val : val;
        if (abs_val > max_abs) {
    event0();
            max_abs = abs_val;
        }
    event1();
    }
    *output = (bfloat16)max_abs;
}

} // extern "C"