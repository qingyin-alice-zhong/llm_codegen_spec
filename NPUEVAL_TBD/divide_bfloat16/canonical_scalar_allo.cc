// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void divide_bfloat16(bfloat16 a[], bfloat16 b[], bfloat16 out[]) {
    constexpr std::int32_t vector_size = 256;
    for (int i = 0; i < vector_size; i++) {
    event0();
        if (b[i] == (bfloat16)0.0f) {
    event0();
            out[i] = (bfloat16)0.0f;
        } else {
    event0();
            out[i] = (bfloat16)((float)a[i] / (float)b[i]);
        }
    event1();
    }
}

} // extern "C"