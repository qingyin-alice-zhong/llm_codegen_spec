// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void relu6_bfloat16(bfloat16 input[], bfloat16 output[]) {
    constexpr std::int32_t N = 256;
    for (int i = 0; i < N; ++i) {
    event0();
        float x = float(input[i]);
        float y = x > 0.0f ? x : 0.0f;
        y = y < 6.0f ? y : 6.0f;
        output[i] = bfloat16(y);
    event1();
    }
}

} // extern "C"