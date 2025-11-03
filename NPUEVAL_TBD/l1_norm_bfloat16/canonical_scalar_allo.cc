// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void l1_norm_bfloat16(bfloat16 in_buffer[], bfloat16 out_buffer[]) {
    constexpr std::int32_t N = 256;
    bfloat16 sum = 0;
    for (int i = 0; i < N; ++i) {
    event0();
        bfloat16 val = in_buffer[i];
        sum += (val >= 0.0f) ? val : -val;
    event1();
    }
    *out_buffer = sum;
}

} // extern "C"