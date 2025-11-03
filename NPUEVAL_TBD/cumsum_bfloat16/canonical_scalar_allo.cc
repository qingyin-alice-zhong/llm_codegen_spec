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

void cumsum_bfloat16(bfloat16 in[], bfloat16 out[]) {
    constexpr int N = 256;
    float sum = 0.0f;
    for (int i = 0; i < N; ++i) {
    event0();
        sum += float(in[i]);
        out[i] = bfloat16(sum);
    event1();
    }
}

} // extern "C"