// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void vectormult_bfloat16(bfloat16 in_buffer1[64], bfloat16 in_buffer2[64], bfloat16 out_buffer[64]) {
    event0();
    
    constexpr std::int32_t N = 128;
    for (std::int32_t i = 0; i < N; ++i) {
        out_buffer[i] = in_buffer1[i] * in_buffer2[i];
    }
    
    event1();
}

} // extern "C"