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

void elementwise_max_bfloat16(bfloat16 input0[], bfloat16 input1[], bfloat16 output[]) {
    constexpr std::int32_t VECTOR_SIZE = 256;
    for (int i = 0; i < VECTOR_SIZE; i++) {
    event0();
        float a = float(input0[i]);
        float b = float(input1[i]);
        output[i] = bfloat16(a > b ? a : b);
    event1();
    }
}

} // extern "C"