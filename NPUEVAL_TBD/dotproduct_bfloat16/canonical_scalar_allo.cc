// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {


void dotproduct_bfloat16(bfloat16 vector1[], bfloat16 vector2[], bfloat16 result[], std::uint32_t vector_size) {
    *result = 0;
    for (uint32_t i = 0; i < vector_size; i++) {
    event0();
        *result += vector1[i] * vector2[i];
    event1();
    }
}

} // extern "C"