// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void relu_bfloat16(bfloat16 in_buffer[], bfloat16 out_buffer[]) {
    constexpr std::int32_t num_elements = 256;
    for (uint32_t i = 0; i < num_elements; ++i) {
    event0();
        out_buffer[i] = in_buffer[i] < 0 ? 0 : in_buffer[i];
    event1();
    }
}

} // extern "C"