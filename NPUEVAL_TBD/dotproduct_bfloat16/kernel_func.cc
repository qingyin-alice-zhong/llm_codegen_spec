// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void dotproduct_bfloat16(bfloat16 vector1[], bfloat16 vector2[], bfloat16 result[], std::uint32_t vector_size) { {
    event0();
    // TODO: Implement the kernel
    event1();
}

} // extern "C"