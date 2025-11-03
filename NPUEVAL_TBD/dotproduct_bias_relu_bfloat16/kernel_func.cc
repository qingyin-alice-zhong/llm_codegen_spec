// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void dotproduct_bias_relu_bfloat16(bfloat16 in0[], bfloat16 in1[], bfloat16 out[], bfloat16 bias, std::uint32_t vector_size) { {
    event0();
    // TODO: Implement the kernel
    event1();
}

} // extern "C"