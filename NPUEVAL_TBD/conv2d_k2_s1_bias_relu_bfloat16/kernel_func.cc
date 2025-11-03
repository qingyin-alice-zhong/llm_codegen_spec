// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void conv2d_k2_s1_bias_relu_bfloat16(bfloat16 input[], bfloat16 kernel[], bfloat16 output[], bfloat16 bias) { {
    event0();
    // TODO: Implement the kernel
    event1();
}

} // extern "C"