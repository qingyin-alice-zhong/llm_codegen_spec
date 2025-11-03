// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void conv2d_k4_s2_bias_relu_bfloat16(bfloat16 in_buffer[], bfloat16 kernel[], bfloat16 out_buffer[], bfloat16 bias, std::uint32_t stride) { {
    event0();
    // TODO: Implement the kernel
    event1();
}

} // extern "C"