//===- conv.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void conv2d_int32(int32_t input[16][16], int32_t kernel[3][3], int32_t output[14][14]) {
    // TODO: Implement the 2D convolution kernel
    // input: 16x16 input feature map (int32 for 4-byte alignment)
    // kernel: 3x3 convolution filter
    // output: 14x14 output feature map (no padding)
}

} // extern "C" 