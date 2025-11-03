//===- gelu.cc -------------------------------------------------*- C++ -*-===//
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

void gelu_float32(float input[4][768], float output[4][768]) {
    // TODO: Implement the GELU activation kernel
    // input: 64x3072 input tensor from feed-forward network
    // output: 64x3072 GELU activated tensor
    // GELU(x) = x * Φ(x) where Φ(x) is the cumulative distribution function of the standard normal distribution
    // Approximate GELU: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
}

} // extern "C" 