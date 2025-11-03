// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void mse_loss_bfloat16(bfloat16 input_a[], bfloat16 input_b[], bfloat16 output[]) {
    constexpr int N = 256;
    float acc = 0.0f;
    for (int i = 0; i < N; i++) {
    event0();
        float a = (float)input_a[i];
        float b = (float)input_b[i];
        float diff = a - b;
        acc += diff * diff;
    event1();
    }
    float mse = acc / N;
    *output = (bfloat16)mse;
}

} // extern "C"