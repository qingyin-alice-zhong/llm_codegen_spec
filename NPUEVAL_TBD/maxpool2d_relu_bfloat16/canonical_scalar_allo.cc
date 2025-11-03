// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void maxpool2d_relu_bfloat16(bfloat16 input[], bfloat16 output[]) {
    constexpr int ROWS = 32;
    constexpr int COLS = 32;
    constexpr int WINDOW_SIZE = 2;
    constexpr int STRIDE = 2;

    for (int i = 0; i < ROWS; i += STRIDE) {
    event0();
        for (int j = 0; j < COLS; j += STRIDE) {
    event0();
            bfloat16 max_val = input[i * COLS + j];
            for (int wi = 0; wi < WINDOW_SIZE; wi++) {
    event0();
                for (int wj = 0; wj < WINDOW_SIZE; wj++) {
    event0();
                    bfloat16 current_val = input[(i + wi) * COLS + (j + wj)];
                    if (current_val > max_val) {
    event0();
                        max_val = current_val;
                    }
                }
            }
            float out_val = float(max_val) > 0.0f ? float(max_val) : 0.0f;
            output[(i/STRIDE) * (COLS/STRIDE) + (j/STRIDE)] = (bfloat16)out_val;
        }
    event1();
    }
}

} // extern "C"