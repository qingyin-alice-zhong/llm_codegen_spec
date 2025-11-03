// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void bitwisexor_uint8(std::uint8_t in_buffer1[], std::uint8_t in_buffer2[], std::uint8_t out_buffer[]) {
    constexpr std::int32_t nbytes = 1024;
    for (uint32_t i = 0; i < nbytes; ++i) {  
    event0();
        out_buffer[i] = in_buffer1[i] ^ in_buffer2[i];
    event1();
    }
}

} // extern "C"