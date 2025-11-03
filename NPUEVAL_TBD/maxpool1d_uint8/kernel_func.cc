// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void maxpool1d_uint8(std::uint8_t in_buffer[], std::uint8_t out_buffer[], std::uint32_t window_size, std::uint32_t stride) { {
    event0();
    // TODO: Implement the kernel
    event1();
}

} // extern "C"