// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void bitcount_uint8(std::uint8_t input_vector[], std::uint8_t output_vector[]) { {
    event0();
    // TODO: Implement the kernel
    event1();
}

} // extern "C"