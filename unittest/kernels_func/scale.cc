//===- scale.cc -------------------------------------------------*- C++ -*-===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void vector_scalar_mul_int16_vector(int16_t a_in[1024], int32_t factor[1], int16_t c_out[1024]) {
    // TODO: Implement the kernel
}

} // extern "C"