//===- mv.cc ----------------------------------------------000---*- C++ -*-===//

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

// Matrix-vector multiplication: A(M x K) * b(K) = c(M)
// Using flattened row-major matrix input
void matvec_vectorized_i16_i32(int16_t a_in[1024], int16_t b_in[32], int32_t c_out[32]) {
    // TODO: Implement the kernel
}

} // extern "C"
