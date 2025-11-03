//===- gemm_kernel.cc -------------------------------------------------*- C++ -*-===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void gemm_kernel_int16(int16_t A[64][64], int16_t B[64][64], int16_t C[64][64]) {
    // TODO: Implement the GEMM kernel
    // C = A @ B
    // A: 64x64 matrix
    // B: 64x64 matrix  
    // C: 64x64 output matrix
}

} // extern "C" 