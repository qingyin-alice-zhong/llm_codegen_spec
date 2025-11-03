//===- conv_small.cc -----------------------------------------*- C++ -*-===//
// This is from LLM-generated code. Vectorized version. 

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void conv2d_int32(int32_t input[3][3], int32_t kernel[3][3], int32_t output[5][5]) {
    event0();

    // Input dimensions
    constexpr int IN_H = 3;
    constexpr int IN_W = 3;
    // Kernel dimensions
    constexpr int K_H = 3;
    constexpr int K_W = 3;
    // Padding
    constexpr int PADDING = 2;
    // Padded input dimensions
    constexpr int PADDED_IN_H = IN_H + 2 * PADDING; // 3 + 2*2 = 7
    constexpr int PADDED_IN_W = IN_W + 2 * PADDING; // 3 + 2*2 = 7
    // Output dimensions
    constexpr int OUT_H = IN_H - K_H + 2 * PADDING + 1; // 3 - 3 + 4 + 1 = 5
    constexpr int OUT_W = IN_W - K_W + 2 * PADDING + 1; // 3 - 3 + 4 + 1 = 5

    // Vectorization factor for int32_t. AIE-ML/XDNA 1 supports 4, 8, 16, 32.
    // Using 4 as it allows processing multiple output pixels per iteration and
    // fits the 5x5 output dimensions (5 = 4 + 1 remainder).
    constexpr int VEC_SIZE = 4; 

    // Padded input buffer (7x7)
    // Ensure alignment for AIE vector operations. Even when using unaligned loads,
    // aligning the base buffer can be beneficial for overall memory access patterns.
    alignas(aie::vector_decl_align) int32_t padded_input[PADDED_IN_H][PADDED_IN_W] = {0};

    // Copy original input to the center of padded_input
    for (int i = 0; i < IN_H; ++i) {
        for (int j = 0; j < IN_W; ++j) {
            padded_input[i + PADDING][j + PADDING] = input[i][j];
        }
    }

    // Loop over output rows
    for (int out_row = 0; out_row < OUT_H; ++out_row) {
        // Loop over output columns, processing VEC_SIZE elements at a time
        for (int out_col = 0; out_col < OUT_W; out_col += VEC_SIZE) {
            // Declare accumulator for VEC_SIZE output pixels.
            // acc64 is the default accumulator type for int32*int32 multiplication on AIE-ML/XDNA 1.
            aie::accum<acc64, VEC_SIZE> acc_vec; 

            // Loop over kernel dimensions (K_H x K_W = 3x3)
            // The outer loop of the convolution sum is pipelined for performance.
            for (int k_row = 0; k_row < K_H; ++k_row)
                chess_prepare_for_pipelining chess_loop_range(K_H, K_H) { // K_H = 3
                for (int k_col = 0; k_col < K_W; ++k_col) { // K_W = 3
                    // Load VEC_SIZE input values from the padded_input window.
                    // The input window for the current kernel element starts at
                    // padded_input[out_row + k_row][out_col + k_col].
                    // These VEC_SIZE values are contiguous in memory.
                    // Use aie::load_unaligned_v because the starting address
                    // (out_col + k_col) might not be aligned to VEC_SIZE boundary (e.g., if k_col is 1 or 2).
                    // The '1' indicates byte alignment for the source pointer.
                    aie::vector<int32_t, VEC_SIZE> v_input_window = 
                        aie::load_unaligned_v<VEC_SIZE>(&padded_input[out_row + k_row][out_col + k_col], 1);
                    
                    // Load the single kernel value and replicate it into a vector of VEC_SIZE.
                    // This is necessary because the kernel value multiplies each of the VEC_SIZE input values.
                    aie::vector<int32_t, VEC_SIZE> v_kernel_val;
                    for (int i = 0; i < VEC_SIZE; ++i) {
                        v_kernel_val[i] = kernel[k_row][k_col];
                    }

                    // Perform multiply-accumulate.
                    // The first multiplication (k_row=0, k_col=0) initializes the accumulator.
                    if (k_row == 0 && k_col == 0) {
                        acc_vec = aie::mul(v_input_window, v_kernel_val);
                    } else {
                        acc_vec = aie::mac(acc_vec, v_input_window, v_kernel_val);
                    }
                }
            }
            // Store the final accumulated results to the output tensor.
            // Handle potential partial vector at the end of a row.
            if (out_col + VEC_SIZE <= OUT_W) {
                // If a full vector can be stored, use aie::store_unaligned_v.
                // The '1' indicates byte alignment for the destination pointer.
                aie::store_unaligned_v(&output[out_row][out_col], acc_vec.to_vector<int32_t>(), 1);
            } else {
                // This handles the "tail" case where fewer than VEC_SIZE elements remain.
                // For OUT_W=5 and VEC_SIZE=4, this will handle the last element (index 4).
                aie::vector<int32_t, VEC_SIZE> temp_out_vec = acc_vec.to_vector<int32_t>();
                for (int i = 0; i < (OUT_W - out_col); ++i) {
                    // Store element by element for the remainder.
                    output[out_row][out_col + i] = temp_out_vec[i];
                }
            }
        }
    }
    event1();
}

} // extern "C"