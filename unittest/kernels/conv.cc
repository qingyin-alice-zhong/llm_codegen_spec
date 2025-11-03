//===- conv.cc -------------------------------------------------*- C++ -*-===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

extern "C" {

void conv2d_int32(int32_t input[16][16], int32_t kernel[3][3], int32_t output[14][14]) {
    event0();
    
    // Direct implementation - flatten arrays for easier access
    int32_t *in = reinterpret_cast<int32_t*>(input);
    int32_t *ker = reinterpret_cast<int32_t*>(kernel);
    int32_t *out = reinterpret_cast<int32_t*>(output);
    
    // Constants for readability
    constexpr int IN_WIDTH = 16;
    constexpr int OUT_WIDTH = 14;
    constexpr int OUT_HEIGHT = 14;
    constexpr int KERNEL_SIZE = 3;
    
    // 2D Convolution: output[i][j] = sum over kernel of input[i+ki][j+kj] * kernel[ki][kj]
    for (int out_row = 0; out_row < OUT_HEIGHT; out_row++)
        chess_prepare_for_pipelining
        chess_loop_range(14, )
        {
            for (int out_col = 0; out_col < OUT_WIDTH; out_col++) {
                int32_t sum = 0;  // Use int32 to avoid overflow during accumulation
                
                // Convolve with 3x3 kernel
                for (int k_row = 0; k_row < KERNEL_SIZE; k_row++) {
                    for (int k_col = 0; k_col < KERNEL_SIZE; k_col++) {
                        // Calculate input coordinates
                        int in_row = out_row + k_row;
                        int in_col = out_col + k_col;
                        
                                                 // Perform multiply-accumulate
                         int32_t input_val = in[in_row * IN_WIDTH + in_col];
                         int32_t kernel_val = ker[k_row * KERNEL_SIZE + k_col];
                         sum += input_val * kernel_val;
                    }
                }
                
                                 // Store result
                 out[out_row * OUT_WIDTH + out_col] = sum;
            }
        }
    
    event1();
}

} // extern "C" 