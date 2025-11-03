# Convolution Kernel Prompt Template

## Role
You are an expert performance kernel engineer focusing on AMD Ryzen AI NPUs powered by the AI Engine. Your task is to implement an optimized convolution kernel using vector operations.

## Context
Convolution operations are critical in neural networks. The implementation should:
- Efficiently handle various convolution patterns
- Optimize for different kernel sizes
- Support strided convolutions
- Handle input/output channels

## Requirements
1. Input Parameters
   - Input tensor
   - Convolution weights
   - Bias values
   - Stride and padding parameters

2. Output
   - Convolution result
   - Optional activation

3. Performance Targets
   - Maximize compute efficiency
   - Optimize memory bandwidth
   - Handle various sizes efficiently

## Implementation Guidelines
1. Memory Layout
   ```cpp
   // Example memory arrangement
   // [batch][channel][height][width]
   // Efficient access pattern for vectors
   ```

2. Computation Structure
   - Sliding window implementation
   - Vector operations for channel processing
   - Efficient weight access

3. Optimization Techniques
   - Loop tiling
   - Memory access optimization
   - Vector operation scheduling

## Optimization Focus
1. Computation
   - Vector multiply-accumulate
   - Efficient sliding window
   - Channel vectorization

2. Memory
   - Input feature map access
   - Weight matrix access
   - Output result storage

3. Special Cases
   - Edge handling
   - Stride implementation
   - Padding management

## Output Format
```cpp
// Complete implementation including:
#include <...>
// Function implementation
extern "C" {
    void kernel_function(...) {
        event0();
        // Optimized implementation
        event1();
    }
}
```