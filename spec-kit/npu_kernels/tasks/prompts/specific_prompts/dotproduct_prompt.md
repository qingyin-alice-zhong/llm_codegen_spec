# Dot Product Kernel Prompt Template

## Role
You are an expert performance kernel engineer focusing on AMD Ryzen AI NPUs powered by the AI Engine. Your task is to implement an optimized dot product kernel using vector operations.

## Context
The dot product operation is a fundamental computation in many AI/ML workloads. The implementation should:
- Utilize vector operations for efficient computation
- Optimize memory access patterns
- Handle bias addition and ReLU activation
- Support INT8 data type

## Requirements
1. Input Parameters
   - Vector arrays (a, b)
   - Bias value
   - Array dimensions

2. Output
   - INT8 result after dot product, bias addition, and ReLU

3. Performance Targets
   - Maximize vector unit utilization
   - Efficient memory access
   - Minimal overhead

## Implementation Guidelines
1. Vector Operations
   ```cpp
   // Example vector declarations and loading
   using vec8_t = aie::vector<int8_t, 16>;
   vec8_t va = aie::load_v(a_ptr);
   vec8_t vb = aie::load_v(b_ptr);

   // For int8*int8, use acc48 accumulator and initialize with {}
   aie::accum<acc48, 16> acc{};
   
   // Use mul/mac without explicit template args when possible
   acc = aie::mul(va, vb);
   ```

2. Memory Access
   - Aligned vector loads
   - Sequential access pattern
   - Proper handling of remainders

3. Computation Flow
   - Dot product calculation
   - Bias addition
   - ReLU activation
   - Result storage

## Optimization Focus
1. Vectorization
   - Full vector utilization
   - Efficient vector operations
   - Proper data alignment

2. Memory
   - Coalesced memory access
   - Minimal cache misses
   - Efficient data movement

3. Performance
   - Low instruction overhead
   - High throughput
   - Efficient resource usage

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