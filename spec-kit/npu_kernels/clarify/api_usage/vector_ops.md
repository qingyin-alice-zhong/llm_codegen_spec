# Vector Operations Guide for AIE API

## Data Types and Vectors
1. Basic Vector Types
   ```cpp
   // Include necessary headers
   #include <aie_api/aie.hpp>
   
   // Vector type definitions
   template<int N>
   using v_int8 = aie::vector<int8_t, N>;
   template<int N>
   using v_int32 = aie::vector<int32_t, N>;
   
   // Common vector sizes
   constexpr int VEC_SIZE = 16;  // Example fixed size
   using vec8_t = v_int8<VEC_SIZE>;
   using vec32_t = v_int32<VEC_SIZE>;
   ```

2. Accumulator Types
   ```cpp
   // Accumulator type definitions
   template<int N>
   using acc32_t = aie::accum<int32_t, N>;
   
   // Create and initialize accumulator
   acc32_t<VEC_SIZE> acc{};  // Default constructor zeros the accumulator
   
   // Alternative initialization
   auto acc = acc32_t<VEC_SIZE>{};
   ```

3. Vector Operations
   ```cpp
   // Basic vector types
   template<int N>
   using v_int8 = aie::vector<int8_t, N>;
   template<int N>
   using v_int32 = aie::vector<int32_t, N>;
   
   // Accumulator type with correct tag
   template<int N>
   using acc32_t = aie::accum<acc32, N>;
   
   // Memory operations with explicit types
   v_int8<16> vec8 = aie::load_v<int8_t>(ptr);     // Load int8 vector
   v_int32<16> vec32 = aie::load_v<int32_t>(ptr);  // Load int32 vector
   aie::store_v(ptr, vec8);                         // Store vector
   
   // Type conversion with saturation - manual implementation since aie::saturate is not available
   v_int8<16> vec8;
   for (int k = 0; k < 16; ++k) {
       int32_t val = vec32[k];
       val = std::min(std::max(val, (int32_t)-128), (int32_t)127);
       vec8[k] = (int8_t)val;
   }
   
   // Accumulation with proper initialization
   acc32_t<16> acc{};  // Default constructor zeros the accumulator
   acc = aie::mul(vec1, vec2);  // For INT8*INT8 use mul instead of mac
   
   // Vector extraction from accumulator
   v_int32<16> result = acc.to_vector<int32_t>();  // Convert acc to vector
   
   // ReLU implementation
   v_int8<16> zero = aie::zeros<int8_t, 16>();
   v_int8<16> result = aie::max(vec, zero);
   ```

## Advanced Operations

1. Type Conversion and Saturation
   ```cpp
   // Converting between vector types with saturation
   aie::vector<int32_t, 16> vec32;
   aie::vector<int8_t, 16> vec8;
   
   // Convert int32 to int8 with manual saturation
   for (int k = 0; k < 16; ++k) {
       int32_t val = vec32[k];
       val = std::min(std::max(val, (int32_t)-128), (int32_t)127);
       vec8[k] = (int8_t)val;
   }
   
   // Initialize vectors and accumulators
   vec8 = aie::vector<int8_t, 16>{};  // Zero-initialized vector
   acc = aie::accum<acc32, 16>{};     // Zero-initialized accumulator
   ```

2. Dot Product Implementation
   ```cpp
   // Dot product with accumulation
   aie::accum<acc32, 16> acc = aie::zeros<acc32, 16>();
   
   for (int i = 0; i < size; i += 16) {
       auto vec1 = aie::load_v(ptr1 + i);
       auto vec2 = aie::load_v(ptr2 + i);
       acc = aie::mac(acc, vec1, vec2);
   }
   
   // Extract final result
   auto result = acc.to_vector();
   ```

3. Special Functions
   ```cpp
   // Maximum/Minimum
   result = aie::max(vec1, vec2);
   result = aie::min(vec1, vec2);
   
   // Absolute value
   result = aie::abs(vec);
   
   // ReLU implementation
   result = aie::max(vec, aie::zeros<int8_t, 16>());
   ```

## Optimization Guidelines

1. Memory Access Patterns
   ```cpp
   // Aligned vector access
   if (aie::is_aligned<16>(ptr)) {
       vec = aie::load_v(ptr);
   }
   
   // Sequential access with vector operations
   for (int i = 0; i < size; i += 16) {
       vec = aie::load_v(ptr + i);
       vec = aie::mul(vec, scale);
       aie::store_v(out_ptr + i, vec);
   }
   ```

2. Performance Best Practices
   - Use vector operations whenever possible
   - Minimize type conversions
   - Utilize accumulator types for sum operations
   - Chain operations to reduce intermediate storage
   - Consider alignment for memory operations

3. Common Patterns
   ```cpp
   // Vectorized reduction
   aie::accum<acc32, 16> acc = aie::zeros<acc32, 16>();
   for (int i = 0; i < size; i += 16) {
       auto vec = aie::load_v(ptr + i);
       acc = aie::mac(acc, vec, vec);
   }
   
   // Vector operation with bias and ReLU
   auto bias_vec = aie::broadcast<int8_t, 16>(bias);
   result = aie::max(aie::add(vec, bias_vec), aie::zeros<int8_t, 16>());
   ```