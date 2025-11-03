# Kernel Implementation Steps

## 1. Analysis Phase
1. Requirements Analysis
   - Understand kernel functionality
   - Identify input/output parameters
   - Determine data types and sizes

2. Reference Code Review
   - Analyze PyTorch/NumPy implementation
   - Identify key operations
   - Note edge cases and special handling

3. Hardware Consideration
   - Review hardware constraints
   - Identify relevant vector operations
   - Consider memory access patterns

## 2. Design Phase
1. Algorithm Design
   - Choose vectorization strategy
   - Plan memory access pattern
   - Design data layout

2. Optimization Planning
   - Identify vectorization opportunities
   - Plan loop transformations
   - Consider memory optimizations

3. Interface Design
   - Match function signature
   - Plan error handling
   - Consider performance monitoring

## 3. Implementation Phase
1. Basic Implementation
   ```cpp
   void kernel_function(params...) {
       event0();
       // Basic implementation
       event1();
   }
   ```

2. Vectorization
   ```cpp
   // Vector operations
   v16int8 vec_a = load_v(a_ptr);
   v16int8 vec_b = load_v(b_ptr);
   v16int8 result = operation_v(vec_a, vec_b);
   ```

3. Memory Optimization
   ```cpp
   // Efficient memory access
   for (int i = 0; i < n; i += VECTOR_SIZE) {
       // Vectorized operations
   }
   ```

## 4. Validation Phase
1. Correctness Testing
   - Compare with reference output
   - Test edge cases
   - Verify numerical accuracy

2. Performance Testing
   - Measure cycle counts
   - Check bandwidth utilization
   - Verify vectorization efficiency

3. Code Review
   - Check coding standards
   - Verify optimization implementation
   - Document key decisions

## 5. Optimization Phase
1. Performance Analysis
   - Identify bottlenecks
   - Measure resource usage
   - Compare with targets

2. Optimization
   - Apply additional optimizations
   - Fine-tune parameters
   - Balance resources

3. Final Validation
   - Verify optimizations
   - Document performance
   - Prepare for integration