# C++ Code Standards for NPU Kernel Development

## General Guidelines
1. Code Structure
   - Clear function organization
   - Consistent naming conventions
   - Proper documentation
   - Version control tags

2. Performance Critical Code
   - Explicit vectorization
   - Memory alignment declarations
   - Performance event markers
   - Optimization annotations

## Coding Rules
1. Function Implementation
   ```cpp
   extern "C" {
     // Function declaration
     void kernel_function(params...) {
       // Performance markers
       event0();
       
       // Implementation
       
       event1();
     }
   }
   ```

2. Memory Management
   - Proper alignment declarations
   - Vector type usage
   - DMA transfer patterns
   - Buffer management

3. Optimization Practices
   - Loop unrolling
   - Vector intrinsics usage
   - Memory access patterns
   - Register allocation hints

## Documentation Requirements
1. Function Headers
   ```cpp
   /**
    * @brief Brief description
    * @param param1 Description of param1
    * @param param2 Description of param2
    * @return Description of return value
    */
   ```

2. Performance Documentation
   - Expected performance metrics
   - Optimization decisions
   - Hardware requirements
   - Known limitations

## Testing Standards
1. Correctness Testing
   - Reference implementation comparison
   - Edge case validation
   - Error handling

2. Performance Testing
   - Cycle count measurement
   - Memory bandwidth utilization
   - Resource usage monitoring