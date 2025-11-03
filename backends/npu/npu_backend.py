import os
import re
import subprocess
from typing import Dict
from backend import Backend

class NPUBackend(Backend):
    """AMD Ryzen NPU backend implementation."""
    
    def __init__(self):
        super().__init__("npu")
        
        self.api_docs_path = "/home/jl4257/.conda/envs/aie/lib/python3.12/site-packages/mlir_aie/include/aie_api/aie_doc.hpp"
        
        # NPU-specific kernels (these would be different from CPU kernels)
        self.npu_kernel_path = {
            "scale": "../mlir-aie/aie_kernels/aie2/scale.cc",
            "mv": "../mlir-aie/aie_kernels/aie2/mv.cc",
            "reduce_add": "../mlir-aie/aie_kernels/aie2/reduce_add.cc",
            "reduce_max": "../mlir-aie/aie_kernels/aie2/reduce_max.cc",
            "reduce_min": "../mlir-aie/aie_kernels/aie2/reduce_min.cc",
        }
        self.npu_kernel_description = {
            "scale": "Scale all elements of a tensor with a scale factor.",
            "mv": "Multiply a matrix with a vector.",
            "reduce_add": "Find the sum of elements in a tensor.",
            "reduce_max": "Find the maximum value in a tensor.",
            "reduce_min": "Find the minimum value in a tensor.",
        }
        self.npu_kernel_build_dir = {
            "scale": "../mlir-aie/programming_examples/basic/vector_scalar_mul",
            "mv": "../mlir-aie/programming_examples/basic/matrix_multiplication/matrix_vector",
            "reduce_add": "../mlir-aie/programming_examples/basic/vector_reduce_add",
            "reduce_max": "../mlir-aie/programming_examples/basic/vector_reduce_max",
            "reduce_min": "../mlir-aie/programming_examples/basic/vector_reduce_min",
        }
        self.npu_kernel_original_time = { # in us (microseconds). warmup_iter=10000, iters=100. 
            "scale": 132.5, 
            "mv": 670.1, 
            "reduce_add": 126.2,
            "reduce_max": 122.2,
            "reduce_min": 126.3,
        }
        
        self.input_example_path = "backends/npu/add_input.cc"
        self.output_example_path = "backends/npu/add_output.cc"
    
    def get_available_kernels(self) -> Dict[str, str]:
        """Get available NPU kernels."""
        return self.npu_kernel_path.copy()
    
    def create_prompt(self, complete_code: str, kernel_name: str) -> str:
        """Create NPU-specific optimization prompt."""
        with open(self.input_example_path, "r") as f:
            input_example = f.read()
        with open(self.output_example_path, "r") as f:
            output_example = f.read()
        
        api_docs = ""
        if os.path.exists(self.api_docs_path):
            with open(self.api_docs_path, "r") as f:
                api_docs = f.read()
        else:
            api_docs = "Documentation file not found."

        wrapped_func = self._extract_wrapped_func_block(complete_code)

        return f"""Act as an experienced performance kernel engineer focusing on AMD Ryzen AI NPUs powered by the AI Engine. Implement low-level kernel implementations in C++ using the AI Engine's API. You will be provided with the AI Engine API documentation. Your task is to finish the optimized kernel implementation given the function signature. 

# Steps
1. **Understand Hardware Specifications**: 
   - Familiarize yourself with the architecture and capabilities of AMD Ryzen AI NPUs.
   - Consult the latest documentation and resources provided by AMD regarding AI Engine specifications.
2. **AI Engine API Utilization**:
   - Explore the AI Engine's API for available functionalities and optimizations.
   - Identify key functions and methods within the API that are relevant to your kernel implementation.
3. **Kernel Implementation**:
   - Write efficient kernel code in C++ that leverages the AI Engine API optimally.
   - Ensure the implementation aligns with best practices in performance and low-level optimizations for NPUs.

# Output Format
Produce C++ code snippets as required for the kernel implementation. You will be provided with the function call in `extern "C"` blocks, which will call the internal function implementation. You should complete the function implementation in the C++ template format. Do not include any other text other than the entire code snippet. ONLY output the entire code snippet. The code should be complete with the necessary includes, function implementation, and extern "C" block. 
   
# AI Engine API Documentation
Here are some relevant parts of the AI Engine API documentation that might be useful:
```
{api_docs}
```

# Examples
## Input
Implement an add kernel that does the pointwise addition of 2 tensors. The function signature wrapped in `extern "C"` is as follows:
```cpp
{input_example}
```

## Output
```cpp
{output_example}
```

# Task
Now implement a {self.npu_kernel_description[kernel_name]} kernel: {self.npu_kernel_description[kernel_name]}. The function signature wrapped in `extern "C"` is as follows:
```cpp
{wrapped_func}
```

"""
    
    
    def create_output(self, input_complete_code: str, llm_gen_code: str, kernel_name: str) -> str:
        """Return the output complete code. 
        Here we directly return the code generated by the LLM, since we assume LLM will generate the complete code here. """
        return llm_gen_code
    
    def get_output_code_name(self, kernel_name: str) -> str:
        """Get the output code name."""
        return f"{kernel_name}.cc"
    
    def compile_and_run(self, output_code_path: str, kernel_name: str, 
                       output_dir: str) -> Dict:
        """Compile and run C++ code on NPU using make with VPATH override."""
        # NPU compilation requires using the mlir-aie build system
        build_dir = self.npu_kernel_build_dir[kernel_name]
        
        compile_results = {
            "source_file": output_code_path,
            "build_directory": build_dir,
            "compilation_successful": False,
            "execution_successful": False,
            "compile_output": "",
            "compile_error": "",
            "run_output": "",
            "run_error": ""
        }
        
        try:
            # make clean first
            clean_cmd = f"cd {build_dir} && make clean"
            compile_process = subprocess.run(
                clean_cmd,
                shell=True,  # Use shell to inherit full environment
                capture_output=True,
                text=True,
            )
            
            # Get the directory containing the optimized source file
            source_dir = os.path.dirname(os.path.abspath(output_code_path))
            
            # Compile using make with VPATH as command-line argument (overrides Makefile)
            compile_cmd = f"cd {build_dir} && make VPATH={source_dir}"
            
            compile_process = subprocess.run(
                compile_cmd,
                shell=True,  # Use shell to inherit full environment
                capture_output=True,
                text=True,
                timeout=120
            )
            
            compile_results["compile_cmd"] = compile_cmd
            compile_results["compile_output"] = compile_process.stdout
            compile_results["compile_error"] = compile_process.stderr
            compile_results["compilation_successful"] = compile_process.returncode == 0
            
            if compile_results["compilation_successful"]:
                # Run the built executable with VPATH override
                run_cmd = f"cd {build_dir} && make VPATH={source_dir} run_perf"
                
                run_process = subprocess.run(
                    run_cmd,
                    shell=True,  # Use shell to inherit full environment
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                compile_results["run_cmd"] = run_cmd
                compile_results["run_output"] = run_process.stdout
                compile_results["run_error"] = run_process.stderr
                compile_results["execution_successful"] = run_process.returncode == 0
                
        except subprocess.TimeoutExpired:
            compile_results["compile_error"] = "Compilation or execution timeout"
        except Exception as e:
            compile_results["compile_error"] = f"Error during compilation/execution: {str(e)}"
        
        return compile_results
    
    def analyze_run_output(self, run_output: str, kernel_name: str) -> Dict:
        """Analyze NPU benchmark output."""
        analysis = {
            "verification_success": False,
            # "runtime": None, # unit in microseconds
            "original_time": self.npu_kernel_original_time[kernel_name], # in us (microseconds). warmup_iter=100, iters=100. Directly get the already saved original time
            "optimized_time": None,
            "speedup": None,
        }
        
        lines = run_output.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Look for verification results - simple PASS! or FAIL!
            if line == "PASS!":
                analysis["verification_success"] = True
            elif line == "FAIL!":
                analysis["verification_success"] = False
            
            # Look for NPU timing information
            elif "Avg NPU time:" in line:
                try:
                    # Extract the time value and unit
                    # Format: "Avg NPU time: 572us."
                    time_part = line.split("Avg NPU time:")[-1].strip()
                    
                    if time_part.endswith("us."):
                        # Remove "us." and convert to float, keep in microseconds
                        time_str = time_part.replace("us.", "").strip()
                        analysis["optimized_time"] = float(time_str)
                    elif time_part.endswith("ms."):
                        # Remove "ms." and convert to microseconds
                        time_str = time_part.replace("ms.", "").strip()
                        time_ms = float(time_str)
                        analysis["optimized_time"] = time_ms * 1000.0  # Convert ms to us
                    elif time_part.endswith("s."):
                        # Remove "s." and convert to microseconds
                        time_str = time_part.replace("s.", "").strip()
                        time_s = float(time_str)
                        analysis["optimized_time"] = time_s * 1_000_000.0  # Convert s to us
                        
                except (ValueError, IndexError):
                    # If parsing fails, leave runtime as None
                    pass
        
        if analysis["optimized_time"] is not None:
            analysis["speedup"] = analysis["original_time"] / analysis["optimized_time"]

        return analysis
    
    def get_system_info(self) -> str:
        """Get NPU system information."""
        return """NPU Architecture:
- AMD Ryzen AI NPU (XDNA architecture)
- Dedicated AI acceleration units
- Optimized for INT8/FP16 computations
- Specialized memory subsystem for AI workloads
- Support for quantized operations""" 
    
    def _extract_wrapped_func_block(self, complete_code: str) -> str:
        """Extract the extern 'C' function block from C++ code."""
        lines = complete_code.split('\n')
        
        # Find the start of extern "C" block
        extern_start_line = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('extern "C"') and '{' in stripped:
                extern_start_line = i
                break
            elif stripped == 'extern "C"':
                # Check if next line has opening brace
                if i + 1 < len(lines) and '{' in lines[i + 1]:
                    extern_start_line = i
                    break
        
        if extern_start_line == -1:
            return "// No extern C block found"
        
        # Find the matching closing brace
        open_braces = 0
        extern_end_line = -1
        found_opening_brace = False
        
        for i in range(extern_start_line, len(lines)):
            line = lines[i]
            
            # Count braces
            for char in line:
                if char == '{':
                    open_braces += 1
                    found_opening_brace = True
                elif char == '}':
                    open_braces -= 1
            
            # If we found the opening brace and braces are balanced, we found the end
            if found_opening_brace and open_braces == 0:
                extern_end_line = i
                break
        
        if extern_end_line == -1:
            return "// Could not find end of extern C block"
        
        # Extract the extern "C" block
        extern_block_lines = lines[extern_start_line:extern_end_line + 1]
        return '\n'.join(extern_block_lines)