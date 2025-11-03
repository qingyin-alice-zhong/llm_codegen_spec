import os
import re
import subprocess
from typing import Dict
from backend import Backend

class CPUBackend(Backend):
    """CPU backend implementation using Intel MKL optimization."""
    
    def __init__(self):
        super().__init__("cpu")
        
        # PolyBench kernels for CPU
        self.polybench_code_path = {
            # datamining
            "correlation": "polybench/datamining/correlation/correlation_simple.c",
            "covariance": "polybench/datamining/covariance/covariance_simple.c",
            # linear-algebra/blas
            "gemm": "polybench/linear-algebra/blas/gemm/gemm_simple.c",
            "gemver": "polybench/linear-algebra/blas/gemver/gemver_simple.c",
            "gesummv": "polybench/linear-algebra/blas/gesummv/gesummv_simple.c",
            "symm": "polybench/linear-algebra/blas/symm/symm_simple.c",
            "syr2k": "polybench/linear-algebra/blas/syr2k/syr2k_simple.c",
            "syrk": "polybench/linear-algebra/blas/syrk/syrk_simple.c",
            "trmm": "polybench/linear-algebra/blas/trmm/trmm_simple.c",
            # linear-algebra/kernels
            "2mm": "polybench/linear-algebra/kernels/2mm/2mm_simple.c",
            "3mm": "polybench/linear-algebra/kernels/3mm/3mm_simple.c",
            "atax": "polybench/linear-algebra/kernels/atax/atax_simple.c",
            "bicg": "polybench/linear-algebra/kernels/bicg/bicg_simple.c",
            "doitgen": "polybench/linear-algebra/kernels/doitgen/doitgen_simple.c",
            "mvt": "polybench/linear-algebra/kernels/mvt/mvt_simple.c",
            # linear-algebra/solvers
            "cholesky": "polybench/linear-algebra/solvers/cholesky/cholesky_simple.c",
            "durbin": "polybench/linear-algebra/solvers/durbin/durbin_simple.c",
            "gramschmidt": "polybench/linear-algebra/solvers/gramschmidt/gramschmidt_simple.c",
            "lu": "polybench/linear-algebra/solvers/lu/lu_simple.c",
            "ludcmp": "polybench/linear-algebra/solvers/ludcmp/ludcmp_simple.c",
            "trisolv": "polybench/linear-algebra/solvers/trisolv/trisolv_simple.c",
            # medley
            "deriche": "polybench/medley/deriche/deriche_simple.c",
            "floyd-warshall": "polybench/medley/floyd-warshall/floyd-warshall_simple.c",
            "nussinov": "polybench/medley/nussinov/nussinov_simple.c",
            # stencils
            "adi": "polybench/stencils/adi/adi_simple.c",
            "fdtd-2d": "polybench/stencils/fdtd-2d/fdtd-2d_simple.c",
            "heat-3d": "polybench/stencils/heat-3d/heat-3d_simple.c",
            "jacobi-1d": "polybench/stencils/jacobi-1d/jacobi-1d_simple.c",
            "jacobi-2d": "polybench/stencils/jacobi-2d/jacobi-2d_simple.c",
            "seidel-2d": "polybench/stencils/seidel-2d/seidel-2d_simple.c" 
        }
    
    def get_available_kernels(self) -> Dict[str, str]:
        """Get available CPU kernels."""
        return self.polybench_code_path.copy()
    
    def create_prompt(self, complete_code: str, kernel_name: str) -> str:
        """Create CPU-specific optimization prompt."""
        # Extract the kernel function implementation and optimized signature
        kernel_func_name = f"kernel_{kernel_name}"
        kernel_func_code = self._extract_function_from_code(complete_code, kernel_func_name)
        
        if not kernel_func_code:
            raise ValueError(f"Could not find function '{kernel_func_name}' in code")
        
        # Extract the optimized function signature
        optimized_func_name = f"kernel_{kernel_name}_optimized"
        optimized_func_signature = self._extract_function_signature_from_code(complete_code, optimized_func_name)

        if not optimized_func_signature:
            raise ValueError(f"Could not find function '{optimized_func_name}' in code")
        
        return f"""You are an expert in high-performance computing and kernel engineering on the CPU. You are familiar with different optimized libraries and their performance characteristics, including Intel MKL. You also know the performance improvement techniques, including vectorization, memory access optimization, tiling, unrolling, etc. Here we focus on the single-threaded performance improvement. Don't use multi-threading. 

Given the following code `{kernel_name}.c`:
```c
{complete_code}
```

with the following kernel implementation: 
```c
{kernel_func_code}
```

Task: Analyze this kernel and generate an optimized kernel implementation to get better performance while maintaining functional equivalence. You should first consider if the kernel can be implemented using a single corresponding MKL function. If not, consider if the kernel can be decomposed into multiple MKL calls. If there is no way to implement the kernel using MKL, or you think MKL cannot get good performance, then consider applying optimizations directly to the kernel, such as vectorization, memory access optimization, tiling, unrolling, etc. You should only use single thread for the optimized kernel implementation. 

Machine we are using: 
- Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz
- L1d cache: 1.5MB
- L1i cache: 1.5MB
- L2 cache: 48MB
- L3 cache: 71.5MB
- Supports SSE, AVX2, AVX512

Requirements:
1. Identify which a single MKL function or a combination of MKL functions can replace this kernel to get better performance.
2. Provide the equivalent MKL implementation. 
3. Include necessary headers and initialization code. 
4. Ensure functional equivalence.
5. If there is no available MKL function that can get good performance, then consider applying optimizations directly to the kernel, such as vectorization, memory access optimization, tiling, unrolling, etc.
6. If neither MKL nor optimizations can get good performance, then just fallback to the default implementation.

Output format:
You should only output the optimized kernel implementation which follows the exact function signature as follows: 
```c
{optimized_func_signature}
```

Do not include any other text other than the optimized kernel implementation. ONLY output the optimized kernel implementation within the code block. 

"""
    
    def create_output(self, input_complete_code: str, llm_gen_code: str, kernel_name: str) -> str:
        """Create the final optimized complete code by replacing the optimized function."""
        optimized_func_name = f"kernel_{kernel_name}_optimized"
        return self._replace_function_in_code(input_complete_code, optimized_func_name, llm_gen_code)
    
    def _extract_function_from_code(self, complete_code: str, function_name: str) -> str:
        """Extract C function implementation."""
        # Pattern to match function definition and its body
        pattern = rf'({function_name}\s*\([^{{]*\{{\s*.*?\n}})'
        
        # Search for the function with DOTALL flag to match across newlines
        match = re.search(pattern, complete_code, re.DOTALL)
        
        if match:
            function_code = match.group(1)
            
            # Count braces to find the complete function body
            open_braces = 0
            function_start = complete_code.find(match.group(0))
            
            # Find the actual start of the function (including return type and signature)
            lines = complete_code[:function_start + len(match.group(0))].split('\n')
            
            # Look backwards to find the function signature start
            for i in range(len(lines) - 1, -1, -1):
                if function_name in lines[i]:
                    # Found the line with function name, now find the return type
                    j = i
                    while j >= 0 and not (lines[j].strip().startswith('void') or 
                                         lines[j].strip().startswith('int') or
                                         lines[j].strip().startswith('double') or
                                         lines[j].strip().startswith('float') or
                                         lines[j].strip().startswith('char') or
                                         lines[j].strip().startswith('static')):
                        j -= 1
                    if j >= 0:
                        function_start_line = j
                        break
            
            # Extract from function start to end
            lines_from_start = complete_code.split('\n')[function_start_line:]
            result_lines = []
            open_braces = 0
            found_opening_brace = False
            
            for line in lines_from_start:
                result_lines.append(line)
                
                # Count braces
                for char in line:
                    if char == '{':
                        open_braces += 1
                        found_opening_brace = True
                    elif char == '}':
                        open_braces -= 1
                
                # If we found the opening brace and braces are balanced, we're done
                if found_opening_brace and open_braces == 0:
                    break
            
            return '\n'.join(result_lines)
        
        return ""
    
    def _extract_function_signature_from_code(self, complete_code: str, function_name: str) -> str:
        """Extract C function signature."""
        lines = complete_code.split('\n')
        
        # Find the line containing the function name
        function_start_line = -1
        for i, line in enumerate(lines):
            if function_name in line and '(' in line:
                function_start_line = i
                break
        
        if function_start_line == -1:
            return ""
        
        # Look backwards to find the return type
        signature_start_line = function_start_line
        for i in range(function_start_line, -1, -1):
            line_stripped = lines[i].strip()
            if (line_stripped.startswith('void') or 
                line_stripped.startswith('int') or
                line_stripped.startswith('double') or
                line_stripped.startswith('float') or
                line_stripped.startswith('char') or
                line_stripped.startswith('static')):
                signature_start_line = i
                break
        
        # Extract lines from return type to the opening brace
        signature_lines = []
        for i in range(signature_start_line, len(lines)):
            line = lines[i]
            signature_lines.append(line)
            
            # Stop when we reach the opening brace
            if '{' in line:
                # Remove the opening brace and everything after it
                brace_pos = line.find('{')
                last_line = line[:brace_pos].rstrip()
                signature_lines[-1] = last_line
                break
        
        # Join the lines and clean up
        signature = '\n'.join(signature_lines).strip()
        return signature
    
    def _replace_function_in_code(self, complete_code: str, function_name: str, 
                               new_implementation: str) -> str:
        """Replace C function implementation."""
        lines = complete_code.split('\n')
        
        # Find the function start and end
        function_start_line = -1
        function_end_line = -1
        
        # Find the line containing the function name
        for i, line in enumerate(lines):
            if function_name in line and '(' in line:
                # Look backwards to find the return type
                for j in range(i, -1, -1):
                    line_stripped = lines[j].strip()
                    if (line_stripped.startswith('void') or 
                        line_stripped.startswith('int') or
                        line_stripped.startswith('double') or
                        line_stripped.startswith('float') or
                        line_stripped.startswith('char') or
                        line_stripped.startswith('static')):
                        function_start_line = j
                        break
                break
        
        if function_start_line == -1:
            return complete_code  # Function not found, return original
        
        # Find the end of the function by counting braces
        open_braces = 0
        found_opening_brace = False
        
        for i in range(function_start_line, len(lines)):
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
                function_end_line = i
                break
        
        if function_end_line == -1:
            return complete_code  # Could not find function end, return original
        
        # Replace the function
        new_lines = (lines[:function_start_line] + 
                    [new_implementation] + 
                    lines[function_end_line + 1:])
        
        return '\n'.join(new_lines)
    
    
    def get_output_code_name(self, kernel_name: str) -> str:
        """Get the output code name."""
        return f"{kernel_name}_optimized.c"
    
    
    def compile_and_run(self, output_code_path: str, kernel_name: str, 
                       output_dir: str) -> Dict:
        """Compile and run C code with GCC and MKL."""
        executable_path = os.path.join(output_dir, f"{kernel_name}_optimized")
        
        compile_results = {
            "source_file": output_code_path,
            "executable": executable_path,
            "compilation_successful": False,
            "execution_successful": False,
            "compile_output": "",
            "compile_error": "",
            "run_output": "",
            "run_error": ""
        }
        
        try:
            # Compile the code
            compile_cmd = [
                "gcc", "-o", executable_path, output_code_path, 
                "-I", "${MKL_ROOT}/include", 
                "-L", "${MKL_ROOT}/lib/intel64", 
                "-lmkl_intel_lp64", "-lmkl_sequential", "-lmkl_core", 
                "-lpthread", "-lm", "-march=native"
            ]
            
            compile_process = subprocess.run(
                compile_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            compile_results["compile_cmd"] = " ".join(compile_cmd)
            compile_results["compile_output"] = compile_process.stdout
            compile_results["compile_error"] = compile_process.stderr
            compile_results["compilation_successful"] = compile_process.returncode == 0
            
            if compile_results["compilation_successful"]:
                # Run the executable
                run_process = subprocess.run(
                    [executable_path],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                compile_results["run_output"] = run_process.stdout
                compile_results["run_error"] = run_process.stderr
                compile_results["execution_successful"] = run_process.returncode == 0
                
        except subprocess.TimeoutExpired:
            compile_results["compile_error"] = "Compilation or execution timeout"
        except Exception as e:
            compile_results["compile_error"] = f"Error during compilation/execution: {str(e)}"
        
        return compile_results
    
    def analyze_run_output(self, run_output: str, kernel_name: str) -> Dict:
        """Analyze CPU benchmark output."""
        analysis = {
            "verification_success": False,
            "original_time": None,
            "optimized_time": None, 
            "speedup": None,
        }
        
        lines = run_output.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for verification results
            if "Verification Results:" in line:
                # Check next few lines for PASS/FAIL
                for j in range(i+1, min(i+5, len(lines))):
                    next_line = lines[j].strip()
                    if "PASS:" in next_line:
                        analysis["verification_success"] = True
                        break
                    elif "FAIL:" in next_line:
                        analysis["verification_success"] = False
                        break
            
            # Look for performance results
            elif "Performance Results:" in line:
                # Look for timing information in next few lines
                for j in range(i+1, min(i+10, len(lines))):
                    next_line = lines[j].strip()
                    
                    # Extract original kernel time
                    if "Original kernel average time:" in next_line:
                        try:
                            time_str = next_line.split(":")[-1].replace("seconds", "").strip()
                            analysis["original_time"] = float(time_str)
                        except (ValueError, IndexError):
                            pass
                    
                    # Extract optimized kernel time
                    elif "Optimized kernel average time:" in next_line:
                        try:
                            time_str = next_line.split(":")[-1].replace("seconds", "").strip()
                            analysis["optimized_time"] = float(time_str)
                        except (ValueError, IndexError):
                            pass
                    
                    # Extract speedup
                    elif "Speedup:" in next_line:
                        try:
                            speedup_str = next_line.split(":")[-1].strip()
                            analysis["speedup"] = float(speedup_str)
                        except (ValueError, IndexError):
                            pass
        
        return analysis
    
    def get_system_info(self) -> str:
        """Get CPU system information."""
        return """Machine we are using: 
- Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz
- L1d cache: 1.5MB
- L1i cache: 1.5MB
- L2 cache: 48MB
- L3 cache: 71.5MB
- Supports SSE, AVX2, AVX512""" 