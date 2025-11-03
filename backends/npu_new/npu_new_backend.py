import os
import re
import subprocess
from typing import Dict
from backend import Backend

class NPUNewBackend(Backend):
    def __init__(self):
        super().__init__("npu_new")
        self.spec_kit_root = os.path.join(os.path.dirname(__file__), "..", "..", "spec-kit/npu_kernels")
        self._load_spec_kit_content()
        
        # Load NPU-specific configurations
        from .npueval_config import (
            get_npueval_kernels,
            get_npueval_kernel_descriptions,
            get_npueval_kernel_paths,
            get_npueval_testbench_paths,
            get_npueval_scalar_times
        )
        
        self.npu_kernel_path = get_npueval_kernel_paths()
        self.npu_kernel_description = get_npueval_kernel_descriptions()
        self.npu_kernel_testbench_path = get_npueval_testbench_paths()
        self.npu_kernel_scalar_time = get_npueval_scalar_times()
        
    def _load_spec_kit_content(self):
        """Load all necessary content from spec-kit structure"""
        # Load constitution
        with open(os.path.join(self.spec_kit_root, "constitution/kernel_engineer_role.md")) as f:
            self.engineer_role = f.read()
        with open(os.path.join(self.spec_kit_root, "constitution/hardware_constraints.md")) as f:
            self.hardware_constraints = f.read()
        with open(os.path.join(self.spec_kit_root, "constitution/code_standards.md")) as f:
            self.code_standards = f.read()
            
        # Load implementation steps
        with open(os.path.join(self.spec_kit_root, "plan/kernel_implementation_steps.md")) as f:
            self.implementation_steps = f.read()
        with open(os.path.join(self.spec_kit_root, "plan/optimization_strategies.md")) as f:
            self.optimization_strategies = f.read()
            
        # Load API documentation
        with open(os.path.join(self.spec_kit_root, "clarify/api_usage/vector_ops.md")) as f:
            self.vector_ops_guide = f.read()
        with open(os.path.join(self.spec_kit_root, "clarify/api_usage/memory_access.md")) as f:
            self.memory_access_guide = f.read()
            
        # Load kernel type specific content
        self.kernel_type_prompts = {}
        prompts_dir = os.path.join(self.spec_kit_root, "tasks/prompts/specific_prompts")
        for prompt_file in os.listdir(prompts_dir):
            if prompt_file.endswith("_prompt.md"):
                kernel_type = prompt_file.replace("_prompt.md", "")
                with open(os.path.join(prompts_dir, prompt_file)) as f:
                    self.kernel_type_prompts[kernel_type] = f.read()
                    
    def _get_kernel_type(self, kernel_name: str) -> str:
        """Determine kernel type based on kernel name"""
        if "dotproduct" in kernel_name:
            return "dotproduct"
        elif "conv" in kernel_name:
            return "convolution"
        # Add more kernel type mappings as needed
        return "compute"  # default type
    
    def get_available_kernels(self) -> Dict[str, str]:
        """Get available NPU kernels."""
        return self.npu_kernel_path.copy()
    
    def create_prompt(self, input_code: str, kernel_name: str) -> str:
        """Create NPU-specific optimization prompt using spec-kit structure."""
        # Get kernel type and specific prompt
        kernel_type = self._get_kernel_type(kernel_name)
        specific_prompt = self.kernel_type_prompts.get(kernel_type, "")
        
        # Get reference implementation
        with open(self.npu_kernel_testbench_path[kernel_name], "r") as f:
            testbench_code = f.read()
        ref_code = re.search(r"# Reference code starts\n(.*?)# Reference code ends", 
                           testbench_code, re.DOTALL).group(1)
        
        # Compose the final prompt
        prompt = f"""# Performance Kernel Engineering Task

## Role and Context
{self.engineer_role}

## Hardware Specifications and Constraints
{self.hardware_constraints}

## Implementation Guidelines
{self.implementation_steps}

## Optimization Strategies
{self.optimization_strategies}

## API Reference
{self.vector_ops_guide}

## Memory Access Patterns
{self.memory_access_guide}

## Specific Kernel Requirements
{specific_prompt}

## Reference Implementation
```python
{ref_code}
```

## Input Function Signature
```cpp
{input_code}
```

Please provide the complete optimized implementation following the code standards
and maintaining the exact function signature. Include only the implementation
code without any explanation or comments.
"""
        return prompt
    
    def create_output(self, input_complete_code: str, llm_gen_code: str, kernel_name: str) -> str:
        """Return the output complete code."""
        return llm_gen_code
    
    def get_output_code_name(self, kernel_name: str) -> str:
        """Get the output code name."""
        return f"{kernel_name}.cc"
    
    def compile_and_run(self, output_code_path: str, kernel_name: str, 
                       output_dir: str) -> Dict:
        """Compile and run C++ code on NPU using make with VPATH override."""
        testbench_path = self.npu_kernel_testbench_path[kernel_name]
        
        compile_results = {
            "source_file": output_code_path,
            "compilation_successful": False,
            "execution_successful": False,
            "compile_output": "",
            "compile_error": "",
            "run_output": "",
            "run_error": ""
        }
        
        try:
            compile_cmd = f"python {testbench_path} --kernel_path {os.path.abspath(output_code_path)}"
            compile_process = subprocess.run(
                compile_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            compile_results["compile_cmd"] = compile_cmd
            compile_results["compile_output"] = compile_process.stdout
            compile_results["compile_error"] = self._filter_compile_warnings(compile_process.stderr)
            compile_results["compilation_successful"] = compile_process.returncode == 0
            
            if compile_results["compilation_successful"]:
                compile_results["execution_successful"] = True
                compile_results["run_output"] = compile_results["compile_output"]
                compile_results["run_error"] = ""
                
        except subprocess.TimeoutExpired:
            compile_results["compile_error"] = "Compilation timeout"
        except Exception as e:
            compile_results["compile_error"] = f"Error during compilation: {str(e)}"
        
        return compile_results
    
    def _filter_compile_warnings(self, compile_error: str) -> str:
        """Filter out compiler setup noise."""
        compile_error_lines = compile_error.split('\n')
        filtered = []
        skip_indices = set()
        n = len(compile_error_lines)
        
        # Remove everything before and including 'End of search list.'
        start_idx = 0
        for i, line in enumerate(compile_error_lines):
            if 'End of search list.' in line:
                start_idx = i + 1
                start_idx += 25  # skip warning block
                break

        # Remove everything from Traceback to end
        end_idx = n
        for i in range(start_idx, n):
            if compile_error_lines[i].startswith('Traceback'):
                end_idx = i
                break
                
        for i in range(start_idx, end_idx):
            if i not in skip_indices:
                filtered.append(compile_error_lines[i])
                
        return '\n'.join(filtered)
    
    def analyze_run_output(self, run_output: str, kernel_name: str) -> Dict:
        """Analyze NPU benchmark output."""
        analysis = {
            "verification_success": False,
            "scalar_time": None,
            "vector_time": None,
            "speedup": None,
        }
        
        lines = run_output.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if line == "PASS!":
                analysis["verification_success"] = True
            elif line == "FAIL!":
                analysis["verification_success"] = False
            elif "First/Min/Avg/Max cycles is" in line:
                try:
                    match = re.search(r'First/Min/Avg/Max cycles is\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)', line)
                    if match:
                        analysis["vector_time"] = float(match.group(3))
                    else:
                        parts = [p.strip() for p in line.split("First/Min/Avg/Max cycles is")[-1].split('/')]
                        if len(parts) >= 3:
                            analysis["vector_time"] = float(parts[2])
                except (ValueError, IndexError):
                    pass
        
        if analysis["vector_time"] is not None:
            analysis["scalar_time"] = self.npu_kernel_scalar_time[kernel_name]
            analysis["speedup"] = analysis["scalar_time"] / analysis["vector_time"]

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
        
        # Find start of extern "C" block
        extern_start_line = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('extern "C"') and '{' in stripped:
                extern_start_line = i
                break
            elif stripped == 'extern "C"':
                if i + 1 < len(lines) and '{' in lines[i + 1]:
                    extern_start_line = i
                    break
        
        if extern_start_line == -1:
            return "// No extern C block found"
        
        # Find matching closing brace
        open_braces = 0
        extern_end_line = -1
        found_opening_brace = False
        
        for i in range(extern_start_line, len(lines)):
            line = lines[i]
            for char in line:
                if char == '{':
                    open_braces += 1
                    found_opening_brace = True
                elif char == '}':
                    open_braces -= 1
            
            if found_opening_brace and open_braces == 0:
                extern_end_line = i
                break
        
        if extern_end_line == -1:
            return "// Could not find end of extern C block"
        
        extern_block_lines = lines[extern_start_line:extern_end_line + 1]
        return '\n'.join(extern_block_lines)

    @property
    def show_speedup_in_feedback(self):
        return True