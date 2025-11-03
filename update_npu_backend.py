#!/usr/bin/env python3
"""
Update NPU backend with all working kernels from benchmark results.
"""

import json
import os
import re
from pathlib import Path

def load_benchmark_results(json_file: str):
    """Load benchmark results from JSON."""
    with open(json_file, 'r') as f:
        return json.load(f)

def extract_description_from_generate_py(kernel_dir: str) -> str:
    """Extract description from generate.py file."""
    generate_path = os.path.join("npueval_dataset", kernel_dir, "generate.py")
    
    if not os.path.exists(generate_path):
        return f"A {kernel_dir} kernel."
    
    try:
        with open(generate_path, 'r') as f:
            content = f.read()
        
        # Look for description variable
        desc_match = re.search(r'description\s*=\s*["\']([^"\']+)["\']', content)
        if desc_match:
            return desc_match.group(1)
        
        # Look for docstring patterns
        docstring_match = re.search(r'"""([^"]+)"""', content)
        if docstring_match:
            # Take first line of docstring
            lines = docstring_match.group(1).strip().split('\n')
            return lines[0].strip()
        
        # Look for comment patterns
        comment_match = re.search(r'#\s*([^\n]+)', content)
        if comment_match:
            return comment_match.group(1).strip()
        
        return f"A {kernel_dir} kernel."
        
    except Exception as e:
        print(f"Warning: Could not extract description from {generate_path}: {e}")
        return f"A {kernel_dir} kernel."

def generate_backend_code(successful_kernels: list, kernel_cycles: dict) -> str:
    """Generate the updated backend code."""
    
    # Generate kernel paths dictionary
    kernel_paths = []
    for kernel in successful_kernels:
        kernel_paths.append(f'            "{kernel}": "npueval_dataset/{kernel}/kernel_func.cc",')
    
    # Generate kernel descriptions dictionary
    kernel_descriptions = []
    for kernel in successful_kernels:
        description = extract_description_from_generate_py(kernel)
        kernel_descriptions.append(f'            "{kernel}": "{description}",')
    
    # Generate testbench paths dictionary
    testbench_paths = []
    for kernel in successful_kernels:
        testbench_paths.append(f'            "{kernel}": "npueval_dataset/{kernel}/test.py",')
    
    # Generate scalar times dictionary
    scalar_times = []
    for kernel in successful_kernels:
        cycles = kernel_cycles.get(kernel, 0)
        scalar_times.append(f'            "{kernel}": {cycles},')
    
    return {
        'kernel_paths': '\n'.join(kernel_paths),
        'kernel_descriptions': '\n'.join(kernel_descriptions),
        'testbench_paths': '\n'.join(testbench_paths),
        'scalar_times': '\n'.join(scalar_times)
    }

def update_backend_file(backend_file: str, code_sections: dict):
    """Update the backend file with new kernel information."""
    
    with open(backend_file, 'r') as f:
        content = f.read()
    
    # Update npu_kernel_path
    kernel_path_pattern = r'(self\.npu_kernel_path = \{[^}]*?# ===== NPUEval dataset =====\n)(.*?)(\n        \})'
    replacement = f'\\g<1>{code_sections["kernel_paths"]}\\3'
    content = re.sub(kernel_path_pattern, replacement, content, flags=re.DOTALL)
    
    # Update npu_kernel_description
    desc_pattern = r'(self\.npu_kernel_description = \{[^}]*?# ===== NPUEval dataset =====\n)(.*?)(\n        \})'
    replacement = f'\\g<1>{code_sections["kernel_descriptions"]}\\3'
    content = re.sub(desc_pattern, replacement, content, flags=re.DOTALL)
    
    # Update npu_kernel_testbench_path
    testbench_pattern = r'(self\.npu_kernel_testbench_path = \{[^}]*?# ===== NPUEval dataset =====\n)(.*?)(\n        \})'
    replacement = f'\\g<1>{code_sections["testbench_paths"]}\\3'
    content = re.sub(testbench_pattern, replacement, content, flags=re.DOTALL)
    
    # Update npu_kernel_scalar_time
    scalar_pattern = r'(self\.npu_kernel_scalar_time = \{[^}]*?)(.*?)(\n        \})'
    replacement = f'\\g<1>{code_sections["scalar_times"]}\\3'
    content = re.sub(scalar_pattern, replacement, content, flags=re.DOTALL)
    
    return content

def main():
    print("🔄 Updating NPU backend with benchmark results...")
    
    # Load benchmark results
    benchmark_file = "npueval_dataset/kernel_benchmark_results.json"
    data = load_benchmark_results(benchmark_file)
    
    # Extract successful kernels and their cycle counts
    successful_kernels = []
    kernel_cycles = {}
    
    for result in data['results']:
        if result['success'] and result['metrics'].get('avg_cycles') is not None:
            kernel_name = result['kernel_name']
            cycles = int(result['metrics']['avg_cycles'])
            
            successful_kernels.append(kernel_name)
            kernel_cycles[kernel_name] = cycles
    
    print(f"Found {len(successful_kernels)} successful kernels:")
    for kernel in sorted(successful_kernels):
        cycles = kernel_cycles[kernel]
        print(f"  • {kernel:<35} {cycles:>8} cycles")
    print()
    
    # Generate code sections
    print("🔧 Generating backend code sections...")
    code_sections = generate_backend_code(successful_kernels, kernel_cycles)
    
    # Update backend file
    backend_file = "backends/npu_new/npu_new_backend.py"
    print(f"📝 Updating {backend_file}...")
    
    updated_content = update_backend_file(backend_file, code_sections)
    
    # Write updated content
    with open(backend_file, 'w') as f:
        f.write(updated_content)
    
    print(f"✅ Successfully updated backend with {len(successful_kernels)} kernels!")
    print()
    print("🚀 You can now run:")
    print("python main.py --provider google --model gemini-2.5-flash --backend npu_new --npueval_all --max_iterations 2")

if __name__ == "__main__":
    main()


