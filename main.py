import os
import re
from typing import Dict, List, Optional, Tuple
import json
import argparse
from datetime import datetime
import logging
from agent import KernelCodeGenerator
from llm import LLM
from backend_factory import BackendFactory
from tqdm import tqdm
from plot import plot_speedup

# Configure logging
def setup_logging(output_dir: str):
    """Setup logging to both file and console"""
    log_file = os.path.join(output_dir, f"run.log")
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return log_file

def find_best_iteration_and_create_summary(results_all_iter: Dict, kernel_name: str, backend_name: str) -> Dict:
    """
    Find the best iteration from multiple iterations and create kernel summary.
    
    Args:
        results_all_iter: Dictionary of results from all iterations
        kernel_name: Name of the kernel
        backend_name: Name of the backend
        
    Returns:
        Dictionary containing kernel summary with best iteration results
    """
    # Handle case where only one result is returned (kernel generation failed)
    if not isinstance(results_all_iter, dict) or 'kernel_generation_success' in results_all_iter:
        # Single result returned, use it directly
        best_results = results_all_iter
        best_iteration = 0
    else:
        # Multiple iterations, find the best one
        best_iteration = None
        best_results = None
        
        # Priority order: Better speedup > Verification pass > Execution pass > Compile pass
        best_speedup = -1
        
        for iteration, results in results_all_iter.items():
            # Priority 1: Best speedup
            if (results.get("compilation_success", False) and 
                results.get("execution_success", False) and 
                results.get("verification_success", False) and
                "run_analysis" in results and 
                results["run_analysis"].get("speedup") is not None):
                
                speedup = results["run_analysis"]["speedup"]
                if speedup > best_speedup:
                    best_speedup = speedup
                    best_iteration = iteration
                    best_results = results
                    
        # If no speedup found, try verification pass
        if best_results is None:
            for iteration, results in results_all_iter.items():
                if (results.get("compilation_success", False) and 
                    results.get("execution_success", False) and 
                    results.get("verification_success", False)):
                    best_iteration = iteration
                    best_results = results
                    break
                    
        # If no verification pass, try execution pass
        if best_results is None:
            for iteration, results in results_all_iter.items():
                if (results.get("compilation_success", False) and 
                    results.get("execution_success", False)):
                    best_iteration = iteration
                    best_results = results
                    break
                    
        # If no execution pass, try compilation pass
        if best_results is None:
            for iteration, results in results_all_iter.items():
                if results.get("compilation_success", False):
                    best_iteration = iteration
                    best_results = results
                    break
                    
        # if no compilation pass, try kernel generation pass
        if best_results is None:
            for iteration, results in results_all_iter.items():
                if results.get("kernel_generation_success", False):
                    best_iteration = iteration
                    best_results = results
                    break

        # If nothing passes, use the last iteration
        if best_results is None:
            last_iteration = max(results_all_iter.keys())
            best_results = results_all_iter[last_iteration]
            best_iteration = last_iteration
    
    # Collect status and speedup for each kernel using best results
    kernel_summary = {
        "backend": backend_name,
        "total_iterations": len(results_all_iter) if isinstance(results_all_iter, dict) and 'kernel_generation_success' not in results_all_iter else 1,
        "best_iteration": best_iteration,
        "kernel_generation_success": best_results["kernel_generation_success"],
        "compilation_success": best_results["compilation_success"],
        "execution_success": best_results["execution_success"],
        "verification_success": best_results["verification_success"],
    }
    
    # Add speedup if available
    if "run_analysis" in best_results and best_results["run_analysis"].get("speedup") is not None:
        kernel_summary["speedup"] = best_results["run_analysis"]["speedup"]
        
    # Add iterations breakdown
    if isinstance(results_all_iter, dict) and 'kernel_generation_success' not in results_all_iter:
        iterations_breakdown = {}
        for iteration, results in results_all_iter.items():
            iter_summary = {
                "kernel_generation_success": results.get("kernel_generation_success", False),
                "compilation_success": results.get("compilation_success", False),
                "execution_success": results.get("execution_success", False),
                "verification_success": results.get("verification_success", False),
            }
            if "run_analysis" in results and results["run_analysis"].get("speedup") is not None:
                iter_summary["speedup"] = results["run_analysis"]["speedup"]
            iterations_breakdown[f"iteration_{iteration}"] = iter_summary
        kernel_summary["iterations_breakdown"] = iterations_breakdown
        
    # Log the best iteration selection
    if isinstance(results_all_iter, dict) and 'kernel_generation_success' not in results_all_iter:
        if kernel_summary.get("speedup") is not None:
            logging.info(f"📈 Best iteration for {kernel_name}: Iteration {best_iteration} (Speedup: {kernel_summary['speedup']:.2f}x)")
        elif kernel_summary["verification_success"]:
            logging.info(f"📈 Best iteration for {kernel_name}: Iteration {best_iteration} (Verification passed)")
        elif kernel_summary["execution_success"]:
            logging.info(f"📈 Best iteration for {kernel_name}: Iteration {best_iteration} (Execution passed)")
        elif kernel_summary["compilation_success"]:
            logging.info(f"📈 Best iteration for {kernel_name}: Iteration {best_iteration} (Compilation passed)")
        else:
            logging.info(f"📈 Best iteration for {kernel_name}: Iteration {best_iteration} (Last iteration)")
    else:
        logging.info(f"📈 Single iteration for {kernel_name}")
        
    return kernel_summary


def process_kernels(generator: KernelCodeGenerator, kernel_names: List[str], 
                   available_kernels: Dict[str, str], output_dir: str, 
                   model: str, provider: str, backend_name: str, max_iterations: int = 1) -> Dict:
    """
    Process a list of kernels and save the results.
    
    Args:
        generator: The KernelCodeGenerator instance
        kernel_names: List of kernel names to process
        available_kernels: Dictionary of available kernels for the backend
        output_dir: Base output directory
        model: Model name being used
        provider: Provider name being used
        backend_name: Backend name being used
        max_iterations: Number of feedback max_iterations
        
    Returns:
        Dictionary containing results for all processed kernels
    """
    all_results = {}
    for kernel_name in kernel_names:
        if kernel_name not in available_kernels:
            logging.error(f"Kernel '{kernel_name}' not found for {backend_name} backend. Available kernels: {', '.join(available_kernels.keys())}")
            continue
            
        kernel_code_path = available_kernels[kernel_name]
        kernel_output_dir = os.path.join(output_dir, kernel_name)
        os.makedirs(kernel_output_dir, exist_ok=True)
        
        # Process the kernel
        results_all_iter = generator.process_kernel(kernel_code_path, kernel_name, kernel_output_dir, max_iterations=max_iterations)
        
        # Find best iteration and create kernel summary
        kernel_summary = find_best_iteration_and_create_summary(results_all_iter, kernel_name, backend_name)
        all_results[kernel_name] = kernel_summary
    
    # Save aggregated results
    all_summary_path = os.path.join(output_dir, "all_summary.json")
    with open(all_summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logging.info(f"Summary saved to {all_summary_path}")
    
    return all_results


def main():
    """
    Main function to demonstrate the framework.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate optimized kernel code using LLMs for different hardware backends')
    parser.add_argument('--provider', type=str, required=True, choices=['openai', 'google'],
                      help='LLM provider to use')
    parser.add_argument('--model', type=str, required=True, help='Model name (provider-specific)')
    parser.add_argument('--backend', type=str, required=True, 
                      choices=BackendFactory.get_available_backends(),
                      help='Hardware backend to target')
    parser.add_argument('--kernel', type=str, nargs='+', 
                      help='One or more kernels to process (backend-specific)')
    parser.add_argument('--all', action='store_true',
                      help='Process all available kernels for the selected backend')
    parser.add_argument('--npueval_all', action='store_true',
                      help='Process all working NPUEval kernels (for npu_new backend only)')
    parser.add_argument('--max_iterations', type=int, default=1,
                        help='Number of max_iterations with feedback to fix compilation/runtime errors.')
    args = parser.parse_args()

    # Create output directory first
    output_dir = f"llm_output/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.backend}_{args.model}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_file = setup_logging(output_dir)
    logging.info(f"Logging initialized. Log file: {log_file}")

    try:
        # Create backend instance
        backend = BackendFactory.create_backend(args.backend)
        logging.info(f"Initialized {args.backend} backend")
        
        # Get available kernels for this backend
        available_kernels = backend.get_available_kernels()
        
        # Create LLM instance
        llm = LLM(provider=args.provider, model=args.model)
        logging.info(f"Initialized {llm}")
        
        # Initialize the generator with backend
        generator = KernelCodeGenerator(llm, backend)

        # Determine which kernels to process
        if args.all:
            # Process all available kernels for this backend
            kernels_to_process = list(available_kernels.keys())
            logging.info(f"Processing all {len(kernels_to_process)} kernels for {args.backend} backend using {args.model} by {args.provider.upper()}...")
        elif args.npueval_all:
            # Process all working NPUEval kernels (for npu_new backend)
            if args.backend != 'npu_new':
                logging.error("--npueval_all flag can only be used with npu_new backend")
                return
            
            # Load working kernels from benchmark results
            try:
                import json
                benchmark_file = "npueval_dataset/kernel_benchmark_results.json"
                with open(benchmark_file, 'r') as f:
                    benchmark_data = json.load(f)
                
                working_kernels = []
                for result in benchmark_data['results']:
                    if result['success'] and result['kernel_name'] in available_kernels:
                        working_kernels.append(result['kernel_name'])
                
                kernels_to_process = working_kernels
                logging.info(f"Processing {len(kernels_to_process)} working NPUEval kernels for {args.backend} backend using {args.model} by {args.provider.upper()}...")
            except Exception as e:
                logging.error(f"Error loading NPUEval benchmark results: {e}")
                return
        else:
            # Process specified kernels or default
            if args.kernel:
                kernels_to_process = args.kernel
            else:
                # Use first available kernel as default
                kernels_to_process = [list(available_kernels.keys())[0]] if available_kernels else []
                logging.info(f"No kernels specified, using default: {kernels_to_process}")
            
            logging.info(f"Processing kernels {kernels_to_process} for {args.backend} backend using {args.model} by {args.provider.upper()}...")
        
        if not kernels_to_process:
            logging.error(f"No kernels to process for {args.backend} backend")
            return
            
        # Process kernels
        process_kernels(generator, kernels_to_process, available_kernels, 
                       output_dir, args.model, args.provider, args.backend,
                       max_iterations=args.max_iterations)
            
    except ValueError as e:
        logging.error(f"Error: {e}")
        return

    # Run plotting (if applicable for the backend)
    try:
        plot_speedup(output_dir)
    except Exception as e:
        logging.warning(f"Could not generate plots: {e}")

if __name__ == "__main__":
    main() 