import os
import re
from typing import Dict, List, Tuple, Optional
import json
import logging
from llm import LLM
from backend import Backend


class KernelCodeGenerator:
    """
    A framework for using LLMs to generate optimized kernel code for different hardware backends.
    """
    
    def __init__(self, llm: LLM, backend: Backend):
        """
        Initialize the kernel code generator.
        
        Args:
            llm: LLM instance configured with provider and model
            backend: Backend instance for specific hardware target
        """
        self.llm = llm
        self.backend = backend
        
        # State tracking for current kernel processing
        self.current_kernel = None
        self.current_file_path = None
        self.current_output_dir = None
        self.results_all_iterations = {}
        self.best_iteration = None
        self.best_results = None
        
    def _extract_code_blocks(self, response: str) -> List[str]:
        """
        Extract code blocks from the LLM response.
        
        Args:
            response: The LLM's response
            
        Returns:
            List of code blocks found in the response
        """
        if not response:
            return []
        # Pattern to match code blocks with optional language specification
        pattern = r'```(?:[a-zA-Z]*\n)?(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)
        return [match.strip() for match in matches]
    
    def _reset_kernel_state(self, kernel_name: str, file_path: str, output_dir: str):
        """Reset state for a new kernel processing session."""
        self.current_kernel = kernel_name
        self.current_file_path = file_path
        self.current_output_dir = output_dir
        self.results_all_iterations = {}
        self.best_iteration = None
        self.best_results = None
        
        # Reset LLM conversation history to prevent cross-contamination between kernels
        self.llm.conversation_history = []
    
    def _update_iteration_results(self, iteration: int, results: Dict):
        """Update results for a specific iteration and track best results."""
        self.results_all_iterations[iteration] = results.copy()
        
        # Update best results based on stage progression logic
        if self._is_better_iteration(results, self.best_results):
            self.best_iteration = iteration
            self.best_results = results.copy()
    
    def _get_iteration_status(self, results: Dict) -> Dict[str, bool]:
        """Extract status flags from iteration results."""
        return {
            'kernel_generation_success': results.get('kernel_generation_success', False),
            'compilation_success': results.get('compilation_success', False),
            'execution_success': results.get('execution_success', False),
            'verification_success': results.get('verification_success', False)
        }
    
    def _log_iteration_summary(self, iteration: int, status: Dict[str, bool], speedup: Optional[float] = None):
        """Log a summary of the current iteration."""
        stages_passed = sum(status.values())
        stage_names = ['Kernel Gen', 'Compilation', 'Execution', 'Verification']
        passed_stages = [name for i, name in enumerate(stage_names) if list(status.values())[i]]
        
        logging.info(f"Iteration {iteration}: {stages_passed}/4 stages passed - {', '.join(passed_stages)}")
        if speedup is not None:
            logging.info(f"  Speedup: {speedup:.2f}x")
    
    def call_llm(self, prompt: str) -> Tuple[str, Optional[Dict]]:
        """
        Call the LLM API with the given prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Tuple of (The LLM's response text, entire response dictionary)
        """
        system_prompt = f"You are an expert in high-performance computing, kernel optimization, and hardware acceleration for {self.backend.name}."
        return self.llm.generate_completion(system_prompt, prompt)

    def generate_kernel(self, file_path: str, kernel_name: str, output_dir: str, iteration: int = 0, previous_results: Dict = None) -> Dict:
        """
        Generate an optimized kernel implementation through the LLM pipeline by reading from a file.
        
        Args:
            file_path: Path to the source file
            kernel_name: Name of the kernel. The function name in the code is `kernel_{kernel_name}`
            output_dir: Directory to save outputs
            iteration: Current iteration number (0-based)
            previous_results: Results from previous iteration for feedback
        Returns:
            Dictionary containing the results
        """
        
        # Read the complete code from file
        try:
            with open(file_path, 'r') as f:
                input_complete_code = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find file: {file_path}")
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {str(e)}")
        
        # kernel name: `-` -> `_`
        kernel_name = kernel_name.replace('-', '_')

        # Create backend-specific prompt
        if iteration == 0:
            prompt = self.backend.create_prompt(input_complete_code, kernel_name)
        elif previous_results:
            prompt = self._generate_feedback(previous_results, iteration)
        else:
            raise ValueError(f"This is iteration {iteration}, but no previous results are given")

        # Call LLM
        response, entire_response = self.call_llm(prompt)
        
        # Extract code blocks
        code_blocks = self._extract_code_blocks(response)
        
        # Find the main optimized implementation (usually the longest code block)
        llm_gen_code = None
        kernel_generation_success = False
        if response and code_blocks:
            llm_gen_code = max(code_blocks, key=len)
            # Only set success to True if we have meaningful generated code
            kernel_generation_success = llm_gen_code is not None and len(llm_gen_code.strip()) > 0
        
        # Generate optimized complete code using backend
        output_complete_code = None
        if llm_gen_code:
            output_complete_code = self.backend.create_output(
                input_complete_code, llm_gen_code, kernel_name
            )
        
        output_code_name = self.backend.get_output_code_name(kernel_name)
        output_code_path = os.path.join(output_dir, output_code_name)
        with open(output_code_path, 'w') as f:
            # Handle the case where output_complete_code is None
            if output_complete_code is not None:
                f.write(output_complete_code)
            else:
                f.write("// ERROR: Failed to generate kernel code\n")
        
        return {
            "kernel_name": kernel_name,
            "backend": self.backend.name,
            "input_code_path": file_path,
            "prompt": prompt,
            "llm_response": response,
            "entire_llm_response": entire_response,
            "llm_gen_code": llm_gen_code,
            "num_code_blocks": len(code_blocks),
            "output_complete_code": output_complete_code,
            "output_code_path": output_code_path,
            "kernel_generation_success": kernel_generation_success
        }

    def _generate_feedback(self, previous_results: Dict, iteration: int) -> str:
        """
        Generate feedback based on previous iteration results.
        
        Args:
            previous_results: Results from previous iteration
            
        Returns:
            Feedback string to be added to the prompt
        """
        feedback_parts = []
        
        feedback_parts.append(f"FEEDBACK FROM PREVIOUS ITERATION (ITERATION {iteration - 1}):")
        feedback_parts.append("=" * 50)
        
        # Check for kernel generation failure
        if not previous_results.get('kernel_generation_success', False):
            feedback_parts.append("❌ KERNEL GENERATION FAILED")
            feedback_parts.append("The LLM failed to generate a valid kernel implementation.")
            feedback_parts.append("Try generating the kernel again. ")
            
        # Check for compilation errors
        elif not previous_results.get('compilation_success', False):
            feedback_parts.append("❌ COMPILATION FAILED")
            if 'compile_and_run' in previous_results:
                compile_error = previous_results['compile_and_run'].get('compile_error', '')
                if compile_error:
                    feedback_parts.append("Compilation Error Details:")
                    feedback_parts.append(f"```\n{compile_error}\n```")
            feedback_parts.append("Fix the compilation error in your implementation.")
        
        # Check for runtime errors
        elif not previous_results.get('execution_success', False):
            feedback_parts.append("❌ EXECUTION FAILED")
            if 'compile_and_run' in previous_results:
                run_error = previous_results['compile_and_run'].get('run_error', '')
                if run_error:
                    feedback_parts.append("Runtime Error Details:")
                    feedback_parts.append(f"```\n{run_error}\n```")
            feedback_parts.append("Fix the runtime error in your implementation.")
        
        # Check for verification errors
        elif not previous_results.get('verification_success', False):
            feedback_parts.append("❌ VERIFICATION FAILED")
            feedback_parts.append("The optimized implementation produces incorrect results compared to the original implementation.")
            if 'compile_and_run' in previous_results:
                compile_and_run = previous_results['compile_and_run']
                if 'run_output' in compile_and_run:
                    feedback_parts.append("Run Output:")
                    feedback_parts.append(f"```\n{compile_and_run['run_output']}\n```")
            feedback_parts.append("Fix the verification error by ensuring functional equivalence with the original kernel.")
        
        # If everything passed, provide performance feedback
        else:
            feedback_parts.append("✅ PREVIOUS ITERATION SUCCESSFUL")
            feedback_parts.append("Compilation, execution, and verification all passed.")
            
            # Add performance information if available
            if 'run_analysis' in previous_results:
                run_analysis = previous_results['run_analysis']
                if self.backend.show_speedup_in_feedback:
                    if run_analysis.get('scalar_time') is not None:
                        feedback_parts.append(f"scalar kernel time: {run_analysis['scalar_time']:.6f} cycles")
                    if run_analysis.get('vector_time') is not None:
                        feedback_parts.append(f"vector kernel time: {run_analysis['vector_time']:.6f} cycles")
                    if run_analysis.get('speedup') is not None:
                        speedup = run_analysis['speedup']
                        feedback_parts.append(f"Current speedup: {speedup:.2f}x")
                else:
                    if run_analysis.get('vector_time') is not None:
                        feedback_parts.append(f"Current kernel time: {run_analysis['vector_time']:.6f} cycles")
            
            feedback_parts.append("Try to improve the performance further if possible, while maintaining correctness.")
        
        feedback_parts.append("=" * 50)
        feedback_parts.append("Based on the above feedback, provide an improved implementation.")
        
        return "\n".join(feedback_parts)

    def save_results(self, results: Dict, output_dir: str):
        """
        Save results including all status information.
        
        Args:
            results: The complete results dictionary
            output_dir: Directory to save outputs
        """
        kernel_name = results['kernel_name']
        backend_name = results.get('backend', 'unknown')
        
        # Save the current iteration's entire LLM conversation history
        with open(os.path.join(output_dir, f"{kernel_name}_llm_conversation_history.json"), 'w') as f:
            json.dump(self.llm.conversation_history, f, indent=2)

        # Save the prompt newly added for this iteration
        with open(os.path.join(output_dir, f"{kernel_name}_prompt.txt"), 'w') as f:
            f.write(results['prompt'])
        
        # Save full LLM response
        with open(os.path.join(output_dir, f"{kernel_name}_llm_response.txt"), 'w') as f:
            f.write(results['llm_response'] if results['llm_response'] is not None else "")
        
        # Save compilation and execution logs
        if 'compile_and_run' in results:
            comp_results = results['compile_and_run']
            
            # Save compilation output
            with open(os.path.join(output_dir, f"{kernel_name}_compile_output.txt"), 'w') as f:
                f.write("COMPILATION/EXECUTION COMMAND:\n")
                f.write(comp_results.get('compile_cmd', 'N/A'))
                f.write("\n\nSTDOUT:\n")
                f.write(comp_results.get('compile_output', ''))
                f.write("\n\nSTDERR:\n")
                f.write(comp_results.get('compile_error', ''))
            
            # Save execution output
            with open(os.path.join(output_dir, f"{kernel_name}_execution_output.txt"), 'w') as f:
                f.write("EXECUTION COMMAND:\n")
                f.write(comp_results.get('executable', ''))
                f.write("\n\nSTDOUT:\n")
                f.write(comp_results.get('run_output', ''))
                f.write("\n\nSTDERR:\n")
                f.write(comp_results.get('run_error', ''))
        
        # Save summary
        summary = {
            "kernel_name": kernel_name,
            "backend": backend_name,
            "input_code_path": results.get('input_code_path', ''),
            "output_code_path": results.get('output_code_path', ''),
            "status": {
                "kernel_generation_success": results.get('kernel_generation_success', False),
                "compilation_success": results.get('compilation_success', False),
                "execution_success": results.get('execution_success', False),
                "verification_success": results.get('verification_success', False)
            },
            "statistics": {
                "num_code_blocks_generated": results.get('num_code_blocks', 0),
                "llm_gen_code_length": len(results.get('llm_gen_code') or ''),
                "prompt_length": len(results.get('prompt', '')),
                "llm_response_length": len(results.get('llm_response') or '')
            }, 
            "details": {
                "entire_llm_response": results.get('entire_llm_response', '')
            }
        }
        
        # Add run analysis if available
        if 'run_analysis' in results:
            run_analysis = results['run_analysis']
            
            summary["performance_analysis"] = {
                "verification_success": run_analysis.get('verification_success', False),
                "scalar_time_cycles": run_analysis.get('scalar_time'),
                "vector_time_cycles": run_analysis.get('vector_time'),
                "speedup": run_analysis.get('speedup'),
            }
        
        # Add compilation details if available
        if 'compile_and_run' in results:
            comp_results = results['compile_and_run']
            summary["compilation_details"] = {
                "source_file": comp_results.get('source_file', ''),
                "executable": comp_results.get('executable', ''),
                "compile_output_length": len(comp_results.get('compile_output', '')),
                "compile_error_length": len(comp_results.get('compile_error', '')),
                "run_output_length": len(comp_results.get('run_output', '')),
                "run_error_length": len(comp_results.get('run_error', ''))
            }
        
        with open(os.path.join(output_dir, f"{kernel_name}_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print status summary
        logging.info("RESULTS SUMMARY")
        logging.info("="*20)
        logging.info(f"Kernel: {kernel_name}")
        logging.info(f"Backend: {backend_name}")
        logging.info(f"Kernel Generation Success: {'✓' if summary['status']['kernel_generation_success'] else '✗'}")
        logging.info(f"Compilation Success: {'✓' if summary['status']['compilation_success'] else '✗'}")
        logging.info(f"Execution Success: {'✓' if summary['status']['execution_success'] else '✗'}")
        logging.info(f"Verification Success: {'✓' if summary['status']['verification_success'] else '✗'}")
        
        # Print performance summary if available
        if 'performance_analysis' in summary:
            perf = summary['performance_analysis']
            if perf['speedup'] is not None:
                logging.info(f"Speedup: {perf['speedup']:.2f}x")
        
        logging.info(f"Files saved in: {output_dir}")
        logging.info("="*60)

    def process_kernel(self, file_path: str, kernel_name: str, output_dir: str, max_iterations: int = 1) -> Dict:
        """
        Complete workflow: process kernel, save results, compile and run optimized code.
        
        Args:
            file_path: Path to the source file
            kernel_name: Name of the kernel
            output_dir: Directory to save outputs
            max_iterations: Maximum number of iterations to try
            
        Returns:
            Dictionary containing all results including compilation and execution
        """
        # Initialize state for this kernel
        self._reset_kernel_state(kernel_name, file_path, output_dir)
        
        logging.info(f"Processing kernel '{kernel_name}' from file: {file_path} using {self.backend.name} backend")
        
        # Process each iteration
        for iteration in range(max_iterations):
            logging.info(f"="*10 + f"Iteration {iteration}" + "="*10)
            
            # Process single iteration
            iteration_results = self._process_single_iteration(iteration, max_iterations)
            
            self.results_all_iterations[iteration] = iteration_results.copy()
            
            # Log iteration summary
            status = self._get_iteration_status(iteration_results)
            speedup = iteration_results.get('run_analysis', {}).get('speedup')
            self._log_iteration_summary(iteration, status, speedup)
        
        return self.results_all_iterations
    
    def _process_single_iteration(self, iteration: int, max_iterations: int) -> Dict:
        """Process a single iteration of kernel generation, compilation, and execution."""
        output_iter_dir = os.path.join(self.current_output_dir, f"iteration_{iteration}")
        os.makedirs(output_iter_dir, exist_ok=True)
        
        # Get previous results for feedback
        previous_results = self.results_all_iterations.get(iteration - 1) if iteration > 0 else None
        
        # Generate kernel code
        results = self.generate_kernel(
            self.current_file_path, 
            self.current_kernel, 
            output_iter_dir, 
            iteration, 
            previous_results
        )
        
        # If kernel generation failed, set all flags to False and return
        if not results.get('kernel_generation_success', False):
            logging.warning("✗ Kernel generation failed. Skipping compilation and execution.")
            results.update({
                "compilation_success": False,
                "execution_success": False,
                "verification_success": False,
            })
            self.save_results(results, output_iter_dir)
            return results
        
        # logging.info(f"✓ Kernel generation completed")
        
        # Compile and run if we have optimized code
        if results.get('output_complete_code'):
            self._compile_and_run_kernel(results, output_iter_dir)
        else:
            logging.warning("No optimized code generated to compile")
            results.update({
                "compilation_success": False,
                "execution_success": False,
                "verification_success": False,
            })
        
        # Save results for this iteration
        self.save_results(results, output_iter_dir)
        return results
    
    def _compile_and_run_kernel(self, results: Dict, output_iter_dir: str):
        """Compile and run the generated kernel, updating results with status."""
        compile_results = self.backend.compile_and_run(
            results['output_code_path'], 
            self.current_kernel, 
            output_iter_dir
        )
        results['compile_and_run'] = compile_results
        
        compilation_success = compile_results['compilation_successful']
        execution_success = compile_results['execution_successful']
        
        # Analyze run output if execution was successful
        run_analysis = {}
        if execution_success and compile_results.get('run_output'):
            run_analysis = self.backend.analyze_run_output(compile_results['run_output'], self.current_kernel)
            results['run_analysis'] = run_analysis
        
        # Log compilation results
        if compilation_success:
            # logging.info(f"✓ Compilation successful")
            if execution_success:
                self._log_execution_results(run_analysis)
            else:
                logging.error(f"✗ Execution failed:")
                logging.error(compile_results['run_error'])
        else:
            logging.error(f"✗ Compilation failed:")
            logging.error(compile_results['compile_error'])
        
        # Update results with final status
        results.update({
            "compilation_success": compilation_success,
            "execution_success": execution_success,
            "verification_success": run_analysis.get('verification_success', False),
        })
    
    def _log_execution_results(self, run_analysis: Dict):
        """Log execution and verification results."""
        # logging.info(f"✓ Execution successful")
        
        # Log verification results
        verification_success = run_analysis.get('verification_success')
        if verification_success:
            # logging.info(f"🔍 Verification: ✓ PASS")
            pass
        else:
            logging.warning(f"🔍 Verification: ✗ FAIL")

        # Log performance analysis
        logging.info(f"📊 Performance Analysis:")
        if run_analysis.get('scalar_time') is not None:
            logging.info(f"  Scalar time: {run_analysis['scalar_time']:.6f} cycles")
        if run_analysis.get('vector_time') is not None:
            logging.info(f"  Vector time: {run_analysis['vector_time']:.6f} cycles")
        if run_analysis.get('speedup') is not None:
            logging.info(f"  Speedup: {run_analysis['speedup']:.2f}x")
    
    def _log_final_summary(self):
        """Log final summary of kernel processing."""
        if self.best_results is None:
            logging.warning(f"No successful iterations for kernel '{self.current_kernel}'")
            return
        
        status = self._get_iteration_status(self.best_results)
        speedup = self.best_results.get('run_analysis', {}).get('speedup')
        
        logging.info(f"\n{'='*60}")
        logging.info(f"FINAL SUMMARY for kernel '{self.current_kernel}'")
        logging.info(f"{'='*60}")
        logging.info(f"Best iteration: {self.best_iteration}")
        logging.info(f"Stages passed: {sum(status.values())}/4")
        
        stage_names = ['Kernel Gen', 'Compilation', 'Execution', 'Verification']
        for i, (stage, passed) in enumerate(zip(stage_names, status.values())):
            status_symbol = "✓" if passed else "✗"
            logging.info(f"  {stage}: {status_symbol}")
        
        if speedup is not None:
            logging.info(f"Best speedup: {speedup:.2f}x")
        logging.info(f"{'='*60}\n")
    
    def get_current_state(self) -> Dict:
        """Get current processing state information."""
        return {
            'current_kernel': self.current_kernel,
            'current_file_path': self.current_file_path,
            'current_output_dir': self.current_output_dir,
            'best_iteration': self.best_iteration,
            'best_results': self.best_results,
            'total_iterations': len(self.results_all_iterations),
            'results_all_iterations': self.results_all_iterations
        }
    
    def get_best_results(self) -> Optional[Dict]:
        """Get the best results from the current kernel processing."""
        return self.best_results
    
    def get_iteration_results(self, iteration: int) -> Optional[Dict]:
        """Get results for a specific iteration."""
        return self.results_all_iterations.get(iteration) 