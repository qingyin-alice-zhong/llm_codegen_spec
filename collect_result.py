#!/usr/bin/env python3
"""
Script to collect and analyze results from all_summary.json
Generates plots and statistics including pass rates and geometric mean of speedups.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import seaborn as sns
import os

from plot import plot_speedup

def load_results(json_path: str) -> Dict:
    """Load results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def calculate_stage_success_rates(results: Dict) -> Dict[str, float]:
    """Calculate success rates for each stage across all kernels."""
    stages = ['kernel_generation_success', 'compilation_success', 'execution_success', 'verification_success']
    stage_counts = {stage: 0 for stage in stages}
    total_kernels = len(results)
    
    for kernel_name, kernel_data in results.items():
        for stage in stages:
            if kernel_data.get(stage, False):
                stage_counts[stage] += 1
    
    success_rates = {stage: count / total_kernels * 100 for stage, count in stage_counts.items()}
    return success_rates

def calculate_geometric_mean_speedup(results: Dict) -> Optional[float]:
    """Calculate geometric mean of speedups for kernels that have verification_success=True."""
    speedups = []
    for kernel_name, kernel_data in results.items():
        if kernel_data.get('verification_success', False) and 'speedup' in kernel_data:
            speedups.append(kernel_data['speedup'])
    
    if not speedups:
        return None
    
    # Convert to log space to avoid overflow, then back
    log_speedups = np.log(speedups)
    geometric_mean = np.exp(np.mean(log_speedups))
    return geometric_mean

def calculate_geometric_mean_speedup_clamped(results: Dict) -> Optional[float]:
    """Calculate geometric mean of speedups with values < 1 clamped to 1."""
    speedups = []
    for kernel_name, kernel_data in results.items():
        if kernel_data.get('verification_success', False) and 'speedup' in kernel_data:
            # Clamp speedups < 1 to 1.0
            clamped_speedup = max(1.0, kernel_data['speedup'])
            speedups.append(clamped_speedup)
    
    if not speedups:
        return None
    
    # Convert to log space to avoid overflow, then back
    log_speedups = np.log(speedups)
    geometric_mean = np.exp(np.mean(log_speedups))
    return geometric_mean

def get_stage_progression_stats(results: Dict) -> Dict[str, int]:
    """Get statistics on how many kernels pass each stage."""
    stages = ['kernel_generation_success', 'compilation_success', 'execution_success', 'verification_success']
    stage_progression = {stage: 0 for stage in stages}
    
    for kernel_name, kernel_data in results.items():
        for i, stage in enumerate(stages):
            if kernel_data.get(stage, False):
                stage_progression[stage] += 1
            else:
                # If this stage fails, stop counting for this kernel
                break
    
    return stage_progression

def create_stage_success_plot(success_rates: Dict[str, float], output_dir: str):
    """Create a bar plot showing success rates for each stage."""
    stages = ['Kernel Generation', 'Compilation', 'Execution', 'Verification']
    rates = [success_rates['kernel_generation_success'], 
             success_rates['compilation_success'], 
             success_rates['execution_success'], 
             success_rates['verification_success']]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(stages, rates, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Success Rate by Stage', fontsize=16, fontweight='bold')
    plt.ylabel('Success Rate (%)', fontsize=12)
    plt.xlabel('Stage', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    # Add horizontal line at 50% for reference
    plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/stage_success_rates.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_speedup_distribution_plot(results: Dict, output_dir: str):
    """Create a histogram of speedups for successful kernels, showing both original and clamped distributions."""
    speedups = []
    clamped_speedups = []
    kernel_names = []
    
    for kernel_name, kernel_data in results.items():
        if kernel_data.get('verification_success', False) and 'speedup' in kernel_data:
            original_speedup = kernel_data['speedup']
            speedups.append(original_speedup)
            clamped_speedups.append(max(1.0, original_speedup))
            kernel_names.append(kernel_name)
    
    if not speedups:
        print("No successful kernels with speedup data found.")
        return
    
    plt.figure(figsize=(16, 12))
    
    # Original speedup histogram
    plt.subplot(3, 2, 1)
    plt.hist(speedups, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Original Speedup Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Speedup', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add statistics text for original
    mean_speedup = np.mean(speedups)
    median_speedup = np.median(speedups)
    geometric_mean = np.exp(np.mean(np.log(np.maximum(speedups, 1e-10))))  # Avoid log(0)
    
    stats_text = f'Mean: {mean_speedup:.2f}x\nMedian: {median_speedup:.2f}x\nGeometric Mean: {geometric_mean:.2f}x'
    plt.text(0.7, 0.8, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10, verticalalignment='top')
    
    # Clamped speedup histogram
    plt.subplot(3, 2, 2)
    plt.hist(clamped_speedups, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.title('Clamped Speedup Distribution (min = 1.0)', fontsize=14, fontweight='bold')
    plt.xlabel('Speedup', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add statistics text for clamped
    mean_clamped = np.mean(clamped_speedups)
    median_clamped = np.median(clamped_speedups)
    geometric_mean_clamped = np.exp(np.mean(np.log(clamped_speedups)))
    clamped_count = sum(1 for s in speedups if s < 1.0)
    
    stats_text_clamped = f'Mean: {mean_clamped:.2f}x\nMedian: {median_clamped:.2f}x\nGeometric Mean: {geometric_mean_clamped:.2f}x\nClamped: {clamped_count}'
    plt.text(0.7, 0.8, stats_text_clamped, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.8),
             fontsize=10, verticalalignment='top')
    
    # Side-by-side comparison histogram
    plt.subplot(3, 1, 2)
    bins = np.linspace(min(min(speedups), min(clamped_speedups)), 
                      max(max(speedups), max(clamped_speedups)), 20)
    plt.hist([speedups, clamped_speedups], bins=bins, alpha=0.7, 
             color=['skyblue', 'lightcoral'], edgecolor='black',
             label=['Original', 'Clamped (min=1.0)'])
    plt.title('Speedup Distribution Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Speedup', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Box plots comparison
    plt.subplot(3, 1, 3)
    box_data = [speedups, clamped_speedups]
    box_labels = ['Original', 'Clamped (min=1.0)']
    plt.boxplot(box_data, labels=box_labels)
    plt.title('Speedup Box Plot Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Speedup', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/speedup_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_kernel_success_heatmap(results: Dict, output_dir: str):
    """Create a heatmap showing success/failure for each kernel and stage."""
    kernels = list(results.keys())
    stages = ['kernel_generation_success', 'compilation_success', 'execution_success', 'verification_success']
    stage_labels = ['Kernel Gen', 'Compilation', 'Execution', 'Verification']
    
    # Create matrix: 1 for success, 0 for failure
    success_matrix = np.zeros((len(kernels), len(stages)))
    
    for i, kernel_name in enumerate(kernels):
        for j, stage in enumerate(stages):
            if results[kernel_name].get(stage, False):
                success_matrix[i, j] = 1
    
    plt.figure(figsize=(12, max(8, len(kernels) * 0.3)))
    sns.heatmap(success_matrix, 
                xticklabels=stage_labels, 
                yticklabels=kernels,
                cmap='RdYlGn', 
                cbar_kws={'label': 'Success (1) / Failure (0)'},
                annot=True, 
                fmt='.0f')
    
    plt.title('Kernel Success Heatmap by Stage', fontsize=16, fontweight='bold')
    plt.xlabel('Stage', fontsize=12)
    plt.ylabel('Kernel', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/kernel_success_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_detailed_summary_table(results: Dict, output_dir: str):
    """Create a detailed summary table and save as CSV."""
    summary_data = []
    
    for kernel_name, kernel_data in results.items():
        row = {
            'Kernel': kernel_name,
            'Backend': kernel_data.get('backend', 'N/A'),
            'Total Iterations': kernel_data.get('total_iterations', 0),
            'Best Iteration': kernel_data.get('best_iteration', 'N/A'),
            'Kernel Generation': '✓' if kernel_data.get('kernel_generation_success', False) else '✗',
            'Compilation': '✓' if kernel_data.get('compilation_success', False) else '✗',
            'Execution': '✓' if kernel_data.get('execution_success', False) else '✗',
            'Verification': '✓' if kernel_data.get('verification_success', False) else '✗',
            'Speedup': kernel_data.get('speedup', 'N/A')
        }
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    df.to_csv(f'{output_dir}/detailed_summary.csv', index=False)
    return df

def print_summary_statistics(results: Dict, success_rates: Dict[str, float], 
                           geometric_mean_speedup: Optional[float], stage_progression: Dict[str, int],
                           geometric_mean_speedup_clamped: Optional[float]):
    """Print comprehensive summary statistics."""
    total_kernels = len(results)
    
    print("=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal Kernels Tested: {total_kernels}")
    print(f"Backend: {list(results.values())[0].get('backend', 'N/A') if results else 'N/A'}")
    
    print("\n" + "-" * 50)
    print("STAGE SUCCESS RATES")
    print("-" * 50)
    stage_names = {
        'kernel_generation_success': 'Kernel Generation',
        'compilation_success': 'Compilation', 
        'execution_success': 'Execution',
        'verification_success': 'Verification'
    }
    
    for stage, rate in success_rates.items():
        print(f"{stage_names[stage]:<20}: {rate:>6.1f}%")
    
    print("\n" + "-" * 50)
    print("STAGE PROGRESSION (Cumulative)")
    print("-" * 50)
    for stage, count in stage_progression.items():
        print(f"{stage_names[stage]:<20}: {count:>6d} kernels")
    
    print("\n" + "-" * 50)
    print("SPEEDUP STATISTICS")
    print("-" * 50)
    
    successful_kernels = [k for k, v in results.items() if v.get('verification_success', False)]
    print(f"Successful Kernels: {len(successful_kernels)}/{total_kernels} ({len(successful_kernels)/total_kernels*100:.1f}%)")
    
    # Original speedup statistics
    if geometric_mean_speedup:
        print(f"\nOriginal Speedups:")
        print(f"  Geometric Mean Speedup: {geometric_mean_speedup:.2f}x")
        
        # Additional speedup statistics
        speedups = [v['speedup'] for k, v in results.items() 
                   if v.get('verification_success', False) and 'speedup' in v]
        if speedups:
            print(f"  Arithmetic Mean Speedup: {np.mean(speedups):.2f}x")
            print(f"  Median Speedup: {np.median(speedups):.2f}x")
            print(f"  Min Speedup: {np.min(speedups):.2f}x")
            print(f"  Max Speedup: {np.max(speedups):.2f}x")
    else:
        print("No successful kernels with speedup data found.")
    
    # Clamped speedup statistics (values < 1 set to 1)
    if geometric_mean_speedup_clamped:
        print(f"\nClamped Speedups (values < 1.0 set to 1.0):")
        print(f"  Geometric Mean Speedup (Clamped): {geometric_mean_speedup_clamped:.2f}x")
        
        # Additional clamped speedup statistics
        clamped_speedups = [max(1.0, v['speedup']) for k, v in results.items() 
                           if v.get('verification_success', False) and 'speedup' in v]
        if clamped_speedups:
            print(f"  Arithmetic Mean Speedup (Clamped): {np.mean(clamped_speedups):.2f}x")
            print(f"  Median Speedup (Clamped): {np.median(clamped_speedups):.2f}x")
            print(f"  Min Speedup (Clamped): {np.min(clamped_speedups):.2f}x")
            print(f"  Max Speedup (Clamped): {np.max(clamped_speedups):.2f}x")
            
            # Show how many speedups were clamped
            original_speedups = [v['speedup'] for k, v in results.items() 
                               if v.get('verification_success', False) and 'speedup' in v]
            clamped_count = sum(1 for s in original_speedups if s < 1.0)
            print(f"  Kernels with speedup < 1.0 (clamped): {clamped_count}/{len(original_speedups)}")
    else:
        print("No successful kernels with speedup data found for clamped calculation.")
    
    print("\n" + "-" * 50)
    print("TOP PERFORMING KERNELS")
    print("-" * 50)
    
    # Sort by speedup (descending)
    successful_with_speedup = [(k, v['speedup']) for k, v in results.items() 
                              if v.get('verification_success', False) and 'speedup' in v]
    successful_with_speedup.sort(key=lambda x: x[1], reverse=True)
    
    for i, (kernel, speedup) in enumerate(successful_with_speedup[:10]):
        print(f"{i+1:2d}. {kernel:<30}: {speedup:>8.2f}x")
    
    print("\n" + "=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Collect and analyze results from all_summary.json')
    parser.add_argument('--work_dir', '-i', default='llm_output/full-run-20250904_010759_npu_new_gemini-2.5-flash',
                       help='Path to all_summary.json file')
    # parser.add_argument('--output', '-o', default='analysis_results',
    #                    help='Output directory for plots and analysis')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    input_file = os.path.join(args.work_dir, 'all_summary.json')
    # Create output directory
    # output_dir = Path(args.output)
    # output_dir.mkdir(exist_ok=True)
    output_dir = args.work_dir
    
    # Load results
    print(f"Loading results from {input_file}...")
    results = load_results(input_file)
    
    # Calculate statistics
    print("Calculating statistics...")
    success_rates = calculate_stage_success_rates(results)
    geometric_mean_speedup = calculate_geometric_mean_speedup(results)
    geometric_mean_speedup_clamped = calculate_geometric_mean_speedup_clamped(results)
    stage_progression = get_stage_progression_stats(results)
    
    # Print summary
    print_summary_statistics(results, success_rates, geometric_mean_speedup, stage_progression, geometric_mean_speedup_clamped)
    
    # Generate plots if requested
    if not args.no_plots:
        print(f"\nGenerating plots in {output_dir}...")
        create_stage_success_plot(success_rates, str(output_dir))
        create_speedup_distribution_plot(results, str(output_dir))
        create_kernel_success_heatmap(results, str(output_dir))
        plot_speedup(output_dir)
        print("Plots generated successfully!")
    
    # Create detailed summary table
    print("Creating detailed summary table...")
    df = create_detailed_summary_table(results, str(output_dir))
    print(f"Detailed summary saved to {output_dir}/detailed_summary.csv")
    
    # Save summary statistics to JSON
    summary_stats = {
        'total_kernels': len(results),
        'success_rates': success_rates,
        'geometric_mean_speedup': geometric_mean_speedup,
        'geometric_mean_speedup_clamped': geometric_mean_speedup_clamped,
        'stage_progression': stage_progression,
        'successful_kernels': len([k for k, v in results.items() if v.get('verification_success', False)])
    }
    
    # Add additional speedup statistics if available
    speedups = [v['speedup'] for k, v in results.items() 
               if v.get('verification_success', False) and 'speedup' in v]
    if speedups:
        clamped_speedups = [max(1.0, s) for s in speedups]
        clamped_count = sum(1 for s in speedups if s < 1.0)
        
        summary_stats.update({
            'original_speedup_stats': {
                'arithmetic_mean': float(np.mean(speedups)),
                'median': float(np.median(speedups)),
                'min': float(np.min(speedups)),
                'max': float(np.max(speedups))
            },
            'clamped_speedup_stats': {
                'arithmetic_mean': float(np.mean(clamped_speedups)),
                'median': float(np.median(clamped_speedups)),
                'min': float(np.min(clamped_speedups)),
                'max': float(np.max(clamped_speedups)),
                'kernels_clamped': clamped_count,
                'total_kernels_with_speedup': len(speedups)
            }
        })
    
    with open(f'{output_dir}/summary_statistics.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}/")
    print(f"- Stage success rates plot: {output_dir}/stage_success_rates.png")
    print(f"- Speedup distribution plot: {output_dir}/speedup_distribution.png") 
    print(f"- Kernel success heatmap: {output_dir}/kernel_success_heatmap.png")
    print(f"- Detailed summary table: {output_dir}/detailed_summary.csv")
    print(f"- Summary statistics: {output_dir}/summary_statistics.json")

if __name__ == "__main__":
    main()
