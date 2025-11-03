#!/usr/bin/env python3
"""
Script to compare speedup performance between gemini-2.5-flash and gpt-5 
for overlapped kernels using a side-by-side bar plot.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_json_data(file_path):
    """Load JSON data from file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_speedup_data(gemini_data, gpt5_data):
    """Extract speedup data for overlapped kernels including error information."""
    # Get kernel names from both datasets
    gemini_kernels = set(gemini_data.keys())
    gpt5_kernels = set(gpt5_data.keys())
    
    # Find overlapped kernels (kernels present in both datasets)
    overlapped_kernels = gemini_kernels.intersection(gpt5_kernels)
    overlapped_kernels = sorted(list(overlapped_kernels))
    
    print(f"Found {len(overlapped_kernels)} overlapped kernels:")
    for kernel in overlapped_kernels:
        print(f"  - {kernel}")
    
    # Extract speedup values and error information
    gemini_speedups = []
    gpt5_speedups = []
    gemini_errors = []
    gpt5_errors = []
    kernel_names = []
    
    def get_error_type(kernel_data):
        """Determine error type based on success flags."""
        if not kernel_data.get('kernel_generation_success', True):
            return 'kernel_gen'
        elif not kernel_data.get('compilation_success', True):
            return 'compile'
        elif not kernel_data.get('execution_success', True):
            return 'runtime'
        elif not kernel_data.get('verification_success', True):
            return 'verification'
        else:
            return None
    
    for kernel in overlapped_kernels:
        gemini_kernel_data = gemini_data[kernel]
        gpt5_kernel_data = gpt5_data[kernel]
        
        # Get speedup from gemini data
        gemini_error = get_error_type(gemini_kernel_data)
        if gemini_error:
            gemini_speedup = 1.0  # Use 1.0 as baseline for error markers
        else:
            gemini_speedup = gemini_kernel_data.get('speedup', 1.0)
            if gemini_speedup is None or gemini_speedup <= 0:
                gemini_speedup = 1.0
                
        # Get speedup from gpt5 data  
        gpt5_error = get_error_type(gpt5_kernel_data)
        if gpt5_error:
            gpt5_speedup = 1.0  # Use 1.0 as baseline for error markers
        else:
            gpt5_speedup = gpt5_kernel_data.get('speedup', 1.0)
            if gpt5_speedup is None or gpt5_speedup <= 0:
                gpt5_speedup = 1.0
            
        gemini_speedups.append(gemini_speedup)
        gpt5_speedups.append(gpt5_speedup)
        gemini_errors.append(gemini_error)
        gpt5_errors.append(gpt5_error)
        kernel_names.append(kernel)
        
        # Print status
        gemini_status = f"ERROR({gemini_error})" if gemini_error else f"{gemini_speedup:.2f}"
        gpt5_status = f"ERROR({gpt5_error})" if gpt5_error else f"{gpt5_speedup:.2f}"
        print(f"{kernel}: gemini={gemini_status}, gpt5={gpt5_status}")
    
    return kernel_names, gemini_speedups, gpt5_speedups, gemini_errors, gpt5_errors

def create_speedup_comparison_plot(kernel_names, gemini_speedups, gpt5_speedups, gemini_errors, gpt5_errors):
    """Create side-by-side bar plot comparing speedups with error markers."""
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Set y-axis to log scale
    ax.set_yscale('log')
    
    # Add reference line at y=1
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Baseline (1x)')
    
    # Set the width of bars and positions
    bar_width = 0.35
    x = np.arange(len(kernel_names))
    
    # Create bars for successful kernels only
    gemini_bars = []
    gpt5_bars = []
    
    for i, (gemini_speedup, gpt5_speedup, gemini_error, gpt5_error) in enumerate(
        zip(gemini_speedups, gpt5_speedups, gemini_errors, gpt5_errors)):
        
        # Only create bars for successful kernels
        if not gemini_error:
            bar1 = ax.bar(i - bar_width/2, gemini_speedup, bar_width, 
                         color='#1f77b4', alpha=0.7)
            gemini_bars.extend(bar1)
            
        if not gpt5_error:
            bar2 = ax.bar(i + bar_width/2, gpt5_speedup, bar_width,
                         color='#ff7f0e', alpha=0.7)
            gpt5_bars.extend(bar2)
    
    # Add error markers
    error_colors = {
        'kernel_gen': 'red',
        'compile': 'orange', 
        'runtime': 'purple',
        'verification': 'green'
    }
    
    # Track which error types we actually encounter for legend
    used_errors = set()
    
    for i, (gemini_error, gpt5_error) in enumerate(zip(gemini_errors, gpt5_errors)):
        if gemini_error:
            ax.plot(i - bar_width/2, 1, 'x', color=error_colors[gemini_error], 
                   markersize=10, markeredgewidth=2)
            used_errors.add(gemini_error)
            
        if gpt5_error:
            ax.plot(i + bar_width/2, 1, 'x', color=error_colors[gpt5_error], 
                   markersize=10, markeredgewidth=2)
            used_errors.add(gpt5_error)
    
    # Add value labels on top of successful bars
    def add_value_labels(bars, speedups, errors):
        bar_idx = 0
        for i, (speedup, error) in enumerate(zip(speedups, errors)):
            if not error and bar_idx < len(bars):
                bar = bars[bar_idx]
                height = bar.get_height()
                if height > 1:  # Only show labels for speedups > 1
                    ax.annotate(f'{height:.1f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom',
                               fontsize=8, rotation=0)
                bar_idx += 1
    
    add_value_labels(gemini_bars, gemini_speedups, gemini_errors)
    add_value_labels(gpt5_bars, gpt5_speedups, gpt5_errors)
    
    # Customize the plot
    ax.set_xlabel('Kernel Name', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Speedup Comparison: Gemini-2.5-Flash vs GPT-5\n(Overlapped Kernels)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(kernel_names, rotation=45, ha='right')
    
    # Create legend including models and error types
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color='#1f77b4', alpha=0.7, label='Gemini-2.5-Flash'),
        plt.Rectangle((0, 0), 1, 1, color='#ff7f0e', alpha=0.7, label='GPT-5'),
        plt.Line2D([0], [0], color='gray', linestyle='--', alpha=0.5, label='Baseline (1x)')
    ]
    
    # Add error type legends only for errors that actually occurred
    for error_type in used_errors:
        legend_elements.append(
            plt.Line2D([0], [0], marker='x', color=error_colors[error_type], 
                      markersize=10, markeredgewidth=2, 
                      label=f'Error: {error_type}', linestyle='None')
        )
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Add grid for better readability
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Set reasonable y-axis limits for log scale
    max_speedup = max([s for s, e in zip(gemini_speedups + gpt5_speedups, 
                                        gemini_errors + gpt5_errors) if not e])
    ax.set_ylim(0.1, max_speedup * 2)
    
    return fig, ax

def save_plot(fig, output_path):
    """Save the plot to file."""
    fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Plot saved to: {output_path}")

def print_summary_statistics(kernel_names, gemini_speedups, gpt5_speedups, gemini_errors, gpt5_errors):
    """Print summary statistics for the comparison."""
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Filter out error cases for statistics
    gemini_successful_speedups = [s for s, e in zip(gemini_speedups, gemini_errors) if not e]
    gpt5_successful_speedups = [s for s, e in zip(gpt5_speedups, gpt5_errors) if not e]
    
    gemini_successful_speedups = np.array(gemini_successful_speedups)
    gpt5_successful_speedups = np.array(gpt5_successful_speedups)
    
    print(f"Number of overlapped kernels: {len(kernel_names)}")
    
    # Count error statistics
    gemini_error_count = sum(1 for e in gemini_errors if e)
    gpt5_error_count = sum(1 for e in gpt5_errors if e)
    
    print(f"\nSuccess Rate:")
    print(f"  Gemini-2.5-Flash: {len(gemini_successful_speedups)}/{len(kernel_names)} successful ({(len(gemini_successful_speedups)/len(kernel_names)*100):.1f}%)")
    print(f"  GPT-5: {len(gpt5_successful_speedups)}/{len(kernel_names)} successful ({(len(gpt5_successful_speedups)/len(kernel_names)*100):.1f}%)")
    
    if len(gemini_successful_speedups) > 0:
        print(f"\nGemini-2.5-Flash (successful kernels only):")
        print(f"  Mean speedup: {np.mean(gemini_successful_speedups):.2f}")
        print(f"  Median speedup: {np.median(gemini_successful_speedups):.2f}")
        print(f"  Max speedup: {np.max(gemini_successful_speedups):.2f}")
        print(f"  Min speedup: {np.min(gemini_successful_speedups):.2f}")
    else:
        print(f"\nGemini-2.5-Flash: No successful kernels")
    
    if len(gpt5_successful_speedups) > 0:
        print(f"\nGPT-5 (successful kernels only):")
        print(f"  Mean speedup: {np.mean(gpt5_successful_speedups):.2f}")
        print(f"  Median speedup: {np.median(gpt5_successful_speedups):.2f}")
        print(f"  Max speedup: {np.max(gpt5_successful_speedups):.2f}")
        print(f"  Min speedup: {np.min(gpt5_successful_speedups):.2f}")
    else:
        print(f"\nGPT-5: No successful kernels")
    
    # Calculate win/loss statistics (only among mutually successful kernels)
    mutually_successful = []
    for i, (gemini_error, gpt5_error) in enumerate(zip(gemini_errors, gpt5_errors)):
        if not gemini_error and not gpt5_error:
            mutually_successful.append(i)
    
    if mutually_successful:
        gemini_mutual_speedups = [gemini_speedups[i] for i in mutually_successful]
        gpt5_mutual_speedups = [gpt5_speedups[i] for i in mutually_successful]
        
        gemini_wins = sum(1 for g, p in zip(gemini_mutual_speedups, gpt5_mutual_speedups) if g > p)
        gpt5_wins = sum(1 for g, p in zip(gemini_mutual_speedups, gpt5_mutual_speedups) if p > g)
        ties = sum(1 for g, p in zip(gemini_mutual_speedups, gpt5_mutual_speedups) if g == p)
        
        print(f"\nHead-to-head comparison (mutually successful kernels only):")
        print(f"  Total mutually successful: {len(mutually_successful)} kernels")
        print(f"  Gemini-2.5-Flash wins: {gemini_wins} kernels")
        print(f"  GPT-5 wins: {gpt5_wins} kernels")
        print(f"  Ties: {ties} kernels")
    
    # Print error breakdown
    if gemini_error_count > 0 or gpt5_error_count > 0:
        print(f"\nError breakdown:")
        
        if gemini_error_count > 0:
            gemini_error_types = {}
            for error in gemini_errors:
                if error:
                    gemini_error_types[error] = gemini_error_types.get(error, 0) + 1
            print(f"  Gemini-2.5-Flash errors: {gemini_error_types}")
        
        if gpt5_error_count > 0:
            gpt5_error_types = {}
            for error in gpt5_errors:
                if error:
                    gpt5_error_types[error] = gpt5_error_types.get(error, 0) + 1
            print(f"  GPT-5 errors: {gpt5_error_types}")

def main():
    """Main function to run the comparison analysis."""
    # File paths
    gemini_file = "llm_output/20250905_230641_npu_new_gemini-2.5-flash/all_summary.json"
    gpt5_file = "llm_output/20250909_020427_npu_new_gpt-5/all_summary.json"
    output_file = "speedup_comparison_plot.png"
    
    # Load data
    print("Loading data...")
    gemini_data = load_json_data(gemini_file)
    gpt5_data = load_json_data(gpt5_file)
    
    print(f"Gemini-2.5-Flash: {len(gemini_data)} kernels")
    print(f"GPT-5: {len(gpt5_data)} kernels")
    
    # Extract speedup data for overlapped kernels
    print("\nExtracting speedup data...")
    kernel_names, gemini_speedups, gpt5_speedups, gemini_errors, gpt5_errors = extract_speedup_data(gemini_data, gpt5_data)
    
    # Create the comparison plot
    print("\nCreating comparison plot...")
    fig, ax = create_speedup_comparison_plot(kernel_names, gemini_speedups, gpt5_speedups, gemini_errors, gpt5_errors)
    
    # Save the plot
    save_plot(fig, output_file)
    
    # Print summary statistics
    print_summary_statistics(kernel_names, gemini_speedups, gpt5_speedups, gemini_errors, gpt5_errors)
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()
