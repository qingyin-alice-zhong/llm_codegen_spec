#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// 1D Jacobi stencil computation
// Performs iterative smoothing on a 1D array
void kernel_jacobi_1d(int tsteps, int n, double* A, double* B)
{
    int t, i;

    for (t = 0; t < tsteps; t++) {
        // Update B based on A
        for (i = 1; i < n - 1; i++) {
            B[i] = 0.33333 * (A[i-1] + A[i] + A[i+1]);
        }
        
        // Update A based on B
        for (i = 1; i < n - 1; i++) {
            A[i] = 0.33333 * (B[i-1] + B[i] + B[i+1]);
        }
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_jacobi_1d_optimized(int tsteps, int n, double* A, double* B)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_jacobi_1d(tsteps, n, A, B);
}

// Allocate a 1D array
double* allocate_1d_array(int size)
{
    return (double*)malloc(size * sizeof(double));
}

// Free a 1D array
void free_1d_array(double* array)
{
    if (array) {
        free(array);
    }
}

// Initialize arrays
void init_array(int n, double* A, double* B)
{
    int i;

    for (i = 0; i < n; i++) {
        A[i] = ((double)i + 2) / n;
        B[i] = ((double)i + 3) / n;
    }
}

// Verify if two vectors are equal within tolerance
int verify_results(int n, double* A1, double* A2, double tolerance)
{
    printf("Verifying stencil results with tolerance %e...\n", tolerance);
    
    for (int i = 0; i < n; i++) {
        if (fabs(A1[i] - A2[i]) > tolerance) {
            printf("Verification failed at position [%d]: %f != %f (diff: %e)\n", 
                   i, A1[i], A2[i], fabs(A1[i] - A2[i]));
            return 0;
        }
    }
    return 1;
}

// Copy vector
void copy_vector(int n, double* src, double* dst)
{
    for (int i = 0; i < n; i++) {
        dst[i] = src[i];
    }
}

// Run performance test for a kernel
double run_performance_test(void (*kernel)(int, int, double*, double*),
                          int tsteps, int n, double* A_orig, double* B_orig,
                          double* A, double* B, int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset arrays to initial state
        copy_vector(n, A_orig, A);
        copy_vector(n, B_orig, B);
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(tsteps, n, A, B);
        
        // Stop timing
        clock_gettime(CLOCK_MONOTONIC, &end);
        double time_spent = (end.tv_sec - start.tv_sec) + 
                          (end.tv_nsec - start.tv_nsec) / 1e9;
        total_time += time_spent;
    }
    
    return total_time / num_runs;
}

int main(int argc, char** argv)
{
    // Problem size (using LARGE_DATASET dimensions)
    int n = 4000;    // Array size
    int tsteps = 1000;   // Number of time steps
    int num_runs = 10;   // Number of runs for performance testing

    // Allocate arrays
    double* A_orig = allocate_1d_array(n);
    double* B_orig = allocate_1d_array(n);
    double* A1 = allocate_1d_array(n);
    double* B1 = allocate_1d_array(n);
    double* A2 = allocate_1d_array(n);
    double* B2 = allocate_1d_array(n);

    // Initialize arrays
    init_array(n, A_orig, B_orig);
    copy_vector(n, A_orig, A1);
    copy_vector(n, B_orig, B1);
    copy_vector(n, A_orig, A2);
    copy_vector(n, B_orig, B2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_jacobi_1d, tsteps, n,
                                              A_orig, B_orig, A1, B1, num_runs);
    
    double optimized_time = run_performance_test(kernel_jacobi_1d_optimized, tsteps, n,
                                               A_orig, B_orig, A2, B2, num_runs);

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Original kernel average time: %f seconds\n", original_time);
    printf("Optimized kernel average time: %f seconds\n", optimized_time);
    printf("Speedup: %f\n", original_time / optimized_time);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(n, A1, A2, 1e-4)) {
        printf("PASS: Results match within tolerance\n");
    } else {
        printf("FAIL: Results do not match within tolerance\n");
    }

    // Free memory
    free_1d_array(A_orig);
    free_1d_array(B_orig);
    free_1d_array(A1);
    free_1d_array(B1);
    free_1d_array(A2);
    free_1d_array(B2);

    return 0;
} 
