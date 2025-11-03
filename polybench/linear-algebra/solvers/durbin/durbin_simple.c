#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// Durbin-Levinson Algorithm for solving Toeplitz systems
// r: n-element autocorrelation vector (input)
// y: n-element solution vector (output)
// The algorithm solves the Toeplitz system for the reflection coefficients
void kernel_durbin(int n, double* r, double* y)
{
    double* z = (double*)malloc(n * sizeof(double));
    double alpha;
    double beta;
    double sum;
    int i, k;

    y[0] = -r[0];
    beta = 1.0;
    alpha = -r[0];

    for (k = 1; k < n; k++) {
        beta = (1 - alpha * alpha) * beta;
        sum = 0.0;
        for (i = 0; i < k; i++) {
            sum += r[k - i - 1] * y[i];
        }
        alpha = -(r[k] + sum) / beta;

        for (i = 0; i < k; i++) {
            z[i] = y[i] + alpha * y[k - i - 1];
        }
        for (i = 0; i < k; i++) {
            y[i] = z[i];
        }
        y[k] = alpha;
    }

    free(z);
}

// Placeholder for optimized kernel - to be replaced later
void kernel_durbin_optimized(int n, double* r, double* y)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_durbin(n, r, y);
}

// Initialize arrays with some values
void init_array(int n, double* r)
{
    int i;

    for (i = 0; i < n; i++) {
        r[i] = (double)(n + 1 - i);
    }
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

// Verify if two vectors are equal within a small tolerance
int verify_results(int n, double* y1, double* y2, double tolerance)
{
    for (int i = 0; i < n; i++) {
        if (fabs(y1[i] - y2[i]) > tolerance) {
            printf("Verification failed at position [%d]: %f != %f\n", 
                   i, y1[i], y2[i]);
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

// Initialize vector to zero
void zero_vector(int n, double* vec)
{
    for (int i = 0; i < n; i++) {
        vec[i] = 0.0;
    }
}

// Run performance test for a kernel
double run_performance_test(void (*kernel)(int, double*, double*),
                          int n,
                          double* r, double* y,
                          int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset output vector
        zero_vector(n, y);
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(n, r, y);
        
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
    int n = 2000;  // Size of the problem
    int num_runs = 10;  // Number of runs for performance testing

    // Allocate arrays for original kernel
    double* r = allocate_1d_array(n);
    double* y1 = allocate_1d_array(n);

    // Allocate arrays for optimized kernel
    double* y2 = allocate_1d_array(n);

    // Initialize arrays
    init_array(n, r);
    zero_vector(n, y1);
    zero_vector(n, y2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_durbin, n,
                                              r, y1, num_runs);
    
    double optimized_time = run_performance_test(kernel_durbin_optimized, n,
                                               r, y2, num_runs);

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Original kernel average time: %f seconds\n", original_time);
    printf("Optimized kernel average time: %f seconds\n", optimized_time);
    printf("Speedup: %f\n", original_time / optimized_time);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(n, y1, y2, 1e-4)) {
        printf("PASS: Results match within tolerance (1e-4)\n");
    } else {
        printf("FAIL: Results do not match\n");
    }

    // Free memory
    free_1d_array(r);
    free_1d_array(y1);
    free_1d_array(y2);

    return 0;
} 
