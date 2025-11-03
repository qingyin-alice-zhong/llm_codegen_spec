#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// 2D Seidel stencil computation
// Performs iterative smoothing on a 2D grid using Gauss-Seidel method
void kernel_seidel_2d(int tsteps, int n, double** A)
{
    int t, i, j;

    for (t = 0; t <= tsteps - 1; t++) {
        for (i = 1; i <= n - 2; i++) {
            for (j = 1; j <= n - 2; j++) {
                A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1] +
                          A[i][j-1] + A[i][j] + A[i][j+1] +
                          A[i+1][j-1] + A[i+1][j] + A[i+1][j+1]) / 9.0;
            }
        }
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_seidel_2d_optimized(int tsteps, int n, double** A)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_seidel_2d(tsteps, n, A);
}

// Allocate a 2D array as flat continuous memory with pointer array for row access
double** allocate_2d_array(int rows, int cols)
{
    // Allocate flat continuous memory for the data
    double* data = (double*)malloc(rows * cols * sizeof(double));
    if (!data) return NULL;
    
    // Allocate array of pointers to each row
    double** array = (double**)malloc(rows * sizeof(double*));
    if (!array) {
        free(data);
        return NULL;
    }
    
    // Set up row pointers
    for (int i = 0; i < rows; i++) {
        array[i] = data + i * cols;
    }
    
    return array;
}

// Free a 2D array allocated with flat continuous memory
void free_2d_array(double** array, int rows)
{
    if (array) {
        // Free the continuous data (accessible through array[0])
        free(array[0]);
        // Free the pointer array
        free(array);
    }
}

// Initialize arrays
void init_array(int n, double** A)
{
    int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            // A[i][j] = ((double)i * (j + 2) + 2) / n;
            A[i][j] = (rand() / (double)RAND_MAX) * 1000;
        }
    }
}

// Verify if two matrices are equal within tolerance
int verify_results(int n, double** A1, double** A2, double tolerance)
{
    printf("Verifying Seidel 2D results with tolerance %e...\n", tolerance);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(A1[i][j] - A2[i][j]) > tolerance) {
                printf("Verification failed at position [%d][%d]: %f != %f (diff: %e)\n", 
                       i, j, A1[i][j], A2[i][j], fabs(A1[i][j] - A2[i][j]));
                return 0;
            }
        }
    }
    return 1;
}

// Copy matrix
void copy_matrix(int rows, int cols, double** src, double** dst)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[i][j] = src[i][j];
        }
    }
}

// Run performance test for a kernel
double run_performance_test(void (*kernel)(int, int, double**),
                          int tsteps, int n, double** A_orig, double** A, int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset array to initial state
        copy_matrix(n, n, A_orig, A);
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(tsteps, n, A);
        
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
    int n = 2000;       // Grid size (n x n)
    int tsteps = 500;   // Number of time steps
    int num_runs = 1;   // Number of runs for performance testing

    // Allocate arrays
    double** A_orig = allocate_2d_array(n, n);
    double** A1 = allocate_2d_array(n, n);
    double** A2 = allocate_2d_array(n, n);

    // Initialize arrays
    init_array(n, A_orig);
    copy_matrix(n, n, A_orig, A1);
    copy_matrix(n, n, A_orig, A2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_seidel_2d, tsteps, n,
                                              A_orig, A1, num_runs);
    
    double optimized_time = run_performance_test(kernel_seidel_2d_optimized, tsteps, n,
                                               A_orig, A2, num_runs);

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
    free_2d_array(A_orig, n);
    free_2d_array(A1, n);
    free_2d_array(A2, n);

    return 0;
} 
