#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// Symmetric Rank-K update: C = alpha * A * A^T + beta * C
// A: n x m matrix
// C: n x n symmetric matrix (only lower triangular part is stored)
void kernel_syrk(int n, int m,
                double alpha,
                double beta,
                double** C,
                double** A)
{
    int i, j, k;

    for (i = 0; i < n; i++) {
        // First multiply C by beta (only lower triangular part)
        for (j = 0; j <= i; j++) {
            C[i][j] *= beta;
        }
        // Then add alpha * A * A^T
        for (k = 0; k < m; k++) {
            for (j = 0; j <= i; j++) {
                C[i][j] += alpha * A[i][k] * A[j][k];
            }
        }
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_syrk_optimized(int n, int m,
                          double alpha,
                          double beta,
                          double** C,
                          double** A)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_syrk(n, m, alpha, beta, C, A);
}

// Initialize matrices with some values
void init_array(int n, int m,
               double* alpha,
               double* beta,
               double** C,
               double** A)
{
    int i, j;

    *alpha = 1.5;
    *beta = 1.2;

    // Initialize A
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            A[i][j] = ((i*j+1) % n) / (double)n;
        }
    }

    // Initialize C (only lower triangular part)
    for (i = 0; i < n; i++) {
        for (j = 0; j <= i; j++) {
            C[i][j] = ((i*j+2) % m) / (double)m;
        }
    }
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

// Verify if two matrices are equal within a small tolerance
int verify_results(int n, double** C1, double** C2, double tolerance)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(C1[i][j] - C2[i][j]) > tolerance) {
                printf("Verification failed at position [%d][%d]: %f != %f\n", 
                       i, j, C1[i][j], C2[i][j]);
                return 0;
            }
        }
    }
    return 1;
}

// Run performance test for a kernel
double run_performance_test(void (*kernel)(int, int, double, double, double**, double**),
                          int n, int m,
                          double alpha, double beta,
                          double** C, double** A,
                          int num_runs)
{
    double total_time = 0.0;
    struct timespec start, end;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset C to initial state
        init_array(n, m, &alpha, &beta, C, A);
        
        // Start timing using monotonic clock
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(n, m, alpha, beta, C, A);
        
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
    // Problem size
    int n = 1000;  // Size of square matrix C
    int m = 1200;  // Number of columns in A
    int num_runs = 5;  // Number of runs for performance testing

    // Variables
    double alpha, beta;

    // Allocate matrices for original kernel
    double** C1 = allocate_2d_array(n, n);
    double** A1 = allocate_2d_array(n, m);

    // Allocate matrices for optimized kernel
    double** C2 = allocate_2d_array(n, n);
    double** A2 = allocate_2d_array(n, m);

    // Initialize arrays
    init_array(n, m, &alpha, &beta, C1, A1);
    init_array(n, m, &alpha, &beta, C2, A2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_syrk, n, m, 
                                              alpha, beta, C1, A1, num_runs);
    
    double optimized_time = run_performance_test(kernel_syrk_optimized, n, m, 
                                               alpha, beta, C2, A2, num_runs);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(n, C1, C2, 1e-4)) {
        printf("PASS: Results match within tolerance (1e-4)\n");
    } else {
        printf("FAIL: Results do not match\n");
    }

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Original kernel average time: %f seconds\n", original_time);
    printf("Optimized kernel average time: %f seconds\n", optimized_time);
    printf("Speedup: %f\n", original_time / optimized_time);



    // Free memory
    free_2d_array(C1, n);
    free_2d_array(A1, n);
    free_2d_array(C2, n);
    free_2d_array(A2, n);

    return 0;
} 