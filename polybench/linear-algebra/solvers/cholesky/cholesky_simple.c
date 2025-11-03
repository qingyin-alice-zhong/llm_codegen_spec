#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// Cholesky Decomposition: A = L * L^T
// A: n x n positive definite matrix (input/output)
// The algorithm computes the lower triangular matrix L such that A = L * L^T
void kernel_cholesky(int n, double** A)
{
    int i, j, k;

    for (i = 0; i < n; i++) {
        // For j < i (lower triangular part)
        for (j = 0; j < i; j++) {
            for (k = 0; k < j; k++) {
                A[i][j] -= A[i][k] * A[j][k];
            }
            A[i][j] /= A[j][j];
        }
        
        // For i == j (diagonal element)
        for (k = 0; k < i; k++) {
            A[i][i] -= A[i][k] * A[i][k];
        }
        A[i][i] = sqrt(A[i][i]);
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_cholesky_optimized(int n, double** A)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_cholesky(n, A);
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

// Verify if two matrices are equal within a small tolerance (only lower triangular part)
int verify_results(int n, double** A1, double** A2, double tolerance)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {  // Only check lower triangular part
            if (fabs(A1[i][j] - A2[i][j]) > tolerance) {
                printf("Verification failed at position [%d][%d]: %f != %f\n", 
                       i, j, A1[i][j], A2[i][j]);
                return 0;
            }
        }
    }
    return 1;
}

// Copy matrix (only the parts that matter for Cholesky)
void copy_matrix(int n, double** src, double** dst)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dst[i][j] = src[i][j];
        }
    }
}

// Initialize arrays with some values to create a positive definite matrix
void init_array(int n, double** A)
{
    int i, j, r, s, t;

    // First, create an initial matrix
    for (i = 0; i < n; i++) {
        for (j = 0; j <= i; j++) {
            A[i][j] = (double)(-j % n) / n + 1;
        }
        for (j = i + 1; j < n; j++) {
            A[i][j] = 0;
        }
        A[i][i] = 1;
    }

    // Make the matrix positive semi-definite using A = B^T * B
    double** B = allocate_2d_array(n, n);
    
    // Copy A to B
    for (r = 0; r < n; r++) {
        for (s = 0; s < n; s++) {
            B[r][s] = A[r][s];
        }
    }
    
    // Initialize A to zero
    for (r = 0; r < n; r++) {
        for (s = 0; s < n; s++) {
            A[r][s] = 0;
        }
    }
    
    // Compute A = B^T * B to ensure positive definiteness
    for (t = 0; t < n; t++) {
        for (r = 0; r < n; r++) {
            for (s = 0; s < n; s++) {
                A[r][s] += B[r][t] * B[s][t];
            }
        }
    }
    
    free_2d_array(B, n);
}

// Run performance test for a kernel
double run_performance_test(void (*kernel)(int, double**),
                          int n,
                          double** A_orig, double** A,
                          int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset matrix to initial state
        copy_matrix(n, A_orig, A);
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(n, A);
        
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
    int n = 2000;  // Size of the square matrix
    int num_runs = 1;  // Number of runs for performance testing

    // Allocate arrays for original kernel
    double** A1_orig = allocate_2d_array(n, n);
    double** A1 = allocate_2d_array(n, n);

    // Allocate arrays for optimized kernel
    double** A2 = allocate_2d_array(n, n);

    // Initialize arrays
    init_array(n, A1_orig);
    copy_matrix(n, A1_orig, A1);
    copy_matrix(n, A1_orig, A2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_cholesky, n,
                                              A1_orig, A1, num_runs);
    
    double optimized_time = run_performance_test(kernel_cholesky_optimized, n,
                                               A1_orig, A2, num_runs);

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Original kernel average time: %f seconds\n", original_time);
    printf("Optimized kernel average time: %f seconds\n", optimized_time);
    printf("Speedup: %f\n", original_time / optimized_time);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(n, A1, A2, 1e-4)) {
        printf("PASS: Results match within tolerance (1e-4)\n");
    } else {
        printf("FAIL: Results do not match\n");
    }

    // Free memory
    free_2d_array(A1_orig, n);
    free_2d_array(A1, n);
    free_2d_array(A2, n);

    return 0;
} 
