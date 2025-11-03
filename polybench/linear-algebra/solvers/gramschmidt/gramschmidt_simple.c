#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// QR Decomposition with Modified Gram-Schmidt
// A: m x n matrix (input/modified during computation)
// R: n x n upper triangular matrix (output)
// Q: m x n orthogonal matrix (output)
// The algorithm computes Q and R such that A = Q * R
void kernel_gramschmidt(int m, int n, double** A, double** R, double** Q)
{
    int i, j, k;
    double nrm;

    for (k = 0; k < n; k++) {
        nrm = 0.0;
        for (i = 0; i < m; i++) {
            nrm += A[i][k] * A[i][k];
        }
        R[k][k] = sqrt(nrm);
        
        for (i = 0; i < m; i++) {
            Q[i][k] = A[i][k] / R[k][k];
        }
        
        for (j = k + 1; j < n; j++) {
            R[k][j] = 0.0;
            for (i = 0; i < m; i++) {
                R[k][j] += Q[i][k] * A[i][j];
            }
            for (i = 0; i < m; i++) {
                A[i][j] = A[i][j] - Q[i][k] * R[k][j];
            }
        }
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_gramschmidt_optimized(int m, int n, double** A, double** R, double** Q)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_gramschmidt(m, n, A, R, Q);
}

// Initialize arrays with some values
void init_array(int m, int n, double** A, double** R, double** Q)
{
    int i, j;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            A[i][j] = (double)(((i * j) % m) / (double)m) * 100 + 10;
            Q[i][j] = 0.0;
        }
    }
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            R[i][j] = 0.0;
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
int verify_matrices(int rows, int cols, double** A1, double** A2, double tolerance)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fabs(A1[i][j] - A2[i][j]) > tolerance) {
                printf("Verification failed at position [%d][%d]: %f != %f\n", 
                       i, j, A1[i][j], A2[i][j]);
                return 0;
            }
        }
    }
    return 1;
}

// Verify QR decomposition results
int verify_results(int m, int n, double** Q1, double** R1, double** Q2, double** R2, double tolerance)
{
    printf("Verifying Q matrices...\n");
    if (!verify_matrices(m, n, Q1, Q2, tolerance)) {
        return 0;
    }
    
    printf("Verifying R matrices...\n");
    if (!verify_matrices(n, n, R1, R2, tolerance)) {
        return 0;
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
double run_performance_test(void (*kernel)(int, int, double**, double**, double**),
                          int m, int n,
                          double** A_orig, double** A, double** R, double** Q,
                          int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset matrices to initial state
        copy_matrix(m, n, A_orig, A);
        init_array(m, n, A, R, Q);  // Initialize R and Q to zero
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(m, n, A, R, Q);
        
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
    int m = 1000;  // Number of rows
    int n = 1200;  // Number of columns
    int num_runs = 1;  // Number of runs for performance testing

    // Allocate arrays for original kernel
    double** A1_orig = allocate_2d_array(m, n);
    double** A1 = allocate_2d_array(m, n);
    double** R1 = allocate_2d_array(n, n);
    double** Q1 = allocate_2d_array(m, n);

    // Allocate arrays for optimized kernel
    double** A2 = allocate_2d_array(m, n);
    double** R2 = allocate_2d_array(n, n);
    double** Q2 = allocate_2d_array(m, n);

    // Initialize arrays
    init_array(m, n, A1_orig, R1, Q1);
    copy_matrix(m, n, A1_orig, A1);
    copy_matrix(m, n, A1_orig, A2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_gramschmidt, m, n,
                                              A1_orig, A1, R1, Q1, num_runs);
    
    double optimized_time = run_performance_test(kernel_gramschmidt_optimized, m, n,
                                               A1_orig, A2, R2, Q2, num_runs);

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Original kernel average time: %f seconds\n", original_time);
    printf("Optimized kernel average time: %f seconds\n", optimized_time);
    printf("Speedup: %f\n", original_time / optimized_time);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(m, n, Q1, R1, Q2, R2, 1e-4)) {
        printf("PASS: Results match within tolerance (1e-4)\n");
    } else {
        printf("FAIL: Results do not match\n");
    }

    // Free memory
    free_2d_array(A1_orig, m);
    free_2d_array(A1, m);
    free_2d_array(R1, n);
    free_2d_array(Q1, m);
    free_2d_array(A2, m);
    free_2d_array(R2, n);
    free_2d_array(Q2, m);

    return 0;
} 
