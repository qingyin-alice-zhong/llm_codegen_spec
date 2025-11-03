#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// Triangular Matrix Multiplication: B = alpha * A^T * B
// A: m x m triangular matrix
// B: m x n matrix
void kernel_trmm(int m, int n,
                double alpha,
                double** A,
                double** B)
{
    int i, j, k;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            for (k = i+1; k < m; k++) {
                B[i][j] += A[k][i] * B[k][j];
            }
            B[i][j] = alpha * B[i][j];
        }
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_trmm_optimized(int m, int n,
                          double alpha,
                          double** A,
                          double** B)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_trmm(m, n, alpha, A, B);
}

// Initialize matrices with some values
void init_array(int m, int n,
               double* alpha,
               double** A,
               double** B)
{
    int i, j;

    *alpha = 1.5;
    
    // Initialize A as a lower triangular matrix
    for (i = 0; i < m; i++) {
        for (j = 0; j < i; j++) {
            A[i][j] = ((i+j) % m) / (double)m;
        }
        A[i][i] = 1.0;  // Diagonal elements are 1.0
    }

    // Initialize B
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            B[i][j] = ((n+(i-j)) % n) / (double)n;
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
int verify_results(int m, int n, double** B1, double** B2, double tolerance)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(B1[i][j] - B2[i][j]) > tolerance) {
                printf("Verification failed at position [%d][%d]: %f != %f\n", 
                       i, j, B1[i][j], B2[i][j]);
                return 0;
            }
        }
    }
    return 1;
}

// Run performance test for a kernel
double run_performance_test(void (*kernel)(int, int, double, double**, double**),
                          int m, int n,
                          double alpha,
                          double** A, double** B,
                          int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset B to initial state
        init_array(m, n, &alpha, A, B);
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(m, n, alpha, A, B);
        
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
    int m = 1000;  // Size of square matrix A
    int n = 1200;  // Number of columns in B
    int num_runs = 5;  // Number of runs for performance testing

    // Variables
    double alpha;

    // Allocate matrices for original kernel
    double** A1 = allocate_2d_array(m, m);
    double** B1 = allocate_2d_array(m, n);

    // Allocate matrices for optimized kernel
    double** A2 = allocate_2d_array(m, m);
    double** B2 = allocate_2d_array(m, n);

    // Initialize arrays
    init_array(m, n, &alpha, A1, B1);
    init_array(m, n, &alpha, A2, B2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_trmm, m, n, 
                                              alpha, A1, B1, num_runs);
    
    double optimized_time = run_performance_test(kernel_trmm_optimized, m, n, 
                                               alpha, A2, B2, num_runs);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(m, n, B1, B2, 1e-4)) {
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
    free_2d_array(A1, m);
    free_2d_array(B1, m);
    free_2d_array(A2, m);
    free_2d_array(B2, m);

    return 0;
} 
