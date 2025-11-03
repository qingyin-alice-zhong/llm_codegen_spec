#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// ATAX: Matrix Transpose and Vector Multiplication
// y = A^T * (A * x)
// A: m x n matrix
// x: n-dimensional vector (input)
// y: n-dimensional vector (output)
// tmp: m-dimensional vector (temporary)
void kernel_atax(int m, int n,
                double** A,
                double* x,
                double* y,
                double* tmp)
{
    int i, j;

    // Initialize y to zero
    for (i = 0; i < n; i++) {
        y[i] = 0.0;
    }

    // Compute tmp = A * x and y = A^T * tmp
    for (i = 0; i < m; i++) {
        tmp[i] = 0.0;
        // First: tmp[i] = sum(A[i][j] * x[j]) for j = 0 to n-1
        for (j = 0; j < n; j++) {
            tmp[i] += A[i][j] * x[j];
        }
        // Second: y[j] += A[i][j] * tmp[i] for j = 0 to n-1
        for (j = 0; j < n; j++) {
            y[j] += A[i][j] * tmp[i];
        }
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_atax_optimized(int m, int n,
                          double** A,
                          double* x,
                          double* y,
                          double* tmp)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_atax(m, n, A, x, y, tmp);
}

// Initialize arrays with some values
void init_array(int m, int n,
               double** A,
               double* x)
{
    int i, j;
    double fn = (double)n;

    // Initialize x
    for (i = 0; i < n; i++) {
        x[i] = 1.0 + (i / fn);
    }

    // Initialize A
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            A[i][j] = ((i+j) % n) / (5.0*m);
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

// Allocate a 1D array
double* allocate_1d_array(int size)
{
    return (double*)malloc(size * sizeof(double));
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

// Run performance test for a kernel
double run_performance_test(void (*kernel)(int, int, double**, double*, double*, double*),
                          int m, int n,
                          double** A, double* x, double* y, double* tmp,
                          int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset arrays to initial state
        init_array(m, n, A, x);
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(m, n, A, x, y, tmp);
        
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
    int m = 1900;  // Number of rows in matrix A
    int n = 2100;  // Number of columns in matrix A
    int num_runs = 5;  // Number of runs for performance testing

    // Allocate arrays for original kernel
    double** A1 = allocate_2d_array(m, n);
    double* x1 = allocate_1d_array(n);
    double* y1 = allocate_1d_array(n);
    double* tmp1 = allocate_1d_array(m);

    // Allocate arrays for optimized kernel
    double** A2 = allocate_2d_array(m, n);
    double* x2 = allocate_1d_array(n);
    double* y2 = allocate_1d_array(n);
    double* tmp2 = allocate_1d_array(m);

    // Initialize arrays
    init_array(m, n, A1, x1);
    init_array(m, n, A2, x2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_atax, m, n,
                                              A1, x1, y1, tmp1, num_runs);
    
    double optimized_time = run_performance_test(kernel_atax_optimized, m, n,
                                               A2, x2, y2, tmp2, num_runs);

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
    free_2d_array(A1, m);
    free_1d_array(x1);
    free_1d_array(y1);
    free_1d_array(tmp1);
    free_2d_array(A2, m);
    free_1d_array(x2);
    free_1d_array(y2);
    free_1d_array(tmp2);

    return 0;
} 
