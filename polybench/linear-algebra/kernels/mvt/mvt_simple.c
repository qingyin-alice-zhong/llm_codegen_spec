#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// Matrix Vector Product and Transpose: x1 = x1 + A*y1; x2 = x2 + A^T*y2
// A: n x n matrix (input)
// x1, x2: n-element vectors (input/output)
// y1, y2: n-element vectors (input)
// Computes x1 = x1 + A*y1 and x2 = x2 + A^T*y2
void kernel_mvt(int n, double** A, double* x1, double* x2, double* y_1, double* y_2)
{
    int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            x1[i] = x1[i] + A[i][j] * y_1[j];
        }
    }
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            x2[i] = x2[i] + A[j][i] * y_2[j];
        }
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_mvt_optimized(int n, double** A, double* x1, double* x2, double* y_1, double* y_2)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_mvt(n, A, x1, x2, y_1, y_2);
}

// Initialize arrays with some values
void init_array(int n, double* x1, double* x2, double* y_1, double* y_2, double** A)
{
    int i, j;

    for (i = 0; i < n; i++) {
        x1[i] = (double)(i % n) / n;
        x2[i] = (double)((i + 1) % n) / n;
        y_1[i] = (double)((i + 3) % n) / n;
        y_2[i] = (double)((i + 4) % n) / n;
        for (j = 0; j < n; j++) {
            A[i][j] = (double)(i * j % n) / n;
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
int verify_vectors(int n, double* v1, double* v2, double tolerance)
{
    for (int i = 0; i < n; i++) {
        if (fabs(v1[i] - v2[i]) > tolerance) {
            printf("Verification failed at position [%d]: %f != %f\n", 
                   i, v1[i], v2[i]);
            return 0;
        }
    }
    return 1;
}

// Verify MVT results
int verify_results(int n, double* x1_1, double* x2_1, double* x1_2, double* x2_2, double tolerance)
{
    printf("Verifying x1 vectors...\n");
    if (!verify_vectors(n, x1_1, x1_2, tolerance)) {
        return 0;
    }
    
    printf("Verifying x2 vectors...\n");
    if (!verify_vectors(n, x2_1, x2_2, tolerance)) {
        return 0;
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
double run_performance_test(void (*kernel)(int, double**, double*, double*, double*, double*),
                          int n,
                          double** A, double* x1_orig, double* x2_orig, 
                          double* x1, double* x2, double* y_1, double* y_2,
                          int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset vectors to initial state
        copy_vector(n, x1_orig, x1);
        copy_vector(n, x2_orig, x2);
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(n, A, x1, x2, y_1, y_2);
        
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
    int num_runs = 5;  // Number of runs for performance testing

    // Allocate arrays
    double** A = allocate_2d_array(n, n);
    double* x1_orig = allocate_1d_array(n);
    double* x2_orig = allocate_1d_array(n);
    double* y_1 = allocate_1d_array(n);
    double* y_2 = allocate_1d_array(n);

    // Arrays for original kernel
    double* x1_1 = allocate_1d_array(n);
    double* x2_1 = allocate_1d_array(n);

    // Arrays for optimized kernel
    double* x1_2 = allocate_1d_array(n);
    double* x2_2 = allocate_1d_array(n);

    // Initialize arrays
    init_array(n, x1_orig, x2_orig, y_1, y_2, A);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_mvt, n,
                                              A, x1_orig, x2_orig,
                                              x1_1, x2_1, y_1, y_2, num_runs);
    
    double optimized_time = run_performance_test(kernel_mvt_optimized, n,
                                               A, x1_orig, x2_orig,
                                               x1_2, x2_2, y_1, y_2, num_runs);

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Original kernel average time: %f seconds\n", original_time);
    printf("Optimized kernel average time: %f seconds\n", optimized_time);
    printf("Speedup: %f\n", original_time / optimized_time);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(n, x1_1, x2_1, x1_2, x2_2, 1e-4)) {
        printf("PASS: Results match within tolerance (1e-4)\n");
    } else {
        printf("FAIL: Results do not match\n");
    }

    // Free memory
    free_2d_array(A, n);
    free_1d_array(x1_orig);
    free_1d_array(x2_orig);
    free_1d_array(y_1);
    free_1d_array(y_2);
    free_1d_array(x1_1);
    free_1d_array(x2_1);
    free_1d_array(x1_2);
    free_1d_array(x2_2);

    return 0;
} 
