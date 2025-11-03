#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// Triangular Solver using Forward Substitution
// L: n x n lower triangular matrix (input)
// b: n-element right-hand side vector (input)
// x: n-element solution vector (output)
// Solves Lx = b using forward substitution
void kernel_trisolv(int n, double** L, double* x, double* b)
{
    int i, j;

    for (i = 0; i < n; i++) {
        x[i] = b[i];
        for (j = 0; j < i; j++) {
            x[i] -= L[i][j] * x[j];
        }
        x[i] = x[i] / L[i][i];
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_trisolv_optimized(int n, double** L, double* x, double* b)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_trisolv(n, L, x, b);
}

// Initialize arrays with some values
void init_array(int n, double** L, double* x, double* b)
{
    int i, j;

    for (i = 0; i < n; i++) {
        x[i] = -999;
        b[i] = i;
        for (j = 0; j <= i; j++) {
            L[i][j] = (double)(i + n - j + 1) * 2 / n;
        }
        // Ensure upper triangular part is zero
        for (j = i + 1; j < n; j++) {
            L[i][j] = 0.0;
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
int verify_results(int n, double* x1, double* x2, double tolerance)
{
    for (int i = 0; i < n; i++) {
        if (fabs(x1[i] - x2[i]) > tolerance) {
            printf("Verification failed at position [%d]: %f != %f\n", 
                   i, x1[i], x2[i]);
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

// Zero vector
void zero_vector(int n, double* vec)
{
    for (int i = 0; i < n; i++) {
        vec[i] = 0.0;
    }
}

// Run performance test for a kernel
double run_performance_test(void (*kernel)(int, double**, double*, double*),
                          int n,
                          double** L, double* x, double* b,
                          int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset solution vector (x is output)
        zero_vector(n, x);
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(n, L, x, b);
        
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
    int num_runs = 1;  // Number of runs for performance testing

    // Allocate arrays for original kernel
    double** L = allocate_2d_array(n, n);
    double* x1 = allocate_1d_array(n);
    double* b = allocate_1d_array(n);

    // Allocate arrays for optimized kernel
    double* x2 = allocate_1d_array(n);

    // Initialize arrays
    init_array(n, L, x1, b);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_trisolv, n,
                                              L, x1, b, num_runs);
    
    double optimized_time = run_performance_test(kernel_trisolv_optimized, n,
                                               L, x2, b, num_runs);

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Original kernel average time: %f seconds\n", original_time);
    printf("Optimized kernel average time: %f seconds\n", optimized_time);
    printf("Speedup: %f\n", original_time / optimized_time);

    // Verify results (solution vector x)
    printf("\nVerification Results:\n");
    if (verify_results(n, x1, x2, 1e-4)) {
        printf("PASS: Results match within tolerance (1e-4)\n");
    } else {
        printf("FAIL: Results do not match\n");
    }

    // Free memory
    free_2d_array(L, n);
    free_1d_array(x1);
    free_1d_array(b);
    free_1d_array(x2);

    return 0;
} 
