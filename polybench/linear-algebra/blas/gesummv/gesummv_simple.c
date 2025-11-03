#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// General Matrix-Vector Multiplication: y = alpha * (A * x) + beta * (B * x)
// A, B: n x n matrices
// x: n-dimensional vector
// y: n-dimensional vector
void kernel_gesummv(int n,
                   double alpha,
                   double beta,
                   double** A,
                   double** B,
                   double* tmp,
                   double* x,
                   double* y)
{
    int i, j;

    for (i = 0; i < n; i++) {
        tmp[i] = 0.0;
        y[i] = 0.0;
        for (j = 0; j < n; j++) {
            tmp[i] += A[i][j] * x[j];
            y[i] += B[i][j] * x[j];
        }
        y[i] = alpha * tmp[i] + beta * y[i];
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_gesummv_optimized(int n,
                            double alpha,
                            double beta,
                            double** A,
                            double** B,
                            double* tmp,
                            double* x,
                            double* y)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_gesummv(n, alpha, beta, A, B, tmp, x, y);
}

// Initialize arrays with some values
void init_array(int n,
               double* alpha,
               double* beta,
               double** A,
               double** B,
               double* x)
{
    int i, j;

    *alpha = 1.5;
    *beta = 1.2;

    // Initialize x
    for (i = 0; i < n; i++) {
        x[i] = (i % n) / (double)n;
    }

    // Initialize A and B
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            A[i][j] = ((i*j+1) % n) / (double)n;
            B[i][j] = ((i*j+2) % n) / (double)n;
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
double run_performance_test(void (*kernel)(int, double, double, double**, double**, double*, double*, double*),
                          int n,
                          double alpha, double beta,
                          double** A, double** B,
                          double* tmp, double* x, double* y,
                          int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset arrays to initial state
        init_array(n, &alpha, &beta, A, B, x);
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(n, alpha, beta, A, B, tmp, x, y);
        
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
    int n = 1300;  // Size of matrices and vectors
    int num_runs = 5;  // Number of runs for performance testing

    // Variables
    double alpha, beta;

    // Allocate arrays for original kernel
    double** A1 = allocate_2d_array(n, n);
    double** B1 = allocate_2d_array(n, n);
    double* tmp1 = allocate_1d_array(n);
    double* x1 = allocate_1d_array(n);
    double* y1 = allocate_1d_array(n);

    // Allocate arrays for optimized kernel
    double** A2 = allocate_2d_array(n, n);
    double** B2 = allocate_2d_array(n, n);
    double* tmp2 = allocate_1d_array(n);
    double* x2 = allocate_1d_array(n);
    double* y2 = allocate_1d_array(n);

    // Initialize arrays
    init_array(n, &alpha, &beta, A1, B1, x1);
    init_array(n, &alpha, &beta, A2, B2, x2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_gesummv, n,
                                              alpha, beta, A1, B1,
                                              tmp1, x1, y1, num_runs);
    
    double optimized_time = run_performance_test(kernel_gesummv_optimized, n,
                                               alpha, beta, A2, B2,
                                               tmp2, x2, y2, num_runs);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(n, y1, y2, 1e-4)) {
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
    free_2d_array(A1, n);
    free_2d_array(B1, n);
    free_1d_array(tmp1);
    free_1d_array(x1);
    free_1d_array(y1);
    free_2d_array(A2, n);
    free_2d_array(B2, n);
    free_1d_array(tmp2);
    free_1d_array(x2);
    free_1d_array(y2);

    return 0;
} 
