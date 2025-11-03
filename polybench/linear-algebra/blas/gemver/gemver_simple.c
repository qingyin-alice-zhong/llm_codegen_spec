#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// General Matrix-Vector Multiplication with Vector Update
// A: n x n matrix
// u1, v1, u2, v2, w, x, y, z: n-dimensional vectors
// The operation sequence is:
// 1. A = A + u1 * v1^T + u2 * v2^T
// 2. x = x + beta * A^T * y
// 3. x = x + z
// 4. w = w + alpha * A * x
void kernel_gemver(int n,
                  double alpha,
                  double beta,
                  double** A,
                  double* u1,
                  double* v1,
                  double* u2,
                  double* v2,
                  double* w,
                  double* x,
                  double* y,
                  double* z)
{
    int i, j;

    // Step 1: Update A with rank-2 update
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
        }
    }

    // Step 2: Compute x = x + beta * A^T * y
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            x[i] = x[i] + beta * A[j][i] * y[j];
        }
    }

    // Step 3: Add z to x
    for (i = 0; i < n; i++) {
        x[i] = x[i] + z[i];
    }

    // Step 4: Compute w = w + alpha * A * x
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            w[i] = w[i] + alpha * A[i][j] * x[j];
        }
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_gemver_optimized(int n,
                           double alpha,
                           double beta,
                           double** A,
                           double* u1,
                           double* v1,
                           double* u2,
                           double* v2,
                           double* w,
                           double* x,
                           double* y,
                           double* z)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_gemver(n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
}

// Initialize arrays with some values
void init_array(int n,
               double* alpha,
               double* beta,
               double** A,
               double* u1,
               double* v1,
               double* u2,
               double* v2,
               double* w,
               double* x,
               double* y,
               double* z)
{
    int i, j;

    *alpha = 1.5;
    *beta = 1.2;

    double fn = (double)n;

    for (i = 0; i < n; i++) {
        u1[i] = i;
        u2[i] = ((i+1)/fn)/2.0;
        v1[i] = ((i+1)/fn)/4.0;
        v2[i] = ((i+1)/fn)/6.0;
        y[i] = ((i+1)/fn)/8.0;
        z[i] = ((i+1)/fn)/9.0;
        x[i] = 0.0;
        w[i] = 0.0;
        for (j = 0; j < n; j++) {
            A[i][j] = (i*j % n) / (double)n;
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
int verify_results(int n, double* w1, double* w2, double tolerance)
{
    for (int i = 0; i < n; i++) {
        if (fabs(w1[i] - w2[i]) > tolerance) {
            printf("Verification failed at position [%d]: %f != %f\n", 
                   i, w1[i], w2[i]);
            return 0;
        }
    }
    return 1;
}

// Run performance test for a kernel
double run_performance_test(void (*kernel)(int, double, double, double**, double*, double*, double*, double*, double*, double*, double*, double*),
                          int n,
                          double alpha, double beta,
                          double** A,
                          double* u1, double* v1,
                          double* u2, double* v2,
                          double* w, double* x,
                          double* y, double* z,
                          int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset arrays to initial state
        init_array(n, &alpha, &beta, A, u1, v1, u2, v2, w, x, y, z);
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
        
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
    int n = 2000;  // Size of matrix and vectors
    int num_runs = 5;  // Number of runs for performance testing

    // Variables
    double alpha, beta;

    // Allocate arrays for original kernel
    double** A1 = allocate_2d_array(n, n);
    double* u1_1 = allocate_1d_array(n);
    double* v1_1 = allocate_1d_array(n);
    double* u2_1 = allocate_1d_array(n);
    double* v2_1 = allocate_1d_array(n);
    double* w1 = allocate_1d_array(n);
    double* x1 = allocate_1d_array(n);
    double* y1 = allocate_1d_array(n);
    double* z1 = allocate_1d_array(n);

    // Allocate arrays for optimized kernel
    double** A2 = allocate_2d_array(n, n);
    double* u1_2 = allocate_1d_array(n);
    double* v1_2 = allocate_1d_array(n);
    double* u2_2 = allocate_1d_array(n);
    double* v2_2 = allocate_1d_array(n);
    double* w2 = allocate_1d_array(n);
    double* x2 = allocate_1d_array(n);
    double* y2 = allocate_1d_array(n);
    double* z2 = allocate_1d_array(n);

    // Initialize arrays
    init_array(n, &alpha, &beta, A1, u1_1, v1_1, u2_1, v2_1, w1, x1, y1, z1);
    init_array(n, &alpha, &beta, A2, u1_2, v1_2, u2_2, v2_2, w2, x2, y2, z2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_gemver, n,
                                              alpha, beta, A1,
                                              u1_1, v1_1, u2_1, v2_1,
                                              w1, x1, y1, z1, num_runs);
    
    double optimized_time = run_performance_test(kernel_gemver_optimized, n,
                                               alpha, beta, A2,
                                               u1_2, v1_2, u2_2, v2_2,
                                               w2, x2, y2, z2, num_runs);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(n, w1, w2, 1e-4)) {
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
    free_1d_array(u1_1);
    free_1d_array(v1_1);
    free_1d_array(u2_1);
    free_1d_array(v2_1);
    free_1d_array(w1);
    free_1d_array(x1);
    free_1d_array(y1);
    free_1d_array(z1);
    free_2d_array(A2, n);
    free_1d_array(u1_2);
    free_1d_array(v1_2);
    free_1d_array(u2_2);
    free_1d_array(v2_2);
    free_1d_array(w2);
    free_1d_array(x2);
    free_1d_array(y2);
    free_1d_array(z2);

    return 0;
} 
