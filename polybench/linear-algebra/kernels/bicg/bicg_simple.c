#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// BICG: BiConjugate Gradient
// s = A^T * r (transpose matrix-vector multiplication)
// q = A * p (matrix-vector multiplication)
// A: n x m matrix
// r: n-dimensional input vector
// p: m-dimensional input vector
// s: m-dimensional output vector
// q: n-dimensional output vector
void kernel_bicg(int m, int n,
                double** A,
                double* s,
                double* q,
                double* p,
                double* r)
{
    int i, j;

    // Initialize s to zero
    for (i = 0; i < m; i++) {
        s[i] = 0.0;
    }

    // Main computation
    for (i = 0; i < n; i++) {
        q[i] = 0.0;
        for (j = 0; j < m; j++) {
            s[j] = s[j] + r[i] * A[i][j];  // s += A^T * r
            q[i] = q[i] + A[i][j] * p[j];  // q = A * p
        }
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_bicg_optimized(int m, int n,
                          double** A,
                          double* s,
                          double* q,
                          double* p,
                          double* r)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_bicg(m, n, A, s, q, p, r);
}

// Initialize arrays with some values
void init_array(int m, int n,
               double** A,
               double* r,
               double* p)
{
    int i, j;

    for (i = 0; i < m; i++) {
        p[i] = (double)(i % m) / m;
    }

    for (i = 0; i < n; i++) {
        r[i] = (double)(i % n) / n;
        for (j = 0; j < m; j++) {
            A[i][j] = (double)(i * (j + 1) % n) / n;
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
int verify_results(int size, double* v1, double* v2, double tolerance)
{
    for (int i = 0; i < size; i++) {
        if (fabs(v1[i] - v2[i]) > tolerance) {
            printf("Verification failed at position [%d]: %f != %f\n", 
                   i, v1[i], v2[i]);
            return 0;
        }
    }
    return 1;
}

// Run performance test for a kernel
double run_performance_test(void (*kernel)(int, int, double**, double*, double*, double*, double*),
                          int m, int n,
                          double** A, double* s, double* q, double* p, double* r,
                          int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset arrays to initial state
        init_array(m, n, A, r, p);
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(m, n, A, s, q, p, r);
        
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
    int m = 1900;  // Number of columns in matrix A
    int n = 2100;  // Number of rows in matrix A
    int num_runs = 5;  // Number of runs for performance testing

    // Allocate arrays for original kernel
    double** A1 = allocate_2d_array(n, m);
    double* s1 = allocate_1d_array(m);
    double* q1 = allocate_1d_array(n);
    double* p1 = allocate_1d_array(m);
    double* r1 = allocate_1d_array(n);

    // Allocate arrays for optimized kernel
    double** A2 = allocate_2d_array(n, m);
    double* s2 = allocate_1d_array(m);
    double* q2 = allocate_1d_array(n);
    double* p2 = allocate_1d_array(m);
    double* r2 = allocate_1d_array(n);

    // Initialize arrays
    init_array(m, n, A1, r1, p1);
    init_array(m, n, A2, r2, p2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_bicg, m, n,
                                              A1, s1, q1, p1, r1, num_runs);
    
    double optimized_time = run_performance_test(kernel_bicg_optimized, m, n,
                                               A2, s2, q2, p2, r2, num_runs);

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Original kernel average time: %f seconds\n", original_time);
    printf("Optimized kernel average time: %f seconds\n", optimized_time);
    printf("Speedup: %f\n", original_time / optimized_time);

    // Verify results
    printf("\nVerification Results:\n");
    int s_match = verify_results(m, s1, s2, 1e-4);
    int q_match = verify_results(n, q1, q2, 1e-4);
    
    if (s_match && q_match) {
        printf("PASS: Results match within tolerance (1e-4)\n");
    } else {
        printf("FAIL: Results do not match\n");
    }

    // Free memory
    free_2d_array(A1, n);
    free_1d_array(s1);
    free_1d_array(q1);
    free_1d_array(p1);
    free_1d_array(r1);
    free_2d_array(A2, n);
    free_1d_array(s2);
    free_1d_array(q2);
    free_1d_array(p2);
    free_1d_array(r2);

    return 0;
} 
