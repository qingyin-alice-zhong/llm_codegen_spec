#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// LU Decomposition and Linear System Solver
// A: n x n matrix (input/output - stores LU decomposition)
// b: n-element right-hand side vector (input)
// x: n-element solution vector (output)
// y: n-element intermediate vector (output)
// Solves Ax = b using LU decomposition: A = LU, then Ly = b, Ux = y
void kernel_ludcmp(int n, double** A, double* b, double* x, double* y)
{
    int i, j, k;
    double w;

    // LU Decomposition
    for (i = 0; i < n; i++) {
        for (j = 0; j < i; j++) {
            w = A[i][j];
            for (k = 0; k < j; k++) {
                w -= A[i][k] * A[k][j];
            }
            A[i][j] = w / A[j][j];
        }
        for (j = i; j < n; j++) {
            w = A[i][j];
            for (k = 0; k < i; k++) {
                w -= A[i][k] * A[k][j];
            }
            A[i][j] = w;
        }
    }

    // Forward substitution: Ly = b
    for (i = 0; i < n; i++) {
        w = b[i];
        for (j = 0; j < i; j++) {
            w -= A[i][j] * y[j];
        }
        y[i] = w;
    }

    // Backward substitution: Ux = y
    for (i = n - 1; i >= 0; i--) {
        w = y[i];
        for (j = i + 1; j < n; j++) {
            w -= A[i][j] * x[j];
        }
        x[i] = w / A[i][i];
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_ludcmp_optimized(int n, double** A, double* b, double* x, double* y)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_ludcmp(n, A, b, x, y);
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

// Initialize arrays with some values
void init_array(int n, double** A, double* b, double* x, double* y)
{
    int i, j, r, s, t;
    double fn = (double)n;

    // Initialize vectors
    for (i = 0; i < n; i++) {
        x[i] = 0;
        y[i] = 0;
        b[i] = (i + 1) / fn / 2.0 + 4;
    }

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

    // Make the matrix well-conditioned using A = B^T * B
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
    
    // Compute A = B^T * B to ensure the matrix is well-conditioned
    for (t = 0; t < n; t++) {
        for (r = 0; r < n; r++) {
            for (s = 0; s < n; s++) {
                A[r][s] += B[r][t] * B[s][t];
            }
        }
    }
    
    free_2d_array(B, n);
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

// Copy matrix
void copy_matrix(int n, double** src, double** dst)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dst[i][j] = src[i][j];
        }
    }
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
double run_performance_test(void (*kernel)(int, double**, double*, double*, double*),
                          int n,
                          double** A_orig, double** A, double* b, double* x, double* y,
                          int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset arrays to initial state
        copy_matrix(n, A_orig, A);
        zero_vector(n, x);
        zero_vector(n, y);
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(n, A, b, x, y);
        
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
    double* b = allocate_1d_array(n);
    double* x1 = allocate_1d_array(n);
    double* y1 = allocate_1d_array(n);

    // Allocate arrays for optimized kernel
    double** A2 = allocate_2d_array(n, n);
    double* x2 = allocate_1d_array(n);
    double* y2 = allocate_1d_array(n);

    // Initialize arrays
    init_array(n, A1_orig, b, x1, y1);
    copy_matrix(n, A1_orig, A1);
    copy_matrix(n, A1_orig, A2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_ludcmp, n,
                                              A1_orig, A1, b, x1, y1, num_runs);
    
    double optimized_time = run_performance_test(kernel_ludcmp_optimized, n,
                                               A1_orig, A2, b, x2, y2, num_runs);

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
    free_2d_array(A1_orig, n);
    free_2d_array(A1, n);
    free_1d_array(b);
    free_1d_array(x1);
    free_1d_array(y1);
    free_2d_array(A2, n);
    free_1d_array(x2);
    free_1d_array(y2);

    return 0;
} 
