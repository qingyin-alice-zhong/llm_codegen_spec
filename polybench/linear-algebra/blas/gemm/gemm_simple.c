#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>


// Simple matrix multiplication: C = alpha * A * B + beta * C
// A: ni x nk matrix
// B: nk x nj matrix
// C: ni x nj matrix
void kernel_gemm(int ni, int nj, int nk,
                double alpha,
                double beta,
                double** C,
                double** A,
                double** B)
{
    int i, j, k;

    // First multiply C by beta
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            C[i][j] *= beta;
        }
    }

    // Then add alpha * A * B
    for (i = 0; i < ni; i++) {
        for (k = 0; k < nk; k++) {
            for (j = 0; j < nj; j++) {
                C[i][j] += alpha * A[i][k] * B[k][j];
            }
        }
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_gemm_optimized(int ni, int nj, int nk,
                          double alpha,
                          double beta,
                          double** C,
                          double** A,
                          double** B)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_gemm(ni, nj, nk, alpha, beta, C, A, B);
}

// Initialize matrices with some values
void init_array(int ni, int nj, int nk,
               double* alpha,
               double* beta,
               double** C,
               double** A,
               double** B)
{
    int i, j;

    *alpha = 1.5;
    *beta = 1.2;

    // Initialize C
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            C[i][j] = ((i*j+1) % ni) / (double)ni;
        }
    }

    // Initialize A
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nk; j++) {
            A[i][j] = (i*(j+1) % nk) / (double)nk;
        }
    }

    // Initialize B
    for (i = 0; i < nk; i++) {
        for (j = 0; j < nj; j++) {
            B[i][j] = (i*(j+2) % nj) / (double)nj;
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
int verify_results(int ni, int nj, double** C1, double** C2, double tolerance)
{
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nj; j++) {
            if (fabs(C1[i][j] - C2[i][j]) > tolerance) {
                printf("Verification failed at position [%d][%d]: %f != %f\n", 
                       i, j, C1[i][j], C2[i][j]);
                return 0;
            }
        }
    }
    return 1;
}

// Run performance test for a kernel
double run_performance_test(void (*kernel)(int, int, int, double, double, double**, double**, double**),
                          int ni, int nj, int nk,
                          double alpha, double beta,
                          double** C, double** A, double** B,
                          int num_runs)
{
    double total_time = 0.0;
    struct timespec start, end;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset C to initial state
        init_array(ni, nj, nk, &alpha, &beta, C, A, B);
        
        // Start timing using monotonic clock
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(ni, nj, nk, alpha, beta, C, A, B);
        
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
    int ni = 1000;  // Size of first dimension
    int nj = 1100;  // Size of second dimension
    int nk = 1200;  // Size of third dimension
    int num_runs = 5;  // Number of runs for performance testing

    // Variables
    double alpha, beta;

    // Allocate matrices for original kernel
    double** C1 = allocate_2d_array(ni, nj);
    double** A1 = allocate_2d_array(ni, nk);
    double** B1 = allocate_2d_array(nk, nj);

    // Allocate matrices for optimized kernel
    double** C2 = allocate_2d_array(ni, nj);
    double** A2 = allocate_2d_array(ni, nk);
    double** B2 = allocate_2d_array(nk, nj);

    // Initialize arrays
    init_array(ni, nj, nk, &alpha, &beta, C1, A1, B1);
    init_array(ni, nj, nk, &alpha, &beta, C2, A2, B2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_gemm, ni, nj, nk, 
                                              alpha, beta, C1, A1, B1, num_runs);
    
    double optimized_time = run_performance_test(kernel_gemm_optimized, ni, nj, nk, 
                                               alpha, beta, C2, A2, B2, num_runs);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(ni, nj, C1, C2, 1e-4)) {
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
    free_2d_array(C1, ni);
    free_2d_array(A1, ni);
    free_2d_array(B1, nk);
    free_2d_array(C2, ni);
    free_2d_array(A2, ni);
    free_2d_array(B2, nk);

    return 0;
} 
