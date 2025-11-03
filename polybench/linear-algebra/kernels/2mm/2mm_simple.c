#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// 2mm: Two Matrix Multiplications
// D = alpha * A * B * C + beta * D
// A: ni x nk matrix
// B: nk x nj matrix  
// C: nj x nl matrix
// D: ni x nl matrix (output)
// tmp: ni x nj matrix (temporary)
void kernel_2mm(int ni, int nj, int nk, int nl,
               double alpha,
               double beta,
               double** tmp,
               double** A,
               double** B,
               double** C,
               double** D)
{
    int i, j, k;

    // First multiplication: tmp = alpha * A * B
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            tmp[i][j] = 0.0;
            for (k = 0; k < nk; k++) {
                tmp[i][j] += alpha * A[i][k] * B[k][j];
            }
        }
    }

    // Second multiplication: D = tmp * C + beta * D
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nl; j++) {
            D[i][j] *= beta;
            for (k = 0; k < nj; k++) {
                D[i][j] += tmp[i][k] * C[k][j];
            }
        }
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_2mm_optimized(int ni, int nj, int nk, int nl,
                         double alpha,
                         double beta,
                         double** tmp,
                         double** A,
                         double** B,
                         double** C,
                         double** D)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_2mm(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D);
}

// Initialize arrays with some values
void init_array(int ni, int nj, int nk, int nl,
               double* alpha,
               double* beta,
               double** A,
               double** B,
               double** C,
               double** D)
{
    int i, j;

    *alpha = 1.5;
    *beta = 1.2;

    // Initialize A (ni x nk)
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nk; j++) {
            A[i][j] = ((i*j+1) % ni) / (double)ni;
        }
    }

    // Initialize B (nk x nj)
    for (i = 0; i < nk; i++) {
        for (j = 0; j < nj; j++) {
            B[i][j] = (i*(j+1) % nj) / (double)nj;
        }
    }

    // Initialize C (nj x nl)
    for (i = 0; i < nj; i++) {
        for (j = 0; j < nl; j++) {
            C[i][j] = ((i*(j+3)+1) % nl) / (double)nl;
        }
    }

    // Initialize D (ni x nl)
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nl; j++) {
            D[i][j] = (i*(j+2) % nk) / (double)nk;
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
int verify_results(int ni, int nl, double** D1, double** D2, double tolerance)
{
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            if (fabs(D1[i][j] - D2[i][j]) > tolerance) {
                printf("Verification failed at position [%d][%d]: %f != %f\n", 
                       i, j, D1[i][j], D2[i][j]);
                return 0;
            }
        }
    }
    return 1;
}

// Run performance test for a kernel
double run_performance_test(void (*kernel)(int, int, int, int, double, double, double**, double**, double**, double**, double**),
                          int ni, int nj, int nk, int nl,
                          double alpha, double beta,
                          double** tmp, double** A, double** B, double** C, double** D,
                          int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset arrays to initial state
        init_array(ni, nj, nk, nl, &alpha, &beta, A, B, C, D);
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D);
        
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
    int ni = 800;   // Number of rows in A and D
    int nj = 900;   // Number of columns in B and tmp
    int nk = 1100;  // Number of columns in A and rows in B
    int nl = 1200;  // Number of columns in C and D
    int num_runs = 5;  // Number of runs for performance testing

    // Variables
    double alpha, beta;

    // Allocate matrices for original kernel
    double** tmp1 = allocate_2d_array(ni, nj);
    double** A1 = allocate_2d_array(ni, nk);
    double** B1 = allocate_2d_array(nk, nj);
    double** C1 = allocate_2d_array(nj, nl);
    double** D1 = allocate_2d_array(ni, nl);

    // Allocate matrices for optimized kernel
    double** tmp2 = allocate_2d_array(ni, nj);
    double** A2 = allocate_2d_array(ni, nk);
    double** B2 = allocate_2d_array(nk, nj);
    double** C2 = allocate_2d_array(nj, nl);
    double** D2 = allocate_2d_array(ni, nl);

    // Initialize arrays
    init_array(ni, nj, nk, nl, &alpha, &beta, A1, B1, C1, D1);
    init_array(ni, nj, nk, nl, &alpha, &beta, A2, B2, C2, D2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_2mm, ni, nj, nk, nl,
                                              alpha, beta, tmp1, A1, B1, C1, D1, num_runs);
    
    double optimized_time = run_performance_test(kernel_2mm_optimized, ni, nj, nk, nl,
                                               alpha, beta, tmp2, A2, B2, C2, D2, num_runs);

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Original kernel average time: %f seconds\n", original_time);
    printf("Optimized kernel average time: %f seconds\n", optimized_time);
    printf("Speedup: %f\n", original_time / optimized_time);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(ni, nl, D1, D2, 1e-4)) {
        printf("PASS: Results match within tolerance (1e-4)\n");
    } else {
        printf("FAIL: Results do not match\n");
    }

    // Free memory
    free_2d_array(tmp1, ni);
    free_2d_array(A1, ni);
    free_2d_array(B1, nk);
    free_2d_array(C1, nj);
    free_2d_array(D1, ni);
    free_2d_array(tmp2, ni);
    free_2d_array(A2, ni);
    free_2d_array(B2, nk);
    free_2d_array(C2, nj);
    free_2d_array(D2, ni);

    return 0;
} 
