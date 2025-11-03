#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// 3mm: Three Matrix Multiplications
// E = A * B, F = C * D, G = E * F
// A: ni x nk matrix
// B: nk x nj matrix
// C: nj x nm matrix
// D: nm x nl matrix
// E: ni x nj matrix (temporary)
// F: nj x nl matrix (temporary)
// G: ni x nl matrix (output)
void kernel_3mm(int ni, int nj, int nk, int nl, int nm,
               double** E,
               double** A,
               double** B,
               double** F,
               double** C,
               double** D,
               double** G)
{
    int i, j, k;

    // First multiplication: E = A * B
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nj; j++) {
            E[i][j] = 0.0;
            for (k = 0; k < nk; k++) {
                E[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // Second multiplication: F = C * D
    for (i = 0; i < nj; i++) {
        for (j = 0; j < nl; j++) {
            F[i][j] = 0.0;
            for (k = 0; k < nm; k++) {
                F[i][j] += C[i][k] * D[k][j];
            }
        }
    }

    // Third multiplication: G = E * F
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nl; j++) {
            G[i][j] = 0.0;
            for (k = 0; k < nj; k++) {
                G[i][j] += E[i][k] * F[k][j];
            }
        }
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_3mm_optimized(int ni, int nj, int nk, int nl, int nm,
                         double** E,
                         double** A,
                         double** B,
                         double** F,
                         double** C,
                         double** D,
                         double** G)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_3mm(ni, nj, nk, nl, nm, E, A, B, F, C, D, G);
}

// Initialize arrays with some values
void init_array(int ni, int nj, int nk, int nl, int nm,
               double** A,
               double** B,
               double** C,
               double** D)
{
    int i, j;

    // Initialize A (ni x nk)
    for (i = 0; i < ni; i++) {
        for (j = 0; j < nk; j++) {
            A[i][j] = ((i*j+1) % ni) / (5.0*ni);
        }
    }

    // Initialize B (nk x nj)
    for (i = 0; i < nk; i++) {
        for (j = 0; j < nj; j++) {
            B[i][j] = ((i*(j+1)+2) % nj) / (5.0*nj);
        }
    }

    // Initialize C (nj x nm)
    for (i = 0; i < nj; i++) {
        for (j = 0; j < nm; j++) {
            C[i][j] = (i*(j+3) % nl) / (5.0*nl);
        }
    }

    // Initialize D (nm x nl)
    for (i = 0; i < nm; i++) {
        for (j = 0; j < nl; j++) {
            D[i][j] = ((i*(j+2)+2) % nk) / (5.0*nk);
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
int verify_results(int ni, int nl, double** G1, double** G2, double tolerance)
{
    for (int i = 0; i < ni; i++) {
        for (int j = 0; j < nl; j++) {
            if (fabs(G1[i][j] - G2[i][j]) > tolerance) {
                printf("Verification failed at position [%d][%d]: %f != %f\n", 
                       i, j, G1[i][j], G2[i][j]);
                return 0;
            }
        }
    }
    return 1;
}

// Run performance test for a kernel
double run_performance_test(void (*kernel)(int, int, int, int, int, double**, double**, double**, double**, double**, double**, double**),
                          int ni, int nj, int nk, int nl, int nm,
                          double** E, double** A, double** B, double** F, double** C, double** D, double** G,
                          int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset arrays to initial state
        init_array(ni, nj, nk, nl, nm, A, B, C, D);
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(ni, nj, nk, nl, nm, E, A, B, F, C, D, G);
        
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
    int ni = 800;   // Number of rows in A, E, and G
    int nj = 900;   // Number of columns in B, E, rows in C, F
    int nk = 1000;  // Number of columns in A, rows in B
    int nl = 1100;  // Number of columns in D, F, and G
    int nm = 1200;  // Number of columns in C, rows in D
    int num_runs = 5;  // Number of runs for performance testing

    // Allocate matrices for original kernel
    double** E1 = allocate_2d_array(ni, nj);
    double** A1 = allocate_2d_array(ni, nk);
    double** B1 = allocate_2d_array(nk, nj);
    double** F1 = allocate_2d_array(nj, nl);
    double** C1 = allocate_2d_array(nj, nm);
    double** D1 = allocate_2d_array(nm, nl);
    double** G1 = allocate_2d_array(ni, nl);

    // Allocate matrices for optimized kernel
    double** E2 = allocate_2d_array(ni, nj);
    double** A2 = allocate_2d_array(ni, nk);
    double** B2 = allocate_2d_array(nk, nj);
    double** F2 = allocate_2d_array(nj, nl);
    double** C2 = allocate_2d_array(nj, nm);
    double** D2 = allocate_2d_array(nm, nl);
    double** G2 = allocate_2d_array(ni, nl);

    // Initialize arrays
    init_array(ni, nj, nk, nl, nm, A1, B1, C1, D1);
    init_array(ni, nj, nk, nl, nm, A2, B2, C2, D2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_3mm, ni, nj, nk, nl, nm,
                                              E1, A1, B1, F1, C1, D1, G1, num_runs);
    
    double optimized_time = run_performance_test(kernel_3mm_optimized, ni, nj, nk, nl, nm,
                                               E2, A2, B2, F2, C2, D2, G2, num_runs);

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Original kernel average time: %f seconds\n", original_time);
    printf("Optimized kernel average time: %f seconds\n", optimized_time);
    printf("Speedup: %f\n", original_time / optimized_time);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(ni, nl, G1, G2, 1e-4)) {
        printf("PASS: Results match within tolerance (1e-4)\n");
    } else {
        printf("FAIL: Results do not match\n");
    }

    // Free memory
    free_2d_array(E1, ni);
    free_2d_array(A1, ni);
    free_2d_array(B1, nk);
    free_2d_array(F1, nj);
    free_2d_array(C1, nj);
    free_2d_array(D1, nm);
    free_2d_array(G1, ni);
    free_2d_array(E2, ni);
    free_2d_array(A2, ni);
    free_2d_array(B2, nk);
    free_2d_array(F2, nj);
    free_2d_array(C2, nj);
    free_2d_array(D2, nm);
    free_2d_array(G2, ni);

    return 0;
} 
