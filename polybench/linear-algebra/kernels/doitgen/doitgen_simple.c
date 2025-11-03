#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// DOITGEN: Multi-dimensional matrix transformation
// For each 2D slice A[r][q][:] of the 3D tensor A, multiply by matrix C4
// A: nr x nq x np 3D tensor
// C4: np x np matrix
// sum: np-dimensional temporary vector
void kernel_doitgen(int nr, int nq, int np,
                   double*** A,
                   double** C4,
                   double* sum)
{
    int r, q, p, s;

    for (r = 0; r < nr; r++) {
        for (q = 0; q < nq; q++) {
            // For each slice A[r][q][:], compute A[r][q][:] = A[r][q][:] * C4
            for (p = 0; p < np; p++) {
                sum[p] = 0.0;
                for (s = 0; s < np; s++) {
                    sum[p] += A[r][q][s] * C4[s][p];
                }
            }
            // Copy the result back to A[r][q][:]
            for (p = 0; p < np; p++) {
                A[r][q][p] = sum[p];
            }
        }
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_doitgen_optimized(int nr, int nq, int np,
                             double*** A,
                             double** C4,
                             double* sum)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_doitgen(nr, nq, np, A, C4, sum);
}

// Initialize arrays with some values
void init_array(int nr, int nq, int np,
               double*** A,
               double** C4)
{
    int i, j, k;

    // Initialize 3D tensor A
    for (i = 0; i < nr; i++) {
        for (j = 0; j < nq; j++) {
            for (k = 0; k < np; k++) {
                A[i][j][k] = (double)((i * j + k) % np) / np;
            }
        }
    }

    // Initialize matrix C4
    for (i = 0; i < np; i++) {
        for (j = 0; j < np; j++) {
            C4[i][j] = (double)(i * j % np) / np;
        }
    }
}

// Allocate a 3D array as flat continuous memory with pointer arrays for access
double*** allocate_3d_array(int dim1, int dim2, int dim3)
{
    // Allocate flat continuous memory for the data
    double* data = (double*)malloc(dim1 * dim2 * dim3 * sizeof(double));
    if (!data) return NULL;
    
    // Allocate array of pointers to 2D slices
    double*** array = (double***)malloc(dim1 * sizeof(double**));
    if (!array) {
        free(data);
        return NULL;
    }
    
    // Allocate array of pointers to rows
    double** rows = (double**)malloc(dim1 * dim2 * sizeof(double*));
    if (!rows) {
        free(data);
        free(array);
        return NULL;
    }
    
    // Set up pointers
    for (int i = 0; i < dim1; i++) {
        array[i] = rows + i * dim2;
        for (int j = 0; j < dim2; j++) {
            array[i][j] = data + (i * dim2 + j) * dim3;
        }
    }
    
    return array;
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

// Free a 3D array allocated with flat continuous memory
void free_3d_array(double*** array, int dim1, int dim2)
{
    if (array) {
        // Free the continuous data (accessible through array[0][0])
        free(array[0][0]);
        // Free the row pointers (accessible through array[0])
        free(array[0]);
        // Free the slice pointers
        free(array);
    }
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

// Verify if two 3D tensors are equal within a small tolerance
int verify_results(int nr, int nq, int np, double*** A1, double*** A2, double tolerance)
{
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nq; j++) {
            for (int k = 0; k < np; k++) {
                if (fabs(A1[i][j][k] - A2[i][j][k]) > tolerance) {
                    printf("Verification failed at position [%d][%d][%d]: %f != %f\n", 
                           i, j, k, A1[i][j][k], A2[i][j][k]);
                    return 0;
                }
            }
        }
    }
    return 1;
}

// Copy 3D tensor
void copy_3d_tensor(int nr, int nq, int np, double*** src, double*** dst)
{
    for (int i = 0; i < nr; i++) {
        for (int j = 0; j < nq; j++) {
            for (int k = 0; k < np; k++) {
                dst[i][j][k] = src[i][j][k];
            }
        }
    }
}

// Run performance test for a kernel
double run_performance_test(void (*kernel)(int, int, int, double***, double**, double*),
                          int nr, int nq, int np,
                          double*** A_orig, double*** A, double** C4, double* sum,
                          int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset arrays to initial state
        copy_3d_tensor(nr, nq, np, A_orig, A);
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(nr, nq, np, A, C4, sum);
        
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
    int nr = 150;  // Number of slices in first dimension
    int nq = 140;  // Number of slices in second dimension
    int np = 160;  // Size of each 2D slice
    int num_runs = 1;  // Number of runs for performance testing

    // Allocate arrays for original kernel
    double*** A1_orig = allocate_3d_array(nr, nq, np);
    double*** A1 = allocate_3d_array(nr, nq, np);
    double** C4_1 = allocate_2d_array(np, np);
    double* sum1 = allocate_1d_array(np);

    // Allocate arrays for optimized kernel
    double*** A2 = allocate_3d_array(nr, nq, np);
    double** C4_2 = allocate_2d_array(np, np);
    double* sum2 = allocate_1d_array(np);

    // Initialize arrays
    init_array(nr, nq, np, A1_orig, C4_1);
    copy_3d_tensor(nr, nq, np, A1_orig, A1);
    copy_3d_tensor(nr, nq, np, A1_orig, A2);
    
    // Copy C4 for second test
    for (int i = 0; i < np; i++) {
        for (int j = 0; j < np; j++) {
            C4_2[i][j] = C4_1[i][j];
        }
    }

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_doitgen, nr, nq, np,
                                              A1_orig, A1, C4_1, sum1, num_runs);
    
    double optimized_time = run_performance_test(kernel_doitgen_optimized, nr, nq, np,
                                               A1_orig, A2, C4_2, sum2, num_runs);

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Original kernel average time: %f seconds\n", original_time);
    printf("Optimized kernel average time: %f seconds\n", optimized_time);
    printf("Speedup: %f\n", original_time / optimized_time);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(nr, nq, np, A1, A2, 1e-4)) {
        printf("PASS: Results match within tolerance (1e-4)\n");
    } else {
        printf("FAIL: Results do not match\n");
    }

    // Free memory
    free_3d_array(A1_orig, nr, nq);
    free_3d_array(A1, nr, nq);
    free_2d_array(C4_1, np);
    free_1d_array(sum1);
    free_3d_array(A2, nr, nq);
    free_2d_array(C4_2, np);
    free_1d_array(sum2);

    return 0;
} 
