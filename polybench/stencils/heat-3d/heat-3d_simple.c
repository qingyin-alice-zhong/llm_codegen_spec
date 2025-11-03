#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// 3D Heat equation stencil computation
// Performs iterative heat diffusion simulation on a 3D grid
void kernel_heat_3d(int tsteps, int n, double*** A, double*** B)
{
    int t, i, j, k;

    for (t = 1; t <= tsteps; t++) {
        // Update B based on A using 7-point stencil
        for (i = 1; i < n - 1; i++) {
            for (j = 1; j < n - 1; j++) {
                for (k = 1; k < n - 1; k++) {
                    B[i][j][k] = 0.125 * (A[i+1][j][k] - 2.0 * A[i][j][k] + A[i-1][j][k]) +
                                0.125 * (A[i][j+1][k] - 2.0 * A[i][j][k] + A[i][j-1][k]) +
                                0.125 * (A[i][j][k+1] - 2.0 * A[i][j][k] + A[i][j][k-1]) +
                                A[i][j][k];
                }
            }
        }
        
        // Update A based on B using 7-point stencil
        for (i = 1; i < n - 1; i++) {
            for (j = 1; j < n - 1; j++) {
                for (k = 1; k < n - 1; k++) {
                    A[i][j][k] = 0.125 * (B[i+1][j][k] - 2.0 * B[i][j][k] + B[i-1][j][k]) +
                                0.125 * (B[i][j+1][k] - 2.0 * B[i][j][k] + B[i][j-1][k]) +
                                0.125 * (B[i][j][k+1] - 2.0 * B[i][j][k] + B[i][j][k-1]) +
                                B[i][j][k];
                }
            }
        }
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_heat_3d_optimized(int tsteps, int n, double*** A, double*** B)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_heat_3d(tsteps, n, A, B);
}

// Allocate a 3D array as flat continuous memory with pointer arrays for access
double*** allocate_3d_array(int dim1, int dim2, int dim3)
{
    // Allocate flat continuous memory for the data
    double* data = (double*)malloc(dim1 * dim2 * dim3 * sizeof(double));
    if (!data) return NULL;
    
    // Allocate array of pointers to each 2D slice
    double*** array = (double***)malloc(dim1 * sizeof(double**));
    if (!array) {
        free(data);
        return NULL;
    }
    
    // Allocate arrays of pointers to each row
    double** row_ptrs = (double**)malloc(dim1 * dim2 * sizeof(double*));
    if (!row_ptrs) {
        free(data);
        free(array);
        return NULL;
    }
    
    // Set up pointers
    for (int i = 0; i < dim1; i++) {
        array[i] = row_ptrs + i * dim2;
        for (int j = 0; j < dim2; j++) {
            array[i][j] = data + (i * dim2 * dim3) + (j * dim3);
        }
    }
    
    return array;
}

// Free a 3D array allocated with flat continuous memory
void free_3d_array(double*** array, int dim1)
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

// Initialize arrays
void init_array(int n, double*** A, double*** B)
{
    int i, j, k;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            for (k = 0; k < n; k++) {
                A[i][j][k] = B[i][j][k] = (double)(i + j + (n - k)) * 10 / n;
            }
        }
    }
}

// Verify if two 3D arrays are equal within tolerance
int verify_results(int n, double*** A1, double*** A2, double tolerance)
{
    printf("Verifying Heat 3D results with tolerance %e...\n", tolerance);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                if (fabs(A1[i][j][k] - A2[i][j][k]) > tolerance) {
                    printf("Verification failed at position [%d][%d][%d]: %f != %f (diff: %e)\n", 
                           i, j, k, A1[i][j][k], A2[i][j][k], fabs(A1[i][j][k] - A2[i][j][k]));
                    return 0;
                }
            }
        }
    }
    return 1;
}

// Copy 3D array
void copy_3d_array(int n, double*** src, double*** dst)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                dst[i][j][k] = src[i][j][k];
            }
        }
    }
}

// Run performance test for a kernel
double run_performance_test(void (*kernel)(int, int, double***, double***),
                          int tsteps, int n, double*** A_orig, double*** B_orig,
                          double*** A, double*** B, int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset arrays to initial state
        copy_3d_array(n, A_orig, A);
        copy_3d_array(n, B_orig, B);
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(tsteps, n, A, B);
        
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
    // Problem size (using MEDIUM_DATASET dimensions)
    int n = 120;        // Grid size (n x n x n)
    int tsteps = 500;   // Number of time steps
    int num_runs = 1;   // Number of runs for performance testing

    // Allocate arrays
    double*** A_orig = allocate_3d_array(n, n, n);
    double*** B_orig = allocate_3d_array(n, n, n);
    double*** A1 = allocate_3d_array(n, n, n);
    double*** B1 = allocate_3d_array(n, n, n);
    double*** A2 = allocate_3d_array(n, n, n);
    double*** B2 = allocate_3d_array(n, n, n);

    // Initialize arrays
    init_array(n, A_orig, B_orig);
    copy_3d_array(n, A_orig, A1);
    copy_3d_array(n, B_orig, B1);
    copy_3d_array(n, A_orig, A2);
    copy_3d_array(n, B_orig, B2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_heat_3d, tsteps, n,
                                              A_orig, B_orig, A1, B1, num_runs);
    
    double optimized_time = run_performance_test(kernel_heat_3d_optimized, tsteps, n,
                                               A_orig, B_orig, A2, B2, num_runs);

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Original kernel average time: %f seconds\n", original_time);
    printf("Optimized kernel average time: %f seconds\n", optimized_time);
    printf("Speedup: %f\n", original_time / optimized_time);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(n, A1, A2, 1e-4)) {
        printf("PASS: Results match within tolerance\n");
    } else {
        printf("FAIL: Results do not match within tolerance\n");
    }

    // Free memory
    free_3d_array(A_orig, n);
    free_3d_array(B_orig, n);
    free_3d_array(A1, n);
    free_3d_array(B1, n);
    free_3d_array(A2, n);
    free_3d_array(B2, n);

    return 0;
} 
