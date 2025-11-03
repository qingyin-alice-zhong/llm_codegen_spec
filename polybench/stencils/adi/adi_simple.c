#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// ADI (Alternating Direction Implicit) solver
// Solves 2D heat equation using alternating direction implicit method
void kernel_adi(int tsteps, int n, double** u, double** v, double** p, double** q)
{
    int t, i, j;
    double DX, DY, DT;
    double B1, B2;
    double mul1, mul2;
    double a, b, c, d, e, f;

    DX = 1.0 / (double)n;
    DY = 1.0 / (double)n;
    DT = 1.0 / (double)tsteps;
    B1 = 2.0;
    B2 = 1.0;
    mul1 = B1 * DT / (DX * DX);
    mul2 = B2 * DT / (DY * DY);

    a = -mul1 / 2.0;
    b = 1.0 + mul1;
    c = a;
    d = -mul2 / 2.0;
    e = 1.0 + mul2;
    f = d;

    for (t = 1; t <= tsteps; t++) {
        // Column Sweep
        for (i = 1; i < n - 1; i++) {
            v[0][i] = 1.0;
            p[i][0] = 0.0;
            q[i][0] = v[0][i];
            for (j = 1; j < n - 1; j++) {
                p[i][j] = -c / (a * p[i][j-1] + b);
                q[i][j] = (-d * u[j][i-1] + (1.0 + 2.0 * d) * u[j][i] - f * u[j][i+1] - a * q[i][j-1]) / (a * p[i][j-1] + b);
            }

            v[n-1][i] = 1.0;
            for (j = n - 2; j >= 1; j--) {
                v[j][i] = p[i][j] * v[j+1][i] + q[i][j];
            }
        }
        
        // Row Sweep
        for (i = 1; i < n - 1; i++) {
            u[i][0] = 1.0;
            p[i][0] = 0.0;
            q[i][0] = u[i][0];
            for (j = 1; j < n - 1; j++) {
                p[i][j] = -f / (d * p[i][j-1] + e);
                q[i][j] = (-a * v[i-1][j] + (1.0 + 2.0 * a) * v[i][j] - c * v[i+1][j] - d * q[i][j-1]) / (d * p[i][j-1] + e);
            }
            u[i][n-1] = 1.0;
            for (j = n - 2; j >= 1; j--) {
                u[i][j] = p[i][j] * u[i][j+1] + q[i][j];
            }
        }
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_adi_optimized(int tsteps, int n, double** u, double** v, double** p, double** q)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_adi(tsteps, n, u, v, p, q);
}

// Allocate a 2D array as flat continuous memory with pointer array for access
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
    
    // Set up row pointers to point into the flat memory
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
        // Free the row pointers
        free(array);
    }
}

// Initialize arrays
void init_array(int n, double** u)
{
    int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            u[i][j] = (double)(i + n - j) / n;
        }
    }
}

// Verify if two 2D arrays are equal within tolerance
int verify_results(int n, double** u1, double** u2, double tolerance)
{
    printf("Verifying ADI results with tolerance %e...\n", tolerance);
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(u1[i][j] - u2[i][j]) > tolerance) {
                printf("Verification failed at position [%d][%d]: %f != %f (diff: %e)\n", 
                       i, j, u1[i][j], u2[i][j], fabs(u1[i][j] - u2[i][j]));
                return 0;
            }
        }
    }
    return 1;
}

// Copy 2D array
void copy_2d_array(int rows, int cols, double** src, double** dst)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[i][j] = src[i][j];
        }
    }
}

// Run performance test for a kernel
double run_performance_test(void (*kernel)(int, int, double**, double**, double**, double**),
                          int tsteps, int n, double** u_orig,
                          double** u, double** v, double** p, double** q, int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset arrays to initial state
        copy_2d_array(n, n, u_orig, u);
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(tsteps, n, u, v, p, q);
        
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
    int n = 1000;       // Grid size (n x n)
    int tsteps = 500;   // Number of time steps
    int num_runs = 1;   // Number of runs for performance testing

    // Allocate arrays
    double** u_orig = allocate_2d_array(n, n);
    double** u1 = allocate_2d_array(n, n);
    double** v1 = allocate_2d_array(n, n);
    double** p1 = allocate_2d_array(n, n);
    double** q1 = allocate_2d_array(n, n);
    double** u2 = allocate_2d_array(n, n);
    double** v2 = allocate_2d_array(n, n);
    double** p2 = allocate_2d_array(n, n);
    double** q2 = allocate_2d_array(n, n);

    // Initialize arrays
    init_array(n, u_orig);
    copy_2d_array(n, n, u_orig, u1);
    copy_2d_array(n, n, u_orig, u2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_adi, tsteps, n, u_orig,
                                              u1, v1, p1, q1, num_runs);
    
    double optimized_time = run_performance_test(kernel_adi_optimized, tsteps, n, u_orig,
                                               u2, v2, p2, q2, num_runs);

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Original kernel average time: %f seconds\n", original_time);
    printf("Optimized kernel average time: %f seconds\n", optimized_time);
    printf("Speedup: %f\n", original_time / optimized_time);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(n, u1, u2, 1e-4)) {
        printf("PASS: Results match within tolerance\n");
    } else {
        printf("FAIL: Results do not match within tolerance\n");
    }

    // Free memory
    free_2d_array(u_orig, n);
    free_2d_array(u1, n);
    free_2d_array(v1, n);
    free_2d_array(p1, n);
    free_2d_array(q1, n);
    free_2d_array(u2, n);
    free_2d_array(v2, n);
    free_2d_array(p2, n);
    free_2d_array(q2, n);

    return 0;
} 
