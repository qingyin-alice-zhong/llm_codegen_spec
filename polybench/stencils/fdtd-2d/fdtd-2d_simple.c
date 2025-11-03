#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// FDTD 2D electromagnetic simulation
// Simulates electromagnetic wave propagation using finite difference time domain method
void kernel_fdtd_2d(int tmax, int nx, int ny, double** ex, double** ey, double** hz, double* _fict_)
{
    int t, i, j;

    for (t = 0; t < tmax; t++) {
        // Set boundary conditions
        for (j = 0; j < ny; j++)
            ey[0][j] = _fict_[t];
        
        // Update ey field
        for (i = 1; i < nx; i++)
            for (j = 0; j < ny; j++)
                ey[i][j] = ey[i][j] - 0.5 * (hz[i][j] - hz[i-1][j]);
        
        // Update ex field
        for (i = 0; i < nx; i++)
            for (j = 1; j < ny; j++)
                ex[i][j] = ex[i][j] - 0.5 * (hz[i][j] - hz[i][j-1]);
        
        // Update hz field
        for (i = 0; i < nx - 1; i++)
            for (j = 0; j < ny - 1; j++)
                hz[i][j] = hz[i][j] - 0.7 * (ex[i][j+1] - ex[i][j] + ey[i+1][j] - ey[i][j]);
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_fdtd_2d_optimized(int tmax, int nx, int ny, double** ex, double** ey, double** hz, double* _fict_)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_fdtd_2d(tmax, nx, ny, ex, ey, hz, _fict_);
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

// Allocate a 1D array
double* allocate_1d_array(int size)
{
    return (double*)malloc(size * sizeof(double));
}

// Free a 1D array
void free_1d_array(double* array)
{
    free(array);
}

// Initialize arrays
void init_array(int tmax, int nx, int ny, double** ex, double** ey, double** hz, double* _fict_)
{
    int i, j;

    for (i = 0; i < tmax; i++)
        _fict_[i] = (double) i;
    
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            ex[i][j] = ((double) i * (j + 1)) / nx;
            ey[i][j] = ((double) i * (j + 2)) / ny;
            hz[i][j] = ((double) i * (j + 3)) / nx;
        }
    }
}

// Verify if two 2D arrays are equal within tolerance
int verify_2d_array(int rows, int cols, double** A1, double** A2, double tolerance)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fabs(A1[i][j] - A2[i][j]) > tolerance) {
                printf("Array mismatch at [%d][%d]: %f != %f (diff: %e)\n", 
                       i, j, A1[i][j], A2[i][j], fabs(A1[i][j] - A2[i][j]));
                return 0;
            }
        }
    }
    return 1;
}

// Verify if two 1D arrays are equal within tolerance
int verify_1d_array(int size, double* A1, double* A2, double tolerance)
{
    for (int i = 0; i < size; i++) {
        if (fabs(A1[i] - A2[i]) > tolerance) {
            printf("Array mismatch at [%d]: %f != %f (diff: %e)\n", 
                   i, A1[i], A2[i], fabs(A1[i] - A2[i]));
            return 0;
        }
    }
    return 1;
}

// Verify FDTD 2D results
int verify_results(int nx, int ny, double** ex1, double** ey1, double** hz1,
                  double** ex2, double** ey2, double** hz2, double tolerance)
{
    printf("Verifying FDTD 2D results with tolerance %e...\n", tolerance);
    
    int ex_match = verify_2d_array(nx, ny, ex1, ex2, tolerance);
    int ey_match = verify_2d_array(nx, ny, ey1, ey2, tolerance);
    int hz_match = verify_2d_array(nx, ny, hz1, hz2, tolerance);
    
    return ex_match && ey_match && hz_match;
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

// Copy 1D array
void copy_1d_array(int size, double* src, double* dst)
{
    for (int i = 0; i < size; i++) {
        dst[i] = src[i];
    }
}

// Run performance test for a kernel
double run_performance_test(void (*kernel)(int, int, int, double**, double**, double**, double*),
                          int tmax, int nx, int ny, 
                          double** ex_orig, double** ey_orig, double** hz_orig, double* fict_orig,
                          double** ex, double** ey, double** hz, double* fict, int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset arrays to initial state
        copy_2d_array(nx, ny, ex_orig, ex);
        copy_2d_array(nx, ny, ey_orig, ey);
        copy_2d_array(nx, ny, hz_orig, hz);
        copy_1d_array(tmax, fict_orig, fict);
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(tmax, nx, ny, ex, ey, hz, fict);
        
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
    int tmax = 500;    // Number of time steps
    int nx = 1000;     // Grid size in x direction
    int ny = 1200;     // Grid size in y direction
    int num_runs = 1;  // Number of runs for performance testing

    // Allocate arrays
    double** ex_orig = allocate_2d_array(nx, ny);
    double** ey_orig = allocate_2d_array(nx, ny);
    double** hz_orig = allocate_2d_array(nx, ny);
    double* fict_orig = allocate_1d_array(tmax);
    
    double** ex1 = allocate_2d_array(nx, ny);
    double** ey1 = allocate_2d_array(nx, ny);
    double** hz1 = allocate_2d_array(nx, ny);
    double* fict1 = allocate_1d_array(tmax);
    
    double** ex2 = allocate_2d_array(nx, ny);
    double** ey2 = allocate_2d_array(nx, ny);
    double** hz2 = allocate_2d_array(nx, ny);
    double* fict2 = allocate_1d_array(tmax);

    // Initialize arrays
    init_array(tmax, nx, ny, ex_orig, ey_orig, hz_orig, fict_orig);
    copy_2d_array(nx, ny, ex_orig, ex1);
    copy_2d_array(nx, ny, ey_orig, ey1);
    copy_2d_array(nx, ny, hz_orig, hz1);
    copy_1d_array(tmax, fict_orig, fict1);
    copy_2d_array(nx, ny, ex_orig, ex2);
    copy_2d_array(nx, ny, ey_orig, ey2);
    copy_2d_array(nx, ny, hz_orig, hz2);
    copy_1d_array(tmax, fict_orig, fict2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_fdtd_2d, tmax, nx, ny,
                                              ex_orig, ey_orig, hz_orig, fict_orig,
                                              ex1, ey1, hz1, fict1, num_runs);
    
    double optimized_time = run_performance_test(kernel_fdtd_2d_optimized, tmax, nx, ny,
                                               ex_orig, ey_orig, hz_orig, fict_orig,
                                               ex2, ey2, hz2, fict2, num_runs);

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Original kernel average time: %f seconds\n", original_time);
    printf("Optimized kernel average time: %f seconds\n", optimized_time);
    printf("Speedup: %f\n", original_time / optimized_time);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(nx, ny, ex1, ey1, hz1, ex2, ey2, hz2, 1e-4)) {
        printf("PASS: Results match within tolerance\n");
    } else {
        printf("FAIL: Results do not match within tolerance\n");
    }

    // Free memory
    free_2d_array(ex_orig, nx);
    free_2d_array(ey_orig, nx);
    free_2d_array(hz_orig, nx);
    free_1d_array(fict_orig);
    free_2d_array(ex1, nx);
    free_2d_array(ey1, nx);
    free_2d_array(hz1, nx);
    free_1d_array(fict1);
    free_2d_array(ex2, nx);
    free_2d_array(ey2, nx);
    free_2d_array(hz2, nx);
    free_1d_array(fict2);

    return 0;
} 
