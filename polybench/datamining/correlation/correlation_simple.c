#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// Correlation Matrix Computation
// data: n x m matrix (input/modified during computation)
// corr: m x m correlation matrix (output)
// mean: m-element mean vector (output)
// stddev: m-element standard deviation vector (output)
// The algorithm computes the correlation matrix from the input data
void kernel_correlation(int m, int n, double float_n,
                       double** data, double** corr, double* mean, double* stddev)
{
    int i, j, k;
    double eps = 0.1;

    // Compute mean for each feature (column)
    for (j = 0; j < m; j++) {
        mean[j] = 0.0;
        for (i = 0; i < n; i++) {
            mean[j] += data[i][j];
        }
        mean[j] /= float_n;
    }

    // Compute standard deviation for each feature
    for (j = 0; j < m; j++) {
        stddev[j] = 0.0;
        for (i = 0; i < n; i++) {
            stddev[j] += (data[i][j] - mean[j]) * (data[i][j] - mean[j]);
        }
        stddev[j] /= float_n;
        stddev[j] = sqrt(stddev[j]);
        
        // Handle near-zero std. dev. values to avoid division by zero
        stddev[j] = stddev[j] <= eps ? 1.0 : stddev[j];
    }

    // Center and normalize the data
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            data[i][j] -= mean[j];
            data[i][j] /= sqrt(float_n) * stddev[j];
        }
    }

    // Calculate the m x m correlation matrix
    for (i = 0; i < m - 1; i++) {
        corr[i][i] = 1.0;  // Diagonal elements are 1
        for (j = i + 1; j < m; j++) {
            corr[i][j] = 0.0;
            for (k = 0; k < n; k++) {
                corr[i][j] += data[k][i] * data[k][j];
            }
            corr[j][i] = corr[i][j];  // Symmetric matrix
        }
    }
    corr[m-1][m-1] = 1.0;  // Last diagonal element
}

// Placeholder for optimized kernel - to be replaced later
void kernel_correlation_optimized(int m, int n, double float_n,
                                 double** data, double** corr, double* mean, double* stddev)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_correlation(m, n, float_n, data, corr, mean, stddev);
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
void init_array(int m, int n, double* float_n, double** data)
{
    int i, j;

    *float_n = (double)n;

    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            data[i][j] = (double)(i * j) / m + i;
        }
    }
}

// Verify if two matrices are equal within a small tolerance
int verify_matrices(int rows, int cols, double** A1, double** A2, double tolerance)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fabs(A1[i][j] - A2[i][j]) > tolerance) {
                printf("Verification failed at position [%d][%d]: %f != %f\n", 
                       i, j, A1[i][j], A2[i][j]);
                return 0;
            }
        }
    }
    return 1;
}

// Verify correlation results
int verify_results(int m, double** corr1, double** corr2, double tolerance)
{
    printf("Verifying correlation matrices...\n");
    return verify_matrices(m, m, corr1, corr2, tolerance);
}

// Copy matrix
void copy_matrix(int rows, int cols, double** src, double** dst)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[i][j] = src[i][j];
        }
    }
}

// Copy vector
void copy_vector(int size, double* src, double* dst)
{
    for (int i = 0; i < size; i++) {
        dst[i] = src[i];
    }
}

// Run performance test for a kernel
double run_performance_test(void (*kernel)(int, int, double, double**, double**, double*, double*),
                          int m, int n, double float_n,
                          double** data_orig, double** data, double** corr, 
                          double* mean, double* stddev, int num_runs)
{
    double total_time = 0.0;
    struct timespec start, end;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset data to initial state
        copy_matrix(n, m, data_orig, data);
        
        // Start timing using monotonic clock
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(m, n, float_n, data, corr, mean, stddev);
        
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
    int m = 1200;  // Number of features
    int n = 1400;  // Number of samples
    int num_runs = 1;  // Number of runs for performance testing

    // Allocate arrays for original kernel
    double** data1_orig = allocate_2d_array(n, m);
    double** data1 = allocate_2d_array(n, m);
    double** corr1 = allocate_2d_array(m, m);
    double* mean1 = allocate_1d_array(m);
    double* stddev1 = allocate_1d_array(m);

    // Allocate arrays for optimized kernel
    double** data2 = allocate_2d_array(n, m);
    double** corr2 = allocate_2d_array(m, m);
    double* mean2 = allocate_1d_array(m);
    double* stddev2 = allocate_1d_array(m);

    double float_n;

    // Initialize arrays
    init_array(m, n, &float_n, data1_orig);
    copy_matrix(n, m, data1_orig, data1);
    copy_matrix(n, m, data1_orig, data2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_correlation, m, n, float_n,
                                              data1_orig, data1, corr1, mean1, stddev1, num_runs);
    
    double optimized_time = run_performance_test(kernel_correlation_optimized, m, n, float_n,
                                               data1_orig, data2, corr2, mean2, stddev2, num_runs);

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Original kernel average time: %f seconds\n", original_time);
    printf("Optimized kernel average time: %f seconds\n", optimized_time);
    printf("Speedup: %f\n", original_time / optimized_time);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(m, corr1, corr2, 1e-4)) {
        printf("PASS: Results match within tolerance (1e-4)\n");
    } else {
        printf("FAIL: Results do not match\n");
    }

    // Free memory
    free_2d_array(data1_orig, n);
    free_2d_array(data1, n);
    free_2d_array(corr1, m);
    free_1d_array(mean1);
    free_1d_array(stddev1);
    free_2d_array(data2, n);
    free_2d_array(corr2, m);
    free_1d_array(mean2);
    free_1d_array(stddev2);

    return 0;
} 
