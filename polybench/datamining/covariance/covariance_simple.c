#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// Covariance Matrix Computation
// data: n x m matrix (input/modified during computation)
// cov: m x m covariance matrix (output)
// mean: m-element mean vector (output)
// The algorithm computes the covariance matrix from the input data
void kernel_covariance(int m, int n, double float_n,
                      double** data, double** cov, double* mean)
{
    int i, j, k;

    // Compute mean for each feature (column)
    for (j = 0; j < m; j++) {
        mean[j] = 0.0;
        for (i = 0; i < n; i++) {
            mean[j] += data[i][j];
        }
        mean[j] /= float_n;
    }

    // Center the data by subtracting mean
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            data[i][j] -= mean[j];
        }
    }

    // Compute covariance matrix
    for (i = 0; i < m; i++) {
        for (j = i; j < m; j++) {
            cov[i][j] = 0.0;
            for (k = 0; k < n; k++) {
                cov[i][j] += data[k][i] * data[k][j];
            }
            cov[i][j] /= (float_n - 1.0);
            cov[j][i] = cov[i][j];  // Symmetric matrix
        }
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_covariance_optimized(int m, int n, double float_n,
                                double** data, double** cov, double* mean)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_covariance(m, n, float_n, data, cov, mean);
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
            data[i][j] = ((double)i * j) / m;
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

// Verify covariance results
int verify_results(int m, double** cov1, double** cov2, double tolerance)
{
    printf("Verifying covariance matrices...\n");
    return verify_matrices(m, m, cov1, cov2, tolerance);
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
double run_performance_test(void (*kernel)(int, int, double, double**, double**, double*),
                          int m, int n, double float_n,
                          double** data_orig, double** data, double** cov, double* mean,
                          int num_runs)
{
    double total_time = 0.0;
    struct timespec start, end;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset data to initial state
        copy_matrix(n, m, data_orig, data);
        
        // Start timing using monotonic clock
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(m, n, float_n, data, cov, mean);
        
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
    double** cov1 = allocate_2d_array(m, m);
    double* mean1 = allocate_1d_array(m);

    // Allocate arrays for optimized kernel
    double** data2 = allocate_2d_array(n, m);
    double** cov2 = allocate_2d_array(m, m);
    double* mean2 = allocate_1d_array(m);

    double float_n;

    // Initialize arrays
    init_array(m, n, &float_n, data1_orig);
    copy_matrix(n, m, data1_orig, data1);
    copy_matrix(n, m, data1_orig, data2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_covariance, m, n, float_n,
                                              data1_orig, data1, cov1, mean1, num_runs);
    
    double optimized_time = run_performance_test(kernel_covariance_optimized, m, n, float_n,
                                               data1_orig, data2, cov2, mean2, num_runs);

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Original kernel average time: %f seconds\n", original_time);
    printf("Optimized kernel average time: %f seconds\n", optimized_time);
    printf("Speedup: %f\n", original_time / optimized_time);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(m, cov1, cov2, 1e-4)) {
        printf("PASS: Results match within tolerance (1e-4)\n");
    } else {
        printf("FAIL: Results do not match\n");
    }

    // Free memory
    free_2d_array(data1_orig, n);
    free_2d_array(data1, n);
    free_2d_array(cov1, m);
    free_1d_array(mean1);
    free_2d_array(data2, n);
    free_2d_array(cov2, m);
    free_1d_array(mean2);

    return 0;
} 
