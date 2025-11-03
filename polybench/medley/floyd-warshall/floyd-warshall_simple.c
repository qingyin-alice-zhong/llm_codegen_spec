#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// Floyd-Warshall All-Pairs Shortest Path Algorithm
// path: n x n adjacency matrix (input/output)
// The algorithm finds shortest paths between all pairs of vertices
void kernel_floyd_warshall(int n, int** path)
{
    int i, j, k;

    for (k = 0; k < n; k++) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                path[i][j] = path[i][j] < path[i][k] + path[k][j] ?
                    path[i][j] : path[i][k] + path[k][j];
            }
        }
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_floyd_warshall_optimized(int n, int** path)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_floyd_warshall(n, path);
}

// Allocate a 2D integer array as flat continuous memory with pointer array for row access
int** allocate_2d_int_array(int rows, int cols)
{
    // Allocate flat continuous memory for the data
    int* data = (int*)malloc(rows * cols * sizeof(int));
    if (!data) return NULL;
    
    // Allocate array of pointers to each row
    int** array = (int**)malloc(rows * sizeof(int*));
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

// Free a 2D integer array allocated with flat continuous memory
void free_2d_int_array(int** array, int rows)
{
    if (array) {
        // Free the continuous data (accessible through array[0])
        free(array[0]);
        // Free the pointer array
        free(array);
    }
}

// Initialize arrays with some values to create a graph
void init_array(int n, int** path)
{
    int i, j;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            path[i][j] = i * j % 7 + 1;
            if ((i + j) % 13 == 0 || (i + j) % 7 == 0 || (i + j) % 11 == 0) {
                path[i][j] = 999;  // Large value representing no direct edge
            }
        }
    }
}

// Verify if two integer matrices are equal
int verify_matrices_int(int rows, int cols, int** A1, int** A2)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (A1[i][j] != A2[i][j]) {
                printf("Verification failed at position [%d][%d]: %d != %d\n", 
                       i, j, A1[i][j], A2[i][j]);
                return 0;
            }
        }
    }
    return 1;
}

// Verify floyd-warshall results
int verify_results(int n, int** path1, int** path2)
{
    printf("Verifying shortest path matrices...\n");
    return verify_matrices_int(n, n, path1, path2);
}

// Copy integer matrix
void copy_matrix_int(int rows, int cols, int** src, int** dst)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[i][j] = src[i][j];
        }
    }
}

// Run performance test for a kernel
double run_performance_test(void (*kernel)(int, int**),
                          int n, int** path_orig, int** path, int num_runs)
{
    double total_time = 0.0;
    struct timespec start, end;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset path to initial state
        copy_matrix_int(n, n, path_orig, path);
        
        // Start timing using monotonic clock
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(n, path);
        
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
    int n = 2800;  // Number of vertices in the graph
    int num_runs = 1;  // Number of runs for performance testing

    // Allocate arrays for original kernel
    int** path1_orig = allocate_2d_int_array(n, n);
    int** path1 = allocate_2d_int_array(n, n);

    // Allocate arrays for optimized kernel
    int** path2 = allocate_2d_int_array(n, n);

    // Initialize arrays
    init_array(n, path1_orig);
    copy_matrix_int(n, n, path1_orig, path1);
    copy_matrix_int(n, n, path1_orig, path2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_floyd_warshall, n,
                                              path1_orig, path1, num_runs);
    
    double optimized_time = run_performance_test(kernel_floyd_warshall_optimized, n,
                                               path1_orig, path2, num_runs);

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Original kernel average time: %f seconds\n", original_time);
    printf("Optimized kernel average time: %f seconds\n", optimized_time);
    printf("Speedup: %f\n", original_time / optimized_time);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(n, path1, path2)) {
        printf("PASS: Results match exactly\n");
    } else {
        printf("FAIL: Results do not match\n");
    }

    // Free memory
    free_2d_int_array(path1_orig, n);
    free_2d_int_array(path1, n);
    free_2d_int_array(path2, n);

    return 0;
} 
