#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// RNA bases represented as chars, range is [0,3]
typedef char base;

// Helper macros for the Nussinov algorithm
#define match(b1, b2) (((b1)+(b2)) == 3 ? 1 : 0)
#define max_score(s1, s2) ((s1 >= s2) ? s1 : s2)

// Nussinov RNA Secondary Structure Prediction
// seq: n-element RNA sequence (input)
// table: n x n DP table (output)
// The algorithm uses dynamic programming to find optimal RNA secondary structure
void kernel_nussinov(int n, base* seq, int** table)
{
    int i, j, k;

    for (i = n - 1; i >= 0; i--) {
        for (j = i + 1; j < n; j++) {
            
            if (j - 1 >= 0)
                table[i][j] = max_score(table[i][j], table[i][j-1]);
            if (i + 1 < n)
                table[i][j] = max_score(table[i][j], table[i+1][j]);

            if (j - 1 >= 0 && i + 1 < n) {
                // Don't allow adjacent elements to bond
                if (i < j - 1)
                    table[i][j] = max_score(table[i][j], table[i+1][j-1] + match(seq[i], seq[j]));
                else
                    table[i][j] = max_score(table[i][j], table[i+1][j-1]);
            }

            for (k = i + 1; k < j; k++) {
                table[i][j] = max_score(table[i][j], table[i][k] + table[k+1][j]);
            }
        }
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_nussinov_optimized(int n, base* seq, int** table)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_nussinov(n, seq, table);
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

// Allocate a 1D base array
base* allocate_1d_base_array(int size)
{
    return (base*)malloc(size * sizeof(base));
}

// Free a 1D base array
void free_1d_base_array(base* array)
{
    if (array) {
        free(array);
    }
}

// Initialize arrays with some values
void init_array(int n, base* seq, int** table)
{
    int i, j;

    // RNA bases (A=0, G=1, C=2, T=3/U=3)
    for (i = 0; i < n; i++) {
        seq[i] = (base)((i + 1) % 4);
    }

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            table[i][j] = 0;
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

// Verify nussinov results
int verify_results(int n, int** table1, int** table2)
{
    printf("Verifying DP tables...\n");
    return verify_matrices_int(n, n, table1, table2);
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
double run_performance_test(void (*kernel)(int, base*, int**),
                          int n, base* seq, int** table, int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset table to initial state (all zeros)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                table[i][j] = 0;
            }
        }
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(n, seq, table);
        
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
    int n = 2500;  // Length of RNA sequence
    int num_runs = 1;  // Number of runs for performance testing

    // Allocate arrays for original kernel
    base* seq = allocate_1d_base_array(n);
    int** table1 = allocate_2d_int_array(n, n);

    // Allocate arrays for optimized kernel
    int** table2 = allocate_2d_int_array(n, n);

    // Initialize arrays
    init_array(n, seq, table1);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_nussinov, n, seq, table1, num_runs);
    
    double optimized_time = run_performance_test(kernel_nussinov_optimized, n, seq, table2, num_runs);

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Original kernel average time: %f seconds\n", original_time);
    printf("Optimized kernel average time: %f seconds\n", optimized_time);
    printf("Speedup: %f\n", original_time / optimized_time);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(n, table1, table2)) {
        printf("PASS: Results match exactly\n");
    } else {
        printf("FAIL: Results do not match\n");
    }

    // Free memory
    free_1d_base_array(seq);
    free_2d_int_array(table1, n);
    free_2d_int_array(table2, n);

    return 0;
} 
