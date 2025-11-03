#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <mkl.h>

// Deriche edge detection filter
// Implements a recursive 2D edge detection filter
void kernel_deriche(int w, int h, double alpha,
                   double** imgIn, double** imgOut,
                   double** y1, double** y2)
{
    int i, j;
    double xm1, tm1, ym1, ym2;
    double xp1, xp2;
    double tp1, tp2;
    double yp1, yp2;

    double k;
    double a1, a2, a3, a4, a5, a6, a7, a8;
    double b1, b2, c1, c2;

    // Calculate coefficients
    k = (1.0 - exp(-alpha)) * (1.0 - exp(-alpha)) / 
        (1.0 + 2.0 * alpha * exp(-alpha) - exp(2.0 * alpha));
    a1 = a5 = k;
    a2 = a6 = k * exp(-alpha) * (alpha - 1.0);
    a3 = a7 = k * exp(-alpha) * (alpha + 1.0);
    a4 = a8 = -k * exp(-2.0 * alpha);
    b1 = pow(2.0, -alpha);
    b2 = -exp(-2.0 * alpha);
    c1 = c2 = 1.0;

    // First pass - forward direction along columns
    for (i = 0; i < w; i++) {
        ym1 = 0.0;
        ym2 = 0.0;
        xm1 = 0.0;
        for (j = 0; j < h; j++) {
            y1[i][j] = a1 * imgIn[i][j] + a2 * xm1 + b1 * ym1 + b2 * ym2;
            xm1 = imgIn[i][j];
            ym2 = ym1;
            ym1 = y1[i][j];
        }
    }

    // Second pass - backward direction along columns
    for (i = 0; i < w; i++) {
        yp1 = 0.0;
        yp2 = 0.0;
        xp1 = 0.0;
        xp2 = 0.0;
        for (j = h - 1; j >= 0; j--) {
            y2[i][j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
            xp2 = xp1;
            xp1 = imgIn[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
    }

    // Combine results from first two passes
    for (i = 0; i < w; i++) {
        for (j = 0; j < h; j++) {
            imgOut[i][j] = c1 * (y1[i][j] + y2[i][j]);
        }
    }

    // Third pass - forward direction along rows
    for (j = 0; j < h; j++) {
        tm1 = 0.0;
        ym1 = 0.0;
        ym2 = 0.0;
        for (i = 0; i < w; i++) {
            y1[i][j] = a5 * imgOut[i][j] + a6 * tm1 + b1 * ym1 + b2 * ym2;
            tm1 = imgOut[i][j];
            ym2 = ym1;
            ym1 = y1[i][j];
        }
    }

    // Fourth pass - backward direction along rows
    for (j = 0; j < h; j++) {
        tp1 = 0.0;
        tp2 = 0.0;
        yp1 = 0.0;
        yp2 = 0.0;
        for (i = w - 1; i >= 0; i--) {
            y2[i][j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
            tp2 = tp1;
            tp1 = imgOut[i][j];
            yp2 = yp1;
            yp1 = y2[i][j];
        }
    }

    // Final combination of all passes
    for (i = 0; i < w; i++) {
        for (j = 0; j < h; j++) {
            imgOut[i][j] = c2 * (y1[i][j] + y2[i][j]);
        }
    }
}

// Placeholder for optimized kernel - to be replaced later
void kernel_deriche_optimized(int w, int h, double alpha,
                             double** imgIn, double** imgOut,
                             double** y1, double** y2)
{
    // Currently just calls the original kernel
    // This will be replaced with optimized implementation later
    kernel_deriche(w, h, alpha, imgIn, imgOut, y1, y2);
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

// Initialize arrays
void init_array(int w, int h, double* alpha, double** imgIn, double** imgOut)
{
    int i, j;

    *alpha = 0.25; // Parameter of the filter

    // Input should be between 0 and 1 (grayscale image pixel)
    for (i = 0; i < w; i++) {
        for (j = 0; j < h; j++) {
            imgIn[i][j] = (double)((313 * i + 991 * j) % 65536) / 65535.0;
        }
    }

    // Initialize output to zero
    for (i = 0; i < w; i++) {
        for (j = 0; j < h; j++) {
            imgOut[i][j] = 0.0;
        }
    }
}

// Verify if two matrices are equal within tolerance
int verify_results(int w, int h, double** imgOut1, double** imgOut2, double tolerance)
{
    printf("Verifying edge detection results with tolerance %e...\n", tolerance);
    
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            if (fabs(imgOut1[i][j] - imgOut2[i][j]) > tolerance) {
                printf("Verification failed at position [%d][%d]: %f != %f (diff: %e)\n", 
                       i, j, imgOut1[i][j], imgOut2[i][j], 
                       fabs(imgOut1[i][j] - imgOut2[i][j]));
                return 0;
            }
        }
    }
    return 1;
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

// Run performance test for a kernel
double run_performance_test(void (*kernel)(int, int, double, double**, double**, double**, double**),
                          int w, int h, double alpha, double** imgIn_orig,
                          double** imgIn, double** imgOut, double** y1, double** y2, int num_runs)
{
    double total_time = 0.0;
    
    for (int run = 0; run < num_runs; run++) {
        // Reset input to initial state
        copy_matrix(w, h, imgIn_orig, imgIn);
        
        // Start timing
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);
        
        // Run kernel
        kernel(w, h, alpha, imgIn, imgOut, y1, y2);
        
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
    int w = 4096;   // Image width
    int h = 2160;   // Image height
    double alpha;
    int num_runs = 3;  // Number of runs for performance testing

    // Allocate arrays
    double** imgIn_orig = allocate_2d_array(w, h);
    double** imgIn1 = allocate_2d_array(w, h);
    double** imgOut1 = allocate_2d_array(w, h);
    double** y1_1 = allocate_2d_array(w, h);
    double** y2_1 = allocate_2d_array(w, h);

    double** imgIn2 = allocate_2d_array(w, h);
    double** imgOut2 = allocate_2d_array(w, h);
    double** y1_2 = allocate_2d_array(w, h);
    double** y2_2 = allocate_2d_array(w, h);

    // Initialize arrays
    init_array(w, h, &alpha, imgIn_orig, imgOut1);
    copy_matrix(w, h, imgIn_orig, imgIn1);
    copy_matrix(w, h, imgIn_orig, imgIn2);

    // Run performance tests
    printf("Running performance tests (%d runs each)...\n", num_runs);
    
    double original_time = run_performance_test(kernel_deriche, w, h, alpha,
                                              imgIn_orig, imgIn1, imgOut1, y1_1, y2_1, num_runs);
    
    double optimized_time = run_performance_test(kernel_deriche_optimized, w, h, alpha,
                                               imgIn_orig, imgIn2, imgOut2, y1_2, y2_2, num_runs);

    // Print performance results
    printf("\nPerformance Results:\n");
    printf("Original kernel average time: %f seconds\n", original_time);
    printf("Optimized kernel average time: %f seconds\n", optimized_time);
    printf("Speedup: %f\n", original_time / optimized_time);

    // Verify results
    printf("\nVerification Results:\n");
    if (verify_results(w, h, imgOut1, imgOut2, 1e-4)) {
        printf("PASS: Results match within tolerance\n");
    } else {
        printf("FAIL: Results do not match within tolerance\n");
    }

    // Free memory
    free_2d_array(imgIn_orig, w);
    free_2d_array(imgIn1, w);
    free_2d_array(imgOut1, w);
    free_2d_array(y1_1, w);
    free_2d_array(y2_1, w);
    
    free_2d_array(imgIn2, w);
    free_2d_array(imgOut2, w);
    free_2d_array(y1_2, w);
    free_2d_array(y2_2, w);

    return 0;
} 
