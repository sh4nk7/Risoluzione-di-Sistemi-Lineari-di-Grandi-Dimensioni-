// gauss_seidel_sequential.c

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define N 1000
#define MAX_ITER 1000
#define TOLERANCE 1e-6

void gaussSeidel(float A[N][N], float b[N], float x[N]) {
    float sum;

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        for (int i = 0; i < N; ++i) {
            sum = 0.0f;
            for (int j = 0; j < N; ++j) {
                if (j != i) {
                    sum += A[i][j] * x[j];
                }
            }
            x[i] = (b[i] - sum) / A[i][i];
        }
    }
}

int main() {
    struct timeval start_time, end_time;
    float elapsed_time;

    float A[N][N];
    float b[N];
    float x[N];

    // Initialize A, b, x (omitted for brevity)

    // Start timing
    gettimeofday(&start_time, NULL);

    // Run Gauss-Seidel method
    gaussSeidel(A, b, x);

    // End timing
    gettimeofday(&end_time, NULL);

    // Calculate elapsed time
    elapsed_time = (end_time.tv_sec - start_time.tv_sec) + 
                   (end_time.tv_usec - start_time.tv_usec) / 1e6;

    // Print execution time
    printf("Tempo di esecuzione: %.6f secondi\n", elapsed_time);

    return 0;
}
