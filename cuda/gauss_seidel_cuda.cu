#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define CUDA_CHECK(call) { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "Errore CUDA: %s\n", cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
}

#define FILE_CHECK(file) { \
    if (file == NULL) { \
        perror("Errore nell'apertura del file"); \
        exit(EXIT_FAILURE); \
    } \
}

void load_data(const char *filename, double *data, int size) {
    FILE *file = fopen(filename, "r");
    FILE_CHECK(file);

    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%lf", &data[i]) != 1) {
            fprintf(stderr, "Errore durante il caricamento dei dati dal file %s\n", filename);
            exit(EXIT_FAILURE);
        }
    }

    fclose(file);
}

__global__ void gauss_seidel_kernel(double *A, double *b, double *x, int N, int max_iter, double tol) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        for (int iter = 0; iter < max_iter; iter++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                if (j != idx) {
                    sum += A[idx * N + j] * x[j];
                }
            }
            x[idx] = (b[idx] - sum) / A[idx * N + idx];
            __syncthreads();
        }
    }
}

int main() {
    int N = 1000;
    double *A, *b, *x;
    A = (double *)malloc(N * N * sizeof(double));
    b = (double *)malloc(N * sizeof(double));
    x = (double *)malloc(N * sizeof(double));

    if (A == NULL || b == NULL || x == NULL) {
        fprintf(stderr, "Errore nell'allocazione della memoria\n");
        exit(EXIT_FAILURE);
    }

    load_data("matrix.txt", A, N * N);
    load_data("vector.txt", b, N);

    double *d_A, *d_b, *d_x;
    CUDA_CHECK(cudaMalloc((void **)&d_A, N * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_b, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void **)&d_x, N * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_x, 0, N * sizeof(double))); // Inizializza x a 0

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    gauss_seidel_kernel<<<numBlocks, blockSize>>>(d_A, d_b, d_x, N, 1000, 1e-6);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost));

    printf("Risultati:\n");
    for (int i = 0; i < N; i++) {
        printf("%lf ", x[i]);
    }
    printf("\n");

    free(A);
    free(b);
    free(x);
    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);

    return 0;
}
