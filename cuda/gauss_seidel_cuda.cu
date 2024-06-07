#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

void load_data(double *A, double *b, int N) {
    FILE *file_A = fopen("data/matrix_A.txt", "r");
    FILE *file_b = fopen("data/vector_b.txt", "r");

    if (file_A == NULL || file_b == NULL) {
        perror("Errore nell'apertura dei file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fscanf(file_A, "%lf", &A[i * N + j]);
        }
    }

    for (int i = 0; i < N; i++) {
        fscanf(file_b, "%lf", &b[i]);
    }

    fclose(file_A);
    fclose(file_b);
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
    double *A = (double *)malloc(N * N * sizeof(double));
    double *b = (double *)malloc(N * sizeof(double));
    double *x = (double *)malloc(N * sizeof(double));

    // Caricamento dei dati
    load_data(A, b, N);
    
    // Inizializzazione del vettore soluzione
    for (int i = 0; i < N; i++) {
        x[i] = 0.0;
    }

    double *d_A, *d_b, *d_x;
    cudaMalloc((void **)&d_A, N * N * sizeof(double));
    cudaMalloc((void **)&d_b, N * sizeof(double));
    cudaMalloc((void **)&d_x, N * sizeof(double));

    cudaMemcpy(d_A, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x, N * sizeof(double), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    gauss_seidel_kernel<<<numBlocks, blockSize>>>(d_A, d_b, d_x, N, 1000, 1e-6);

    cudaMemcpy(x, d_x, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_b);
    cudaFree(d_x);
    free(A);
    free(b);
    free(x);

    return 0;
}
