#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

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

void parallel_gauss_seidel(double *A, double *b, double *x, int N, int max_iter, double tol) {
    #pragma omp parallel
    {
        for (int iter = 0; iter < max_iter; iter++) {
            #pragma omp for
            for (int i = 0; i < N; i++) {
                double sum = 0.0;
                for (int j = 0; j < N; j++) {
                    if (j != i) {
                        sum += A[i * N + j] * x[j];
                    }
                }
                x[i] = (b[i] - sum) / A[i * N + i];
            }

            double norm = 0.0;
            #pragma omp for reduction(+:norm)
            for (int i = 0; i < N; i++) {
                norm += (b[i] - A[i * N + i] * x[i]) * (b[i] - A[i * N + i] * x[i]);
            }

            if (norm < tol) break;
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
    
    parallel_gauss_seidel(A, b, x, N, 1000, 1e-6);

    free(A);
    free(b);
    free(x);
    
    return 0;
}
