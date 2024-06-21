#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void load_data(double *A, double *b, int N) {
    FILE *file_A = fopen("matrix.txt", "r");
    FILE *file_b = fopen("vector.txt", "r");

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

void gauss_seidel_sequential(double *A, double *b, double *x, int N, int max_iter, double tol) {
    for (int iter = 0; iter < max_iter; iter++) {
        double norm = 0.0;

        for (int i = 0; i < N; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                if (j != i) {
                    sum += A[i * N + j] * x[j];
                }
            }
            double new_xi = (b[i] - sum) / A[i * N + i];
            norm += (new_xi - x[i]) * (new_xi - x[i]);
            x[i] = new_xi;
        }

        norm = sqrt(norm);
        if (norm < tol) break;
    }
}

int main(int argc, char *argv[]) {
    int N = 1000; // Cambia questo numero se il tuo input ha una dimensione diversa
    int max_iter = 1000;
    double tol = 1e-6;
    double *A = (double *)malloc(N * N * sizeof(double));
    double *b = (double *)malloc(N * sizeof(double));
    double *x = (double *)malloc(N * sizeof(double));

    load_data(A, b, N);

    // Inizializzazione del vettore soluzione
    for (int i = 0; i < N; i++) {
        x[i] = 0.0;
    }

    gauss_seidel_sequential(A, b, x, N, max_iter, tol);

    printf("Soluzione trovata:\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", x[i]);
    }
    printf("\n");

    free(A);
    free(b);
    free(x);

    return 0;
}
