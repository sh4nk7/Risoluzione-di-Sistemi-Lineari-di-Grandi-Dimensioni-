#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>  // Includi math.h per dichiarare la funzione fabs

#define N 1000
#define MAX_ITER 1000
#define TOL 1e-6

void read_matrix(const char *filename, double A[N][N]) {
    // Funzione per leggere la matrice da file
    // Implementazione omessa per brevità
}

void read_vector(const char *filename, double b[N]) {
    // Funzione per leggere il vettore da file
    // Implementazione omessa per brevità
}

void sequential_gauss_seidel(double A[N][N], double b[N], double x[N]) {
    int i, j, iter;
    double norm, sum, x_new;

    for (iter = 0; iter < MAX_ITER; iter++) {
        norm = 0.0;

        for (i = 0; i < N; i++) {
            sum = b[i];
            for (j = 0; j < N; j++) {
                if (j != i) {
                    sum -= A[i][j] * x[j];
                }
            }
            x_new = sum / A[i][i];
            norm += fabs(x_new - x[i]);  // Utilizzo corretto di fabs
            x[i] = x_new;
        }

        if (norm < TOL) {
            break;
        }
    }
}

int main() {
    double A[N][N], b[N], x[N] = {0};

    read_matrix("matrix.txt", A);
    read_vector("vector.txt", b);

    clock_t start_time = clock();  // Inizio misurazione tempo di esecuzione

    sequential_gauss_seidel(A, b, x);

    clock_t end_time = clock();  // Fine misurazione tempo di esecuzione

    double sequential_execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Stampaggio dei risultati
    printf("Tempo di esecuzione sequenziale: %.6f secondi\n", sequential_execution_time);

    return 0;
}
