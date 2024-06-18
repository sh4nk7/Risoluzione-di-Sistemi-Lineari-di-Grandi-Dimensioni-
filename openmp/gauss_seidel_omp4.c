#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 1000
#define MAX_ITER 1000
#define TOL 1e-6

void read_matrix(const char *filename, double A[N][N]) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening matrix file");
        exit(EXIT_FAILURE);
    }

    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            if (fscanf(file, "%lf", &A[i][j]) != 1) {
                perror("Error reading matrix file");
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(file);
}

void read_vector(const char *filename, double b[N]) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening vector file");
        exit(EXIT_FAILURE);
    }

    int i;
    for (i = 0; i < N; i++) {
        if (fscanf(file, "%lf", &b[i]) != 1) {
            perror("Error reading vector file");
            fclose(file);
            exit(EXIT_FAILURE);
        }
    }
    fclose(file);
}

void parallel_gauss_seidel(double A[N][N], double b[N], double x[N]) {
    int i, j, iter;
    double norm, sum, x_new;

    for (iter = 0; iter < MAX_ITER; iter++) {
        norm = 0.0;

        #pragma omp parallel for private(j, sum, x_new) reduction(+:norm)
        for (i = 0; i < N; i++) {
            sum = b[i];
            for (j = 0; j < N; j++) {
                if (j != i) {
                    sum -= A[i][j] * x[j];
                }
            }
            x_new = sum / A[i][i];
            norm += fabs(x_new - x[i]);
            x[i] = x_new;
        }

        if (norm < TOL) {
            break;
        }
    }
}

int main() {
    double A[N][N], b[N], x[N] = {0};
    const char *matrix_file = "matrix.txt";
    const char *vector_file = "vector.txt";

    // Lettura della matrice e del vettore da file
    read_matrix(matrix_file, A);
    read_vector(vector_file, b);

    // Inizio misurazione del tempo di esecuzione
    double start_time = omp_get_wtime();

    // Risoluzione del sistema con il metodo di Gauss-Seidel parallelo
    parallel_gauss_seidel(A, b, x);

    // Fine misurazione del tempo di esecuzione
    double end_time = omp_get_wtime();
    double execution_time = end_time - start_time;

    // Stampa del tempo di esecuzione
    printf("Tempo di esecuzione: %f secondi\n", execution_time);

    // Esempio di stampa di altre informazioni di report
    // Puoi aggiungere ulteriori informazioni a seconda delle tue esigenze
    printf("Dettagli delle prestazioni:\n");
    printf("Numero di thread OpenMP: %d\n", omp_get_max_threads());
    // Altre informazioni di monitoraggio delle prestazioni...

    return 0;
}

		
		