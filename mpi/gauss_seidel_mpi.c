#include <mpi.h>
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
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int rows_per_proc = N / size;
    double *local_A = (double *)malloc(rows_per_proc * N * sizeof(double));
    double *local_b = (double *)malloc(rows_per_proc * sizeof(double));
    double *local_x = (double *)malloc(rows_per_proc * sizeof(double));
    
    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE, local_A, rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(b, rows_per_proc, MPI_DOUBLE, local_b, rows_per_proc, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int iter = 0; iter < max_iter; iter++) {
        for (int i = 0; i < rows_per_proc; i++) {
            double sum = 0.0;
            for (int j = 0; j < N; j++) {
                if (j != rank * rows_per_proc + i) {
                    sum += local_A[i * N + j] * x[j];
                }
            }
            local_x[i] = (local_b[i] - sum) / local_A[i * N + (rank * rows_per_proc + i)];
        }

        MPI_Allgather(local_x, rows_per_proc, MPI_DOUBLE, x, rows_per_proc, MPI_DOUBLE, MPI_COMM_WORLD);
        
        double norm = 0.0;
        for (int i = 0; i < rows_per_proc; i++) {
            norm += (local_b[i] - local_A[i * N + (rank * rows_per_proc + i)] * local_x[i]) * 
                    (local_b[i] - local_A[i * N + (rank * rows_per_proc + i)] * local_x[i]);
        }
        double global_norm;
        MPI_Allreduce(&norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        if (global_norm < tol) break;
    }

    free(local_A);
    free(local_b);
    free(local_x);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
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

    MPI_Finalize();
    return 0;
}
