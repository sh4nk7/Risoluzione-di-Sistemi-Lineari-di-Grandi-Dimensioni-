import numpy as np

def generate_dataset(N, filename_A='matrix_A.txt', filename_b='vector_b.txt'):
    # Genera una matrice casuale NxN
    A = np.random.rand(N, N)
    
    # Rendere A diagonale dominante
    for i in range(N):
        A[i, i] = sum(np.abs(A[i])) + 1
    
    # Genera un vettore casuale Nx1
    b = np.random.rand(N)
    
    # Salva la matrice A e il vettore b in file di testo
    np.savetxt(filename_A, A, fmt='%.6f')
    np.savetxt(filename_b, b, fmt='%.6f')

# Esempio di utilizzo
generate_dataset(1000)
