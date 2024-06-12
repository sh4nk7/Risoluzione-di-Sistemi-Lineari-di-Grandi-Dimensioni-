import numpy as np

#Risultati dei tre algoritmi in tre file separati
mpi_result = np.loadtxt("gauss_seidel_mpi.out.txt")
cuda_result = np.loadtxt("gauss_seidel_cuda.out.txt")
omp_result = np.loadtxt("gauss_seidel_omp.out.txt")

# Confronta i risultati
print("Differenza tra MPI e CUDA:", np.linalg.norm(mpi_result - cud_result))
print("Differenza tra MPI e OMP:", np.linalg.norm(mpi_result - omp_result))
print("Differenza tra CUDA e OMP:", np.linalg.norm(cuda_result - omp_result))

# Imposta una tolleranza per considerare i risultati come equivalenti
tolerance = 1e-6

if (np.allclose(mpi_result, cud_result, atol=tolerance) and 
    np.allclose(mpi_result, omp_result, atol=tolerance) and 
    np.allclose(cuda_result, omp_result, atol=tolerance)):
    print("I risultati sono equivalenti entro la tolleranza specificata.")
else:
    print("I risultati differiscono oltre la tolleranza specificata.")
