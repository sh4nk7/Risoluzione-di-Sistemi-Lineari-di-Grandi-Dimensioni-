import numpy as np

def load_numeric_data(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            try:
                # Prova a convertire la linea in numeri
                data.append([float(x) for x in line.split()])
            except ValueError:
                # Ignora le righe che non possono essere convertite
                continue
    return np.array(data)

# Risultati dei tre algoritmi in tre file separati
mpi_result = load_numeric_data("gauss_seidel_mpi.out")
cuda_result = load_numeric_data("gauss_seidel_cuda.out")
omp_result = load_numeric_data("gauss_seidel_omp.out")

# Confronta i risultati
diff_mpi_cuda = np.linalg.norm(mpi_result - cuda_result)
diff_mpi_omp = np.linalg.norm(mpi_result - omp_result)
diff_cuda_omp = np.linalg.norm(cuda_result - omp_result)

print("Differenza tra MPI e CUDA:", diff_mpi_cuda)
print("Differenza tra MPI e OMP:", diff_mpi_omp)
print("Differenza tra CUDA e OMP:", diff_cuda_omp)

# Imposta una tolleranza per considerare i risultati come equivalenti
tolerance = 1e-2

if (np.allclose(mpi_result, cuda_result, atol=tolerance) and 
    np.allclose(mpi_result, omp_result, atol=tolerance) and 
    np.allclose(cuda_result, omp_result, atol=tolerance)):
    result_msg = "I risultati sono equivalenti entro la tolleranza specificata."
else:
    result_msg = "I risultati differiscono oltre la tolleranza specificata."

# Scrivi il report in un file
report_filename = "confronto_risultati.txt"
with open(report_filename, 'w') as report_file:
    report_file.write("Differenza tra MPI e CUDA: {}\n".format(diff_mpi_cuda))
    report_file.write("Differenza tra MPI e OMP: {}\n".format(diff_mpi_omp))
    report_file.write("Differenza tra CUDA e OMP: {}\n".format(diff_cuda_omp))
    report_file.write("\n")
    report_file.write(result_msg + "\n")

print("Report salvato in:", report_filename)
