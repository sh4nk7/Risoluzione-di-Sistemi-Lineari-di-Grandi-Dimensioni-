#!/bin/bash
#SBATCH --job-name=gauss_seidel_cuda     # Nome del job
#SBATCH --output=gauss_seidel_cuda.out   # File di output
#SBATCH --error=gauss_seidel_cuda.err    # File di errori
#SBATCH --nodes=1                        # Numero di nodi
#SBATCH --ntasks=1                       # Numero di task MPI (non applicabile qui, impostato a 1)
#SBATCH --cpus-per-task=4                # Numero di core CPU per task
#SBATCH --partition=gpu                  # Partizione GPU
#SBATCH --qos=gpu                        # QoS per GPU
#SBATCH --gres=gpu:p100:1                # Specifica la GPU da utilizzare (ad esempio, P100)
#SBATCH --time=01:00:00                  # Limite di tempo

# Carica i moduli necessari
module purge
module load gcc
module load gnu8/8.3.0
module load openmpi4/4.1.1
module load cuda/11.6.0

# Imposta il valore di N per il dataset
N=1000

# Informazioni sulla GPU
gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader)

# Avvio del cronometro per il job
start_time=$(date +%s.%N)

# Esegui il programma CUDA
srun ./gauss_seidel_cuda matrix.txt vector.txt

# Fine del cronometro
end_time=$(date +%s.%N)

# Calcolo del tempo di esecuzione come float
runtime=$(echo "$end_time - $start_time" | bc -l)

# Scrivi il report completo nel file di output
echo "Dataset dimensione (N): $N" > gauss_seidel_cuda_report.txt
echo "Tempo di esecuzione: $runtime secondi" >> gauss_seidel_cuda_report.txt
echo "GPU utilizzata: $gpu_info" >> gauss_seidel_cuda_report.txt
echo "Numero di thread per blocco: 256" >> gauss_seidel_cuda_report.txt
echo "Numero di blocchi: $(( (N + 256 - 1) / 256 ))" >> gauss_seidel_cuda_report.txt
echo "Moduli caricati:" >> gauss_seidel_cuda_report.txt
module list >> gauss_seidel_cuda_report.txt
echo "" >> gauss_seidel_cuda_report.txt
