#!/bin/bash
#SBATCH --job-name=gauss_seidel_omp
#SBATCH --output=gauss_seidel_omp_%A_%a.out
#SBATCH --error=gauss_seidel_omp_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --array=1-10

module load gcc
module load gnu8/8.3.0
module load openmpi4/4.1.1
module load cuda/11.6.0

# Imposta il numero di thread OpenMP
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Compila il programma
gcc -fopenmp gauss_seidel_omp4.c -o gauss_seidel_omp4

# Inizio della misurazione del tempo di esecuzione
start_time=$(date +%s%N)

# Esecuzione del programma
./gauss_seidel_omp4 matrix.txt vector.txt

# Fine della misurazione del tempo di esecuzione
end_time=$(date +%s%N)

# Calcola il tempo di esecuzione in secondi con frazioni
execution_time=$(echo "scale=6; ($end_time - $start_time) / 1000000000" | bc -l)

# Scrivi il report completo nel file di output
{
  echo "Tempo di esecuzione: $execution_time secondi"
  echo "Numero di thread OpenMP: $OMP_NUM_THREADS"
  echo "Numero di core utilizzati: $SLURM_CPUS_PER_TASK"
  echo "Moduli caricati:"
  module list
  echo ""
} > gauss_seidel_omp_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out
