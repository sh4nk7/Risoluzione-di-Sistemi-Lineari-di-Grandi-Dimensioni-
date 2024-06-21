#!/bin/bash
#SBATCH --job-name=gauss_seidel_mpi
#SBATCH --output=gauss_seidel_mpi_%A_%a.out
#SBATCH --error=gauss_seidel_mpi_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --partition=cpu
#SBATCH --qos=cpu
#SBATCH --array=1-10

# Imposta il valore di N per il dataset
N=1000

module purge
module load gcc
module load gnu8/8.3.0
module load openmpi4/4.1.1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

# Avvio del cronometro per il job
start_time=$(date +%s.%N)

# Avvio del programma MPI
mpirun --mca opal_warn_on_missing_libcuda 0 -np $SLURM_NTASKS ./gauss_seidel_mpi

# Fine del cronometro
end_time=$(date +%s.%N)

# Calcolo del tempo di esecuzione come float
runtime=$(echo "$end_time - $start_time" | bc -l)

# Scrivi il report completo nel file di output
report_filename="gauss_seidel_mpi_report_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt"
echo "Dataset dimensione (N): $N" > $report_filename
echo "Tempo di esecuzione: $runtime secondi" >> $report_filename
echo "Numero di processi MPI: $SLURM_NTASKS" >> $report_filename
echo "Numero di core utilizzati per processo: $SLURM_CPUS_PER_TASK" >> $report_filename
echo "Moduli caricati:" >> $report_filename
module list >> $report_filename
echo "" >> $report_filename

# Fine dello script
