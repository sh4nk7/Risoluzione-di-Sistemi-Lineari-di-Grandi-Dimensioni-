#!/bin/bash
#SBATCH --job-name=gauss_seidel_omp
#SBATCH --output=gauss_seidel_omp.out
#SBATCH --error=gauss_seidel_omp.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu
#SBATCH --qos=cpu

module load gcc
module load gnu8/8.3.0
module load openmpi4/4.1.1  
module load cuda/11.6.0

srun ./gauss_seidel_omp matrix.txt vector.txt
