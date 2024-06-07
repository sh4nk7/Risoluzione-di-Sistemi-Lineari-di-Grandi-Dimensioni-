#!/bin/bash
#SBATCH --job-name=gauss_seidel_omp
#SBATCH --output=gauss_seidel_omp_output.txt
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --partition=standard

module load gcc
./gauss_seidel_omp
