#!/bin/bash
#SBATCH --job-name=gauss_seidel_omp
#SBATCH --output=gauss_seidel_omp_output.txt
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --partition=compute

module load gcc

gcc -fopenmp -o gauss_seidel_omp gauss_seidel_omp.c
export OMP_NUM_THREADS=4
./gauss_seidel_omp
