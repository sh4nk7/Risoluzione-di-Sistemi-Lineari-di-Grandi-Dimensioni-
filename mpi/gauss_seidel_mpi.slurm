#!/bin/bash
#SBATCH --job-name=gauss_seidel_mpi
#SBATCH --output=gauss_seidel_mpi_output.txt
#SBATCH --ntasks=4
#SBATCH --time=00:30:00
#SBATCH --partition=compute

module load mpi

mpicc -o gauss_seidel_mpi gauss_seidel_mpi.c
mpirun -np 4 ./gauss_seidel_mpi

