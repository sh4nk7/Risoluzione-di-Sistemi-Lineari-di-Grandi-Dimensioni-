#!/bin/bash
#SBATCH --job-name=gauss_seidel_mpi
#SBATCH --output=gauss_seidel_mpi_output.txt
#SBATCH --ntasks=4
#SBATCH --time=01:00:00
#SBATCH --partition=standard

module load mpi
mpirun ./gauss_seidel_mpi
