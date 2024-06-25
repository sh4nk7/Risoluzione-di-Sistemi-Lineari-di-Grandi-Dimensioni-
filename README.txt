# Progetto: Risoluzione di Sistemi Lineari di Grandi Dimensioni Utilizzando Calcolo Parallelo su Cluster HPC

Questo progetto dimostra l'implementazione e l'esecuzione di algoritmi paralleli per la risoluzione di sistemi lineari di grandi dimensioni utilizzando il metodo di Gauss-Seidel su un cluster HPC (High-Performance Computing) presso l'Università di Parma (UniPr). Vengono esplorati tre approcci: MPI, OpenMP e CUDA.

## Struttura del Progetto

Il progetto è organizzato nelle seguenti directory:

- `mpi/`
  - `gauss_seidel_mpi.c` - Codice sorgente C per l'implementazione MPI.
  - `gauss_seidel_mpi.slurm` - Script SLURM per eseguire il codice MPI sul cluster.
  - `gauss_seidel_mpi_output.txt` - File di output generato dall'esecuzione del codice MPI.

- `openmp/`
  - `gauss_seidel_omp4.c` - Codice sorgente C per l'implementazione OpenMP.
  - `gauss_seidel_omp.slurm` - Script SLURM per eseguire il codice OpenMP sul cluster.
  - `gauss_seidel_omp_output.txt` - File di output generato dall'esecuzione del codice OpenMP.

- `cuda/`
  - `gauss_seidel_cuda.cu` - Codice sorgente CUDA per l'implementazione su GPU.
  - `gauss_seidel_cuda.slurm` - Script SLURM per eseguire il codice CUDA sul cluster.
  - `gauss_seidel_cuda_output.txt` - File di output generato dall'esecuzione del codice CUDA.

- `report/`
  - `relazione_progetto.pdf` - Relazione realizzata in LaTeX che descrive il progetto.

## Requisiti

- Accesso a un cluster HPC presso l'Università di Parma (UniPr).
- Compilatori C/C++ con supporto per MPI, OpenMP e CUDA.
- Ambiente di scripting SLURM per la gestione dei job sul cluster.

## Istruzioni per l'Esecuzione

### Passaggi Generali

1. **Accedere al Cluster HPC**: Utilizzare le credenziali fornite per accedere al cluster HPC di UniPr.

2. **Caricare i Moduli Necessari**: Prima di compilare ed eseguire i programmi, assicurarsi di caricare i moduli necessari per MPI, OpenMP e CUDA. Esempio:
    ```sh
    module purge
    module load gcc
    module load gnu8/8.3.0
    module load openmpi4/4.1.1
    module load cuda/11.6.0
    ```

## Generazione del Dataset

Per generare un dataset sintetico, utilizzare lo script Python fornito:

```sh
python generate_dataset.py

### Implementazione MPI

1. **Compilare il Codice MPI**:
    ```sh
    mpicc -o gauss_seidel_mpi gauss_seidel_mpi.c
    ```

2. **Eseguire il Codice MPI**:
    Utilizzare lo script SLURM per inviare il job al cluster:
    ```sh
    cd mpi
    sbatch gauss_seidel_mpi.slurm
    ```
3. **Verificare l'Output**:
    L'output sarà disponibile nel file `gauss_seidel_mpi_output.txt`.

### Implementazione OpenMP

1. **Compilare il Codice OpenMP**:
    ```sh
    gcc -fopenmp -o gauss_seidel_omp gauss_seidel_omp.c
    ```

2. **Eseguire il Codice OpenMP**:
    Utilizzare lo script SLURM per inviare il job al cluster:
    ```sh
    cd ../openmp
    sbatch gauss_seidel_omp.slurm
    ```
3. **Verificare l'Output**:
    L'output sarà disponibile nel file `gauss_seidel_omp_output.txt`.

### Implementazione CUDA

1. **Compilare il Codice CUDA**:
    ```sh
    nvcc -arch=compute_80 -o gauss_seidel_cuda gauss_seidel_cuda.cu
    ```

2. **Eseguire il Codice CUDA**:
    Utilizzare lo script SLURM per inviare il job al cluster:
    ```sh
    cd ../cuda
    sbatch gauss_seidel_cuda.slurm
    ```
3. **Verificare l'Output**:
    L'output sarà disponibile nel file `gauss_seidel_cuda_output.txt`.

## Confronto dei Risultati

L'implementazione CUDA ha ottenuto il tempo di esecuzione più basso tra tutte le implementazioni, con soli 0.424990 secondi. Questo è dovuto all'efficace utilizzo della potenza computazionale della GPU Tesla P100-PCIE-12GB, sfruttando 256 thread per blocco distribuiti su 4 blocchi. La parallela parallelizzazione su GPU si è dimostrata molto efficace per questo tipo di calcolo intensivo.

L'implementazione OpenMP ha mostrato un tempo di esecuzione di 0.957124292 secondi, utilizzando 4 thread su 4 core. Sebbene l'efficienza sia buona, risulta essere più lenta rispetto alla versione CUDA. Questo suggerisce che la parallelizzazione su CPU tramite OpenMP è meno performante rispetto alla GPU per questo tipo di algoritmo.

L'implementazione MPI ha registrato un tempo di esecuzione di 3 secondi, utilizzando 4 processi con 1 core ciascuno. Questo è significativamente più lento rispetto a entrambe le altre implementazioni. La divisione del lavoro tra più processi su nodi separati può introdurre un overhead significativo nella comunicazione e nella coordinazione, influenzando le prestazioni complessive.

## Relazione

La relazione del progetto è contenuta nel file `report/relazione_progetto.pdf`. 

## Presentazione 

Le slide della presentazione del progetto sono contenute nel file ' report/presentazione_progetto.pdf'.
