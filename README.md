# Lanczos-Lib

**Lanczos-Lib** is a high-performance library implementing classical and block Lanczos algorithms on both CPU and GPU in C and Fortran, optimized for Intel CPU and NVIDIA GPU.

## Features

- **MATLAB Prototypes**: Provided for deeper insight into the Lanczos algorithm.
- **Intel CPU Implementation**: Utilizes Intel MKL for high-performance numerical routines.
- **Reproducibility Control**: Implements Intel Conditional Numerical Reproducibility (CNR) for bitwise-consistent results across runs. Automatically detect and set to the optimal reproducibility.
- **NVIDIA GPU Implementation**: Utilizes NVIDIA cuBLAS for high-performance numerical routines.
- [fortran] (TBC)
- [eigen-system and solver]??? (TBC)

## Prerequisites

- Make
- Intel MKL (implemented with intel-oneapi-hpc-toolkit-2025.1.0.666)
- Intel C and Fortran compilers (tested with intel-oneapi-hpc-toolkit-2025.1.0.666)
- NVIDIA HPC SDK (tested with nvhpc_2025_253_Linux_x86_64_cuda_12.8, Driver 570.133.07)
- NVIDIA cuBLAS library (implemented with nvhpc_2025_253_Linux_x86_64_cuda_12.8)

## Quick Start

1. **Clone the repository**

```bash
git clone https://github.com/taiwan-jjl/Lanczos-Lib.git
cd Lanczos-Lib
```

2. **Configure** (edit `Makefile` if needed)

```bash
# Compile and output object files and executables
make

# Clean build artifacts
make clean
```

## Function specification

**basic_lanczos_cpu** :

```C
void basic_lanczos_cpu(
    const double* A,
    double* nu,
    double* omega,
    double* alpha,
    double* beta,
    const int A_dim,
    const double Lanczos_stop_crit,
    const int Lanczos_stop_check_freq,
    int* Lanczos_iter
)
```

Output: none.

## Future Work

- More error detect and handing code.

    Currently, it is omitted for code readability and simplicity.

- SYMV (symmetric matrix-vector product) issue on GPU:
  - <https://icl.utk.edu/files/publications/2012/icl-utk-530-2012.pdf>
  - <https://www.researchgate.net/publication/220782116_Optimizing_symmetric_dense_matrix-vector_multiplication_on_GPUS>
  - <https://arxiv.org/abs/1410.1726>
  - <https://siboehm.com/articles/22/CUDA-MMM>

- More initial vector (nu) methods.

- Advanced memory allocation method in GPU version.
