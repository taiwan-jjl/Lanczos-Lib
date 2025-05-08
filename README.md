# Lanczos-Lib

**Lanczos-Lib** is a high-performance library implementing classical and block Lanczos algorithms on both CPU and GPU in C and Fortran, optimized for Intel CPU and NVIDIA GPU.

## Features

- **MATLAB Prototypes**: Provided for deeper insight into the Lanczos algorithm.
- **CPU Implementation**: Utilizes Intel MKL for high-performance numerical routines.
- **Reproducibility Control**: Implements Intel Conditional Numerical Reproducibility (CNR) for bitwise-consistent results across runs. Automatically detect and set to the optimal reproducibility.
- [gpu] (TBC)
- [fortran] (TBC)
- [eigen-system and solver]??? (TBC)

## Prerequisites

- Make
- Intel MKL (implemented with intel-oneapi-hpc-toolkit-2025.1.0.666)
- Intel C and Fortran compilers (tested with intel-oneapi-hpc-toolkit-2025.1.0.666)

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
(TBC)