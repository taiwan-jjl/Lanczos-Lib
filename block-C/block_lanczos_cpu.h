#ifndef BLOCK_LANCZOS_CPU   // include guard
#define BLOCK_LANCZOS_CPU   // include guard

// Function declaration
// void block_lanczos_cpu(void);

void block_lanczos_cpu(
    const double* A,
    double* nu,
    double* omega,
    double* alpha,
    double* beta,
    const int A_dim,
    const double Lanczos_stop_crit,
    const int Lanczos_stop_check_freq,
    int* Lanczos_iter
);


#endif                  // include guard
