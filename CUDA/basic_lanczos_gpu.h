#ifndef BASIC_LANCZOS_GPU   // include guard
#define BASIC_LANCZOS_GPU   // include guard

// Function declaration
// void basic_lanczos(void);

void basic_lanczos_gpu(
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
