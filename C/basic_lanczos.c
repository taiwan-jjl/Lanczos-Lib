#include <stdio.h>
#include "basic_lanczos.h"              // Include the declaration
#include "mkl.h"

void basic_lanczos(
    const double* A,
    double* nu,
    double* omega,
    double* alpha,
    double* beta,
    const int A_dim,
    const double Lanczos_stop_crit,
    const int Lanczos_stop_check_freq,
    int* Lanczos_iter
) {
    printf("\nHello, basic_lanczos!\n");

    // Start main algorithm:
    for (int i=0; i<A_dim; i++) {
        // omega_{i} = A*nu_{i+1} - beta_{i}*nu_{i}
        // BLAS L2 cblas_dsymv, Matrix-vector product using a symmetric matrix.
        cblas_dsymv(
            CblasRowMajor,   // ← C‐style storage
            CblasUpper,      // still indicates you’re using the upper triangle
            A_dim,           // rows of A
            1.0,             // alpha
            A,               // your row‐major array
            n,               // “leading dimension”: the number of columns in each row
            nu,              // x vector
            1,               // incx
            0.0,             // beta
            y,               // y vector
            1                // incy
        );







    }

    *Lanczos_iter = 5;












}
