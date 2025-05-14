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

        // STEP-0: calculate the index.
        int idx1 = A_dim*i;             // for _{i}
        int idx2 = A_dim*(i+1);         // for _{i+1}
        int idx3 = A_dim*(i+2);         // for _{i+2}

        // STEP-1:
        // omega_{i} = A*nu_{i+1} - beta_{i}*nu_{i}
        
        // STEP-1.1: BLAS L2 cblas_dsymv, Matrix-vector product using a symmetric matrix. (y := alpha*A*x + beta*y)
        // omega_{i} = A*nu_{i+1} + beta*omega_{i}
        cblas_dsymv(
            CblasRowMajor,              // C‐style storage
            CblasUpper,                 // indicates using the upper triangle
            A_dim,                      // rows of A
            1.0,                        // alpha
            A,                          // your row‐major array
            A_dim,                      // “leading dimension”: the number of columns in each row
            &nu[idx2],                  // x vector
            1,                          // stride. Normally 1 if your vector is contiguous; use a larger stride if picking out every k-th element.
            0.0,                        // beta
            &omega[idx1],               // y vector
            1                           // stride. Normally 1 if your vector is contiguous; use a larger stride if picking out every k-th element.
        );







    }

    *Lanczos_iter = 5;












}
