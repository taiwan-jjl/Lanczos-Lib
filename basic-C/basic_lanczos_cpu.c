#include <stdio.h>
#include "basic_lanczos_cpu.h"              // Include the declaration
#include "mkl.h"

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
) {
    printf("\nHello, basic_lanczos!\n");

    int Lanczos_stop_check_counter = 0; // Set the Lanczos_stop_check_counter.

    // Start main algorithm:
    for (int i=0; i<A_dim; i++) {

        // STEP-0: calculate the index.
        int idx1 = A_dim*i;             // for _{i}
        int idx2 = A_dim*(i+1);         // for _{i+1}
        int idx3 = A_dim*(i+2);         // for _{i+2}

        // STEP-1:
        // "omega_{i} = A*nu_{i+1} - beta_{i}*nu_{i}"
        
        // STEP-1.1: BLAS L2 cblas_dsymv, Matrix-vector product using a symmetric matrix. (y := alpha*A*x + beta*y)
        // omega_{i} = A*nu_{i+1} + 0.0*omega_{i}
        // cblas_dsymv( // This is for upper or lower triangular storage scheme.
        //     CblasRowMajor,              // C‐style storage
        //     CblasUpper,                 // indicates using the upper triangle
        //     A_dim,                      // rows of A
        //     1.0,                        // alpha
        //     A,                          // your row‐major array
        //     A_dim,                      // “leading dimension”: the number of columns in each row
        //     &nu[idx2],                  // x vector
        //     1,                          // stride. Normally 1 if your vector is contiguous; use a larger stride if picking out every k-th element.
        //     0.0,                        // beta
        //     &omega[idx1],               // y vector
        //     1                           // stride. Normally 1 if your vector is contiguous; use a larger stride if picking out every k-th element.
        // );
        cblas_dgemv( // Use GEMV when A is full-matrix storage scheme.
            CblasRowMajor,                 // memory layout: row-major
            CblasNoTrans,                  // whether to transpose A
            A_dim,                         // number of rows of A
            A_dim,                         // number of cols of A
            1.0,                           // scalar multiplier for A*x
            A,                             // pointer to first element of A
            A_dim,                         // leading dimension of A
            &nu[idx2],                     // pointer to first element of x
            1,                             // stride between elements of x
            0.0,                           // scalar multiplier for y
            &omega[idx1],                  // pointer to first element of y
            1                              // stride between elements of y
        );

        // STEP-2: (it is wise to use the intermediate result of STEP-1.1) (out of order execution)
        // "alpha_{i} = nu_{i+1}^{T} * A * nu_{i+1}"

        // STEP-2.1: BLAS L1 cblas_ddot, vector-vector reduction operation. (input is double, accumulation is double, output is double)
        // alpha_{i} = nu_{i+1}^{T} * omega_{i}
        alpha[i] = cblas_ddot(  
            A_dim,                      // Number of elements  
            &omega[idx1],               // Pointer to first element of X  
            1,                          // Stride between elements of X  
            &nu[idx2],                  // Pointer to first element of Y  
            1                           // Stride between elements of Y  
        );

        // STEP-1.2: BLAS L1 cblas_daxpy, vector-vector operation. (y := a*x + y)
        // omega_{i} = -beta_{i}*nu_{i} + omega_{i}
        cblas_daxpy(
            A_dim,                      // length
            -1*beta[i],                 // alpha
            &nu[idx1],                  // x-vector in daxpy
            1,                          // stride. Normally 1 if your vector is contiguous; use a larger stride if picking out every k-th element.
            &omega[idx1],               // y-vector in daxpy
            1                           // stride. Normally 1 if your vector is contiguous; use a larger stride if picking out every k-th element.
        );

        // STEP-3:
        // "omega_{i} = omega_{i} - alpha_{i} * nu_{i+1}"

        // STEP-3.1: BLAS L1 cblas_daxpy, vector-vector operation. (y := a*x + y)
        cblas_daxpy(
            A_dim,                      // length
            -1*alpha[i],                // alpha
            &nu[idx2],                  // x-vector in daxpy
            1,                          // stride. Normally 1 if your vector is contiguous; use a larger stride if picking out every k-th element.
            &omega[idx1],               // y-vector in daxpy
            1                           // stride. Normally 1 if your vector is contiguous; use a larger stride if picking out every k-th element.
        );

        // STEP-4:
        // "beta_{i+1} = || omega_{i} ||"

        // STEP-4.1: BLAS L1 cblas_dnrm2, computes the Euclidean norm of a vector. (res = ||x||)
        beta[i+1] = cblas_dnrm2(  
            A_dim,                      // Number of elements  
            &omega[idx1],               // Pointer to the first element of X  
            1                           // Stride between elements of X  
        );

        // Check Lanczos iteration stop criterion:
        if (Lanczos_stop_check_counter < Lanczos_stop_check_freq) {
            Lanczos_stop_check_counter++;
        } else if (beta[i+1] < Lanczos_stop_crit) {
            break;
        } else {
            Lanczos_stop_check_counter = 0;
        }

        // STEP-5:
        // "nu_{i+2} = omega_{i} / beta_{i+1}"

        // STEP-5.1: BLAS L1 cblas_dcopy, copies a vector to another vector. (y = x)
        // nu_{i+2} = omega_{i}
        cblas_dcopy(
            A_dim,                      // Number of elements  
            &omega[idx1],               // source vector x
            1,                          // stride = 1
            &nu[idx3],                  // dest   vector y
            1                           // stride = 1
        );

        // STEP-5.2: BLAS L1 cblas_dscal, computes the product of a vector by a scalar. (x = a*x)
        // nu_{i+2} = nu_{i+2} / beta_{i+1}
        cblas_dscal(
            A_dim,                      // Number of elements  
            1.0/beta[i+1],              // Scalar multiplier  
            &nu[idx3],                  // Pointer to the first element of X  
            1                           // Stride between elements of X  
        );

        *Lanczos_iter = i+1;

    }




}
