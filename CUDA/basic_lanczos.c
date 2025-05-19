#include "basic_lanczos.h"              // Include the declaration
/////
#include <stdio.h>
#include <stdlib.h>
/////
#include <cuda_runtime.h>
#include <cublas_v2.h>


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

    int Lanczos_stop_check_counter = 0; // Set the Lanczos_stop_check_counter.

    // Create and initialize cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Create cuBLAS parameter which are allowed on host memory
    double cublas_alpha = 0.0;
    double cublas_beta = 0.0;

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
        cublas_alpha = 1.0;
        cublas_beta = 0.0;
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);                                     // important! set the right mode based on your input location.
        cublasDsymv(
            handle,                     // cuBLAS handle
            CUBLAS_FILL_MODE_UPPER      // indicates using the upper triangle
            A_dim,                      // rows of A
            &cublas_alpha,              // alpha
            A,                          // your row‐major array
            A_dim,                      // “leading dimension”: the number of columns in each row
            &nu[idx2],                  // x vector
            1,                          // stride. Normally 1 if your vector is contiguous; use a larger stride if picking out every k-th element.
            &cublas_beta,               // beta
            &omega[idx1],               // y vector
            1                           // stride. Normally 1 if your vector is contiguous; use a larger stride if picking out every k-th element.
        );

        // STEP-2: (it is wise to use the intermediate result of STEP-1.1) (out of order execution)
        // "alpha_{i} = nu_{i+1}^{T} * A * nu_{i+1}"

        // STEP-2.1: BLAS L1 cblas_ddot, vector-vector reduction operation. (input is double, accumulation is double, output is double)
        // alpha_{i} = nu_{i+1}^{T} * omega_{i}
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);                                   // important! set the right mode based on your input location.
        cublasDdot(
            handle,                     // cuBLAS handle  
            A_dim,                      // Number of elements  
            &omega[idx1],               // Pointer to first element of X  
            1,                          // Stride between elements of X  
            &nu[idx2],                  // Pointer to first element of Y  
            1,                          // Stride between elements of Y  
            &alpha[i]                   // result address
        );

        // STEP-1.2: BLAS L1 cblas_daxpy, vector-vector operation. (y := a*x + y)
        // omega_{i} = -beta_{i}*nu_{i} + omega_{i}
        cudaMemcpyFromSymbol(&cublas_alpha, &beta[i], sizeof(double), 0, cudaMemcpyDeviceToHost);   // copy a symbol variable from device to host.
        cublas_alpha = -1*cublas_alpha;                                                             // -beta_{i}
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);                                     // important! set the right mode based on your input location.
        cublasDaxpy(
            handle,                     // cuBLAS handle  
            A_dim,                      // length
            &cublas_alpha,              // alpha
            &nu[idx1],                  // x-vector in daxpy
            1,                          // stride. Normally 1 if your vector is contiguous; use a larger stride if picking out every k-th element.
            &omega[idx1],               // y-vector in daxpy
            1                           // stride. Normally 1 if your vector is contiguous; use a larger stride if picking out every k-th element.
        );

        // STEP-3:
        // "omega_{i} = omega_{i} - alpha_{i} * nu_{i+1}"

        // STEP-3.1: BLAS L1 cblas_daxpy, vector-vector operation. (y := a*x + y)
        cudaMemcpyFromSymbol(&cublas_alpha, &alpha[i], sizeof(double), 0, cudaMemcpyDeviceToHost);  // copy a symbol variable from device to host.
        cublas_alpha = -1*cublas_alpha;                                                             // - alpha_{i}
        cublasDaxpy(
            handle,                     // cuBLAS handle  
            A_dim,                      // length
            &cublas_alpha,              // alpha
            &nu[idx2],                  // x-vector in daxpy
            1,                          // stride. Normally 1 if your vector is contiguous; use a larger stride if picking out every k-th element.
            &omega[idx1],               // y-vector in daxpy
            1                           // stride. Normally 1 if your vector is contiguous; use a larger stride if picking out every k-th element.
        );

        // STEP-4:
        // "beta_{i+1} = || omega_{i} ||"

        // STEP-4.1: BLAS L1 cblas_dnrm2, computes the Euclidean norm of a vector. (res = ||x||)
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);                                   // important! set the right mode based on your input location.
        cublasDnrm2(
            handle,                     // cuBLAS handle    
            A_dim,                      // Number of elements  
            &omega[idx1],               // Pointer to the first element of X  
            1,                          // Stride between elements of X  
            &beta[i+1]                  // result address
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
