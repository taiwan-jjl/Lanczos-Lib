#include "basic_lanczos.h"  // Include the header for the basic_lancaos function.
#include <stdio.h>
#include <string.h>
#include "mkl.h"
#include "helper.h"         // Include the helper functions.




int main(void) {

    //########## START Intel MKL Conditional Numerical Reproducibility Control #####
    printf("\nSTART Intel MKL Conditional Numerical Reproducibility Control\n");
    int current = mkl_cbwr_get(MKL_CBWR_ALL );              // detect current CNR status.
    int suggested = mkl_cbwr_get_auto_branch();             // detect suggested CNR status.

    print_cbwr_status("Current CBWR", current);             // helper fun.
    print_cbwr_status("Suggested CBWR", suggested);         // helper fun.
    mkl_cbwr_set (suggested);                               // set to suggested CNR status.
    current = mkl_cbwr_get(MKL_CBWR_ALL );                  // check CNR change succeeded.
    print_cbwr_status("Current CBWR", current);             // helper fun.
    printf("END Intel MKL Conditional Numerical Reproducibility Control\n\n");
    //########## END Intel MKL Conditional Numerical Reproducibility Control #####
    

    //########## init values ##########

    // Lanczos algorithm requires: A, omega, alpha, beta, nu,  

    // small dense matrix example: A
    int A_dim = 3;                                                          // A matrix dim = 3 .
    int A_ent = A_dim*A_dim;                                                // A elements = 9 .
    double *A = (double*) mkl_calloc(A_ent, sizeof(double), 64);            // use "mkl_calloc" and aligned to 64 bytes for AVX-512 .
    double vals[9] = {4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0};         // a workaround to fastly hand put-in a matrix.
    memcpy(A, vals, sizeof(vals));                                          // init "A" matrix via "memcpy" from "val" matrix.
    print_array_float("A", A, 0, A_ent);                                    // helper fun, verify "A" matrix.

    // initial vector: nu(0)
    // whole nu vectors: nu 1d array
    double *nu = (double*) mkl_calloc(A_dim*(A_dim+2), sizeof(double), 64);


    // meeting note: A must be "symmetric" or Lanczos will break down. So, gemv or semv are noth fine. just fix to one kind. 






    double *x = (double*) mkl_calloc(A_dim, sizeof(double), 64);
    double *y = (double*) mkl_calloc(A_dim, sizeof(double), 64);
    double vals2[3] = {1.0, 0.0, 0.0};
    memcpy(x, vals2, sizeof(vals2));
    double alpha = 1.0;
    double beta = 0.0;

    cblas_dgemv(CblasRowMajor,   // Matrix layout
                CblasNoTrans,    // No transpose
                A_dim, A_dim,    // Dimensions of A
                alpha,           // alpha
                A, A_dim,        // A and leading dimension (lda = A_dim)
                x, 1,            // x and incx
                beta,            // beta
                y, 1);           // y and incy

    printf("\ntest output\n");
    for (int i = 0; i < A_dim; ++i) {
        printf("A[%d] = %.1f\n", i, y[i]);
    }







    basic_lanczos();




    //########## free memory ##########
    mkl_free(A);
    mkl_free(x);
    mkl_free(y);

    mkl_finalize();

    return 0;
}

/* notes:

    for (int i = 0; i < A_ent; ++i) {                                       // verify A matrix
        printf("A[%d] = %.1f\n", i, A[i]);
    }



*/