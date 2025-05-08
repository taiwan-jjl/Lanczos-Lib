#include "basic_lanczos.h"  // Include the header for the basic_lancaos function
#include <stdio.h>
#include <string.h>
#include "mkl.h"
#include "helper.h"         // Include the helper functions




int main(void) {

    //########## START Intel MKL Conditional Numerical Reproducibility Control #####
    printf("\nSTART Intel MKL Conditional Numerical Reproducibility Control\n");
    int current = mkl_cbwr_get(MKL_CBWR_ALL );              // detect current CNR status.
    int suggested = mkl_cbwr_get_auto_branch();             // detect suggested CNR status.

    print_cbwr_status("Current CBWR", current);             // helper fun.
    print_cbwr_status("Suggested CBWR", suggested);         // helper fun.
    mkl_cbwr_set (suggested);                               // set to suggested CNR status
    current = mkl_cbwr_get(MKL_CBWR_ALL );                  // check CNR change succeeded.
    print_cbwr_status("Current CBWR", current);             // helper fun.
    printf("END Intel MKL Conditional Numerical Reproducibility Control\n\n");
    //########## END Intel MKL Conditional Numerical Reproducibility Control #####
    

    //########## init values ##########
    int A_dim = 3;
    int A_ent = A_dim*A_dim;
    double *A = (double*) mkl_calloc(A_ent, sizeof(double), 64);
    double vals[9] = {4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0};
    memcpy(A, vals, sizeof(vals));

    for (int i = 0; i < A_ent; ++i) {
        printf("A[%d] = %.1f\n", i, A[i]);
    }

    double *x = (double*) mkl_calloc(A_dim, sizeof(double), 64);
    double *y = (double*) mkl_calloc(A_dim, sizeof(double), 64);
    double vals2[3] = {1.0, 0,0, 0,0};
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