#include "basic_lanczos.h"  // Include the header for the basic_lancaos function
#include <stdio.h>
#include <string.h>
#include "mkl.h"
#include "helper.h"         // Include the helper functions




int main(void) {

    //########## START Intel MKL Conditional Numerical Reproducibility Control #####
    printf("\nSTART Intel MKL Conditional Numerical Reproducibility Control\n");
    int current = mkl_cbwr_get(MKL_CBWR_ALL );
    int suggested = mkl_cbwr_get_auto_branch();

    print_cbwr_status("Current CBWR", current);
    print_cbwr_status("Suggested CBWR", suggested);
    mkl_cbwr_set (suggested);
    current = mkl_cbwr_get(MKL_CBWR_ALL );
    print_cbwr_status("Current CBWR", current);
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



    // void cblas_dgemv (const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE trans, const MKL_INT m, const MKL_INT n, const double alpha, const double *a, const MKL_INT lda, const double *x, const MKL_INT incx, const double beta, double *y, const MKL_INT incy);









    basic_lanczos();




    //########## free memory ##########
    mkl_free(A);

    mkl_finalize();

    return 0;
}