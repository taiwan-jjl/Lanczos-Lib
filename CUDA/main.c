#include "basic_lanczos.h"  // Include the header for the basic_lancaos function.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>          // To get machine precision for double.
#include "helper.h"         // Include the helper functions.




int main(void) {

    //########## START init values ##########

    // Lanczos algorithm requires: A(input matrix), omega(pre-Lanczos vector), alpha(tridia-matrix), beta(tridia-matrix), nu(Lanczos vector),  

    // Small dense matrix example: A
    int A_dim = 3;                                                          // A matrix dim = 3 .
    int A_ent = A_dim*A_dim;                                                // A elements = 9 .
    double *A = (double*) calloc(A_ent, sizeof(double));                    // Not use "aligned_alloc" in C11 this time.
    double A_vals[9] = {4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0};       // a workaround to fastly hand put-in a matrix.
    memcpy(A, A_vals, sizeof(A_vals));                                      // init "A" matrix via "memcpy" from "A_vals" matrix.
    print_array_float("A", A, 0, A_ent);                                    // helper fun, verify "A" matrix.

    /*
    Put future large size input here and comment out the small A section.
    */

    // initial vector: nu
    // whole nu vectors: nu 1d array
    double *nu = (double*) calloc(A_dim*(A_dim+2), sizeof(double));
    // initial vector: omega
    // whole omega vectors: omega 1d array
    double *omega = (double*) calloc(A_dim*(A_dim), sizeof(double));
    // initial element: alpha
    // whole alpha elements: alpha 1d array
    double *alpha = (double*) calloc(A_dim, sizeof(double));
    // initial element: beta
    // whole beta elements: beta 1d array
    double *beta = (double*) calloc(A_dim+1, sizeof(double));

    // Simple start vector nu(0)=[0,0,0] and nu(1)=[1,0,0]
    double nu_vals[3] = {1.0, 0.0, 0.0};
    memcpy(&nu[3], nu_vals, sizeof(nu_vals));                               // init "nu(1)" vector via "memcpy" from "nu_vals" vector.
    print_array_float("nu(0)", nu, 0, A_dim);                               // helper fun, verify "nu(0)" vector.
    print_array_float("nu(1)", nu, A_dim, (A_dim*2));                       // helper fun, verify "nu(1)" vector.

    // Set Lanczos iteration stop criterion. To avoid "devide by zero" and numerical instability.
    printf("\nMachine epsilon for double: %.20e\n", DBL_EPSILON);           // print out the machine epsilon for double on current environment.
    const double Lanczos_stop_crit = 10.0*DBL_EPSILON;                      // set stop criterion = 10X "DBL_EPSILON".
    printf("Lanczos iteration stop criterion: %.20e\n", Lanczos_stop_crit); // print out the Lanczos iteration stop criterion.

    // Set Lanczos stop criterion check frequency. It is a balance between performance and criterion check.
    const int Lanczos_stop_check_freq = 0;                                  // 0 = check every loop. 1 = check every 2 loops.

    // helper variable "int Lanczos_iter": how many iterations executed
    int Lanczos_iter = 0;

    //########## END init values ##########

    //########## START GPU memory allocation and copy ##########

    //########## END GPU memory allocation and copy ##########


    // Run Lanczos algorithm.
    basic_lanczos(A, nu, omega, alpha, beta, A_dim, Lanczos_stop_crit, Lanczos_stop_check_freq, &Lanczos_iter);


    //########## START verification ##########

    printf("Lanczos completed in %d iterations.\n", Lanczos_iter);          // Check Lanczos iteration completed number.
    print_array_float("omega", omega, 0, A_ent);                            // verify "omega".
    print_array_float("alpha", alpha, 0, A_dim);                            // verify "alpha".
    print_array_float("beta", beta, 0, A_dim+1);                            // verify "beta".
    print_array_float("nu", nu, 0, A_dim*(A_dim+2));                        // verify "nu".

    //########## END verification ##########


    //########## free memory ##########
    free(A);
    free(nu);
    free(omega);
    free(alpha);
    free(beta);


    return 0;
}

/* notes:

    for (int i = 0; i < A_ent; ++i) {                                       // verify A matrix
        printf("A[%d] = %.1f\n", i, A[i]);
    }


    double *x = (double*) mkl_calloc(A_dim, sizeof(double), 64);            // blas example section
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

    mkl_free(x);
    mkl_free(y);


    // meeting note: A(in real space) must be "symmetric" or Lanczos will break down. So, gemv or semv are noth fine. just fix to one kind. 




*/