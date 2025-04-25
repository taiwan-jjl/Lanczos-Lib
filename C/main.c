#include "basic_lanczos.h"  // Include the header for the basic_lancaos function
#include <stdio.h>
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
    













    basic_lanczos();


    return 0;
}