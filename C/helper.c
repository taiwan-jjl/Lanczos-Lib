#include <stdio.h>
#include "helper.h"     // Include the declaration
#include "mkl.h"

void print_cbwr_status(const char *label, int status) {
    printf("%s: ", label);
    switch (status) {
        case MKL_CBWR_OFF:               printf("OFF (Disable CNR mode)\n"); break;
        case MKL_CBWR_BRANCH_OFF:        printf("BRANCH_OFF (CNR mode is disabled)\n"); break;
        case MKL_CBWR_AUTO:              printf("AUTO (Choose branch automatically)\n"); break;
        case MKL_CBWR_COMPATIBLE:        printf("COMPATIBLE (Intel SSE2 without rcpps/rsqrtps)\n"); break;
        case MKL_CBWR_SSE2:              printf("SSE2 (Intel SSE2)\n"); break;
        case MKL_CBWR_SSE3:              printf("SSE3 (Deprecated; equivalent to SSE2)\n"); break;
        case MKL_CBWR_SSSE3:             printf("SSSE3 (Supplemental SSE3)\n"); break;
        case MKL_CBWR_SSE4_1:            printf("SSE4.1\n"); break;
        case MKL_CBWR_SSE4_2:            printf("SSE4.2\n"); break;
        case MKL_CBWR_AVX:               printf("AVX\n"); break;
        case MKL_CBWR_AVX2:              printf("AVX2\n"); break;
        case MKL_CBWR_AVX512_MIC:        printf("AVX512_MIC (Deprecated; equivalent to AVX2)\n"); break;
        case MKL_CBWR_AVX512:            printf("AVX512\n"); break;
        case MKL_CBWR_AVX512_MIC_E1:     printf("AVX512_MIC_E1 (Deprecated; equivalent to AVX2)\n"); break;
        case MKL_CBWR_AVX512_E1:         printf("AVX512_E1 (AVX-512 with VNNI)\n"); break;
        default:
            if (status & MKL_CBWR_STRICT) printf("STRICT mode enabled\n");
            else                          printf("UNKNOWN (0x%X)\n", status);
            break;
    }
}
