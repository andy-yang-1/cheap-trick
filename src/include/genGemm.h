#include "../gemm.cu"

void run_genGemm(int M , int N , int K , 
                float* A , float* B, float *C, float alpha , float beta){


        const int BM = 128, BN = 128, TM = 8, TN = 8;
        dim3 blockDim(32, 16);
        dim3 gridDim(16, 32);

        mm_kernel0<<<gridDim, blockDim>>>(A,B,C);
        // tvmGenerateV5<<<64,256>>>(A,B,C);


}