#include "../gemm.cu"

void run_genGemm(int M , int N , int K , 
                float* A , float* B, float *C, float alpha , float beta){


        const int BM = 128, BN = 128, TM = 8, TN = 8;

        dim3 gridDim(10, 5);
        dim3 blockDim(10, 25);
        

        mm_kernel0<<<gridDim, blockDim>>>(A,B,C);
        // mm_kernel0<<<200,125>>>(A,B,C);


}