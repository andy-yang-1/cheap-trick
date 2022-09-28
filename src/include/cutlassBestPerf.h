#include "cutlass/gemm/device/gemm.h" 

using CutlassBestPerf = cutlass::gemm::device::Gemm<float,
                                                ColumnMajor,
                                                float,
                                                ColumnMajor,
                                                float,
                                                RowMajor,
                                                float
                                                ,                    
                                                cutlass::arch::OpClassSimt,
                                                cutlass::arch::Sm50,
                                                cutlass::gemm::GemmShape<128, 128, 8>,
                                                cutlass::gemm::GemmShape<32, 64, 8>,
                                                cutlass::gemm::GemmShape<1, 1, 1>
                                                ,
                                                cutlass::epilogue::thread::LinearCombination<
                                                  float,
                                                  1,
                                                  float,
                                                  float
                                                >
                                                // ,
                                                // cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
                                                // 2,
                                                // cutlass::arch::OpMultiplyAdd
                                                >;
                                                
 

void run_bestPerf(int M , int N , int K , 
                 float* A ,  float* B, float *C, float alpha , float beta){

    int lda = K;
    int ldb = K;
    int ldc = N;
    int ldd = N;

    CutlassBestPerf gemm_operator;
    CutlassBestPerf::Arguments args({M, N, K},      // Gemm Problem dimensions
                                {A, lda},     // source matrix A
                                {B, ldb},     // source matrix B
                                {C, ldc},     // source matrix C
                                {C, ldd},     // destination matrix D
                                {alpha, beta}); // alpha & beta
    gemm_operator(args); //运行Gemm

}