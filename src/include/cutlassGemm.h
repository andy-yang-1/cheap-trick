#include "cutlass/gemm/device/gemm.h"  

using ColumnMajor = cutlass::layout::ColumnMajor;             // 列主序存储方式
using RowMajor    = cutlass::layout::RowMajor;                // 行主序存储方式
 
using CutlassGemm = cutlass::gemm::device::Gemm<float,        // A矩阵数据类型
                                                RowMajor,     // A矩阵存储方式
                                                float,        // B矩阵数据类型
                                                RowMajor,     // B矩阵存储方式
                                                float,        // C矩阵数据类型
                                                RowMajor>;

void run_cutlass(int M , int N , int K , 
                const float* A , const float* B, float *C, float alpha , float beta){

    int lda = K;
    int ldb = K;
    int ldc = N;
    int ldd = N;

    CutlassGemm gemm_operator;
    CutlassGemm::Arguments args({M, N, K},      // Gemm Problem dimensions
                                {A, lda},     // source matrix A
                                {B, ldb},     // source matrix B
                                {C, ldc},     // source matrix C
                                {C, ldd},     // destination matrix D
                                {alpha, beta}); // alpha & beta
    gemm_operator(args); //运行Gemm

}