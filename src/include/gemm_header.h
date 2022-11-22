#include "cutlassGemm.h"
#include "cutlassBestPerf.h"
#include "v1Gemm.h"
#include "v2Gemm.h"
#include "v3Gemm.h"
#include "v4Gemm.h"
#include "v5Gemm.h"
#include "tvmGemm.h"


#include <iostream>
#include <cmath>

using std::cin ;
using std::cout ;
using std::endl ;


void generate_tensor_2D(float *ptr, int i_M, int i_N){        // 二维矩阵填充函数（此处全部填充1）
    for(int i = 0; i < i_M; i++){
        for(int j = 0; j < i_N; j++){
            float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
            *(ptr + i*i_N + j ) = r;
        }
    }
}

void generate_const_2D(float *ptr, int i_M,int i_N,int val){
    for(int i = 0; i < i_M; i++){
        for(int j = 0; j < i_N; j++){
            *(ptr + i*i_N + j ) = val;
        }
    }
}

float get_Gflops(int round, void(*kernel)(int  , int  , int  , 
                 float*  ,  float* , float *, float  , float )){
    
    int M = 1024;           //M
    int N = 1024;           //N
    int K = 1024;           //K

    float alpha = 1.0;      //alpha
    float beta = 0;       //beta

    float *A;               //申明A矩阵host端指针
    float *B;               //申明B矩阵host端指针
    float *C;               //申明C矩阵host端指针
    float *D;               //申明D矩阵host端指针


    size_t A_mem_size = sizeof(float) * M * K; //memory size of matrix A = M * K * sizeof(float)
    size_t B_mem_size = sizeof(float) * K * N; //memory size of matrix B = K * N * sizeof(float)
    size_t C_mem_size = sizeof(float) * M * N; //memory size of matrix C = M * N * sizeof(float)
    size_t D_mem_size = sizeof(float) * M * N; //memory size of matrix C = M * N * sizeof(float)
 
    A = (float*)malloc(A_mem_size);  // host端A矩阵分配内存
    B = (float*)malloc(B_mem_size);  // host端B矩阵分配内存
    C = (float*)malloc(C_mem_size);  // host端C矩阵分配内存
    D = (float*)malloc(D_mem_size);  // host端D矩阵分配内存

    float *d_A;            // 申明device端A矩阵的指针
    float *d_B;            // 申明device端B矩阵的指针
    float *d_C;            // 申明device端C矩阵的指针
    float *d_D;            // 申明device端D矩阵的指针

    cudaMalloc((void**)&d_A, A_mem_size);  // device端为A矩阵分配内存
    cudaMalloc((void**)&d_B, B_mem_size);  // device端为B矩阵分配内存
    cudaMalloc((void**)&d_C, C_mem_size);  // device端为C矩阵分配内存
    cudaMalloc((void**)&d_D, D_mem_size);  // device端为D矩阵分配内存

    generate_tensor_2D(A, M, K);     // 填充A矩阵
    generate_tensor_2D(B, K, N);     // 填充B矩阵  
    generate_tensor_2D(C, M, N);     // 填充C矩阵

    cudaMemcpy(d_A, A, A_mem_size, cudaMemcpyHostToDevice); // 将矩阵A的数据传递到device端
    cudaMemcpy(d_B, B, B_mem_size, cudaMemcpyHostToDevice); // 将矩阵B的数据传递到device端
    cudaMemcpy(d_C, C, C_mem_size, cudaMemcpyHostToDevice); // 将矩阵C的数据传递到device端


    cudaEvent_t start , stop ; // declare time stamp
    cudaEventCreate(&start) ; // start to record
    cudaEventCreate(&stop) ;
    cudaEventRecord(start,0) ; // record the time

    for (int i = 0 ;  i < round ; i++){


        kernel(M,N,K,d_A,d_B,d_C,alpha,beta) ;


    }    

    cudaEventRecord(stop,0) ;
    cudaEventSynchronize(stop) ; // synchronize time stamp

    float whole_time , flops ;

    cudaEventElapsedTime(&whole_time,start,stop) ; // calculate the time
    cudaEventDestroy(start) ; // destroy the events 
    cudaEventDestroy(stop) ;

    flops = 2.0 * M * N * K / 1024 / 1024 / 1024 / whole_time * 1000 * round ;
    // flops = (2.0 * M * N * K + 2 * M * N) / 1000 / 1000 / 1000 / whole_time * 1000 * round ;

    return flops ;


}


// compare with cutlass
float get_max_error(void(*kernel)(int  , int  , int  , 
                 float*  ,  float* , float *, float  , float ), bool isRowMajor){

    int M = 1024;           //M
    int N = 1024;           //N
    int K = 1024;           //K

    float alpha = 1.0;      //alpha
    float beta = 0;       //beta

    float *A;               //申明A矩阵host端指针
    float *B;               //申明B矩阵host端指针
    float *C;               //申明C矩阵host端指针
    float *D;               //申明D矩阵host端指针


    size_t A_mem_size = sizeof(float) * M * K; //memory size of matrix A = M * K * sizeof(float)
    size_t B_mem_size = sizeof(float) * K * N; //memory size of matrix B = K * N * sizeof(float)
    size_t C_mem_size = sizeof(float) * M * N; //memory size of matrix C = M * N * sizeof(float)
    size_t D_mem_size = sizeof(float) * M * N; //memory size of matrix C = M * N * sizeof(float)
 
    A = (float*)malloc(A_mem_size);  // host端A矩阵分配内存
    B = (float*)malloc(B_mem_size);  // host端B矩阵分配内存
    C = (float*)malloc(C_mem_size);  // host端C矩阵分配内存
    D = (float*)malloc(D_mem_size);  // host端D矩阵分配内存

    float *d_A;            // 申明device端A矩阵的指针
    float *d_B;            // 申明device端B矩阵的指针
    float *d_C;            // 申明device端C矩阵的指针
    float *d_D;            // 申明device端D矩阵的指针


    cudaMalloc((void**)&d_A, A_mem_size);  // device端为A矩阵分配内存
    cudaMalloc((void**)&d_B, B_mem_size);  // device端为B矩阵分配内存
    cudaMalloc((void**)&d_C, C_mem_size);  // device端为C矩阵分配内存
    cudaMalloc((void**)&d_D, D_mem_size);  // device端为D矩阵分配内存


    generate_tensor_2D(A, M, K);     // 填充A矩阵
    generate_tensor_2D(B, K, N);     // 填充B矩阵  
    // generate_const_2D(A,M,K,1) ;
    // generate_const_2D(B,K,N,1) ;


    cudaMemcpy(d_A, A, A_mem_size, cudaMemcpyHostToDevice); // 将矩阵A的数据传递到device端
    cudaMemcpy(d_B, B, B_mem_size, cudaMemcpyHostToDevice); // 将矩阵B的数据传递到device端
    

    int lda = K;
    int ldb = K;
    int ldc = N;
    int ldd = N;



    if (isRowMajor){
        run_cutlass(M,N,K,d_A,d_B,d_C,alpha,beta) ;
    }else{
        run_bestPerf(M,N,K,d_A,d_B,d_C,alpha,beta) ;
    }
    


    kernel(M,N,K,d_A,d_B,d_D,alpha,beta) ;

    cudaMemcpy(D,d_D,D_mem_size,cudaMemcpyDeviceToHost) ;
    cudaMemcpy(C,d_C,C_mem_size,cudaMemcpyDeviceToHost) ;

    float max_err = 0 ;

    for (int i = 0 ; i < M * N ; i++){

        max_err = abs(D[i]-C[i]) > max_err ? abs(D[i]-C[i]) : max_err ;
    }

    // //  debug
    // cout << "debug: C[0]: "  << C[0] << " C[1024]: " << C[1024] << endl ;
    // cout << "debug: D[0]: "  << D[0] << " D[1024]: " << D[1024] << endl ;
    
    return max_err ;
}


float get_dynamic_Gflops(int round, int M, int N, int K, float alpha, float beta, void(*kernel)(int  , int  , int  , 
                 float*  ,  float* , float *, float  , float )){

    float *A;               //申明A矩阵host端指针
    float *B;               //申明B矩阵host端指针
    float *C;               //申明C矩阵host端指针
    float *D;               //申明D矩阵host端指针


    size_t A_mem_size = sizeof(float) * M * K; //memory size of matrix A = M * K * sizeof(float)
    size_t B_mem_size = sizeof(float) * K * N; //memory size of matrix B = K * N * sizeof(float)
    size_t C_mem_size = sizeof(float) * M * N; //memory size of matrix C = M * N * sizeof(float)
    size_t D_mem_size = sizeof(float) * M * N; //memory size of matrix C = M * N * sizeof(float)
 
    A = (float*)malloc(A_mem_size);  // host端A矩阵分配内存
    B = (float*)malloc(B_mem_size);  // host端B矩阵分配内存
    C = (float*)malloc(C_mem_size);  // host端C矩阵分配内存
    D = (float*)malloc(D_mem_size);  // host端D矩阵分配内存

    float *d_A;            // 申明device端A矩阵的指针
    float *d_B;            // 申明device端B矩阵的指针
    float *d_C;            // 申明device端C矩阵的指针
    float *d_D;            // 申明device端D矩阵的指针

    cudaMalloc((void**)&d_A, A_mem_size);  // device端为A矩阵分配内存
    cudaMalloc((void**)&d_B, B_mem_size);  // device端为B矩阵分配内存
    cudaMalloc((void**)&d_C, C_mem_size);  // device端为C矩阵分配内存
    cudaMalloc((void**)&d_D, D_mem_size);  // device端为D矩阵分配内存

    generate_tensor_2D(A, M, K);     // 填充A矩阵
    generate_tensor_2D(B, K, N);     // 填充B矩阵  
    generate_tensor_2D(C, M, N);     // 填充C矩阵

    cudaMemcpy(d_A, A, A_mem_size, cudaMemcpyHostToDevice); // 将矩阵A的数据传递到device端
    cudaMemcpy(d_B, B, B_mem_size, cudaMemcpyHostToDevice); // 将矩阵B的数据传递到device端
    cudaMemcpy(d_C, C, C_mem_size, cudaMemcpyHostToDevice); // 将矩阵C的数据传递到device端


    cudaEvent_t start , stop ; // declare time stamp
    cudaEventCreate(&start) ; // start to record
    cudaEventCreate(&stop) ;
    cudaEventRecord(start,0) ; // record the time

    for (int i = 0 ;  i < round ; i++){


        kernel(M,N,K,d_A,d_B,d_C,alpha,beta) ;


    }    

    cudaEventRecord(stop,0) ;
    cudaEventSynchronize(stop) ; // synchronize time stamp

    float whole_time , flops ;

    cudaEventElapsedTime(&whole_time,start,stop) ; // calculate the time
    cudaEventDestroy(start) ; // destroy the events 
    cudaEventDestroy(stop) ;

    flops = 2.0 * M * N * K / 1024 / 1024 / 1024 / whole_time * 1000 * round ;
    // flops = (2.0 * M * N * K + 2 * M * N) / 1000 / 1000 / 1000 / whole_time * 1000 * round ;

    return flops ;


}

// compare with cutlass
float get_dynamic_max_error(int M, int N, int K, float alpha, float beta, void(*kernel)(int  , int  , int  , 
                 float*  ,  float* , float *, float  , float ), bool isRowMajor){

    // int M = 1024;           //M
    // int N = 1024;           //N
    // int K = 1024;           //K

    // float alpha = 1.0;      //alpha
    // float beta = 0;       //beta

    float *A;               //申明A矩阵host端指针
    float *B;               //申明B矩阵host端指针
    float *C;               //申明C矩阵host端指针
    float *D;               //申明D矩阵host端指针


    size_t A_mem_size = sizeof(float) * M * K; //memory size of matrix A = M * K * sizeof(float)
    size_t B_mem_size = sizeof(float) * K * N; //memory size of matrix B = K * N * sizeof(float)
    size_t C_mem_size = sizeof(float) * M * N; //memory size of matrix C = M * N * sizeof(float)
    size_t D_mem_size = sizeof(float) * M * N; //memory size of matrix C = M * N * sizeof(float)
 
    A = (float*)malloc(A_mem_size);  // host端A矩阵分配内存
    B = (float*)malloc(B_mem_size);  // host端B矩阵分配内存
    C = (float*)malloc(C_mem_size);  // host端C矩阵分配内存
    D = (float*)malloc(D_mem_size);  // host端D矩阵分配内存

    float *d_A;            // 申明device端A矩阵的指针
    float *d_B;            // 申明device端B矩阵的指针
    float *d_C;            // 申明device端C矩阵的指针
    float *d_D;            // 申明device端D矩阵的指针


    cudaMalloc((void**)&d_A, A_mem_size);  // device端为A矩阵分配内存
    cudaMalloc((void**)&d_B, B_mem_size);  // device端为B矩阵分配内存
    cudaMalloc((void**)&d_C, C_mem_size);  // device端为C矩阵分配内存
    cudaMalloc((void**)&d_D, D_mem_size);  // device端为D矩阵分配内存


    generate_tensor_2D(A, M, K);     // 填充A矩阵
    generate_tensor_2D(B, K, N);     // 填充B矩阵  
    // generate_const_2D(A,M,K,1) ;
    // generate_const_2D(B,K,N,1) ;


    cudaMemcpy(d_A, A, A_mem_size, cudaMemcpyHostToDevice); // 将矩阵A的数据传递到device端
    cudaMemcpy(d_B, B, B_mem_size, cudaMemcpyHostToDevice); // 将矩阵B的数据传递到device端
    

    int lda = K;
    int ldb = K;
    int ldc = N;
    int ldd = N;



    if (isRowMajor){
        run_cutlass(M,N,K,d_A,d_B,d_C,alpha,beta) ;
    }else{
        run_bestPerf(M,N,K,d_A,d_B,d_C,alpha,beta) ;
    }
    


    kernel(M,N,K,d_A,d_B,d_D,alpha,beta) ;

    cudaMemcpy(D,d_D,D_mem_size,cudaMemcpyDeviceToHost) ;
    cudaMemcpy(C,d_C,C_mem_size,cudaMemcpyDeviceToHost) ;

    float max_err = 0 ;

    for (int i = 0 ; i < M * N ; i++){

        max_err = abs(D[i]-C[i]) > max_err ? abs(D[i]-C[i]) : max_err ;
    }

    // //  debug
    // cout << "debug: C[0]: "  << C[0] << " C[1024]: " << C[1024] << endl ;
    // cout << "debug: D[0]: "  << D[0] << " D[1024]: " << D[1024] << endl ;
    
    return max_err ;
}