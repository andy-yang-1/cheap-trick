#include "../include/gemm_header.h"
#include "../include/cutlassMultiStage.h"

int main(){
    // // cout << "hello world" << endl ;
    // // cout << "max error: " << get_max_error(run_v4gemm,false) << endl  ;
    // cout << "max error: " << get_max_error(run_v4gemm,true) << endl  ;
    cout << "gflops: " << get_Gflops(100,run_v5gemm) << endl  ;
    cout << "error rate: " << get_max_error(run_v5gemm,true) << endl ;

    // cout << "gflops: " << get_Gflops(100,run_cutlassMultiStage) << endl  ;
    // cout << "error rate: " << get_max_error(run_cutlassMultiStage,false) << endl ;
    // cout << "max error: " << get_max_error(run_v4gemm,true) << endl  ;

    // int M = 1024;           //M
    // int N = 1024;           //N
    // int K = 1024;           //K

    // float alpha = 10;      //alpha
    // float beta = 10.0;       //beta

    // float *A;               //申明A矩阵host端指针
    // float *B;               //申明B矩阵host端指针
    // float *C;               //申明C矩阵host端指针
    // float *D;               //申明D矩阵host端指针


    // size_t A_mem_size = sizeof(float) * M * K; //memory size of matrix A = M * K * sizeof(float)
    // size_t B_mem_size = sizeof(float) * K * N; //memory size of matrix B = K * N * sizeof(float)
    // size_t C_mem_size = sizeof(float) * M * N; //memory size of matrix C = M * N * sizeof(float)
    // size_t D_mem_size = sizeof(float) * M * N; //memory size of matrix C = M * N * sizeof(float)
 
    // A = (float*)malloc(A_mem_size);  // host端A矩阵分配内存
    // B = (float*)malloc(B_mem_size);  // host端B矩阵分配内存
    // C = (float*)malloc(C_mem_size);  // host端C矩阵分配内存
    // D = (float*)malloc(D_mem_size);  // host端D矩阵分配内存

    // float *d_A;            // 申明device端A矩阵的指针
    // float *d_B;            // 申明device端B矩阵的指针
    // float *d_C;            // 申明device端C矩阵的指针
    // float *d_D;            // 申明device端D矩阵的指针
    // // float *d_zero ;          // 申明device端全0矩阵

    // cudaMalloc((void**)&d_A, A_mem_size);  // device端为A矩阵分配内存
    // cudaMalloc((void**)&d_B, B_mem_size);  // device端为B矩阵分配内存
    // cudaMalloc((void**)&d_C, C_mem_size);  // device端为C矩阵分配内存
    // cudaMalloc((void**)&d_D, D_mem_size);  // device端为D矩阵分配内存

    // // generate_tensor_2D(A, M, K);     // 填充A矩阵
    // // generate_tensor_2D(B, K, N);     // 填充B矩阵  
    // generate_const_2D(A,M,K,1) ;
    // generate_const_2D(B,K,N,1) ;

    // cudaMemcpy(d_A, A, A_mem_size, cudaMemcpyHostToDevice); // 将矩阵A的数据传递到device端
    // cudaMemcpy(d_B, B, B_mem_size, cudaMemcpyHostToDevice); // 将矩阵B的数据传递到device端
    // // cudaMemcpy(d_zero, zero, D_mem_size, cudaMemcpyHostToDevice); // 将矩阵C的数据传递到device端

    // // v4 change A & B

   
    // run_cutlass(M,N,K,d_A,d_B,d_D,alpha,beta) ;
    // run_v4gemm(M,N,K,d_A,d_B,d_C,alpha,beta) ;


    // cudaMemcpy(C,d_C,C_mem_size,cudaMemcpyDeviceToHost) ;
    // cudaMemcpy(D,d_D,D_mem_size,cudaMemcpyDeviceToHost) ;


    // float max_err = 0 ;

    // for (int i = 0 ; i < M * N ; i++){
    //     if (abs(C[i]-D[i])>3){
    //         // cout << "i: " << i << " C[i] = " << C[i] << endl ;
    //         // cout << "i: " << i << " D[i] = " << D[i] << endl ;
            
    //         // break ;
    //     }
    //     max_err = max_err > abs(C[i]-D[i]) ? max_err : abs(C[i]-D[i]) ;
            
    // }

    // cout << "max error: " << max_err << endl ;
    // cout << "v4: " <<endl ;
    // cout << C[0] << endl << C[1024] << endl ;
    // cout << "cutlass: " << endl ;
    // cout << D[0] << endl << D[1024] << endl ;

}