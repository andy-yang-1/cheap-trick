#include "../include/gemm_header.h"
#include "../include/genGemm.h"

int main(){

    // argv: M N K

    float flops , max_err ;
    int round = 1 ;

    int M = 1009 , N = 1009 , K = 1009 ;
    // int M = 1024 , N = 1024 , K = 1024 ;

    cout << "M: " << M << endl << "N: " << N << endl << "K: " << K << endl ;

    float alpha = 1.0 , beta = 0;


    cout << "<-------------cutlass bestPerf-------------->" << endl ;

    max_err = get_dynamic_max_error(M,N,K,alpha,beta,run_cutlass,true);
    flops = get_dynamic_Gflops(round,M,N,K,alpha,beta,run_cutlass);
    cout << "max error: " << max_err << endl ;
    cout << "rounds: " << round << endl ;
    cout << "average gflops: " << flops << endl ;

    cout << "<--------------------genGemm------------------>" << endl ;

    max_err = get_dynamic_max_error(M,N,K,alpha,beta,run_paddinggemm,true);
    flops = get_dynamic_Gflops(round,M,N,K,alpha,beta,run_paddinggemm);
    // max_err = get_dynamic_max_error(M,N,K,alpha,beta,run_v4gemm,true);
    // flops = get_dynamic_Gflops(round,M,N,K,alpha,beta,run_v4gemm);
    cout << "max error: " << max_err << endl ;
    cout << "rounds: " << round << endl ;
    cout << "average gflops: " << flops << endl ;

    cout << "<-----------------end---------------->" << endl ;

}