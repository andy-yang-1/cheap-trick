#include "../include/gemm_header.h"
#include "../include/genGemm.h"

int main(){

    // argv: M N K

    float flops , max_err ;
    int round = 1 ;

    int M = 1000 , N = 2000 , K = 3000 ;
    // int M = 2000 , N = 2024 , K = 2024 ;

    cout << "M: " << M << endl << "N: " << N << endl << "K: " << K << endl ;

    float alpha = 1.0 , beta = 0;


    cout << "<-------------cutlass bestPerf-------------->" << endl ;

    max_err = get_dynamic_max_error(M,N,K,alpha,beta,run_cutlass,true);
    flops = get_dynamic_Gflops(round,M,N,K,alpha,beta,run_cutlass);
    cout << "max error: " << max_err << endl ;
    cout << "rounds: " << round << endl ;
    cout << "average gflops: " << flops << endl ;

    cout << "<--------------------genGemm------------------>" << endl ;

    max_err = get_dynamic_max_error(M,N,K,alpha,beta,run_genGemm,true);
    flops = get_dynamic_Gflops(round,M,N,K,alpha,beta,run_genGemm);
    cout << "max error: " << max_err << endl ;
    cout << "rounds: " << round << endl ;
    cout << "average gflops: " << flops << endl ;

    cout << "<-----------------end---------------->" << endl ;

}