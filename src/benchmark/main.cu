#include "../include/gemm_header.h"



int main(){

    float flops ;
    int round = 100 ;

    // test cutlass
    cout << "<--------------cutlass--------------->" << endl ;

    flops = get_Gflops(round,run_cutlass) ;
    cout << "rounds: " << round << endl ;
    cout << "average gflops: " << flops << endl ;

    cout << "<-----------------tvm---------------->" << endl ;

    flops = get_Gflops(round,run_tvm) ;
    cout << "rounds: " << round << endl ;
    cout << "average gflops: " << flops << endl ;


    cout << "<----------------v1gemm-------------->" << endl ;

    flops = get_Gflops(round,run_v1gemm) ;
    cout << "rounds: " << round << endl ;
    cout << "average gflops: " << flops << endl ;

    cout << "<----------------v2gemm-------------->" << endl ;

    flops = get_Gflops(round,run_v2gemm) ;
    cout << "rounds: " << round << endl ;
    cout << "average gflops: " << flops << endl ;

    cout << "<-----------------end---------------->" << endl ;

    // test err

    cout << "max error: " << get_max_error(run_v1gemm) << endl ; 
    // get_max_error(run_cutlass) ;

}