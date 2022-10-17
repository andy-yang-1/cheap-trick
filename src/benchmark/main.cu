#include "../include/gemm_header.h"



int main(){

    float flops , max_err ;
    int round = 100 ;

    // test cutlass
    cout << "<--------------cutlass default--------------->" << endl ;

    flops = get_Gflops(round,run_cutlass) ;
    max_err = get_max_error(run_cutlass,true) ;
    cout << "max error: " << max_err << endl ;
    cout << "rounds: " << round << endl ;
    cout << "average gflops: " << flops << endl ;

    cout << "<-------------cutlass bestPerf-------------->" << endl ;

    flops = get_Gflops(round,run_bestPerf) ;
    max_err = get_max_error(run_bestPerf,false) ;
    cout << "max error: " << max_err << endl ;
    cout << "rounds: " << round << endl ;
    cout << "average gflops: " << flops << endl ;

    cout << "<---------------------tvm-------------------->" << endl ;

    flops = get_Gflops(round,run_tvm) ;
    max_err = get_max_error(run_tvm,true) ;
    cout << "max error: " << max_err << endl ;
    cout << "rounds: " << round << endl ;
    cout << "average gflops: " << flops << endl ;


    cout << "<--------------------v1gemm------------------>" << endl ;

    flops = get_Gflops(round,run_v1gemm) ;
    max_err = get_max_error(run_v1gemm,true) ;
    cout << "max error: " << max_err << endl ;
    cout << "rounds: " << round << endl ;
    cout << "average gflops: " << flops << endl ;

    cout << "<--------------------v2gemm------------------>" << endl ;

    flops = get_Gflops(round,run_v2gemm) ;
    max_err = get_max_error(run_v2gemm,true) ;
    cout << "max error: " << max_err << endl ;
    cout << "rounds: " << round << endl ;
    cout << "average gflops: " << flops << endl ;

    cout << "<--------------------v3gemm------------------>" << endl ;

    flops = get_Gflops(round,run_v3gemm) ;
    max_err = get_max_error(run_v3gemm,true) ;
    cout << "max error: " << max_err << endl ;
    cout << "rounds: " << round << endl ;
    cout << "average gflops: " << flops << endl ;

    cout << "<--------------------v4gemm------------------>" << endl ;

    flops = get_Gflops(round,run_v4gemm) ;
    max_err = get_max_error(run_v4gemm,true) ;
    cout << "max error: " << max_err << endl ;
    cout << "rounds: " << round << endl ;
    cout << "average gflops: " << flops << endl ;

    cout << "<-----------------end---------------->" << endl ;

    // test err

    // cout << "max error: " << get_max_error(run_v1gemm) << endl ; 
    // get_max_error(run_cutlass) ;

}