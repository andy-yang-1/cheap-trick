

__global__ void paddingKernel(float * __restrict__ a, float * __restrict__ b, float * __restrict__ c){

    // M = N = K = 1009

    const int BM = 128;
    const int BN = 128;
    const int BK = 128;
    const int TK = 8;
    const int TM = 8;
    const int TN = 8;

    // Allocate shared memory for A and B 
    __shared__ float s_a[1024];
    __shared__ float s_b[1024];

    // Allocate registers for C
    float r_c[8][8] = {0.0f};    

        // flag: 1 -> blockIdx.y == 8 && blockIdx.x != 8 ; 2 -> blockIdx.y != 8 && blockIdx.x == 8 ; 3 -> blockIdx.y == 8 && blockIdx.x == 8 ; 0 -> others
        int flag = 0;
        flag = blockIdx.y == 7 || blockIdx.x == 7 ;

        // flag = 1 ;

        int tid = threadIdx.y * 16 + threadIdx.x;
        int load_a_smem_m = tid / 2;
        int load_a_smem_k = (tid & 1) << 2;
        int load_b_smem_k = tid / 32;
        int load_b_smem_n = (tid & 31) << 2;

        int load_a_gmem_m = blockIdx.y * 128 + load_a_smem_m;
        int load_b_gmem_n = blockIdx.x * 128 + load_b_smem_n;

        
        // flag == 0
        if (flag == 0){
            // 1009 / 8 = 126
            for ( int bk = 0 ; bk < 126 ; bk++ ){ 

                // load A and B to shared memory
                // *(float4*)( s_a + (load_a_smem_m * 8 + load_a_smem_k) ) = *(float4*)( a + (load_a_gmem_m * 1009 + bk * 8 + load_a_smem_k) );
                // *(float4*)( s_b + (load_b_smem_k * 128 + load_b_smem_n) ) = *(float4*)( b + (bk * 8 + load_b_smem_k ) * 1009 + load_b_gmem_n );

                // FLOAT4(s_a[load_a_smem_m * 8 + load_a_smem_k]) = FLOAT4(a[(load_a_gmem_m) * 1009 + bk * 8 + load_a_smem_k]);
                // FLOAT4(s_b[load_b_smem_k * 128 + load_b_smem_n]) = FLOAT4(b[(load_b_smem_k + bk * 8) * 1009 + load_b_gmem_n]);

                for (int i = 0 ; i < 4; i++){

                    s_a[load_a_smem_m * 8 + load_a_smem_k + i] = a[(load_a_gmem_m) * 1009 + bk * 8 + load_a_smem_k + i];
                    s_b[load_b_smem_k * 128 + load_b_smem_n + i] = b[(load_b_smem_k + bk * 8) * 1009 + load_b_gmem_n + i];
                    
                }


                __syncthreads();

                // compute C
                #pragma unroll
                for ( int k = 0 ; k < 8 ; k++ ){
                    for ( int m = 0 ; m < 8 ; m++ ){
                        for ( int n = 0 ; n < 8 ; n++ ){
                            r_c[m][n] += s_a[(threadIdx.y * 8 + m)*8 + k] * s_b[k*128 + threadIdx.x * 8 + n];
                        }
                    }
                }
            }

            // 1009 % 8 = 126
            for (int m = 0 ; m < 8 ; m++){
                for (int n = 0 ; n < 8; n++){
                    r_c[m][n] += a[(blockIdx.y * 128 + threadIdx.y * 8 + m) * 1009 + 1008] * b[1008 * 1009 + blockIdx.x * 128 + threadIdx.x * 8 + n];
                }
            }

            // store C to global memory
            for ( int m = 0 ; m < 8 ; m++ ){
                for ( int n = 0 ; n < 8 ; n++ ){
                    c[(blockIdx.y * 128 + threadIdx.y * 8 + m) * 1009 + blockIdx.x * 128 + threadIdx.x * 8 + n] = r_c[m][n];
                    // c[(blockIdx.y * 128 + threadIdx.y * 8 + m) * 1009 + blockIdx.x * 128 + threadIdx.x * 8 + n] = 1;
                }
            }


        }

        // flag == 1
        if (flag == 1){
            // 1009 / 8 = 126
            for ( int bk = 0 ; bk < 126 ; bk++ ){ 

                // load A and B to shared memory

                // todo vector load error 
                // if (load_a_gmem_m < 1009 )
                //     *(float4*)( s_a + (load_a_smem_m * 8 + load_a_smem_k) ) = *(float4*)( a + (load_a_gmem_m * 1009 + bk * 8 + load_a_smem_k) );
                // if (load_b_gmem_n < 1009 )
                //     *(float4*)( s_b + (load_b_smem_k * 128 + load_b_smem_n) ) = *(float4*)( b + (load_b_smem_k + bk * 8) * 1009 + load_b_gmem_n );

                for (int i = 0 ; i < 4; i++){
                    if (load_a_gmem_m < 1009 && load_a_smem_k + bk * 8 + i < 1009 ){
                        s_a[load_a_smem_m * 8 + load_a_smem_k + i] = a[(load_a_gmem_m) * 1009 + bk * 8 + load_a_smem_k + i];
                    }else{
                        s_a[load_a_smem_m * 8 + load_a_smem_k + i] = 0;
                    }
                    if (load_b_gmem_n + i < 1009 && load_b_smem_k + bk * 8 < 1009 ){
                        s_b[load_b_smem_k * 128 + load_b_smem_n + i] = b[(load_b_smem_k + bk * 8) * 1009 + load_b_gmem_n + i];
                    }else{
                        s_b[load_b_smem_k * 128 + load_b_smem_n + i] = 0;
                    }
                }


                __syncthreads();

                // compute C
                #pragma unroll
                for ( int k = 0 ; k < 8 ; k++ ){
                    for ( int m = 0 ; m < 8 ; m++ ){
                        for ( int n = 0 ; n < 8 ; n++ ){
                            r_c[m][n] += s_a[(threadIdx.y * 8 + m)*8 + k] * s_b[k*128 + threadIdx.x * 8 + n];
                        }
                    }
                }
            }

            // 1009 % 8 = 126
            for (int m = 0 ; m < 8 ; m++){
                for (int n = 0 ; n < 8; n++){
                    if (blockIdx.y * 128 + threadIdx.y * 8 + m < 1009 && blockIdx.x * 128 + threadIdx.x * 8 + n < 1009)
                        r_c[m][n] += a[(blockIdx.y * 128 + threadIdx.y * 8 + m) * 1009 + 1008] * b[1008 * 1009 + blockIdx.x * 128 + threadIdx.x * 8 + n];
                }
            }

            // store C to global memory
            for ( int m = 0 ; m < 8 ; m++ ){
                for ( int n = 0 ; n < 8 ; n++ ){
                    if (blockIdx.y * 128 + threadIdx.y * 8 + m < 1009 && blockIdx.x * 128 + threadIdx.x * 8 + n < 1009)
                        c[(blockIdx.y * 128 + threadIdx.y * 8 + m) * 1009 + blockIdx.x * 128 + threadIdx.x * 8 + n] = r_c[m][n];
                        // c[(blockIdx.y * 128 + threadIdx.y * 8 + m) * 1009 + blockIdx.x * 128 + threadIdx.x * 8 + n] = 1;
                }
            }

        }

        
}






void run_paddinggemm(int M , int N , int K , 
                float* A , float* B, float *C, float alpha , float beta){

                    dim3 blockDim(16,16);
                    dim3 gridDim(8,8);

                    paddingKernel<<<gridDim,blockDim>>>(A,B,C);


                }