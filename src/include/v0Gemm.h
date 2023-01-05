
// __global__ void padding_kernel(
//     float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
//     const int M, const int N, const int K, const float alpha, const float beta) {

//     // M = N = K = 1009

//     const int BM = 128;
//     const int BN = 128;
//     const int BK = 128;
//     const int TK = 8;
//     const int TM = 8;
//     const int TN = 8;

//     // Allocate shared memory for A and B 
//     __shared__ float s_a[1024];
//     __shared__ float s_b[1024];

//     // Allocate registers for C
//     float r_c[TM][TN] = {0.0f};

//     // add local stage
//     float r_comp_a[8];
//     float r_comp_b[8];

//     // add load stage
//     float r_load_a[4];
//     float r_load_b[4];

//     // flag: 1 -> blockIdx.y == 8 && blockIdx.x != 8 ; 2 -> blockIdx.y != 8 && blockIdx.x == 8 ; 3 -> blockIdx.y == 8 && blockIdx.x == 8 ; 0 -> others
//     int flag = 0;
//     if (blockIdx.y == 8 && blockIdx.x != 8) {
//         flag = 1;
//     } else if (blockIdx.y != 8 && blockIdx.x == 8) {
//         flag = 2;
//     } else if (blockIdx.y == 8 && blockIdx.x == 8) {
//         flag = 3;
//     }

//     int tid = threadIdx.y * 8 + threadIdx.x;
//     int load_a_smem_m = tid >> 1;
//     int load_a_smem_k = (tid & 1) << 2;
//     int load_b_smem_k = tid >> 5;
//     int load_b_smem_n = (tid & 31) << 2;

//     int load_a_gmem_m = blockIdx.y * 128 + load_a_smem_m;
//     int load_b_gmem_n = blockIdx.x * 128 + load_b_smem_n;

//     // flag == 0

//     if (flag == 0){

//         // 1009 / 8 = 126
//         for ( int bk = 0 ; bk < 126 ; bk++ ){ 

//             // load A and B to local stage registers
//             *(float4*)( r_load_a ) = *(float4*)( a + (load_a_gmem_m * 1009 + bk * 8 + load_a_smem_k) );
//             *(float4*)( r_load_b ) = *(float4*)( b + (bk * 1009 * 8 + load_b_gmem_n + load_b_smem_k) );

//             // local stage registers to shared memory
//             // *(float4*)( s_a + (load_a_smem_m * 128 + load_a_smem_k) ) = *(float4*)( r_load_a );
//             s_a[load_a_smem_k * 128 + load_a_smem_m] = r_load_a[0];
//             s_a[(load_a_smem_k + 1) * 128 + load_a_smem_m] = r_load_a[1];
//             s_a[(load_a_smem_k + 2) * 128 + load_a_smem_m] = r_load_a[2];
//             s_a[(load_a_smem_k + 3) * 128 + load_a_smem_m] = r_load_a[3];
//             *(float4*)( s_b + (load_b_smem_k * 128 + load_b_smem_n) ) = *(float4*)( r_load_b );

//             // wait for all threads to finish loading
//             __syncthreads();

//             for (int tk = 0 ; tk < TK ; tk++){

//                 // load A and B from shared memory to local stage registers
                
//                 *(float4*)( r_comp_a ) = *(float4*)( s_a + (tk * 128 + threadIdx.y * 4) );
//                 *(float4*)( r_comp_a + 4 ) = *(float4*)( s_a + (tk * 128 + threadIdx.y * 4 + 64) );
//                 *(float4*)( r_comp_b ) = *(float4*)( s_b + (tk * 128 + threadIdx.x * 4) );
//                 *(float4*)( r_comp_b + 4 ) = *(float4*)( s_b + (tk * 128 + threadIdx.x * 4 + 64) );


//                 // compute
//                 # pragma unroll
//                 for (int i = 0 ; i < 8 ; i++){
//                     for (int j = 0 ; j < 8 ; j++){
//                         r_c[i][j] += r_comp_a[i] * r_comp_b[j];
//                     }
//                 }
//             }

//         }

//         // 1009 % 8 = 1
//         {

//             // compute
//             #pragma unroll
//             for (int q1 = 0 ; q1 < 2 ; q1++){
//                 for (int q2 = 0 ; q2 < 2 ; q2++){
//                     for (int i = 0 ; i < 4 ; i++){
//                         for (int j = 0 ; j < 4 ; j++){
//                             r_c[i + q1 * 4][j + q2 * 4] += a[(blockIdx.y * 128 + i + q1 * 64) * 1009 + 1008] * b[1008*1009+(blockIdx.x * 128 + j + q2 * 64)];
//                         }
//                     }
//                 }
//             }

//             // load to global memory
//             int store_c_addr = (blockIdx.y * 128 + threadIdx.y * 4) * 1009 + blockIdx.x * 128 + threadIdx.x * 4;
            
//             *(float4*)( c + store_c_addr ) = *(float4*)( r_c[0] );
//             *(float4*)( c + store_c_addr + 1009 ) = *(float4*)( r_c[1] );
//             *(float4*)( c + store_c_addr + 2018 ) = *(float4*)( r_c[2] );
//             *(float4*)( c + store_c_addr + 3027 ) = *(float4*)( r_c[3] );
//             *(float4*)( c + store_c_addr + 64 * 1009 ) = *(float4*)( r_c[4] );
//             *(float4*)( c + store_c_addr + 64 * 1009 + 1009 ) = *(float4*)( r_c[5] );
//             *(float4*)( c + store_c_addr + 64 * 1009 + 2018 ) = *(float4*)( r_c[6] );
//             *(float4*)( c + store_c_addr + 64 * 1009 + 3027 ) = *(float4*)( r_c[7] );
//             *(float4*)( c + store_c_addr + 64 ) = *(float4*)( r_c[0] + 4 );
//             *(float4*)( c + store_c_addr + 1009 + 64 ) = *(float4*)( r_c[1] + 4 );
//             *(float4*)( c + store_c_addr + 2018 + 64 ) = *(float4*)( r_c[2] + 4 );
//             *(float4*)( c + store_c_addr + 3027 + 64 ) = *(float4*)( r_c[3] + 4 );
//             *(float4*)( c + store_c_addr + 64 * 1009 + 64 ) = *(float4*)( r_c[4] + 4 );
//             *(float4*)( c + store_c_addr + 64 * 1009 + 1009 + 64 ) = *(float4*)( r_c[5] + 4 );
//             *(float4*)( c + store_c_addr + 64 * 1009 + 2018 + 64 ) = *(float4*)( r_c[6] + 4 );
//             *(float4*)( c + store_c_addr + 64 * 1009 + 3027 + 64 ) = *(float4*)( r_c[7] + 4 );


//         }

//     }

//     // blockIdx.y == 8 && blockIdx.x != 8
//     if (flag == 1){

//         // 1009 / 8 = 126


//     }





// }