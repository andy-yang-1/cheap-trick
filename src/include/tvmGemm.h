
#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(64) main_kernel0( float* __restrict__ A, float* __restrict__ B, float* __restrict__ C) {
  float C_local[64];
  __shared__ float A_shared[2048];
  __shared__ float B_shared[2048];
  C_local[0] = 0.000000e+00f;
  C_local[1] = 0.000000e+00f;
  C_local[4] = 0.000000e+00f;
  C_local[5] = 0.000000e+00f;
  C_local[8] = 0.000000e+00f;
  C_local[9] = 0.000000e+00f;
  C_local[12] = 0.000000e+00f;
  C_local[13] = 0.000000e+00f;
  C_local[16] = 0.000000e+00f;
  C_local[17] = 0.000000e+00f;
  C_local[20] = 0.000000e+00f;
  C_local[21] = 0.000000e+00f;
  C_local[24] = 0.000000e+00f;
  C_local[25] = 0.000000e+00f;
  C_local[28] = 0.000000e+00f;
  C_local[29] = 0.000000e+00f;
  C_local[32] = 0.000000e+00f;
  C_local[33] = 0.000000e+00f;
  C_local[36] = 0.000000e+00f;
  C_local[37] = 0.000000e+00f;
  C_local[40] = 0.000000e+00f;
  C_local[41] = 0.000000e+00f;
  C_local[44] = 0.000000e+00f;
  C_local[45] = 0.000000e+00f;
  C_local[48] = 0.000000e+00f;
  C_local[49] = 0.000000e+00f;
  C_local[52] = 0.000000e+00f;
  C_local[53] = 0.000000e+00f;
  C_local[56] = 0.000000e+00f;
  C_local[57] = 0.000000e+00f;
  C_local[60] = 0.000000e+00f;
  C_local[61] = 0.000000e+00f;
  C_local[2] = 0.000000e+00f;
  C_local[3] = 0.000000e+00f;
  C_local[6] = 0.000000e+00f;
  C_local[7] = 0.000000e+00f;
  C_local[10] = 0.000000e+00f;
  C_local[11] = 0.000000e+00f;
  C_local[14] = 0.000000e+00f;
  C_local[15] = 0.000000e+00f;
  C_local[18] = 0.000000e+00f;
  C_local[19] = 0.000000e+00f;
  C_local[22] = 0.000000e+00f;
  C_local[23] = 0.000000e+00f;
  C_local[26] = 0.000000e+00f;
  C_local[27] = 0.000000e+00f;
  C_local[30] = 0.000000e+00f;
  C_local[31] = 0.000000e+00f;
  C_local[34] = 0.000000e+00f;
  C_local[35] = 0.000000e+00f;
  C_local[38] = 0.000000e+00f;
  C_local[39] = 0.000000e+00f;
  C_local[42] = 0.000000e+00f;
  C_local[43] = 0.000000e+00f;
  C_local[46] = 0.000000e+00f;
  C_local[47] = 0.000000e+00f;
  C_local[50] = 0.000000e+00f;
  C_local[51] = 0.000000e+00f;
  C_local[54] = 0.000000e+00f;
  C_local[55] = 0.000000e+00f;
  C_local[58] = 0.000000e+00f;
  C_local[59] = 0.000000e+00f;
  C_local[62] = 0.000000e+00f;
  C_local[63] = 0.000000e+00f;

  
  for (int k_0 = 0; k_0 < 32; ++k_0) {
    __syncthreads();
    *(float4*)(A_shared + (((int)threadIdx.x) * 4)) = *(float4*)(A + (((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 256)) = *(float4*)(A + ((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 8192));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(A + ((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 16384));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 768)) = *(float4*)(A + ((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 24576));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1024)) = *(float4*)(A + ((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 32768));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1280)) = *(float4*)(A + ((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 40960));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1536)) = *(float4*)(A + ((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 49152));
    *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1792)) = *(float4*)(A + ((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 3) * 1024)) + (k_0 * 32)) + ((((int)threadIdx.x) & 7) * 4)) + 57344));
    B_shared[((int)threadIdx.x)] = B[(((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x))];
    B_shared[(((int)threadIdx.x) + 64)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 1024)];
    B_shared[(((int)threadIdx.x) + 128)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 2048)];
    B_shared[(((int)threadIdx.x) + 192)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 3072)];
    B_shared[(((int)threadIdx.x) + 256)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 4096)];
    B_shared[(((int)threadIdx.x) + 320)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 5120)];
    B_shared[(((int)threadIdx.x) + 384)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 6144)];
    B_shared[(((int)threadIdx.x) + 448)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 7168)];
    B_shared[(((int)threadIdx.x) + 512)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 8192)];
    B_shared[(((int)threadIdx.x) + 576)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 9216)];
    B_shared[(((int)threadIdx.x) + 640)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 10240)];
    B_shared[(((int)threadIdx.x) + 704)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 11264)];
    B_shared[(((int)threadIdx.x) + 768)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 12288)];
    B_shared[(((int)threadIdx.x) + 832)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 13312)];
    B_shared[(((int)threadIdx.x) + 896)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 14336)];
    B_shared[(((int)threadIdx.x) + 960)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 15360)];
    B_shared[(((int)threadIdx.x) + 1024)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 16384)];
    B_shared[(((int)threadIdx.x) + 1088)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 17408)];
    B_shared[(((int)threadIdx.x) + 1152)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 18432)];
    B_shared[(((int)threadIdx.x) + 1216)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 19456)];
    B_shared[(((int)threadIdx.x) + 1280)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 20480)];
    B_shared[(((int)threadIdx.x) + 1344)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 21504)];
    B_shared[(((int)threadIdx.x) + 1408)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 22528)];
    B_shared[(((int)threadIdx.x) + 1472)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 23552)];
    B_shared[(((int)threadIdx.x) + 1536)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 24576)];
    B_shared[(((int)threadIdx.x) + 1600)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 25600)];
    B_shared[(((int)threadIdx.x) + 1664)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 26624)];
    B_shared[(((int)threadIdx.x) + 1728)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 27648)];
    B_shared[(((int)threadIdx.x) + 1792)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 28672)];
    B_shared[(((int)threadIdx.x) + 1856)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 29696)];
    B_shared[(((int)threadIdx.x) + 1920)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 30720)];
    B_shared[(((int)threadIdx.x) + 1984)] = B[((((k_0 * 32768) + ((((int)blockIdx.x) & 15) * 64)) + ((int)threadIdx.x)) + 31744)];
    __syncthreads();
    for (int k_1 = 0; k_1 < 32; ++k_1) {
      C_local[0] = (C_local[0] + (A_shared[(((((int)threadIdx.x) >> 4) * 512) + k_1)] * B_shared[((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4))]));
      C_local[1] = (C_local[1] + (A_shared[(((((int)threadIdx.x) >> 4) * 512) + k_1)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 1)]));
      C_local[4] = (C_local[4] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 32)] * B_shared[((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4))]));
      C_local[5] = (C_local[5] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 32)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 1)]));
      C_local[8] = (C_local[8] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 64)] * B_shared[((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4))]));
      C_local[9] = (C_local[9] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 64)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 1)]));
      C_local[12] = (C_local[12] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 96)] * B_shared[((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4))]));
      C_local[13] = (C_local[13] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 96)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 1)]));
      C_local[16] = (C_local[16] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 128)] * B_shared[((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4))]));
      C_local[17] = (C_local[17] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 128)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 1)]));
      C_local[20] = (C_local[20] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 160)] * B_shared[((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4))]));
      C_local[21] = (C_local[21] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 160)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 1)]));
      C_local[24] = (C_local[24] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 192)] * B_shared[((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4))]));
      C_local[25] = (C_local[25] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 192)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 1)]));
      C_local[28] = (C_local[28] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 224)] * B_shared[((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4))]));
      C_local[29] = (C_local[29] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 224)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 1)]));
      C_local[32] = (C_local[32] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 256)] * B_shared[((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4))]));
      C_local[33] = (C_local[33] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 256)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 1)]));
      C_local[36] = (C_local[36] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 288)] * B_shared[((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4))]));
      C_local[37] = (C_local[37] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 288)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 1)]));
      C_local[40] = (C_local[40] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 320)] * B_shared[((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4))]));
      C_local[41] = (C_local[41] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 320)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 1)]));
      C_local[44] = (C_local[44] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 352)] * B_shared[((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4))]));
      C_local[45] = (C_local[45] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 352)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 1)]));
      C_local[48] = (C_local[48] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 384)] * B_shared[((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4))]));
      C_local[49] = (C_local[49] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 384)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 1)]));
      C_local[52] = (C_local[52] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 416)] * B_shared[((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4))]));
      C_local[53] = (C_local[53] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 416)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 1)]));
      C_local[56] = (C_local[56] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 448)] * B_shared[((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4))]));
      C_local[57] = (C_local[57] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 448)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 1)]));
      C_local[60] = (C_local[60] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 480)] * B_shared[((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4))]));
      C_local[61] = (C_local[61] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 480)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 1)]));
      C_local[2] = (C_local[2] + (A_shared[(((((int)threadIdx.x) >> 4) * 512) + k_1)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 2)]));
      C_local[3] = (C_local[3] + (A_shared[(((((int)threadIdx.x) >> 4) * 512) + k_1)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 3)]));
      C_local[6] = (C_local[6] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 32)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 2)]));
      C_local[7] = (C_local[7] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 32)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 3)]));
      C_local[10] = (C_local[10] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 64)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 2)]));
      C_local[11] = (C_local[11] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 64)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 3)]));
      C_local[14] = (C_local[14] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 96)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 2)]));
      C_local[15] = (C_local[15] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 96)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 3)]));
      C_local[18] = (C_local[18] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 128)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 2)]));
      C_local[19] = (C_local[19] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 128)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 3)]));
      C_local[22] = (C_local[22] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 160)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 2)]));
      C_local[23] = (C_local[23] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 160)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 3)]));
      C_local[26] = (C_local[26] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 192)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 2)]));
      C_local[27] = (C_local[27] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 192)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 3)]));
      C_local[30] = (C_local[30] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 224)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 2)]));
      C_local[31] = (C_local[31] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 224)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 3)]));
      C_local[34] = (C_local[34] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 256)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 2)]));
      C_local[35] = (C_local[35] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 256)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 3)]));
      C_local[38] = (C_local[38] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 288)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 2)]));
      C_local[39] = (C_local[39] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 288)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 3)]));
      C_local[42] = (C_local[42] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 320)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 2)]));
      C_local[43] = (C_local[43] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 320)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 3)]));
      C_local[46] = (C_local[46] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 352)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 2)]));
      C_local[47] = (C_local[47] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 352)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 3)]));
      C_local[50] = (C_local[50] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 384)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 2)]));
      C_local[51] = (C_local[51] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 384)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 3)]));
      C_local[54] = (C_local[54] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 416)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 2)]));
      C_local[55] = (C_local[55] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 416)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 3)]));
      C_local[58] = (C_local[58] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 448)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 2)]));
      C_local[59] = (C_local[59] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 448)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 3)]));
      C_local[62] = (C_local[62] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 480)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 2)]));
      C_local[63] = (C_local[63] + (A_shared[((((((int)threadIdx.x) >> 4) * 512) + k_1) + 480)] * B_shared[(((k_1 * 64) + ((((int)threadIdx.x) & 15) * 4)) + 3)]));
    }
  }
  C[(((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4))] = C_local[0];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 1)] = C_local[1];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 2)] = C_local[2];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 3)] = C_local[3];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 1024)] = C_local[4];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 1025)] = C_local[5];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 1026)] = C_local[6];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 1027)] = C_local[7];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 2048)] = C_local[8];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 2049)] = C_local[9];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 2050)] = C_local[10];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 2051)] = C_local[11];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 3072)] = C_local[12];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 3073)] = C_local[13];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 3074)] = C_local[14];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 3075)] = C_local[15];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 4096)] = C_local[16];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 4097)] = C_local[17];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 4098)] = C_local[18];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 4099)] = C_local[19];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 5120)] = C_local[20];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 5121)] = C_local[21];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 5122)] = C_local[22];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 5123)] = C_local[23];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 6144)] = C_local[24];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 6145)] = C_local[25];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 6146)] = C_local[26];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 6147)] = C_local[27];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 7168)] = C_local[28];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 7169)] = C_local[29];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 7170)] = C_local[30];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 7171)] = C_local[31];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 8192)] = C_local[32];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 8193)] = C_local[33];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 8194)] = C_local[34];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 8195)] = C_local[35];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 9216)] = C_local[36];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 9217)] = C_local[37];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 9218)] = C_local[38];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 9219)] = C_local[39];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 10240)] = C_local[40];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 10241)] = C_local[41];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 10242)] = C_local[42];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 10243)] = C_local[43];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 11264)] = C_local[44];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 11265)] = C_local[45];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 11266)] = C_local[46];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 11267)] = C_local[47];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 12288)] = C_local[48];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 12289)] = C_local[49];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 12290)] = C_local[50];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 12291)] = C_local[51];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 13312)] = C_local[52];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 13313)] = C_local[53];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 13314)] = C_local[54];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 13315)] = C_local[55];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 14336)] = C_local[56];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 14337)] = C_local[57];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 14338)] = C_local[58];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 14339)] = C_local[59];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 15360)] = C_local[60];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 15361)] = C_local[61];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 15362)] = C_local[62];
  C[((((((((int)blockIdx.x) >> 4) * 65536) + ((((int)threadIdx.x) >> 4) * 16384)) + ((((int)blockIdx.x) & 15) * 64)) + ((((int)threadIdx.x) & 15) * 4)) + 15363)] = C_local[63];
}


void run_tvm(int M , int N , int K , 
                 float* A ,  float* B, float *C, float alpha , float beta){
            

    main_kernel0<<<256,64>>>(A,B,C) ;


}