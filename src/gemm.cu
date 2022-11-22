
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
extern "C" __global__ void __launch_bounds__(64) mm_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ Y) {
  float Y_local[40];
  float A_local[1];
  float B_local[1];
  __shared__ float A_shared[512];
  __shared__ float B_shared[192];
  float A_shared_local[10];
  float B_shared_local[4];
  float A_shared_local_1[10];
  float B_shared_local_1[4];
  for (int i0_2_init = 0; i0_2_init < 10; ++i0_2_init) {
    for (int i1_2_init = 0; i1_2_init < 4; ++i1_2_init) {
      Y_local[((i0_2_init * 4) + i1_2_init)] = 0.000000e+00f;
    }
  }
  if (((int)threadIdx.y) < 10) {
    A_local[0] = A[(((((int)blockIdx.x) * 26680) + (((int)threadIdx.y) * 2668)) + (((int)threadIdx.x) * 667))];
  }
  B_local[0] = B[(((((int)blockIdx.y) * 64) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x))];
  if (((int)threadIdx.y) < 10) {
    A_shared[((((((((((int)blockIdx.x) * 5) + (((int)threadIdx.y) >> 1)) >> 4) * 128) + ((((int)threadIdx.y) & 1) * 64)) + ((((((int)blockIdx.x) * 5) + (((int)threadIdx.y) >> 1)) & 15) * 4)) + ((int)threadIdx.x)) - (((((int)blockIdx.x) * 40) >> 7) * 128))] = A_local[0];
  }
  B_shared[((((((int)threadIdx.y) & 1) * 64) + ((((int)threadIdx.y) >> 1) * 4)) + ((int)threadIdx.x))] = B_local[0];
  for (int i2_0 = 0; i2_0 < 666; ++i2_0) {
    if (((int)threadIdx.y) < 10) {
      A_local[0] = A[(((((((int)blockIdx.x) * 26680) + (((int)threadIdx.y) * 2668)) + (((int)threadIdx.x) * 667)) + i2_0) + 1)];
    }
    B_local[0] = B[(((((i2_0 * 1152) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 4)) + ((int)threadIdx.x)) + 1152)];
    __syncthreads();
    for (int ax0_ax1_fused_1_s = 0; ax0_ax1_fused_1_s < 4; ++ax0_ax1_fused_1_s) {
      A_shared_local[ax0_ax1_fused_1_s] = A_shared[(((((((i2_0 & 1) * 256) + (((((((int)blockIdx.x) * 40) + (((int)threadIdx.x) * 10)) + ax0_ax1_fused_1_s) >> 7) * 128)) + (((((ax0_ax1_fused_1_s >> 1) + ((int)threadIdx.x)) & 3) >> 1) * 64)) + ((((((int)blockIdx.x) * 5) + (((((int)threadIdx.x) * 10) + ax0_ax1_fused_1_s) >> 3)) & 15) * 4)) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_1_s) & 3)) - (((((int)blockIdx.x) * 40) >> 7) * 128))];
    }
    for (int ax0_ax1_fused_1_s_1 = 0; ax0_ax1_fused_1_s_1 < 4; ++ax0_ax1_fused_1_s_1) {
      A_shared_local[(ax0_ax1_fused_1_s_1 + 4)] = A_shared[(((((((i2_0 & 1) * 256) + ((((((((int)blockIdx.x) * 40) + (((int)threadIdx.x) * 10)) + ax0_ax1_fused_1_s_1) + 4) >> 7) * 128)) + ((((((((int)threadIdx.x) * 5) + (ax0_ax1_fused_1_s_1 >> 1)) >> 1) + 1) & 1) * 64)) + ((((((int)blockIdx.x) * 5) + ((((((int)threadIdx.x) * 10) + ax0_ax1_fused_1_s_1) + 4) >> 3)) & 15) * 4)) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_1_s_1) & 3)) - (((((int)blockIdx.x) * 40) >> 7) * 128))];
    }
    for (int ax0_ax1_fused_1_s_2 = 0; ax0_ax1_fused_1_s_2 < 4; ++ax0_ax1_fused_1_s_2) {
      if (ax0_ax1_fused_1_s_2 < 2) {
        A_shared_local[(ax0_ax1_fused_1_s_2 + 8)] = A_shared[((((((((i2_0 & 1) * 256) + ((((((((int)blockIdx.x) * 40) + (((int)threadIdx.x) * 10)) + ax0_ax1_fused_1_s_2) + 8) >> 7) * 128)) + ((((int)threadIdx.x) >> 1) * 64)) + (((((((int)blockIdx.x) * 5) + ((int)threadIdx.x)) + 1) & 15) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + ax0_ax1_fused_1_s_2) - (((((int)blockIdx.x) * 40) >> 7) * 128))];
      }
    }
    *(float4*)(B_shared_local + 0) = *(float4*)(B_shared + ((((i2_0 & 1) * 96) + ((((int)threadIdx.y) & 1) * 64)) + ((((int)threadIdx.y) >> 1) * 4)));
    for (int i0_2 = 0; i0_2 < 10; ++i0_2) {
      for (int i1_2 = 0; i1_2 < 4; ++i1_2) {
        Y_local[((i0_2 * 4) + i1_2)] = (Y_local[((i0_2 * 4) + i1_2)] + (A_shared_local[i0_2] * B_shared_local[i1_2]));
      }
    }
    __syncthreads();
    if (((int)threadIdx.y) < 10) {
      A_shared[((((((((i2_0 + 1) & 1) * 256) + ((((((int)blockIdx.x) * 5) + (((int)threadIdx.y) >> 1)) >> 4) * 128)) + ((((int)threadIdx.y) & 1) * 64)) + ((((((int)blockIdx.x) * 5) + (((int)threadIdx.y) >> 1)) & 15) * 4)) + ((int)threadIdx.x)) - (((((int)blockIdx.x) * 40) >> 7) * 128))] = A_local[0];
    }
    B_shared[((((((i2_0 + 1) & 1) * 96) + ((((int)threadIdx.y) & 1) * 64)) + ((((int)threadIdx.y) >> 1) * 4)) + ((int)threadIdx.x))] = B_local[0];
  }
  __syncthreads();
  for (int ax0_ax1_fused_1_s_3 = 0; ax0_ax1_fused_1_s_3 < 4; ++ax0_ax1_fused_1_s_3) {
    A_shared_local_1[ax0_ax1_fused_1_s_3] = A_shared[(((((((((((int)blockIdx.x) * 40) + (((int)threadIdx.x) * 10)) + ax0_ax1_fused_1_s_3) >> 7) * 128) + (((((ax0_ax1_fused_1_s_3 >> 1) + ((int)threadIdx.x)) & 3) >> 1) * 64)) + ((((((int)blockIdx.x) * 5) + (((((int)threadIdx.x) * 10) + ax0_ax1_fused_1_s_3) >> 3)) & 15) * 4)) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_1_s_3) & 3)) - (((((int)blockIdx.x) * 40) >> 7) * 128))];
  }
  for (int ax0_ax1_fused_1_s_4 = 0; ax0_ax1_fused_1_s_4 < 4; ++ax0_ax1_fused_1_s_4) {
    A_shared_local_1[(ax0_ax1_fused_1_s_4 + 4)] = A_shared[((((((((((((int)blockIdx.x) * 40) + (((int)threadIdx.x) * 10)) + ax0_ax1_fused_1_s_4) + 4) >> 7) * 128) + ((((((((int)threadIdx.x) * 5) + (ax0_ax1_fused_1_s_4 >> 1)) >> 1) + 1) & 1) * 64)) + ((((((int)blockIdx.x) * 5) + ((((((int)threadIdx.x) * 10) + ax0_ax1_fused_1_s_4) + 4) >> 3)) & 15) * 4)) + (((((int)threadIdx.x) * 2) + ax0_ax1_fused_1_s_4) & 3)) - (((((int)blockIdx.x) * 40) >> 7) * 128))];
  }
  for (int ax0_ax1_fused_1_s_5 = 0; ax0_ax1_fused_1_s_5 < 4; ++ax0_ax1_fused_1_s_5) {
    if (ax0_ax1_fused_1_s_5 < 2) {
      A_shared_local_1[(ax0_ax1_fused_1_s_5 + 8)] = A_shared[(((((((((((((int)blockIdx.x) * 40) + (((int)threadIdx.x) * 10)) + ax0_ax1_fused_1_s_5) + 8) >> 7) * 128) + ((((int)threadIdx.x) >> 1) * 64)) + (((((((int)blockIdx.x) * 5) + ((int)threadIdx.x)) + 1) & 15) * 4)) + ((((int)threadIdx.x) & 1) * 2)) + ax0_ax1_fused_1_s_5) - (((((int)blockIdx.x) * 40) >> 7) * 128))];
    }
  }
  *(float4*)(B_shared_local_1 + 0) = *(float4*)(B_shared + (((((int)threadIdx.y) & 1) * 64) + ((((int)threadIdx.y) >> 1) * 4)));
  for (int i0_2_1 = 0; i0_2_1 < 10; ++i0_2_1) {
    for (int i1_2_1 = 0; i1_2_1 < 4; ++i1_2_1) {
      Y_local[((i0_2_1 * 4) + i1_2_1)] = (Y_local[((i0_2_1 * 4) + i1_2_1)] + (A_shared_local_1[i0_2_1] * B_shared_local_1[i1_2_1]));
    }
  }
  for (int ax0 = 0; ax0 < 10; ++ax0) {
    *(float4*)(Y + (((((((int)blockIdx.x) * 46080) + (((int)threadIdx.x) * 11520)) + (ax0 * 1152)) + (((int)blockIdx.y) * 64)) + (((int)threadIdx.y) * 4))) = *(float4*)(Y_local + (ax0 * 4));
  }
}


