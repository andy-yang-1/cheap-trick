
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
  float Y_local[64];
  float A_local[256];
  float2 B_local[1024];
  __shared__ float A_local_shared[1152];
  __shared__ float B_local_shared[4096];
  float A_local_shared_local[1];
  float B_local_shared_local[4];
  float A_local_shared_local_1[1];
  float B_local_shared_local_1[4];
  for (int i0_3_init = 0; i0_3_init < 16; ++i0_3_init) {
    Y_local[i0_3_init] = 0.000000e+00f;
    Y_local[(i0_3_init + 16)] = 0.000000e+00f;
    Y_local[(i0_3_init + 32)] = 0.000000e+00f;
    Y_local[(i0_3_init + 48)] = 0.000000e+00f;
  }
  for (int ax0_ax1_fused_3_s = 0; ax0_ax1_fused_3_s < 4; ++ax0_ax1_fused_3_s) {
    if (((int)threadIdx.y) < 32) {
      A_local[((((((int)threadIdx.y) * 4) + ax0_ax1_fused_3_s) - min(((((int)threadIdx.y) & 1) * 4), ((((int)threadIdx.y) & 3) * 2))) - (min((((int)threadIdx.y) >> 1), (((int)threadIdx.y) >> 2)) * 8))] = A[((((((int)blockIdx.x) * 16384) + ((((int)threadIdx.y) >> 1) * 1024)) + ((((int)threadIdx.y) & 1) * 4)) + ax0_ax1_fused_3_s)];
    }
  }
  for (int ax0_ax1_fused_0 = 0; ax0_ax1_fused_0 < 16; ++ax0_ax1_fused_0) {
    B_local[((ax0_ax1_fused_0 * 64) + ((int)threadIdx.y))] = *(float2*)(B + (((((ax0_ax1_fused_0 >> 1) * 1024) + (((int)blockIdx.y) * 256)) + ((ax0_ax1_fused_0 & 1) * 128)) + (((int)threadIdx.y) * 2)));
  }
    int2 __1 = make_int2(((((((((int)threadIdx.y) & 3) * 144) + (((((int)threadIdx.y) & 31) >> 4) * 64)) + ((((int)threadIdx.y) >> 5) * 4)) + ((((int)threadIdx.y) & 15) >> 2)))+(72*0), ((((((((int)threadIdx.y) & 3) * 144) + (((((int)threadIdx.y) & 31) >> 4) * 64)) + ((((int)threadIdx.y) >> 5) * 4)) + ((((int)threadIdx.y) & 15) >> 2)))+(72*1));
    float2 __2 = *(float2*)(A_local + (((((int)threadIdx.y) * 2) - min(((((int)threadIdx.y) & 1) * 4), ((((int)threadIdx.y) & 3) * 2))) - (min((((int)threadIdx.y) >> 1), (((int)threadIdx.y) >> 2)) * 8)));
    A_local_shared[__1.x] = __2.x;
    A_local_shared[__1.y] = __2.y;
  for (int ax0_ax1_fused_0_1 = 0; ax0_ax1_fused_0_1 < 16; ++ax0_ax1_fused_0_1) {
    *(float2*)(B_local_shared + ((((ax0_ax1_fused_0_1 * 128) + (((((int)threadIdx.y) & 3) >> 1) * 64)) + ((((int)threadIdx.y) >> 2) * 4)) + ((((int)threadIdx.y) & 1) * 2))) = B_local[((ax0_ax1_fused_0_1 * 64) + ((int)threadIdx.y))];
  }
  for (int i2_0 = 0; i2_0 < 127; ++i2_0) {
    for (int ax0_ax1_fused_3_s_1 = 0; ax0_ax1_fused_3_s_1 < 4; ++ax0_ax1_fused_3_s_1) {
      if (((int)threadIdx.y) < 32) {
        A_local[((((((int)threadIdx.y) * 4) + ax0_ax1_fused_3_s_1) - min(((((int)threadIdx.y) & 1) * 4), ((((int)threadIdx.y) & 3) * 2))) - (min((((int)threadIdx.y) >> 1), (((int)threadIdx.y) >> 2)) * 8))] = A[((((((((int)blockIdx.x) * 16384) + ((((int)threadIdx.y) >> 1) * 1024)) + (i2_0 * 8)) + ((((int)threadIdx.y) & 1) * 4)) + ax0_ax1_fused_3_s_1) + 8)];
      }
    }
    for (int ax0_ax1_fused_0_2 = 0; ax0_ax1_fused_0_2 < 16; ++ax0_ax1_fused_0_2) {
      B_local[((ax0_ax1_fused_0_2 * 64) + ((int)threadIdx.y))] = *(float2*)(B + ((((((i2_0 * 8192) + ((ax0_ax1_fused_0_2 >> 1) * 1024)) + (((int)blockIdx.y) * 256)) + ((ax0_ax1_fused_0_2 & 1) * 128)) + (((int)threadIdx.y) * 2)) + 8192));
    }
    __syncthreads();
    for (int i0_3 = 0; i0_3 < 16; ++i0_3) {
      for (int ax0_ax1_fused_1_s = 0; ax0_ax1_fused_1_s < 3; ++ax0_ax1_fused_1_s) {
        if (ax0_ax1_fused_1_s < 1) {
          A_local_shared_local[0] = A_local_shared[(((((i2_0 & 1) * 576) + (((i0_3 & 7) >> 2) * 64)) + ((i0_3 >> 3) * 4)) + (i0_3 & 3))];
        }
      }
      for (int ax0_ax1_fused_1_s_1 = 0; ax0_ax1_fused_1_s_1 < 4; ++ax0_ax1_fused_1_s_1) {
        if (ax0_ax1_fused_1_s_1 < 1) {
          B_local_shared_local[0] = B_local_shared[(((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3))];
          B_local_shared_local[1] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 32)];
          B_local_shared_local[2] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 128)];
          B_local_shared_local[3] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 160)];
        }
      }
      Y_local[i0_3] = (Y_local[i0_3] + (A_local_shared_local[0] * B_local_shared_local[0]));
      Y_local[(i0_3 + 16)] = (Y_local[(i0_3 + 16)] + (A_local_shared_local[0] * B_local_shared_local[1]));
      Y_local[(i0_3 + 32)] = (Y_local[(i0_3 + 32)] + (A_local_shared_local[0] * B_local_shared_local[2]));
      Y_local[(i0_3 + 48)] = (Y_local[(i0_3 + 48)] + (A_local_shared_local[0] * B_local_shared_local[3]));
      for (int ax0_ax1_fused_1_s_2 = 0; ax0_ax1_fused_1_s_2 < 3; ++ax0_ax1_fused_1_s_2) {
        if (ax0_ax1_fused_1_s_2 < 1) {
          A_local_shared_local[0] = A_local_shared[((((((i2_0 & 1) * 576) + (((i0_3 & 7) >> 2) * 64)) + ((i0_3 >> 3) * 4)) + (i0_3 & 3)) + 72)];
        }
      }
      for (int ax0_ax1_fused_1_s_3 = 0; ax0_ax1_fused_1_s_3 < 4; ++ax0_ax1_fused_1_s_3) {
        if (ax0_ax1_fused_1_s_3 < 1) {
          B_local_shared_local[0] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 256)];
          B_local_shared_local[1] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 288)];
          B_local_shared_local[2] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 384)];
          B_local_shared_local[3] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 416)];
        }
      }
      Y_local[i0_3] = (Y_local[i0_3] + (A_local_shared_local[0] * B_local_shared_local[0]));
      Y_local[(i0_3 + 16)] = (Y_local[(i0_3 + 16)] + (A_local_shared_local[0] * B_local_shared_local[1]));
      Y_local[(i0_3 + 32)] = (Y_local[(i0_3 + 32)] + (A_local_shared_local[0] * B_local_shared_local[2]));
      Y_local[(i0_3 + 48)] = (Y_local[(i0_3 + 48)] + (A_local_shared_local[0] * B_local_shared_local[3]));
    }
    for (int i0_3_1 = 0; i0_3_1 < 16; ++i0_3_1) {
      for (int ax0_ax1_fused_1_s_4 = 0; ax0_ax1_fused_1_s_4 < 3; ++ax0_ax1_fused_1_s_4) {
        if (ax0_ax1_fused_1_s_4 < 1) {
          A_local_shared_local[0] = A_local_shared[((((((i2_0 & 1) * 576) + (((i0_3_1 & 7) >> 2) * 64)) + ((i0_3_1 >> 3) * 4)) + (i0_3_1 & 3)) + 144)];
        }
      }
      for (int ax0_ax1_fused_1_s_5 = 0; ax0_ax1_fused_1_s_5 < 4; ++ax0_ax1_fused_1_s_5) {
        if (ax0_ax1_fused_1_s_5 < 1) {
          B_local_shared_local[0] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 512)];
          B_local_shared_local[1] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 544)];
          B_local_shared_local[2] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 640)];
          B_local_shared_local[3] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 672)];
        }
      }
      Y_local[i0_3_1] = (Y_local[i0_3_1] + (A_local_shared_local[0] * B_local_shared_local[0]));
      Y_local[(i0_3_1 + 16)] = (Y_local[(i0_3_1 + 16)] + (A_local_shared_local[0] * B_local_shared_local[1]));
      Y_local[(i0_3_1 + 32)] = (Y_local[(i0_3_1 + 32)] + (A_local_shared_local[0] * B_local_shared_local[2]));
      Y_local[(i0_3_1 + 48)] = (Y_local[(i0_3_1 + 48)] + (A_local_shared_local[0] * B_local_shared_local[3]));
      for (int ax0_ax1_fused_1_s_6 = 0; ax0_ax1_fused_1_s_6 < 3; ++ax0_ax1_fused_1_s_6) {
        if (ax0_ax1_fused_1_s_6 < 1) {
          A_local_shared_local[0] = A_local_shared[((((((i2_0 & 1) * 576) + (((i0_3_1 & 7) >> 2) * 64)) + ((i0_3_1 >> 3) * 4)) + (i0_3_1 & 3)) + 216)];
        }
      }
      for (int ax0_ax1_fused_1_s_7 = 0; ax0_ax1_fused_1_s_7 < 4; ++ax0_ax1_fused_1_s_7) {
        if (ax0_ax1_fused_1_s_7 < 1) {
          B_local_shared_local[0] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 768)];
          B_local_shared_local[1] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 800)];
          B_local_shared_local[2] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 896)];
          B_local_shared_local[3] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 928)];
        }
      }
      Y_local[i0_3_1] = (Y_local[i0_3_1] + (A_local_shared_local[0] * B_local_shared_local[0]));
      Y_local[(i0_3_1 + 16)] = (Y_local[(i0_3_1 + 16)] + (A_local_shared_local[0] * B_local_shared_local[1]));
      Y_local[(i0_3_1 + 32)] = (Y_local[(i0_3_1 + 32)] + (A_local_shared_local[0] * B_local_shared_local[2]));
      Y_local[(i0_3_1 + 48)] = (Y_local[(i0_3_1 + 48)] + (A_local_shared_local[0] * B_local_shared_local[3]));
    }
    for (int i0_3_2 = 0; i0_3_2 < 16; ++i0_3_2) {
      for (int ax0_ax1_fused_1_s_8 = 0; ax0_ax1_fused_1_s_8 < 3; ++ax0_ax1_fused_1_s_8) {
        if (ax0_ax1_fused_1_s_8 < 1) {
          A_local_shared_local[0] = A_local_shared[((((((i2_0 & 1) * 576) + (((i0_3_2 & 7) >> 2) * 64)) + ((i0_3_2 >> 3) * 4)) + (i0_3_2 & 3)) + 288)];
        }
      }
      for (int ax0_ax1_fused_1_s_9 = 0; ax0_ax1_fused_1_s_9 < 4; ++ax0_ax1_fused_1_s_9) {
        if (ax0_ax1_fused_1_s_9 < 1) {
          B_local_shared_local[0] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 1024)];
          B_local_shared_local[1] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 1056)];
          B_local_shared_local[2] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 1152)];
          B_local_shared_local[3] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 1184)];
        }
      }
      Y_local[i0_3_2] = (Y_local[i0_3_2] + (A_local_shared_local[0] * B_local_shared_local[0]));
      Y_local[(i0_3_2 + 16)] = (Y_local[(i0_3_2 + 16)] + (A_local_shared_local[0] * B_local_shared_local[1]));
      Y_local[(i0_3_2 + 32)] = (Y_local[(i0_3_2 + 32)] + (A_local_shared_local[0] * B_local_shared_local[2]));
      Y_local[(i0_3_2 + 48)] = (Y_local[(i0_3_2 + 48)] + (A_local_shared_local[0] * B_local_shared_local[3]));
      for (int ax0_ax1_fused_1_s_10 = 0; ax0_ax1_fused_1_s_10 < 3; ++ax0_ax1_fused_1_s_10) {
        if (ax0_ax1_fused_1_s_10 < 1) {
          A_local_shared_local[0] = A_local_shared[((((((i2_0 & 1) * 576) + (((i0_3_2 & 7) >> 2) * 64)) + ((i0_3_2 >> 3) * 4)) + (i0_3_2 & 3)) + 360)];
        }
      }
      for (int ax0_ax1_fused_1_s_11 = 0; ax0_ax1_fused_1_s_11 < 4; ++ax0_ax1_fused_1_s_11) {
        if (ax0_ax1_fused_1_s_11 < 1) {
          B_local_shared_local[0] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 1280)];
          B_local_shared_local[1] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 1312)];
          B_local_shared_local[2] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 1408)];
          B_local_shared_local[3] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 1440)];
        }
      }
      Y_local[i0_3_2] = (Y_local[i0_3_2] + (A_local_shared_local[0] * B_local_shared_local[0]));
      Y_local[(i0_3_2 + 16)] = (Y_local[(i0_3_2 + 16)] + (A_local_shared_local[0] * B_local_shared_local[1]));
      Y_local[(i0_3_2 + 32)] = (Y_local[(i0_3_2 + 32)] + (A_local_shared_local[0] * B_local_shared_local[2]));
      Y_local[(i0_3_2 + 48)] = (Y_local[(i0_3_2 + 48)] + (A_local_shared_local[0] * B_local_shared_local[3]));
    }
    for (int i0_3_3 = 0; i0_3_3 < 16; ++i0_3_3) {
      for (int ax0_ax1_fused_1_s_12 = 0; ax0_ax1_fused_1_s_12 < 3; ++ax0_ax1_fused_1_s_12) {
        if (ax0_ax1_fused_1_s_12 < 1) {
          A_local_shared_local[0] = A_local_shared[((((((i2_0 & 1) * 576) + (((i0_3_3 & 7) >> 2) * 64)) + ((i0_3_3 >> 3) * 4)) + (i0_3_3 & 3)) + 432)];
        }
      }
      for (int ax0_ax1_fused_1_s_13 = 0; ax0_ax1_fused_1_s_13 < 4; ++ax0_ax1_fused_1_s_13) {
        if (ax0_ax1_fused_1_s_13 < 1) {
          B_local_shared_local[0] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 1536)];
          B_local_shared_local[1] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 1568)];
          B_local_shared_local[2] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 1664)];
          B_local_shared_local[3] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 1696)];
        }
      }
      Y_local[i0_3_3] = (Y_local[i0_3_3] + (A_local_shared_local[0] * B_local_shared_local[0]));
      Y_local[(i0_3_3 + 16)] = (Y_local[(i0_3_3 + 16)] + (A_local_shared_local[0] * B_local_shared_local[1]));
      Y_local[(i0_3_3 + 32)] = (Y_local[(i0_3_3 + 32)] + (A_local_shared_local[0] * B_local_shared_local[2]));
      Y_local[(i0_3_3 + 48)] = (Y_local[(i0_3_3 + 48)] + (A_local_shared_local[0] * B_local_shared_local[3]));
      for (int ax0_ax1_fused_1_s_14 = 0; ax0_ax1_fused_1_s_14 < 3; ++ax0_ax1_fused_1_s_14) {
        if (ax0_ax1_fused_1_s_14 < 1) {
          A_local_shared_local[0] = A_local_shared[((((((i2_0 & 1) * 576) + (((i0_3_3 & 7) >> 2) * 64)) + ((i0_3_3 >> 3) * 4)) + (i0_3_3 & 3)) + 504)];
        }
      }
      for (int ax0_ax1_fused_1_s_15 = 0; ax0_ax1_fused_1_s_15 < 4; ++ax0_ax1_fused_1_s_15) {
        if (ax0_ax1_fused_1_s_15 < 1) {
          B_local_shared_local[0] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 1792)];
          B_local_shared_local[1] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 1824)];
          B_local_shared_local[2] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 1920)];
          B_local_shared_local[3] = B_local_shared[((((((i2_0 & 1) * 2048) + (((((int)threadIdx.y) & 7) >> 2) * 64)) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 1952)];
        }
      }
      Y_local[i0_3_3] = (Y_local[i0_3_3] + (A_local_shared_local[0] * B_local_shared_local[0]));
      Y_local[(i0_3_3 + 16)] = (Y_local[(i0_3_3 + 16)] + (A_local_shared_local[0] * B_local_shared_local[1]));
      Y_local[(i0_3_3 + 32)] = (Y_local[(i0_3_3 + 32)] + (A_local_shared_local[0] * B_local_shared_local[2]));
      Y_local[(i0_3_3 + 48)] = (Y_local[(i0_3_3 + 48)] + (A_local_shared_local[0] * B_local_shared_local[3]));
    }
    __syncthreads();
      int2 __3 = make_int2(((((((((i2_0 + 1) & 1) * 576) + ((((int)threadIdx.y) & 3) * 144)) + (((((int)threadIdx.y) & 31) >> 4) * 64)) + ((((int)threadIdx.y) >> 5) * 4)) + ((((int)threadIdx.y) & 15) >> 2)))+(72*0), ((((((((i2_0 + 1) & 1) * 576) + ((((int)threadIdx.y) & 3) * 144)) + (((((int)threadIdx.y) & 31) >> 4) * 64)) + ((((int)threadIdx.y) >> 5) * 4)) + ((((int)threadIdx.y) & 15) >> 2)))+(72*1));
      float2 __4 = *(float2*)(A_local + (((((int)threadIdx.y) * 2) - min(((((int)threadIdx.y) & 1) * 4), ((((int)threadIdx.y) & 3) * 2))) - (min((((int)threadIdx.y) >> 1), (((int)threadIdx.y) >> 2)) * 8)));
      A_local_shared[__3.x] = __4.x;
      A_local_shared[__3.y] = __4.y;
    for (int ax0_ax1_fused_0_3 = 0; ax0_ax1_fused_0_3 < 16; ++ax0_ax1_fused_0_3) {
      *(float2*)(B_local_shared + (((((((i2_0 + 1) & 1) * 2048) + (ax0_ax1_fused_0_3 * 128)) + (((((int)threadIdx.y) & 3) >> 1) * 64)) + ((((int)threadIdx.y) >> 2) * 4)) + ((((int)threadIdx.y) & 1) * 2))) = B_local[((ax0_ax1_fused_0_3 * 64) + ((int)threadIdx.y))];
    }
  }
  __syncthreads();
  for (int i0_3_4 = 0; i0_3_4 < 16; ++i0_3_4) {
    for (int ax0_ax1_fused_1_s_16 = 0; ax0_ax1_fused_1_s_16 < 3; ++ax0_ax1_fused_1_s_16) {
      if (ax0_ax1_fused_1_s_16 < 1) {
        A_local_shared_local_1[0] = A_local_shared[((((((i0_3_4 & 7) >> 2) * 64) + ((i0_3_4 >> 3) * 4)) + (i0_3_4 & 3)) + 576)];
      }
    }
    for (int ax0_ax1_fused_1_s_17 = 0; ax0_ax1_fused_1_s_17 < 4; ++ax0_ax1_fused_1_s_17) {
      if (ax0_ax1_fused_1_s_17 < 1) {
        B_local_shared_local_1[0] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 2048)];
        B_local_shared_local_1[1] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 2080)];
        B_local_shared_local_1[2] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 2176)];
        B_local_shared_local_1[3] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 2208)];
      }
    }
    Y_local[i0_3_4] = (Y_local[i0_3_4] + (A_local_shared_local_1[0] * B_local_shared_local_1[0]));
    Y_local[(i0_3_4 + 16)] = (Y_local[(i0_3_4 + 16)] + (A_local_shared_local_1[0] * B_local_shared_local_1[1]));
    Y_local[(i0_3_4 + 32)] = (Y_local[(i0_3_4 + 32)] + (A_local_shared_local_1[0] * B_local_shared_local_1[2]));
    Y_local[(i0_3_4 + 48)] = (Y_local[(i0_3_4 + 48)] + (A_local_shared_local_1[0] * B_local_shared_local_1[3]));
    for (int ax0_ax1_fused_1_s_18 = 0; ax0_ax1_fused_1_s_18 < 3; ++ax0_ax1_fused_1_s_18) {
      if (ax0_ax1_fused_1_s_18 < 1) {
        A_local_shared_local_1[0] = A_local_shared[((((((i0_3_4 & 7) >> 2) * 64) + ((i0_3_4 >> 3) * 4)) + (i0_3_4 & 3)) + 648)];
      }
    }
    for (int ax0_ax1_fused_1_s_19 = 0; ax0_ax1_fused_1_s_19 < 4; ++ax0_ax1_fused_1_s_19) {
      if (ax0_ax1_fused_1_s_19 < 1) {
        B_local_shared_local_1[0] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 2304)];
        B_local_shared_local_1[1] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 2336)];
        B_local_shared_local_1[2] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 2432)];
        B_local_shared_local_1[3] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 2464)];
      }
    }
    Y_local[i0_3_4] = (Y_local[i0_3_4] + (A_local_shared_local_1[0] * B_local_shared_local_1[0]));
    Y_local[(i0_3_4 + 16)] = (Y_local[(i0_3_4 + 16)] + (A_local_shared_local_1[0] * B_local_shared_local_1[1]));
    Y_local[(i0_3_4 + 32)] = (Y_local[(i0_3_4 + 32)] + (A_local_shared_local_1[0] * B_local_shared_local_1[2]));
    Y_local[(i0_3_4 + 48)] = (Y_local[(i0_3_4 + 48)] + (A_local_shared_local_1[0] * B_local_shared_local_1[3]));
  }
  for (int i0_3_5 = 0; i0_3_5 < 16; ++i0_3_5) {
    for (int ax0_ax1_fused_1_s_20 = 0; ax0_ax1_fused_1_s_20 < 3; ++ax0_ax1_fused_1_s_20) {
      if (ax0_ax1_fused_1_s_20 < 1) {
        A_local_shared_local_1[0] = A_local_shared[((((((i0_3_5 & 7) >> 2) * 64) + ((i0_3_5 >> 3) * 4)) + (i0_3_5 & 3)) + 720)];
      }
    }
    for (int ax0_ax1_fused_1_s_21 = 0; ax0_ax1_fused_1_s_21 < 4; ++ax0_ax1_fused_1_s_21) {
      if (ax0_ax1_fused_1_s_21 < 1) {
        B_local_shared_local_1[0] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 2560)];
        B_local_shared_local_1[1] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 2592)];
        B_local_shared_local_1[2] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 2688)];
        B_local_shared_local_1[3] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 2720)];
      }
    }
    Y_local[i0_3_5] = (Y_local[i0_3_5] + (A_local_shared_local_1[0] * B_local_shared_local_1[0]));
    Y_local[(i0_3_5 + 16)] = (Y_local[(i0_3_5 + 16)] + (A_local_shared_local_1[0] * B_local_shared_local_1[1]));
    Y_local[(i0_3_5 + 32)] = (Y_local[(i0_3_5 + 32)] + (A_local_shared_local_1[0] * B_local_shared_local_1[2]));
    Y_local[(i0_3_5 + 48)] = (Y_local[(i0_3_5 + 48)] + (A_local_shared_local_1[0] * B_local_shared_local_1[3]));
    for (int ax0_ax1_fused_1_s_22 = 0; ax0_ax1_fused_1_s_22 < 3; ++ax0_ax1_fused_1_s_22) {
      if (ax0_ax1_fused_1_s_22 < 1) {
        A_local_shared_local_1[0] = A_local_shared[((((((i0_3_5 & 7) >> 2) * 64) + ((i0_3_5 >> 3) * 4)) + (i0_3_5 & 3)) + 792)];
      }
    }
    for (int ax0_ax1_fused_1_s_23 = 0; ax0_ax1_fused_1_s_23 < 4; ++ax0_ax1_fused_1_s_23) {
      if (ax0_ax1_fused_1_s_23 < 1) {
        B_local_shared_local_1[0] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 2816)];
        B_local_shared_local_1[1] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 2848)];
        B_local_shared_local_1[2] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 2944)];
        B_local_shared_local_1[3] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 2976)];
      }
    }
    Y_local[i0_3_5] = (Y_local[i0_3_5] + (A_local_shared_local_1[0] * B_local_shared_local_1[0]));
    Y_local[(i0_3_5 + 16)] = (Y_local[(i0_3_5 + 16)] + (A_local_shared_local_1[0] * B_local_shared_local_1[1]));
    Y_local[(i0_3_5 + 32)] = (Y_local[(i0_3_5 + 32)] + (A_local_shared_local_1[0] * B_local_shared_local_1[2]));
    Y_local[(i0_3_5 + 48)] = (Y_local[(i0_3_5 + 48)] + (A_local_shared_local_1[0] * B_local_shared_local_1[3]));
  }
  for (int i0_3_6 = 0; i0_3_6 < 16; ++i0_3_6) {
    for (int ax0_ax1_fused_1_s_24 = 0; ax0_ax1_fused_1_s_24 < 3; ++ax0_ax1_fused_1_s_24) {
      if (ax0_ax1_fused_1_s_24 < 1) {
        A_local_shared_local_1[0] = A_local_shared[((((((i0_3_6 & 7) >> 2) * 64) + ((i0_3_6 >> 3) * 4)) + (i0_3_6 & 3)) + 864)];
      }
    }
    for (int ax0_ax1_fused_1_s_25 = 0; ax0_ax1_fused_1_s_25 < 4; ++ax0_ax1_fused_1_s_25) {
      if (ax0_ax1_fused_1_s_25 < 1) {
        B_local_shared_local_1[0] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 3072)];
        B_local_shared_local_1[1] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 3104)];
        B_local_shared_local_1[2] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 3200)];
        B_local_shared_local_1[3] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 3232)];
      }
    }
    Y_local[i0_3_6] = (Y_local[i0_3_6] + (A_local_shared_local_1[0] * B_local_shared_local_1[0]));
    Y_local[(i0_3_6 + 16)] = (Y_local[(i0_3_6 + 16)] + (A_local_shared_local_1[0] * B_local_shared_local_1[1]));
    Y_local[(i0_3_6 + 32)] = (Y_local[(i0_3_6 + 32)] + (A_local_shared_local_1[0] * B_local_shared_local_1[2]));
    Y_local[(i0_3_6 + 48)] = (Y_local[(i0_3_6 + 48)] + (A_local_shared_local_1[0] * B_local_shared_local_1[3]));
    for (int ax0_ax1_fused_1_s_26 = 0; ax0_ax1_fused_1_s_26 < 3; ++ax0_ax1_fused_1_s_26) {
      if (ax0_ax1_fused_1_s_26 < 1) {
        A_local_shared_local_1[0] = A_local_shared[((((((i0_3_6 & 7) >> 2) * 64) + ((i0_3_6 >> 3) * 4)) + (i0_3_6 & 3)) + 936)];
      }
    }
    for (int ax0_ax1_fused_1_s_27 = 0; ax0_ax1_fused_1_s_27 < 4; ++ax0_ax1_fused_1_s_27) {
      if (ax0_ax1_fused_1_s_27 < 1) {
        B_local_shared_local_1[0] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 3328)];
        B_local_shared_local_1[1] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 3360)];
        B_local_shared_local_1[2] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 3456)];
        B_local_shared_local_1[3] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 3488)];
      }
    }
    Y_local[i0_3_6] = (Y_local[i0_3_6] + (A_local_shared_local_1[0] * B_local_shared_local_1[0]));
    Y_local[(i0_3_6 + 16)] = (Y_local[(i0_3_6 + 16)] + (A_local_shared_local_1[0] * B_local_shared_local_1[1]));
    Y_local[(i0_3_6 + 32)] = (Y_local[(i0_3_6 + 32)] + (A_local_shared_local_1[0] * B_local_shared_local_1[2]));
    Y_local[(i0_3_6 + 48)] = (Y_local[(i0_3_6 + 48)] + (A_local_shared_local_1[0] * B_local_shared_local_1[3]));
  }
  for (int i0_3_7 = 0; i0_3_7 < 16; ++i0_3_7) {
    for (int ax0_ax1_fused_1_s_28 = 0; ax0_ax1_fused_1_s_28 < 3; ++ax0_ax1_fused_1_s_28) {
      if (ax0_ax1_fused_1_s_28 < 1) {
        A_local_shared_local_1[0] = A_local_shared[((((((i0_3_7 & 7) >> 2) * 64) + ((i0_3_7 >> 3) * 4)) + (i0_3_7 & 3)) + 1008)];
      }
    }
    for (int ax0_ax1_fused_1_s_29 = 0; ax0_ax1_fused_1_s_29 < 4; ++ax0_ax1_fused_1_s_29) {
      if (ax0_ax1_fused_1_s_29 < 1) {
        B_local_shared_local_1[0] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 3584)];
        B_local_shared_local_1[1] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 3616)];
        B_local_shared_local_1[2] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 3712)];
        B_local_shared_local_1[3] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 3744)];
      }
    }
    Y_local[i0_3_7] = (Y_local[i0_3_7] + (A_local_shared_local_1[0] * B_local_shared_local_1[0]));
    Y_local[(i0_3_7 + 16)] = (Y_local[(i0_3_7 + 16)] + (A_local_shared_local_1[0] * B_local_shared_local_1[1]));
    Y_local[(i0_3_7 + 32)] = (Y_local[(i0_3_7 + 32)] + (A_local_shared_local_1[0] * B_local_shared_local_1[2]));
    Y_local[(i0_3_7 + 48)] = (Y_local[(i0_3_7 + 48)] + (A_local_shared_local_1[0] * B_local_shared_local_1[3]));
    for (int ax0_ax1_fused_1_s_30 = 0; ax0_ax1_fused_1_s_30 < 3; ++ax0_ax1_fused_1_s_30) {
      if (ax0_ax1_fused_1_s_30 < 1) {
        A_local_shared_local_1[0] = A_local_shared[((((((i0_3_7 & 7) >> 2) * 64) + ((i0_3_7 >> 3) * 4)) + (i0_3_7 & 3)) + 1080)];
      }
    }
    for (int ax0_ax1_fused_1_s_31 = 0; ax0_ax1_fused_1_s_31 < 4; ++ax0_ax1_fused_1_s_31) {
      if (ax0_ax1_fused_1_s_31 < 1) {
        B_local_shared_local_1[0] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 3840)];
        B_local_shared_local_1[1] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 3872)];
        B_local_shared_local_1[2] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 3968)];
        B_local_shared_local_1[3] = B_local_shared[((((((((int)threadIdx.y) & 7) >> 2) * 64) + ((((int)threadIdx.y) >> 3) * 4)) + (((int)threadIdx.y) & 3)) + 4000)];
      }
    }
    Y_local[i0_3_7] = (Y_local[i0_3_7] + (A_local_shared_local_1[0] * B_local_shared_local_1[0]));
    Y_local[(i0_3_7 + 16)] = (Y_local[(i0_3_7 + 16)] + (A_local_shared_local_1[0] * B_local_shared_local_1[1]));
    Y_local[(i0_3_7 + 32)] = (Y_local[(i0_3_7 + 32)] + (A_local_shared_local_1[0] * B_local_shared_local_1[2]));
    Y_local[(i0_3_7 + 48)] = (Y_local[(i0_3_7 + 48)] + (A_local_shared_local_1[0] * B_local_shared_local_1[3]));
  }
  for (int ax0 = 0; ax0 < 16; ++ax0) {
    Y[((((((int)blockIdx.x) * 16384) + (ax0 * 1024)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y))] = Y_local[ax0];
    Y[(((((((int)blockIdx.x) * 16384) + (ax0 * 1024)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 64)] = Y_local[(ax0 + 16)];
    Y[(((((((int)blockIdx.x) * 16384) + (ax0 * 1024)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 128)] = Y_local[(ax0 + 32)];
    Y[(((((((int)blockIdx.x) * 16384) + (ax0 * 1024)) + (((int)blockIdx.y) * 256)) + ((int)threadIdx.y)) + 192)] = Y_local[(ax0 + 48)];
  }
}


