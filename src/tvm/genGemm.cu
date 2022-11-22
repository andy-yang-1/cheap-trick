
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
extern "C" __global__ void __launch_bounds__(256) mm_kernel0(float* __restrict__ A, float* __restrict__ B, float* __restrict__ Y) {
  float Y_local[64];
  float4 A_local[1];
  float4 B_local[32];
  __shared__ float A_shared[2048];
  __shared__ float B_shared[2048];
  float B_shared_local[8];
  float A_shared_local[8];
  float B_shared_local1[8];
  float A_shared_local1[8];
  for (int i0_2_init = 0; i0_2_init < 8; ++i0_2_init) {
    for (int i1_2_init = 0; i1_2_init < 8; ++i1_2_init) {
      Y_local[((i0_2_init * 8) + i1_2_init)] = 0.000000e+00f;
    }
  }
  A_local[0] = *(float4*)(A + ((((((int)blockIdx.x) * 131072) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 1) * 1024)) + ((((int)threadIdx.x) & 1) * 4)));
  B_local[(((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) - ((((int)threadIdx.y) >> 1) * 32))] = *(float4*)(B + (((((((int)threadIdx.y) >> 1) * 1024) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 4)));
    int4 _1 = make_int4(((((((((int)threadIdx.x) & 1) * 512) + ((((int)threadIdx.x) >> 3) * 64)) + (((int)threadIdx.y) * 4)) + ((((int)threadIdx.x) & 7) >> 1)))+(128*0), ((((((((int)threadIdx.x) & 1) * 512) + ((((int)threadIdx.x) >> 3) * 64)) + (((int)threadIdx.y) * 4)) + ((((int)threadIdx.x) & 7) >> 1)))+(128*1), ((((((((int)threadIdx.x) & 1) * 512) + ((((int)threadIdx.x) >> 3) * 64)) + (((int)threadIdx.y) * 4)) + ((((int)threadIdx.x) & 7) >> 1)))+(128*2), ((((((((int)threadIdx.x) & 1) * 512) + ((((int)threadIdx.x) >> 3) * 64)) + (((int)threadIdx.y) * 4)) + ((((int)threadIdx.x) & 7) >> 1)))+(128*3));
    float4 _2 = A_local[0];
    A_shared[_1.x] = _2.x;
    A_shared[_1.y] = _2.y;
    A_shared[_1.z] = _2.z;
    A_shared[_1.w] = _2.w;
  *(float4*)(B_shared + (((((((int)threadIdx.y) >> 1) * 128) + ((((int)threadIdx.x) & 1) * 64)) + ((((int)threadIdx.y) & 1) * 32)) + ((((int)threadIdx.x) >> 1) * 4))) = B_local[(((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) - ((((int)threadIdx.y) >> 1) * 32))];
  for (int i2_0 = 0; i2_0 < 127; ++i2_0) {
    A_local[(((((i2_0 * 8) + ((((int)threadIdx.x) & 1) * 4)) + 8) - max(0, ((((i2_0 * 8) + (((int)threadIdx.x) * 4)) + 8) - ((((int)threadIdx.x) >> 1) * 8)))) / 4)] = *(float4*)(A + ((((((((int)blockIdx.x) * 131072) + (((int)threadIdx.y) * 8192)) + ((((int)threadIdx.x) >> 1) * 1024)) + (i2_0 * 8)) + ((((int)threadIdx.x) & 1) * 4)) + 8));
    B_local[(((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) - ((((int)threadIdx.y) >> 1) * 32))] = *(float4*)(B + ((((((i2_0 * 8192) + ((((int)threadIdx.y) >> 1) * 1024)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.y) & 1) * 64)) + (((int)threadIdx.x) * 4)) + 8192));
    __syncthreads();
    *(float4*)(B_shared_local + 0) = *(float4*)(B_shared + (((i2_0 & 1) * 1024) + (((int)threadIdx.y) * 4)));
    *(float4*)(B_shared_local + 4) = *(float4*)(B_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.y) * 4)) + 64));
    *(float4*)(A_shared_local + 0) = *(float4*)(A_shared + (((i2_0 & 1) * 1024) + (((int)threadIdx.x) * 4)));
    *(float4*)(A_shared_local + 4) = *(float4*)(A_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.x) * 4)) + 64));
    for (int i0_2 = 0; i0_2 < 8; ++i0_2) {
      for (int i1_2 = 0; i1_2 < 8; ++i1_2) {
        Y_local[((i0_2 * 8) + i1_2)] = (Y_local[((i0_2 * 8) + i1_2)] + (A_shared_local[i0_2] * B_shared_local[i1_2]));
      }
    }
    *(float4*)(B_shared_local + 0) = *(float4*)(B_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.y) * 4)) + 128));
    *(float4*)(B_shared_local + 4) = *(float4*)(B_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.y) * 4)) + 192));
    *(float4*)(A_shared_local + 0) = *(float4*)(A_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.x) * 4)) + 128));
    *(float4*)(A_shared_local + 4) = *(float4*)(A_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.x) * 4)) + 192));
    for (int i0_21 = 0; i0_21 < 8; ++i0_21) {
      for (int i1_21 = 0; i1_21 < 8; ++i1_21) {
        Y_local[((i0_21 * 8) + i1_21)] = (Y_local[((i0_21 * 8) + i1_21)] + (A_shared_local[i0_21] * B_shared_local[i1_21]));
      }
    }
    *(float4*)(B_shared_local + 0) = *(float4*)(B_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.y) * 4)) + 256));
    *(float4*)(B_shared_local + 4) = *(float4*)(B_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.y) * 4)) + 320));
    *(float4*)(A_shared_local + 0) = *(float4*)(A_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.x) * 4)) + 256));
    *(float4*)(A_shared_local + 4) = *(float4*)(A_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.x) * 4)) + 320));
    for (int i0_22 = 0; i0_22 < 8; ++i0_22) {
      for (int i1_22 = 0; i1_22 < 8; ++i1_22) {
        Y_local[((i0_22 * 8) + i1_22)] = (Y_local[((i0_22 * 8) + i1_22)] + (A_shared_local[i0_22] * B_shared_local[i1_22]));
      }
    }
    *(float4*)(B_shared_local + 0) = *(float4*)(B_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.y) * 4)) + 384));
    *(float4*)(B_shared_local + 4) = *(float4*)(B_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.y) * 4)) + 448));
    *(float4*)(A_shared_local + 0) = *(float4*)(A_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.x) * 4)) + 384));
    *(float4*)(A_shared_local + 4) = *(float4*)(A_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.x) * 4)) + 448));
    for (int i0_23 = 0; i0_23 < 8; ++i0_23) {
      for (int i1_23 = 0; i1_23 < 8; ++i1_23) {
        Y_local[((i0_23 * 8) + i1_23)] = (Y_local[((i0_23 * 8) + i1_23)] + (A_shared_local[i0_23] * B_shared_local[i1_23]));
      }
    }
    *(float4*)(B_shared_local + 0) = *(float4*)(B_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.y) * 4)) + 512));
    *(float4*)(B_shared_local + 4) = *(float4*)(B_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.y) * 4)) + 576));
    *(float4*)(A_shared_local + 0) = *(float4*)(A_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.x) * 4)) + 512));
    *(float4*)(A_shared_local + 4) = *(float4*)(A_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.x) * 4)) + 576));
    for (int i0_24 = 0; i0_24 < 8; ++i0_24) {
      for (int i1_24 = 0; i1_24 < 8; ++i1_24) {
        Y_local[((i0_24 * 8) + i1_24)] = (Y_local[((i0_24 * 8) + i1_24)] + (A_shared_local[i0_24] * B_shared_local[i1_24]));
      }
    }
    *(float4*)(B_shared_local + 0) = *(float4*)(B_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.y) * 4)) + 640));
    *(float4*)(B_shared_local + 4) = *(float4*)(B_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.y) * 4)) + 704));
    *(float4*)(A_shared_local + 0) = *(float4*)(A_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.x) * 4)) + 640));
    *(float4*)(A_shared_local + 4) = *(float4*)(A_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.x) * 4)) + 704));
    for (int i0_25 = 0; i0_25 < 8; ++i0_25) {
      for (int i1_25 = 0; i1_25 < 8; ++i1_25) {
        Y_local[((i0_25 * 8) + i1_25)] = (Y_local[((i0_25 * 8) + i1_25)] + (A_shared_local[i0_25] * B_shared_local[i1_25]));
      }
    }
    *(float4*)(B_shared_local + 0) = *(float4*)(B_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.y) * 4)) + 768));
    *(float4*)(B_shared_local + 4) = *(float4*)(B_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.y) * 4)) + 832));
    *(float4*)(A_shared_local + 0) = *(float4*)(A_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.x) * 4)) + 768));
    *(float4*)(A_shared_local + 4) = *(float4*)(A_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.x) * 4)) + 832));
    for (int i0_26 = 0; i0_26 < 8; ++i0_26) {
      for (int i1_26 = 0; i1_26 < 8; ++i1_26) {
        Y_local[((i0_26 * 8) + i1_26)] = (Y_local[((i0_26 * 8) + i1_26)] + (A_shared_local[i0_26] * B_shared_local[i1_26]));
      }
    }
    *(float4*)(B_shared_local + 0) = *(float4*)(B_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.y) * 4)) + 896));
    *(float4*)(B_shared_local + 4) = *(float4*)(B_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.y) * 4)) + 960));
    *(float4*)(A_shared_local + 0) = *(float4*)(A_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.x) * 4)) + 896));
    *(float4*)(A_shared_local + 4) = *(float4*)(A_shared + ((((i2_0 & 1) * 1024) + (((int)threadIdx.x) * 4)) + 960));
    for (int i0_27 = 0; i0_27 < 8; ++i0_27) {
      for (int i1_27 = 0; i1_27 < 8; ++i1_27) {
        Y_local[((i0_27 * 8) + i1_27)] = (Y_local[((i0_27 * 8) + i1_27)] + (A_shared_local[i0_27] * B_shared_local[i1_27]));
      }
    }
    __syncthreads();
      int4 _3 = make_int4(((((((((i2_0 + 1) & 1) * 1024) + ((((int)threadIdx.x) & 1) * 512)) + ((((int)threadIdx.x) >> 3) * 64)) + (((int)threadIdx.y) * 4)) + ((((int)threadIdx.x) & 7) >> 1)))+(128*0), ((((((((i2_0 + 1) & 1) * 1024) + ((((int)threadIdx.x) & 1) * 512)) + ((((int)threadIdx.x) >> 3) * 64)) + (((int)threadIdx.y) * 4)) + ((((int)threadIdx.x) & 7) >> 1)))+(128*1), ((((((((i2_0 + 1) & 1) * 1024) + ((((int)threadIdx.x) & 1) * 512)) + ((((int)threadIdx.x) >> 3) * 64)) + (((int)threadIdx.y) * 4)) + ((((int)threadIdx.x) & 7) >> 1)))+(128*2), ((((((((i2_0 + 1) & 1) * 1024) + ((((int)threadIdx.x) & 1) * 512)) + ((((int)threadIdx.x) >> 3) * 64)) + (((int)threadIdx.y) * 4)) + ((((int)threadIdx.x) & 7) >> 1)))+(128*3));
      float4 _4 = A_local[(((((i2_0 * 8) + ((((int)threadIdx.x) & 1) * 4)) + 8) - max(0, ((((i2_0 * 8) + (((int)threadIdx.x) * 4)) + 8) - ((((int)threadIdx.x) >> 1) * 8)))) / 4)];
      A_shared[_3.x] = _4.x;
      A_shared[_3.y] = _4.y;
      A_shared[_3.z] = _4.z;
      A_shared[_3.w] = _4.w;
    *(float4*)(B_shared + (((((((i2_0 + 1) & 1) * 1024) + ((((int)threadIdx.y) >> 1) * 128)) + ((((int)threadIdx.x) & 1) * 64)) + ((((int)threadIdx.y) & 1) * 32)) + ((((int)threadIdx.x) >> 1) * 4))) = B_local[(((((int)threadIdx.y) * 16) + ((int)threadIdx.x)) - ((((int)threadIdx.y) >> 1) * 32))];
  }
  __syncthreads();
  *(float4*)(B_shared_local1 + 0) = *(float4*)(B_shared + ((((int)threadIdx.y) * 4) + 1024));
  *(float4*)(B_shared_local1 + 4) = *(float4*)(B_shared + ((((int)threadIdx.y) * 4) + 1088));
  *(float4*)(A_shared_local1 + 0) = *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1024));
  *(float4*)(A_shared_local1 + 4) = *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1088));
  for (int i0_28 = 0; i0_28 < 8; ++i0_28) {
    for (int i1_28 = 0; i1_28 < 8; ++i1_28) {
      Y_local[((i0_28 * 8) + i1_28)] = (Y_local[((i0_28 * 8) + i1_28)] + (A_shared_local1[i0_28] * B_shared_local1[i1_28]));
    }
  }
  *(float4*)(B_shared_local1 + 0) = *(float4*)(B_shared + ((((int)threadIdx.y) * 4) + 1152));
  *(float4*)(B_shared_local1 + 4) = *(float4*)(B_shared + ((((int)threadIdx.y) * 4) + 1216));
  *(float4*)(A_shared_local1 + 0) = *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1152));
  *(float4*)(A_shared_local1 + 4) = *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1216));
  for (int i0_29 = 0; i0_29 < 8; ++i0_29) {
    for (int i1_29 = 0; i1_29 < 8; ++i1_29) {
      Y_local[((i0_29 * 8) + i1_29)] = (Y_local[((i0_29 * 8) + i1_29)] + (A_shared_local1[i0_29] * B_shared_local1[i1_29]));
    }
  }
  *(float4*)(B_shared_local1 + 0) = *(float4*)(B_shared + ((((int)threadIdx.y) * 4) + 1280));
  *(float4*)(B_shared_local1 + 4) = *(float4*)(B_shared + ((((int)threadIdx.y) * 4) + 1344));
  *(float4*)(A_shared_local1 + 0) = *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1280));
  *(float4*)(A_shared_local1 + 4) = *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1344));
  for (int i0_210 = 0; i0_210 < 8; ++i0_210) {
    for (int i1_210 = 0; i1_210 < 8; ++i1_210) {
      Y_local[((i0_210 * 8) + i1_210)] = (Y_local[((i0_210 * 8) + i1_210)] + (A_shared_local1[i0_210] * B_shared_local1[i1_210]));
    }
  }
  *(float4*)(B_shared_local1 + 0) = *(float4*)(B_shared + ((((int)threadIdx.y) * 4) + 1408));
  *(float4*)(B_shared_local1 + 4) = *(float4*)(B_shared + ((((int)threadIdx.y) * 4) + 1472));
  *(float4*)(A_shared_local1 + 0) = *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1408));
  *(float4*)(A_shared_local1 + 4) = *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1472));
  for (int i0_211 = 0; i0_211 < 8; ++i0_211) {
    for (int i1_211 = 0; i1_211 < 8; ++i1_211) {
      Y_local[((i0_211 * 8) + i1_211)] = (Y_local[((i0_211 * 8) + i1_211)] + (A_shared_local1[i0_211] * B_shared_local1[i1_211]));
    }
  }
  *(float4*)(B_shared_local1 + 0) = *(float4*)(B_shared + ((((int)threadIdx.y) * 4) + 1536));
  *(float4*)(B_shared_local1 + 4) = *(float4*)(B_shared + ((((int)threadIdx.y) * 4) + 1600));
  *(float4*)(A_shared_local1 + 0) = *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1536));
  *(float4*)(A_shared_local1 + 4) = *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1600));
  for (int i0_212 = 0; i0_212 < 8; ++i0_212) {
    for (int i1_212 = 0; i1_212 < 8; ++i1_212) {
      Y_local[((i0_212 * 8) + i1_212)] = (Y_local[((i0_212 * 8) + i1_212)] + (A_shared_local1[i0_212] * B_shared_local1[i1_212]));
    }
  }
  *(float4*)(B_shared_local1 + 0) = *(float4*)(B_shared + ((((int)threadIdx.y) * 4) + 1664));
  *(float4*)(B_shared_local1 + 4) = *(float4*)(B_shared + ((((int)threadIdx.y) * 4) + 1728));
  *(float4*)(A_shared_local1 + 0) = *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1664));
  *(float4*)(A_shared_local1 + 4) = *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1728));
  for (int i0_213 = 0; i0_213 < 8; ++i0_213) {
    for (int i1_213 = 0; i1_213 < 8; ++i1_213) {
      Y_local[((i0_213 * 8) + i1_213)] = (Y_local[((i0_213 * 8) + i1_213)] + (A_shared_local1[i0_213] * B_shared_local1[i1_213]));
    }
  }
  *(float4*)(B_shared_local1 + 0) = *(float4*)(B_shared + ((((int)threadIdx.y) * 4) + 1792));
  *(float4*)(B_shared_local1 + 4) = *(float4*)(B_shared + ((((int)threadIdx.y) * 4) + 1856));
  *(float4*)(A_shared_local1 + 0) = *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1792));
  *(float4*)(A_shared_local1 + 4) = *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1856));
  for (int i0_214 = 0; i0_214 < 8; ++i0_214) {
    for (int i1_214 = 0; i1_214 < 8; ++i1_214) {
      Y_local[((i0_214 * 8) + i1_214)] = (Y_local[((i0_214 * 8) + i1_214)] + (A_shared_local1[i0_214] * B_shared_local1[i1_214]));
    }
  }
  *(float4*)(B_shared_local1 + 0) = *(float4*)(B_shared + ((((int)threadIdx.y) * 4) + 1920));
  *(float4*)(B_shared_local1 + 4) = *(float4*)(B_shared + ((((int)threadIdx.y) * 4) + 1984));
  *(float4*)(A_shared_local1 + 0) = *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1920));
  *(float4*)(A_shared_local1 + 4) = *(float4*)(A_shared + ((((int)threadIdx.x) * 4) + 1984));
  for (int i0_215 = 0; i0_215 < 8; ++i0_215) {
    for (int i1_215 = 0; i1_215 < 8; ++i1_215) {
      Y_local[((i0_215 * 8) + i1_215)] = (Y_local[((i0_215 * 8) + i1_215)] + (A_shared_local1[i0_215] * B_shared_local1[i1_215]));
    }
  }
  for (int ax0 = 0; ax0 < 8; ++ax0) {
    *(ulonglong4*)(Y + (((((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 8192)) + (ax0 * 1024)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 8))) = *(ulonglong4*)(Y_local + (ax0 * 8));
  }
}


