
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
  float2 A_local[1];
  ulonglong4 B_local[16];
  __shared__ float A_shared[256];
  __shared__ float B_shared[1024];
  float A_shared_local[16];
  float B_shared_local[4];
  float A_shared_local_1[16];
  float B_shared_local_1[4];
  for (int i0_2_init = 0; i0_2_init < 16; ++i0_2_init) {
    for (int i1_2_init = 0; i1_2_init < 4; ++i1_2_init) {
      Y_local[((i0_2_init * 4) + i1_2_init)] = 0.000000e+00f;
    }
  }
  A_local[0] = *(float2*)(A + (((((int)blockIdx.x) * 36864) + (((int)threadIdx.y) * 1152)) + (((int)threadIdx.x) * 2)));
  B_local[(((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) - ((((int)threadIdx.y) >> 3) * 16))] = *(ulonglong4*)(B + (((((((int)threadIdx.y) >> 3) * 1152) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.y) & 7) * 16)) + (((int)threadIdx.x) * 8)));
    int2 __1 = make_int2((((((int)threadIdx.x) * 64) + ((int)threadIdx.y)))+(32*0), (((((int)threadIdx.x) * 64) + ((int)threadIdx.y)))+(32*1));
    float2 __2 = A_local[0];
    A_shared[__1.x] = __2.x;
    A_shared[__1.y] = __2.y;
  *(ulonglong4*)(B_shared + ((((int)threadIdx.y) * 16) + (((int)threadIdx.x) * 8))) = B_local[(((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) - ((((int)threadIdx.y) >> 3) * 16))];
  for (int i2_0 = 0; i2_0 < 287; ++i2_0) {
    A_local[0] = *(float2*)(A + (((((((int)blockIdx.x) * 36864) + (((int)threadIdx.y) * 1152)) + (i2_0 * 4)) + (((int)threadIdx.x) * 2)) + 4));
    B_local[(((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) - ((((int)threadIdx.y) >> 3) * 16))] = *(ulonglong4*)(B + ((((((i2_0 * 4608) + ((((int)threadIdx.y) >> 3) * 1152)) + (((int)blockIdx.y) * 128)) + ((((int)threadIdx.y) & 7) * 16)) + (((int)threadIdx.x) * 8)) + 4608));
    __syncthreads();
    A_shared_local[0] = A_shared[(((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16))];
    A_shared_local[1] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 1)];
    A_shared_local[2] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 2)];
    A_shared_local[3] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 3)];
    A_shared_local[4] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 4)];
    A_shared_local[5] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 5)];
    A_shared_local[6] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 6)];
    A_shared_local[7] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 7)];
    A_shared_local[8] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 8)];
    A_shared_local[9] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 9)];
    A_shared_local[10] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 10)];
    A_shared_local[11] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 11)];
    A_shared_local[12] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 12)];
    A_shared_local[13] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 13)];
    A_shared_local[14] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 14)];
    A_shared_local[15] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 15)];
    *(float4*)(B_shared_local + 0) = *(float4*)(B_shared + (((i2_0 & 1) * 512) + (((int)threadIdx.y) * 4)));
    for (int i0_2 = 0; i0_2 < 16; ++i0_2) {
      for (int i1_2 = 0; i1_2 < 4; ++i1_2) {
        Y_local[((i0_2 * 4) + i1_2)] = (Y_local[((i0_2 * 4) + i1_2)] + (A_shared_local[i0_2] * B_shared_local[i1_2]));
      }
    }
    A_shared_local[0] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 32)];
    A_shared_local[1] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 33)];
    A_shared_local[2] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 34)];
    A_shared_local[3] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 35)];
    A_shared_local[4] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 36)];
    A_shared_local[5] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 37)];
    A_shared_local[6] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 38)];
    A_shared_local[7] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 39)];
    A_shared_local[8] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 40)];
    A_shared_local[9] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 41)];
    A_shared_local[10] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 42)];
    A_shared_local[11] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 43)];
    A_shared_local[12] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 44)];
    A_shared_local[13] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 45)];
    A_shared_local[14] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 46)];
    A_shared_local[15] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 47)];
    *(float4*)(B_shared_local + 0) = *(float4*)(B_shared + ((((i2_0 & 1) * 512) + (((int)threadIdx.y) * 4)) + 128));
    for (int i0_2_1 = 0; i0_2_1 < 16; ++i0_2_1) {
      for (int i1_2_1 = 0; i1_2_1 < 4; ++i1_2_1) {
        Y_local[((i0_2_1 * 4) + i1_2_1)] = (Y_local[((i0_2_1 * 4) + i1_2_1)] + (A_shared_local[i0_2_1] * B_shared_local[i1_2_1]));
      }
    }
    A_shared_local[0] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 64)];
    A_shared_local[1] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 65)];
    A_shared_local[2] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 66)];
    A_shared_local[3] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 67)];
    A_shared_local[4] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 68)];
    A_shared_local[5] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 69)];
    A_shared_local[6] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 70)];
    A_shared_local[7] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 71)];
    A_shared_local[8] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 72)];
    A_shared_local[9] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 73)];
    A_shared_local[10] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 74)];
    A_shared_local[11] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 75)];
    A_shared_local[12] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 76)];
    A_shared_local[13] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 77)];
    A_shared_local[14] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 78)];
    A_shared_local[15] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 79)];
    *(float4*)(B_shared_local + 0) = *(float4*)(B_shared + ((((i2_0 & 1) * 512) + (((int)threadIdx.y) * 4)) + 256));
    for (int i0_2_2 = 0; i0_2_2 < 16; ++i0_2_2) {
      for (int i1_2_2 = 0; i1_2_2 < 4; ++i1_2_2) {
        Y_local[((i0_2_2 * 4) + i1_2_2)] = (Y_local[((i0_2_2 * 4) + i1_2_2)] + (A_shared_local[i0_2_2] * B_shared_local[i1_2_2]));
      }
    }
    A_shared_local[0] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 96)];
    A_shared_local[1] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 97)];
    A_shared_local[2] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 98)];
    A_shared_local[3] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 99)];
    A_shared_local[4] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 100)];
    A_shared_local[5] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 101)];
    A_shared_local[6] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 102)];
    A_shared_local[7] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 103)];
    A_shared_local[8] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 104)];
    A_shared_local[9] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 105)];
    A_shared_local[10] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 106)];
    A_shared_local[11] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 107)];
    A_shared_local[12] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 108)];
    A_shared_local[13] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 109)];
    A_shared_local[14] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 110)];
    A_shared_local[15] = A_shared[((((i2_0 & 1) * 128) + (((int)threadIdx.x) * 16)) + 111)];
    *(float4*)(B_shared_local + 0) = *(float4*)(B_shared + ((((i2_0 & 1) * 512) + (((int)threadIdx.y) * 4)) + 384));
    for (int i0_2_3 = 0; i0_2_3 < 16; ++i0_2_3) {
      for (int i1_2_3 = 0; i1_2_3 < 4; ++i1_2_3) {
        Y_local[((i0_2_3 * 4) + i1_2_3)] = (Y_local[((i0_2_3 * 4) + i1_2_3)] + (A_shared_local[i0_2_3] * B_shared_local[i1_2_3]));
      }
    }
    __syncthreads();
      int2 __3 = make_int2(((((((i2_0 + 1) & 1) * 128) + (((int)threadIdx.x) * 64)) + ((int)threadIdx.y)))+(32*0), ((((((i2_0 + 1) & 1) * 128) + (((int)threadIdx.x) * 64)) + ((int)threadIdx.y)))+(32*1));
      float2 __4 = A_local[0];
      A_shared[__3.x] = __4.x;
      A_shared[__3.y] = __4.y;
    *(ulonglong4*)(B_shared + (((((i2_0 + 1) & 1) * 512) + (((int)threadIdx.y) * 16)) + (((int)threadIdx.x) * 8))) = B_local[(((((int)threadIdx.y) * 2) + ((int)threadIdx.x)) - ((((int)threadIdx.y) >> 3) * 16))];
  }
  __syncthreads();
  A_shared_local_1[0] = A_shared[((((int)threadIdx.x) * 16) + 128)];
  A_shared_local_1[1] = A_shared[((((int)threadIdx.x) * 16) + 129)];
  A_shared_local_1[2] = A_shared[((((int)threadIdx.x) * 16) + 130)];
  A_shared_local_1[3] = A_shared[((((int)threadIdx.x) * 16) + 131)];
  A_shared_local_1[4] = A_shared[((((int)threadIdx.x) * 16) + 132)];
  A_shared_local_1[5] = A_shared[((((int)threadIdx.x) * 16) + 133)];
  A_shared_local_1[6] = A_shared[((((int)threadIdx.x) * 16) + 134)];
  A_shared_local_1[7] = A_shared[((((int)threadIdx.x) * 16) + 135)];
  A_shared_local_1[8] = A_shared[((((int)threadIdx.x) * 16) + 136)];
  A_shared_local_1[9] = A_shared[((((int)threadIdx.x) * 16) + 137)];
  A_shared_local_1[10] = A_shared[((((int)threadIdx.x) * 16) + 138)];
  A_shared_local_1[11] = A_shared[((((int)threadIdx.x) * 16) + 139)];
  A_shared_local_1[12] = A_shared[((((int)threadIdx.x) * 16) + 140)];
  A_shared_local_1[13] = A_shared[((((int)threadIdx.x) * 16) + 141)];
  A_shared_local_1[14] = A_shared[((((int)threadIdx.x) * 16) + 142)];
  A_shared_local_1[15] = A_shared[((((int)threadIdx.x) * 16) + 143)];
  *(float4*)(B_shared_local_1 + 0) = *(float4*)(B_shared + ((((int)threadIdx.y) * 4) + 512));
  for (int i0_2_4 = 0; i0_2_4 < 16; ++i0_2_4) {
    for (int i1_2_4 = 0; i1_2_4 < 4; ++i1_2_4) {
      Y_local[((i0_2_4 * 4) + i1_2_4)] = (Y_local[((i0_2_4 * 4) + i1_2_4)] + (A_shared_local_1[i0_2_4] * B_shared_local_1[i1_2_4]));
    }
  }
  A_shared_local_1[0] = A_shared[((((int)threadIdx.x) * 16) + 160)];
  A_shared_local_1[1] = A_shared[((((int)threadIdx.x) * 16) + 161)];
  A_shared_local_1[2] = A_shared[((((int)threadIdx.x) * 16) + 162)];
  A_shared_local_1[3] = A_shared[((((int)threadIdx.x) * 16) + 163)];
  A_shared_local_1[4] = A_shared[((((int)threadIdx.x) * 16) + 164)];
  A_shared_local_1[5] = A_shared[((((int)threadIdx.x) * 16) + 165)];
  A_shared_local_1[6] = A_shared[((((int)threadIdx.x) * 16) + 166)];
  A_shared_local_1[7] = A_shared[((((int)threadIdx.x) * 16) + 167)];
  A_shared_local_1[8] = A_shared[((((int)threadIdx.x) * 16) + 168)];
  A_shared_local_1[9] = A_shared[((((int)threadIdx.x) * 16) + 169)];
  A_shared_local_1[10] = A_shared[((((int)threadIdx.x) * 16) + 170)];
  A_shared_local_1[11] = A_shared[((((int)threadIdx.x) * 16) + 171)];
  A_shared_local_1[12] = A_shared[((((int)threadIdx.x) * 16) + 172)];
  A_shared_local_1[13] = A_shared[((((int)threadIdx.x) * 16) + 173)];
  A_shared_local_1[14] = A_shared[((((int)threadIdx.x) * 16) + 174)];
  A_shared_local_1[15] = A_shared[((((int)threadIdx.x) * 16) + 175)];
  *(float4*)(B_shared_local_1 + 0) = *(float4*)(B_shared + ((((int)threadIdx.y) * 4) + 640));
  for (int i0_2_5 = 0; i0_2_5 < 16; ++i0_2_5) {
    for (int i1_2_5 = 0; i1_2_5 < 4; ++i1_2_5) {
      Y_local[((i0_2_5 * 4) + i1_2_5)] = (Y_local[((i0_2_5 * 4) + i1_2_5)] + (A_shared_local_1[i0_2_5] * B_shared_local_1[i1_2_5]));
    }
  }
  A_shared_local_1[0] = A_shared[((((int)threadIdx.x) * 16) + 192)];
  A_shared_local_1[1] = A_shared[((((int)threadIdx.x) * 16) + 193)];
  A_shared_local_1[2] = A_shared[((((int)threadIdx.x) * 16) + 194)];
  A_shared_local_1[3] = A_shared[((((int)threadIdx.x) * 16) + 195)];
  A_shared_local_1[4] = A_shared[((((int)threadIdx.x) * 16) + 196)];
  A_shared_local_1[5] = A_shared[((((int)threadIdx.x) * 16) + 197)];
  A_shared_local_1[6] = A_shared[((((int)threadIdx.x) * 16) + 198)];
  A_shared_local_1[7] = A_shared[((((int)threadIdx.x) * 16) + 199)];
  A_shared_local_1[8] = A_shared[((((int)threadIdx.x) * 16) + 200)];
  A_shared_local_1[9] = A_shared[((((int)threadIdx.x) * 16) + 201)];
  A_shared_local_1[10] = A_shared[((((int)threadIdx.x) * 16) + 202)];
  A_shared_local_1[11] = A_shared[((((int)threadIdx.x) * 16) + 203)];
  A_shared_local_1[12] = A_shared[((((int)threadIdx.x) * 16) + 204)];
  A_shared_local_1[13] = A_shared[((((int)threadIdx.x) * 16) + 205)];
  A_shared_local_1[14] = A_shared[((((int)threadIdx.x) * 16) + 206)];
  A_shared_local_1[15] = A_shared[((((int)threadIdx.x) * 16) + 207)];
  *(float4*)(B_shared_local_1 + 0) = *(float4*)(B_shared + ((((int)threadIdx.y) * 4) + 768));
  for (int i0_2_6 = 0; i0_2_6 < 16; ++i0_2_6) {
    for (int i1_2_6 = 0; i1_2_6 < 4; ++i1_2_6) {
      Y_local[((i0_2_6 * 4) + i1_2_6)] = (Y_local[((i0_2_6 * 4) + i1_2_6)] + (A_shared_local_1[i0_2_6] * B_shared_local_1[i1_2_6]));
    }
  }
  A_shared_local_1[0] = A_shared[((((int)threadIdx.x) * 16) + 224)];
  A_shared_local_1[1] = A_shared[((((int)threadIdx.x) * 16) + 225)];
  A_shared_local_1[2] = A_shared[((((int)threadIdx.x) * 16) + 226)];
  A_shared_local_1[3] = A_shared[((((int)threadIdx.x) * 16) + 227)];
  A_shared_local_1[4] = A_shared[((((int)threadIdx.x) * 16) + 228)];
  A_shared_local_1[5] = A_shared[((((int)threadIdx.x) * 16) + 229)];
  A_shared_local_1[6] = A_shared[((((int)threadIdx.x) * 16) + 230)];
  A_shared_local_1[7] = A_shared[((((int)threadIdx.x) * 16) + 231)];
  A_shared_local_1[8] = A_shared[((((int)threadIdx.x) * 16) + 232)];
  A_shared_local_1[9] = A_shared[((((int)threadIdx.x) * 16) + 233)];
  A_shared_local_1[10] = A_shared[((((int)threadIdx.x) * 16) + 234)];
  A_shared_local_1[11] = A_shared[((((int)threadIdx.x) * 16) + 235)];
  A_shared_local_1[12] = A_shared[((((int)threadIdx.x) * 16) + 236)];
  A_shared_local_1[13] = A_shared[((((int)threadIdx.x) * 16) + 237)];
  A_shared_local_1[14] = A_shared[((((int)threadIdx.x) * 16) + 238)];
  A_shared_local_1[15] = A_shared[((((int)threadIdx.x) * 16) + 239)];
  *(float4*)(B_shared_local_1 + 0) = *(float4*)(B_shared + ((((int)threadIdx.y) * 4) + 896));
  for (int i0_2_7 = 0; i0_2_7 < 16; ++i0_2_7) {
    for (int i1_2_7 = 0; i1_2_7 < 4; ++i1_2_7) {
      Y_local[((i0_2_7 * 4) + i1_2_7)] = (Y_local[((i0_2_7 * 4) + i1_2_7)] + (A_shared_local_1[i0_2_7] * B_shared_local_1[i1_2_7]));
    }
  }
  for (int ax0 = 0; ax0 < 16; ++ax0) {
    *(float4*)(Y + (((((((int)blockIdx.x) * 36864) + (((int)threadIdx.x) * 18432)) + (ax0 * 1152)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.y) * 4))) = *(float4*)(Y_local + (ax0 * 4));
  }
}


