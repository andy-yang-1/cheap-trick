template <typename T, typename Tv, int TWO_BUFFER>
__global__
__launch_bounds__(256) 
void gemm_128x128(int m, int n, int k, T alpha,
                                         const T *A, int lda, const T *B,
                                         int ldb, T beta, T *C, int ldc) {
  const int TvSize = sizeof(Tv) / sizeof(T), idx = threadIdx.x,
            x_range = ((int)blockIdx.x + 1 << 7) - m,
            y_range = ((int)blockIdx.y + 1 << 7) - n, x_a = idx & 31,
            y_a = idx >> 5, x_b = idx & 1, y_b = idx >> 1, x_c = idx & 15,
            y_c = idx >> 4;
  if (x_range > 0) {
    A -= x_range;
    C -= x_range;
  }
  if (y_range > 0) {
    B -= y_range * ldb;
    C -= y_range * ldc;
  }

  A += blockIdx.x << 7;
  B += (blockIdx.y << 7) * ldb;
  C +=
      ((blockIdx.x << 7) + (x_c << 2)) + ((blockIdx.y << 7) + (y_c << 2)) * ldc;

  __shared__ T s_buffer[2048];
  T *s_A = s_buffer;
  T *s_B = s_buffer + 1024;

#if TWO_BUFFER
  __shared__ T s_tbuffer[2048];
  T *s_tA = s_tbuffer;
  T *s_tB = s_tbuffer + 1024;
#endif
  Tv ansC[2][2][TvSize];
  memset(ansC, 0, sizeof(ansC));

  Tv resA = ((Tv *)A)[x_a + y_a * (lda >> 2)];
  Tv resB = ((Tv *)B)[x_b + y_b * (ldb >> 2)];

  for (int l = 0; l < k; l += 8) {
    ((Tv *)s_A)[idx] = resA;
    for (int i = 0; i < TvSize; ++i)
      s_B[(((x_b << 2) + i) << 7) + y_b] = ((T *)&resB)[i];
    __syncthreads();

    if (l + 8 < k) {
      A += lda << 3;
      B += 8;
      resA = ((Tv *)A)[x_a + y_a * (lda >> 2)];
      resB = ((Tv *)B)[x_b + y_b * (ldb >> 2)];
    }

#pragma unroll(4)
    for (int j = 0; j < 8; ++j) {
      Tv a[2], b[2];
      for (int p = 0; p < 2; ++p)
        a[p] = ((Tv *)(s_A + (j << 7) + (x_c << 2)))[p << 4];
      for (int p = 0; p < 2; ++p)
        b[p] = ((Tv *)(s_B + (j << 7) + (y_c << 2)))[p << 4];
      for (int p = 0; p < 2; ++p)
        for (int q = 0; q < 2; ++q)
          for (int i = 0; i < TvSize; ++i)
            for (int j = 0; j < TvSize; ++j)
              ((T *)&ansC[p][q][i])[j] += ((T *)&a[p])[j] * ((T *)&b[q])[i];
    }

#if TWO_BUFFER
    {
      T *tmp_A = s_A;
      s_A = s_tA;
      s_tA = tmp_A;
    }
    {
      T *tmp_B = s_B;
      s_B = s_tB;
      s_tB = tmp_B;
    }
#else
    __syncthreads();
#endif
  }

  for (int p = 0; p < 2; ++p)
    for (int q = 0; q < 2; ++q)
      for (int i = 0; i < TvSize; ++i) {
        Tv *devC = ((Tv *)(C + ldc * (q * 64 + i))) + (p << 4), resC = *devC;
        for (int j = 0; j < TvSize; ++j)
          ((T *)&resC)[j] =
              alpha * ((T *)&ansC[p][q][i])[j] + beta * ((T *)&resC)[j];
        *devC = resC;
      }
}

void run_v1gemm(int M , int N , int K , 
                const float* A , const float* B, float *C, float alpha , float beta){

    int lda = K;
    int ldb = K;
    int ldc = N;
    int ldd = N;
    

    const int TILE_WIDTH = 128;
    const dim3 blockDim(256), gridDim((M + TILE_WIDTH - 1) / TILE_WIDTH,
                                      (N + TILE_WIDTH - 1) / TILE_WIDTH);

    gemm_128x128<float, float, 1><<<gridDim, blockDim>>>(
        M, N, K, alpha, A, lda, B, ldb, beta,
        C, ldc);
}