// #pragma once

#include <type_traits>
#include <cstdint>
#include <cstdio>
#include <unistd.h>

#define CUTLASS_HOST_DEVICE __forceinline__ __device__ __host__
#define CUTLASS_DEVICE __forceinline__ __device__

#define CUTLASS_ENABLE_L2_PREFETCH 1
#define CUDA_CP_ASYNC_ACTIVATED 1

struct IntPair { // pair
    int x; // continuous / row
    int y; // strided / column

    CUTLASS_HOST_DEVICE
    IntPair swap() const {
        return {y, x};
    }

    CUTLASS_HOST_DEVICE
    IntPair operator+(const IntPair &o) const {
        return {x + o.x, y + o.y};
    }

    CUTLASS_HOST_DEVICE
    IntPair &operator+=(const IntPair &o) {
        x += o.x, y += o.y;
        return *this;
    }
};

template <typename T>
struct sizeof_bits {
    static int const value = int(sizeof(T) * 8);
};

template<typename T, int N, int Align = 16>
struct AlignedBuffer {
    using Storage = uint8_t;
    static int const kBytes = (sizeof_bits<T>::value * N + 7) / 8;
    alignas(Align) Storage storage[kBytes];

    CUTLASS_HOST_DEVICE
    T *data() {
        return reinterpret_cast<T *>(storage);
    }
};

template<typename T, int N>
class Array {
public:
    T storage[N];

    CUTLASS_HOST_DEVICE
    void fill(const T &value) {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            storage[i] = value; // static_cast<Storage>(value) ???
        }
    }

    CUTLASS_HOST_DEVICE
    void clear() {
        fill(T(0));
    }

    CUTLASS_HOST_DEVICE
    void print() const {
        for (int i = 0; i < N; i++) printf("%f ", storage[i]);
        printf("\n");
    }
};

template <typename T, int N, int Alignment = sizeof_bits<T>::value * N / 8>
class alignas(Alignment) AlignedArray: public Array<T, N> {};

using LongIndex = int64_t;
using Index = int32_t;

enum LayoutType {
    ColumnMajorType, RowMajorType
};

template<typename T, LayoutType Layout>
class TensorRef {
private:
    T *ptr_;
    LongIndex stride_;

public:
    CUTLASS_HOST_DEVICE
    TensorRef(T *ptr, LongIndex stride) : ptr_(ptr), stride_(stride) {}

    CUTLASS_HOST_DEVICE
    LongIndex offset(const IntPair &coord) const {
        if (Layout == RowMajorType) return LongIndex(coord.x) * stride_ + coord.y;
        else return coord.x + LongIndex(coord.y) * stride_;
    }
    
    CUTLASS_HOST_DEVICE
    T &at(IntPair coord) {
        return ptr_[offset(coord)];
    }
};

inline __device__ unsigned cutlass_get_smem_pointer(void *ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

inline __device__ unsigned cutlass_get_smem_pointer(void const *ptr) {
    return cutlass_get_smem_pointer(const_cast<void *>(ptr));
}

CUTLASS_DEVICE
void shared_load(Array<float, 4> &D, void const *ptr) {
    unsigned addr = cutlass_get_smem_pointer(ptr);
    uint4 v;
    asm volatile ("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];" : 
    "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
    D = reinterpret_cast<Array<float, 4> const &>(v);
}

template<int SizeInBytes = 4>
CUTLASS_DEVICE
void cp_async(void *smem_ptr, void const *global_ptr, bool pred_guard = true) {
#if CUDA_CP_ASYNC_ACTIVATED
    unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %0, 0;\n"
#if CUTLASS_ENABLE_L2_PREFETCH
        "  @p cp.async.ca.shared.global.L2::128B [%1], [%2], %3;\n"
#else
        "  @p cp.async.ca.shared.global [%1], [%2], %3;\n"
#endif
        "}\n" ::"r"((int)pred_guard),
        "r"(smem_int_ptr), "l"(global_ptr), "n"(SizeInBytes)
    );
#else
    using AccessType = Array<uint8_t, SizeInBytes>;
    if (pred_guard) *static_cast<AccessType *>(smem_ptr) = *static_cast<AccessType const *>(global_ptr);
#endif
}

template<int SizeInBytes = 4>
CUTLASS_DEVICE
void cp_async_zfill(void *smem_ptr, void const *global_ptr, bool pred_guard = true) {
#if CUDA_CP_ASYNC_ACTIVATED
    unsigned smem_int_ptr = cutlass_get_smem_pointer(smem_ptr);
    int src_in_bytes = (pred_guard ? SizeInBytes : 0);
    asm volatile(
#if CUTLASS_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
#else
        "cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
#endif
        "l"(global_ptr), "n"(SizeInBytes), "r"(src_in_bytes)
    );
#else
    using AccessType = Array<uint8_t, SizeInBytes>;
    if (pred_guard) *static_cast<AccessType *>(smem_ptr) = *static_cast<AccessType const *>(global_ptr);
    else {
        AccessType zeros;
        zeros.clear();
        *static_cast<AccessType *>(smem_ptr) = zeros;
    }
#endif
}

CUTLASS_DEVICE
void cp_async_fence() {
#if CUDA_CP_ASYNC_ACTIVATED
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template<int N>
CUTLASS_DEVICE
void cp_async_wait() {
#if CUDA_CP_ASYNC_ACTIVATED
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

CUTLASS_HOST_DEVICE
void warp_mma(Array<float, 64> &D, const Array<float, 8> &A, const Array<float, 8> &B, const Array<float, 64> &C) {
    TensorRef<float const, ColumnMajorType> a_ref(reinterpret_cast<float const *>(&A), 8);
    TensorRef<float const, RowMajorType> b_ref(reinterpret_cast<float const *>(&B), 8);
    TensorRef<float, RowMajorType> d_ref(reinterpret_cast<float *>(&D), 8);
    D = C;
    int k = 0;
    #pragma unroll
    for (int n = 0; n < 8; n += 2) {
        #pragma unroll
        for (int m = 0; m < 8; m += 2) {
            int m_serpentine = (n % 4) ? (6 - m) : m;
            {
                IntPair mn = {m_serpentine, n};
                IntPair mk = {m_serpentine, k};
                IntPair kn = {k, n};
                d_ref.at(mn) = a_ref.at(mk) * b_ref.at(kn) + d_ref.at(mn);
            }
            {
                IntPair mn = {m_serpentine + 1, n};
                IntPair mk = {m_serpentine + 1, k};
                IntPair kn = {k, n};
                d_ref.at(mn) = a_ref.at(mk) * b_ref.at(kn) + d_ref.at(mn);
            }
            {
                IntPair mn = {m_serpentine + 1, n + 1};
                IntPair mk = {m_serpentine + 1, k};
                IntPair kn = {k, n + 1};
                d_ref.at(mn) = a_ref.at(mk) * b_ref.at(kn) + d_ref.at(mn);
            }
            {
                IntPair mn = {m_serpentine, n + 1};
                IntPair mk = {m_serpentine, k};
                IntPair kn = {k, n + 1};
                d_ref.at(mn) = a_ref.at(mk) * b_ref.at(kn) + d_ref.at(mn);
            }
        }
    }
}

CUTLASS_DEVICE
void load_warp_frag(Array<float, 8> &frag, Array<float, 4> *pointer_, int inc) {
    Array<float, 4> *dst_ptr = reinterpret_cast<Array<float, 4> *>(&frag);
    #pragma unroll
    for (int n = 0; n < 2; n++) {
        void const *ptr = pointer_ + n * inc;
        shared_load(dst_ptr[n], ptr);
    }
}

class MmaSharedStorage {
private:
    AlignedBuffer<float, 4096> operand_A;
    AlignedBuffer<float, 4096> operand_B;

public:
    CUTLASS_HOST_DEVICE
    float *operand_A_ptr() {
        return operand_A.data();
    }

    CUTLASS_HOST_DEVICE
    float *operand_B_ptr() {
        return operand_B.data();
    }
};

CUTLASS_DEVICE
void global_store(const AlignedArray<float, 1> &D, void *ptr, bool pred_guard) {
    uint32_t const &data = reinterpret_cast<uint32_t const &>(D);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %2, 0;\n"
        "  @p st.global.u32 [%0], %1;\n"
        "}\n"
        :
        : "l"(ptr), "r"(data), "r"((int)pred_guard)
    );
}

class EpilogueSharedStorage {
private:
    AlignedBuffer<float, 2320> storage; // <16, 145>

public:
    CUTLASS_DEVICE
    float *data() {
        return storage.data();
    }
};

union SharedStorage {
    MmaSharedStorage main_loop;
    EpilogueSharedStorage epilogue;
};

__global__ 
void kernel(float * A, float * B, float * C) { // C = A * B
    extern __shared__ int SharedStorageBase[];
    SharedStorage *shared_storage_ptr = reinterpret_cast<SharedStorage *>(SharedStorageBase);
    SharedStorage &shared_storage = *shared_storage_ptr;

    IntPair threadblock_tile_offset = {int(blockIdx.x / 8), int(blockIdx.x % 8)};
    IntPair threadblock_offset = {threadblock_tile_offset.x * 128, threadblock_tile_offset.y * 128};
    __syncthreads();
    int thread_idx = threadIdx.x;
    AlignedArray<float, 1> *global_memory_A = reinterpret_cast<AlignedArray<float, 1> *>(A) + threadblock_offset.x + thread_idx % 128 + 1024 * (thread_idx / 128);
    AlignedArray<float, 1> *global_memory_B = reinterpret_cast<AlignedArray<float, 1> *>(B) + threadblock_offset.y + thread_idx % 128 + 1024 * (thread_idx / 128);
    int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    int lane_idx = threadIdx.x % 32;

    float *smem_iterator_A_ = shared_storage.main_loop.operand_A_ptr() + thread_idx;
    float *smem_iterator_B_ = shared_storage.main_loop.operand_B_ptr() + thread_idx;
    Array<float, 4> *warp_tile_iterator_A_ = reinterpret_cast<Array<float, 4> *>(shared_storage.main_loop.operand_A_ptr() + ((lane_idx / 16) * 2 + lane_idx % 2) * 4) + (warp_idx % 4) * 8;
    Array<float, 4> *warp_tile_iterator_B_ = reinterpret_cast<Array<float, 4> *>(shared_storage.main_loop.operand_B_ptr() + ((lane_idx % 16) / 2) * 4) + (warp_idx / 4) * 16;
    Array<float, 64> accumulators;
    accumulators.clear();
    int gemm_k_iterations = 128;
    int smem_write_stage_idx = 0;
    int smem_read_stage_idx = 0;

    #pragma unroll
    for (int stage = 0; stage < 3; ++stage) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            AlignedArray<float, 1> *dst_ptr = reinterpret_cast<AlignedArray<float, 1> *>(smem_iterator_A_ + 256 * j + 1024 * smem_write_stage_idx);
            cp_async_zfill<4>(dst_ptr, global_memory_A + (128 - gemm_k_iterations) * 8192 + j * 2048);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            AlignedArray<float, 1> *dst_ptr = reinterpret_cast<AlignedArray<float, 1> *>(smem_iterator_B_ + 256 * j + 1024 * smem_write_stage_idx);
            cp_async_zfill<4>(dst_ptr, global_memory_B + (128 - gemm_k_iterations) * 8192 + j * 2048);
        }
        ++smem_write_stage_idx;
        --gemm_k_iterations;
        cp_async_fence();
    }
    cp_async_wait<2>();
    __syncthreads();

    Array<float, 8> warp_loaded_frag_A[2];
    Array<float, 8> warp_loaded_frag_B[2];
    load_warp_frag(warp_loaded_frag_A[0], warp_tile_iterator_A_, 4);
    load_warp_frag(warp_loaded_frag_B[0], warp_tile_iterator_B_, 8);
    #pragma unroll 1
    for ( ; gemm_k_iterations > -3; ) {
        #pragma unroll
        for (int warp_mma_k = 0; warp_mma_k < 8; ++warp_mma_k) {
            load_warp_frag(warp_loaded_frag_A[(warp_mma_k + 1) % 2], warp_tile_iterator_A_ + ((warp_mma_k + 1) % 8 + smem_read_stage_idx * 8) * 32, 4);
            load_warp_frag(warp_loaded_frag_B[(warp_mma_k + 1) % 2], warp_tile_iterator_B_ + ((warp_mma_k + 1) % 8 + smem_read_stage_idx * 8) * 32, 8);

            warp_mma(accumulators, warp_loaded_frag_A[warp_mma_k % 2], warp_loaded_frag_B[warp_mma_k % 2], accumulators);

            if (warp_mma_k < 4 && gemm_k_iterations > 0) {
                AlignedArray<float, 1> *dst_ptr_A = reinterpret_cast<AlignedArray<float, 1> *>(smem_iterator_A_ + 256 * warp_mma_k + 1024 * smem_write_stage_idx);
                cp_async<4>(dst_ptr_A, global_memory_A + (128 - gemm_k_iterations) * 8192 + warp_mma_k * 2048);
                AlignedArray<float, 1> *dst_ptr_B = reinterpret_cast<AlignedArray<float, 1> *>(smem_iterator_B_ + 256 * warp_mma_k + 1024 * smem_write_stage_idx);
                cp_async<4>(dst_ptr_B, global_memory_B + (128 - gemm_k_iterations) * 8192 + warp_mma_k * 2048);
            }
            if (warp_mma_k == 6) {
                cp_async_fence();
                cp_async_wait<2>();
                __syncthreads();

                if (smem_write_stage_idx == 3) smem_write_stage_idx = 0;
                else ++smem_write_stage_idx;
                if (smem_read_stage_idx == 3) smem_read_stage_idx = 0;
                else ++smem_read_stage_idx;
                --gemm_k_iterations;
            }
        }
    }



    AlignedArray<float, 1> *warp_tile_iterator_ = reinterpret_cast<AlignedArray<float, 1> *>(shared_storage.epilogue.data())
        + ((lane_idx / 16) * 2 + lane_idx % 2) * 145 + ((lane_idx % 16) / 2) * 4 + (warp_idx % 4) * 4 * 145 + (warp_idx / 4) * 64;
    AlignedArray<float, 1> const *shared_load_iterator_ = reinterpret_cast<AlignedArray<float, 1> const *>(shared_storage.epilogue.data())
        + ((warp_idx / 2) * 4 + (warp_idx % 2)) * 145 + lane_idx;
    AlignedArray<float, 1> *global_memory_C = reinterpret_cast<AlignedArray<float, 1> *>(
        C + ((warp_idx / 2) * 32 + (warp_idx % 2) * 4 + threadblock_offset.x) * 1024 + lane_idx + threadblock_offset.y);
    Array<float, 4> const * accum_fragment_iterator = reinterpret_cast<Array<float, 4> const *>(&accumulators);

    #pragma unroll(8)
    for (int iter = 0; iter < 8; ++iter) {
        __syncthreads();
        Array<float, 8> accum_fragment;
        Array<float, 4> *frag_ptr_0 = reinterpret_cast<Array<float, 4> *>(&accum_fragment);
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            frag_ptr_0[n] = accum_fragment_iterator[iter * 2 + n];
        }
        AlignedArray<float, 1> const *scalar_frag_ptr = reinterpret_cast<AlignedArray<float, 1> const *>(&accum_fragment);
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                warp_tile_iterator_[n * 32 + s] = scalar_frag_ptr[n * 4 + s];
            }
        }

        __syncthreads();
        Array<float, 8> aligned_accum_fragment;
        AlignedArray<float, 1> *frag_ptr_1 = reinterpret_cast<AlignedArray<float, 1> *>(&aligned_accum_fragment);
        #pragma unroll
        for (int group = 0; group < 2; ++group) {
            AlignedArray<float, 1> const *memory_pointer = shared_load_iterator_ + group * 2 * 145;
            #pragma unroll
            for (int column = 0; column < 4; ++column) {
                frag_ptr_1[group * 4 + column] = memory_pointer[column * 32];
            }
        }
        AlignedArray<float, 1> *pointer = global_memory_C + (iter % 4) * 1024 + (iter / 4) * 16384;
        AlignedArray<float, 1> const *frag_ptr_2 = reinterpret_cast<AlignedArray<float, 1> const *>(&aligned_accum_fragment);
        #pragma unroll
        for (int group = 0; group < 2; ++group) {
            AlignedArray<float, 1> *memory_pointer = reinterpret_cast<AlignedArray<float, 1> *>(pointer + group * 8192);
            #pragma unroll
            for (int column = 0; column < 4; ++column) {
                global_store(frag_ptr_2[group * 4 + column], (void *)&memory_pointer[0], true);
                memory_pointer += 32;
            }
        }
    }
}

void run_cutlassMultiStage(int M, int N, int K, float * A, float * B, float * C, float alpha, float beta) {
    dim3 grid(64, 1, 1), block(256, 1, 1);
    int smem_size = int(sizeof(SharedStorage));
    kernel<<<grid, block, smem_size, nullptr>>>(A, B, C);
}

