#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
#include <cuda_fp16.h>
__device__ half max(half a, half b)
{
  return __hgt(__half(a), __half(b)) ? a : b;
}
__device__ half min(half a, half b)
{
  return __hlt(__half(a), __half(b)) ? a : b;
}
#else

typedef unsigned short uint16_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef int int32_t;
typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

#define TVM_FORCE_INLINE inline __attribute__((always_inline))
#define TVM_XINLINE TVM_FORCE_INLINE __device__ __host__
#define TVM_ALIGNED(x) __attribute__ ((aligned(x)))
#define TVM_HALF_OPERATOR(RTYPE, OP)                              \
  TVM_XINLINE RTYPE operator OP (half a, half b) {                \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (half a, T b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE RTYPE operator OP (T a, half b) {                   \
    return RTYPE(float(a) OP float(b));                           \
  }

#define TVM_HALF_ASSIGNOP(AOP, OP)                                \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const T& a) {                    \
    return *this = half(float(*this) OP float(a));                \
  }                                                               \
  template<typename T>                                            \
  TVM_XINLINE half operator AOP (const volatile T& a) volatile {  \
    return *this = half(float(*this) OP float(a));                \
  }

class TVM_ALIGNED(2) half {
 public:
  uint16_t half_;

  static TVM_XINLINE half Binary(uint16_t value) {
    half res;
    res.half_ = value;
    return res;
  }

  TVM_XINLINE half() {}

  TVM_XINLINE half(const float& value) { constructor(value); }
  TVM_XINLINE explicit half(const double& value) { constructor(value); }
  TVM_XINLINE explicit half(const int8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint8_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const int32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint32_t& value) { constructor(value); }
  TVM_XINLINE explicit half(const long long& value) { constructor(value); }
  TVM_XINLINE explicit half(const uint64_t& value) { constructor(value); }

  TVM_XINLINE operator float() const {                          \
    return float(half2float(half_));                            \
  }                                                             \
  TVM_XINLINE operator float() const volatile {                 \
    return float(half2float(half_));                            \
  }


  TVM_HALF_ASSIGNOP(+=, +)
  TVM_HALF_ASSIGNOP(-=, -)
  TVM_HALF_ASSIGNOP(*=, *)
  TVM_HALF_ASSIGNOP(/=, /)

  TVM_XINLINE half operator+() {
    return *this;
  }

  TVM_XINLINE half operator-() {
    return half(-float(*this));
  }

  TVM_XINLINE half operator=(const half& a) {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) {
    return *this = half(a);
  }

  TVM_XINLINE half operator=(const half& a) volatile {
    half_ = a.half_;
    return a;
  }

  template<typename T>
  TVM_XINLINE half operator=(const T& a) volatile {
    return *this = half(a);
  }

 private:
  union Bits {
    float f;
    int32_t si;
    uint32_t ui;
  };

  static int const fp16FractionBits = 10;
  static int const fp32FractionBits = 23;
  static int32_t const fp32FractionMask = ~(~0u << fp32FractionBits);   // == 0x7fffff
  static int32_t const fp32HiddenBit = 1 << fp32FractionBits;   // == 0x800000
  static int const shift = fp32FractionBits - fp16FractionBits;   // == 13
  static int const shiftSign = 16;
  static int32_t const expAdjust = 127 - 15;   // exp32-127 = exp16-15, so exp16 = exp32 - (127-15)

  static int32_t const infN = 0x7F800000;   // flt32 infinity
  static int32_t const maxN = 0x477FFFFF;   // max flt32 that's a flt16 normal after >> by shift
  static int32_t const minN = 0x38800000;   // min flt16 normal as a flt32
  static int32_t const maxZ = 0x33000000;   // max fp32 number that's still rounded to zero in fp16
  static int32_t const signN = 0x80000000;  // flt32 sign bit

  static int32_t const infC = infN >> shift;
  static int32_t const nanN = (infC + 1) << shift;   // minimum flt16 nan as a flt32
  static int32_t const maxC = maxN >> shift;
  static int32_t const minC = minN >> shift;
  static int32_t const signC = signN >> shiftSign;  // flt16 sign bit

  static int32_t const mulN = 0x52000000;  // (1 << 23) / minN
  static int32_t const mulC = 0x33800000;  // minN / (1 << (23 - shift))

  static int32_t const subC = 0x003FF;  // max flt32 subnormal down shifted
  static int32_t const norC = 0x00400;  // min flt32 normal down shifted

  static int32_t const maxD = infC - maxC - 1;
  static int32_t const minD = minC - subC - 1;

  TVM_XINLINE uint16_t float2half(const float& value) const {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  // Same as above routine, except for addition of volatile keyword
  TVM_XINLINE uint16_t float2half(
    const volatile float& value) const volatile {
    Bits v;
    v.f = value;
    uint32_t sign = v.si & signN;    // grab sign bit
    v.si ^= sign;                    // clear sign bit from v
    sign >>= shiftSign;              // logical shift sign to fp16 position

    if (v.si <= maxZ) {
      // Handle eventual zeros here to ensure
      // vshift will not exceed 32 below.
      v.ui = 0;
    } else if (v.si < minN) {
      // Handle denorms
      uint32_t exp32 = v.ui >> fp32FractionBits;
      int32_t exp16 = exp32 - expAdjust;
      // If exp16 == 0 (just into the denorm range), then significant should be shifted right 1.
      // Smaller (so negative) exp16 values should result in greater right shifts.
      uint32_t vshift = 1 - exp16;
      uint32_t significand = fp32HiddenBit | (v.ui & fp32FractionMask);
      v.ui = significand >> vshift;
      v.ui += (v.ui & 0x3fff) != 0x1000 || (significand & 0x7ff) ? 0x1000 : 0;
    } else if (v.si <= maxN) {
      // Handle norms
      v.ui += (v.ui & 0x3fff) != 0x1000 ? 0x1000 : 0;
      v.ui -= expAdjust << fp32FractionBits;
    } else if (v.si <= infN) {
      v.si = infN;
    } else if (v.si < nanN) {
      v.si = nanN;
    }

    v.ui >>= shift;
    return sign | (v.ui & 0x7fff);
  }

  TVM_XINLINE float half2float(const uint16_t& value) const {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  TVM_XINLINE float half2float(
    const volatile uint16_t& value) const volatile {
    Bits v;
    v.ui = value;
    int32_t sign = v.si & signC;
    v.si ^= sign;
    sign <<= shiftSign;
    v.si ^= ((v.si + minD) ^ v.si) & -(v.si > subC);
    v.si ^= ((v.si + maxD) ^ v.si) & -(v.si > maxC);
    Bits s;
    s.si = mulC;
    s.f *= v.si;
    int32_t mask = -(norC > v.si);
    v.si <<= shift;
    v.si ^= (s.si ^ v.si) & mask;
    v.si |= sign;
    return v.f;
  }

  template<typename T>
  TVM_XINLINE void constructor(const T& value) {
    half_ = float2half(float(value));
  }
};

TVM_HALF_OPERATOR(half, +)
TVM_HALF_OPERATOR(half, -)
TVM_HALF_OPERATOR(half, *)
TVM_HALF_OPERATOR(half, /)
TVM_HALF_OPERATOR(bool, >)
TVM_HALF_OPERATOR(bool, <)
TVM_HALF_OPERATOR(bool, >=)
TVM_HALF_OPERATOR(bool, <=)

TVM_XINLINE half __float2half_rn(const float a) {
  return half(a);
}
#endif


// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

#define CUDA_UNSUPPORTED_HALF_MATH_BINARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x, half y) {   \
  float tmp_x = __half2float(x);                                          \
  float tmp_y = __half2float(y);                                          \
  float result = FP32_MATH_NAME(tmp_x, tmp_y);                            \
  return __float2half(result);                                            \
}

#define CUDA_UNSUPPORTED_HALF_MATH_UNARY(HALF_MATH_NAME, FP32_MATH_NAME) \
static inline __device__ __host__ half HALF_MATH_NAME(half x) {          \
  float tmp_x = __half2float(x);                                         \
  float result = FP32_MATH_NAME(tmp_x);                                  \
  return __float2half(result);                                           \
}

// Some fp16 math functions are not supported in cuda_fp16.h,
// so we define them here to make sure the generated CUDA code
// is valid.
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ >= 530)
CUDA_UNSUPPORTED_HALF_MATH_BINARY(hpow, powf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htanh, tanhf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(htan, tanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hatan, atanf)
CUDA_UNSUPPORTED_HALF_MATH_UNARY(herf, erf)
#else
CUDA_UNSUPPORTED_HALF_MATH_UNARY(hexp, exp)
#endif
#endif

#undef CUDA_UNSUPPORTED_HALF_MATH_BINARY
#undef CUDA_UNSUPPORTED_HALF_MATH_UNARY

struct __align__(8) half4 {
  __half x, y, z, w;
  __host__ __device__ half4() : x(__half(0)), y(__half(0)), z(__half(0)), w(__half(0)) {}
  __host__ __device__ half4(__half x, __half y, __half z, __half w) : x(x), y(y), z(z), w(w) {}

};
__host__ __device__ half4 make_half4(__half x, __half y, __half z, __half w) {
    return half4(x, y, z, w);
}

#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

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
extern "C" __global__ void __launch_bounds__(32) default_function_kernel(half* __restrict__ k, int* __restrict__ kv_indptr, float* __restrict__ lse, int* __restrict__ m_array, int* __restrict__ mask, int* __restrict__ mn_indptr, half* __restrict__ output, half* __restrict__ q, int* __restrict__ q_indptr, int* __restrict__ q_rope_position, half* __restrict__ v, float attn_score_scaling_factor, int batch_size, int kv_indptr_elem_offset, int kv_len, int q_indptr_elem_offset, int q_rope_position_elem_offset, int qo_len, float rope_scale, float rope_theta, int rotary_mode, int tree_size);
extern "C" __global__ void __launch_bounds__(32) default_function_kernel(half* __restrict__ k, int* __restrict__ kv_indptr, float* __restrict__ lse, int* __restrict__ m_array, int* __restrict__ mask, int* __restrict__ mn_indptr, half* __restrict__ output, half* __restrict__ q, int* __restrict__ q_indptr, int* __restrict__ q_rope_position, half* __restrict__ v, float attn_score_scaling_factor, int batch_size, int kv_indptr_elem_offset, int kv_len, int q_indptr_elem_offset, int q_rope_position_elem_offset, int qo_len, float rope_scale, float rope_theta, int rotary_mode, int tree_size) {
  int tile_id[1];
  int batch_idx[1];
  int batch_rows[1];
  int batch_tiles[1];
  int kv_chunk_len[1];
  __shared__ float m_smem[32];
  __shared__ float d_smem[32];
  float O_local[1];
  __shared__ half Q_smem[32];
  __shared__ half K_smem[16];
  __shared__ half V_smem[16];
  float S_local[16];
  __shared__ float S_smem[512];
  float m_prev[1];
  float m_new[1];
  float d_new[1];
  __shared__ float m_prev_smem[32];
  tile_id[0] = ((int)blockIdx.x);
  batch_idx[0] = 0;
  batch_rows[0] = (q_indptr[(q_indptr_elem_offset + 1)] - q_indptr[q_indptr_elem_offset]);
  batch_tiles[0] = ((batch_rows[0] + 31) >> 5);
  while (1) {
    if (!(((batch_idx[0] < batch_size)))) { break; }
    while (1) {
      if (!(((batch_tiles[0] <= tile_id[0]) && (batch_idx[0] < batch_size)))) { break; }
      tile_id[0] = (tile_id[0] - batch_tiles[0]);
      batch_idx[0] = (batch_idx[0] + 1);
      if (batch_idx[0] < batch_size) {
        int b_idx = batch_idx[0];
        batch_rows[0] = (q_indptr[((b_idx + q_indptr_elem_offset) + 1)] - q_indptr[(b_idx + q_indptr_elem_offset)]);
        batch_tiles[0] = ((batch_rows[0] + 31) >> 5);
      }
    }
    if ((batch_idx[0] < batch_size)) {
      int b_idx_1 = batch_idx[0];
      int L_start = ((tile_id[0] * 32) + q_indptr[(b_idx_1 + q_indptr_elem_offset)]);
      kv_chunk_len[0] = (kv_indptr[((b_idx_1 + kv_indptr_elem_offset) + 1)] - kv_indptr[(b_idx_1 + kv_indptr_elem_offset)]);
      __syncthreads();
      m_smem[((int)threadIdx.x)] = -5.000000e+04f;
      d_smem[((int)threadIdx.x)] = 1.000000e+00f;
      O_local[0] = 0.000000e+00f;
      __syncthreads();
      if (((int)threadIdx.x) < 8) {
        for (int li_lj_fused_3_s = 0; li_lj_fused_3_s < 4; ++li_lj_fused_3_s) {
          if ((((((int)threadIdx.x) * 4) + L_start) + li_lj_fused_3_s) < q_indptr[((b_idx_1 + q_indptr_elem_offset) + 1)]) {
            half condval;
            if ((rotary_mode == 1)) {
              condval = ((((half)__cosf(((((float)q_rope_position[((((((int)threadIdx.x) * 4) + li_lj_fused_3_s) + L_start) + q_rope_position_elem_offset)]) * rope_scale) / powf(rope_theta, 0.000000e+00f)))) * q[(((((int)threadIdx.x) * 4) + li_lj_fused_3_s) + L_start)]) + (((half)__sinf(((((float)q_rope_position[((((((int)threadIdx.x) * 4) + li_lj_fused_3_s) + L_start) + q_rope_position_elem_offset)]) * rope_scale) / powf(rope_theta, 0.000000e+00f)))) * q[(((((int)threadIdx.x) * 4) + li_lj_fused_3_s) + L_start)]));
            } else {
              condval = q[(((((int)threadIdx.x) * 4) + li_lj_fused_3_s) + L_start)];
            }
            Q_smem[((((int)threadIdx.x) * 4) + li_lj_fused_3_s)] = condval;
          } else {
            Q_smem[((((int)threadIdx.x) * 4) + li_lj_fused_3_s)] = __float2half_rn(0.000000e+00f);
          }
        }
      }
      __syncthreads();
      for (int iterator = 0; iterator < ((kv_chunk_len[0] + 15) >> 4); ++iterator) {
        int L_kv_base = kv_indptr[(b_idx_1 + kv_indptr_elem_offset)];
        __syncthreads();
        if (((int)threadIdx.x) < 4) {
          for (int lz_ly_fused_3_s = 0; lz_ly_fused_3_s < 4; ++lz_ly_fused_3_s) {
            if ((((iterator * 16) + (((int)threadIdx.x) * 4)) + lz_ly_fused_3_s) < kv_chunk_len[0]) {
              half condval_1;
              if ((rotary_mode == 1)) {
                condval_1 = ((((half)__cosf(((((float)q_rope_position[(((((iterator * 16) + (((int)threadIdx.x) * 4)) + lz_ly_fused_3_s) + L_kv_base) + q_rope_position_elem_offset)]) * rope_scale) / powf(rope_theta, 0.000000e+00f)))) * k[((((iterator * 16) + (((int)threadIdx.x) * 4)) + lz_ly_fused_3_s) + L_kv_base)]) + (((half)__sinf(((((float)q_rope_position[(((((iterator * 16) + (((int)threadIdx.x) * 4)) + lz_ly_fused_3_s) + L_kv_base) + q_rope_position_elem_offset)]) * rope_scale) / powf(rope_theta, 0.000000e+00f)))) * k[((((iterator * 16) + (((int)threadIdx.x) * 4)) + lz_ly_fused_3_s) + L_kv_base)]));
              } else {
                condval_1 = k[((((iterator * 16) + (((int)threadIdx.x) * 4)) + lz_ly_fused_3_s) + L_kv_base)];
              }
              K_smem[((((int)threadIdx.x) * 4) + lz_ly_fused_3_s)] = condval_1;
              V_smem[((((int)threadIdx.x) * 4) + lz_ly_fused_3_s)] = v[((((iterator * 16) + (((int)threadIdx.x) * 4)) + lz_ly_fused_3_s) + L_kv_base)];
            } else {
              K_smem[((((int)threadIdx.x) * 4) + lz_ly_fused_3_s)] = __float2half_rn(0.000000e+00f);
              V_smem[((((int)threadIdx.x) * 4) + lz_ly_fused_3_s)] = __float2half_rn(0.000000e+00f);
            }
          }
        }
        __syncthreads();
        // print Q, K, V
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            // print Q
            printf(">>>>>>>>>>>>>> Q\n");
            for (int row = 0; row < 32; ++row) {
              for (int col = 0; col < 1; ++col) {
                printf("%f ", __half2float(Q_smem[row * 1 + col]));
              }
              printf("\n");
            }
            // print K
            printf(">>>>>>>>>>>>>> K\n");
            for (int row = 0; row < 16; ++row) {
              for (int col = 0; col < 1; ++col) {
                printf("%f ", __half2float(K_smem[row * 1 + col]));
              }
              printf("\n");
            }
            // print V
            printf(">>>>>>>>>>>>>> V\n");
            for (int row = 0; row < 16; ++row) {
              for (int col = 0; col < 1; ++col) {
                printf("%f ", __half2float(V_smem[row * 1 + col]));
              }
              printf("\n");
            }
        }
        for (int li_1_init = 0; li_1_init < 4; ++li_1_init) {
          for (int lj_1_init = 0; lj_1_init < 4; ++lj_1_init) {
            S_local[((li_1_init * 4) + lj_1_init)] = 0.000000e+00f;
          }
        }
        for (int li_1 = 0; li_1 < 4; ++li_1) {
          for (int lj_1 = 0; lj_1 < 4; ++lj_1) {
            for (int lk_1 = 0; lk_1 < 8; ++lk_1) {
              if (lk_1 < 1) {
                S_local[((li_1 * 4) + lj_1)] = (S_local[((li_1 * 4) + lj_1)] + (((((float)Q_smem[(((((int)threadIdx.x) >> 2) * 4) + li_1)]) * ((float)K_smem[(((((int)threadIdx.x) & 3) * 4) + lj_1)])) * attn_score_scaling_factor) * 1.442695e+00f));
              }
            }
          }
        }
        __syncthreads();
        for (int li_1_1 = 0; li_1_1 < 4; ++li_1_1) {
          for (int lj_1_1 = 0; lj_1_1 < 4; ++lj_1_1) {
            S_smem[(((((((int)threadIdx.x) >> 2) * 64) + (li_1_1 * 16)) + ((((int)threadIdx.x) & 3) * 4)) + lj_1_1)] = S_local[((li_1_1 * 4) + lj_1_1)];
          }
        }
        __syncthreads();
        // print S
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf(">>>>>>>>>>>>>> S\n");
            for (int row = 0; row < 32; ++row) {
              for (int col = 0; col < 16; ++col) {
                printf("%f ", S_smem[row * 16 + col]);
              }
              printf("\n");
            }
            // print kv_chunk_len
            printf(">>>>>>>>>>>>>> kv_chunk_len\n");
            printf("%d\n", kv_chunk_len[0]);
        }
        m_prev[0] = m_smem[((int)threadIdx.x)];
        m_new[0] = m_smem[((int)threadIdx.x)];
        for (int j = 0; j < 16; ++j) {
          if ((((iterator * 16) + j) < kv_chunk_len[0]) && (mask[((((iterator * 16) + mn_indptr[b_idx_1]) + (((tile_id[0] * 32) + ((int)threadIdx.x)) * m_array[b_idx_1])) + j)] == 1)) {
            m_new[0] = max(m_new[0], S_smem[((((int)threadIdx.x) * 16) + j)]);
          }
        }
        d_new[0] = (d_smem[((int)threadIdx.x)] * exp2f((m_prev[0] - m_new[0])));
        __syncthreads();
        for (int j_1 = 0; j_1 < 16; ++j_1) {
          if (blockIdx.x == 0 && threadIdx.x == 0) {
            // print ((((iterator * 16) + mn_indptr[b_idx_1]) + (((tile_id[0] * 32) + ((int)threadIdx.x)) * m_array[b_idx_1])) + j_1)
            auto x = ((((iterator * 16) + mn_indptr[b_idx_1]) + (((tile_id[0] * 32) + ((int)threadIdx.x)) * m_array[b_idx_1])) + j_1);
            printf("%d\n", m_array[b_idx_1]);
            printf(">>%d, %d\n", x, mask[x]);
          }
          if ((((iterator * 16) + j_1) < kv_chunk_len[0]) && (mask[((((iterator * 16) + mn_indptr[b_idx_1]) + (((tile_id[0] * 32) + ((int)threadIdx.x)) * m_array[b_idx_1])) + j_1)] == 1)) {
            S_smem[((((int)threadIdx.x) * 16) + j_1)] = exp2f((S_smem[((((int)threadIdx.x) * 16) + j_1)] - m_new[0]));
          } else {
            S_smem[((((int)threadIdx.x) * 16) + j_1)] = exp2f((-5.000000e+04f - m_new[0]));
          }
        }
        __syncthreads();
        // print S
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf(">>>>>>>>>>>>>> S\n");
            for (int row = 0; row < 32; ++row) {
              for (int col = 0; col < 16; ++col) {
                printf("%f ", S_smem[row * 16 + col]);
              }
              printf("\n");
            }
        }
        for (int j_2 = 0; j_2 < 16; ++j_2) {
          d_new[0] = (d_new[0] + S_smem[((((int)threadIdx.x) * 16) + j_2)]);
        }
        m_smem[((int)threadIdx.x)] = m_new[0];
        d_smem[((int)threadIdx.x)] = d_new[0];
        m_prev_smem[((int)threadIdx.x)] = m_prev[0];
        __syncthreads();
        O_local[0] = (O_local[0] * exp2f((m_prev_smem[((int)threadIdx.x)] - m_smem[((int)threadIdx.x)])));
        for (int lk_0 = 0; lk_0 < 2; ++lk_0) {
          for (int lk_1_1 = 0; lk_1_1 < 8; ++lk_1_1) {
            O_local[0] = (O_local[0] + (S_smem[(((((int)threadIdx.x) * 16) + (lk_0 * 8)) + lk_1_1)] * ((float)V_smem[((lk_0 * 8) + lk_1_1)])));
          }
        }
      }
      if ((L_start + ((int)threadIdx.x)) < q_indptr[((b_idx_1 + q_indptr_elem_offset) + 1)]) {
        output[(((int)threadIdx.x) + L_start)] = ((half)(O_local[0] / d_smem[((int)threadIdx.x)]));
      }
      if ((L_start + ((int)threadIdx.x)) < q_indptr[((b_idx_1 + q_indptr_elem_offset) + 1)]) {
        lse[(((int)threadIdx.x) + L_start)] = (m_smem[((int)threadIdx.x)] + __log2f(d_smem[((int)threadIdx.x)]));
      }
      tile_id[0] = (tile_id[0] + 16);
    }
  }
}
