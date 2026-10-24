// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Added 2026-10-23 by ElectroHobo for the qwen38-flash-next-3bpw-1x-cmp170hx-recipe:
// RecoverSSM for Qwen's GDN layers (patches/README.md).
// Written for Morrowmake/vllm-cmp170hx at 18305948acba3f4c4d912cdd56f2202d2ebd94e8.
//
// RecoverSSM verify for Qwen's GDN layers on sm_80, built at first use by
// vllm/sm80/kernels/gdn/recover.py with torch.utils.cpp_extension. The
// helpers (lines between the markers) are copied unchanged from
// csrc/gdn/fused_gdn_decode_kernel.cu of the engine image, whose fused
// spec-decode kernel this follows line for line except where noted.

#include <cstdint>
#include <string>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {
// --- copied from fused_gdn_decode_kernel.cu ---

template <typename StateT>
__device__ __forceinline__ void cp_async_16b(StateT* smem_ptr,
                                             const StateT* gmem_ptr) {
  const uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               :
               : "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n" ::: "memory");
}

template <typename StateT, int ChunkV, int DimK, int Stages>
__device__ __forceinline__ void copy_state_chunk(StateT* shared_state,
                                                 const StateT* state, int chunk,
                                                 int thread, int threads) {
  constexpr int kElementsPerCopy = 16 / sizeof(StateT);
  constexpr int kCopiesPerChunk = ChunkV * DimK / kElementsPerCopy;
  const int stage = chunk % Stages;
  for (int copy = thread; copy < kCopiesPerChunk; copy += threads) {
    const int element = copy * kElementsPerCopy;
    cp_async_16b(shared_state + stage * ChunkV * DimK + element,
                 state + chunk * ChunkV * DimK + element);
  }
  cp_async_commit();
}

template <typename StateT>
__device__ __forceinline__ float4 load_state4(const StateT* state);

template <>
__device__ __forceinline__ float4 load_state4<float>(const float* state) {
  return *reinterpret_cast<const float4*>(state);
}

template <>
__device__ __forceinline__ float4
load_state4<__nv_bfloat16>(const __nv_bfloat16* state) {
  const __nv_bfloat162 lo = *reinterpret_cast<const __nv_bfloat162*>(state);
  const __nv_bfloat162 hi = *reinterpret_cast<const __nv_bfloat162*>(state + 2);
  return make_float4(__bfloat162float(lo.x), __bfloat162float(lo.y),
                     __bfloat162float(hi.x), __bfloat162float(hi.y));
}

template <typename StateT>
__device__ __forceinline__ void store_state4(StateT* state, float4 value);

template <>
__device__ __forceinline__ void store_state4<float>(float* state,
                                                    float4 value) {
  *reinterpret_cast<float4*>(state) = value;
}

template <>
__device__ __forceinline__ void store_state4<__nv_bfloat16>(
    __nv_bfloat16* state, float4 value) {
  *reinterpret_cast<__nv_bfloat162*>(state) =
      __floats2bfloat162_rn(value.x, value.y);
  *reinterpret_cast<__nv_bfloat162*>(state + 2) =
      __floats2bfloat162_rn(value.z, value.w);
}

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kChunkV = 32;
constexpr int kNumChunks = kDimV / kChunkV;
constexpr int kRowsPerWarp = kChunkV / kWarps;
constexpr int kMaxMtpTokens = 8;
constexpr int kDtBiasFloat32 = 0;
constexpr int kDtBiasBFloat16 = 1;
constexpr int kDtBiasFloat16 = 2;

struct GdnDecodeStrides {
  int64_t mixed_row;
  int64_t a_row;
  int64_t b_row;
  int64_t gate_row;
  int64_t state_slot;
};

__device__ __forceinline__ float sigmoid_fast(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float silu_fast(float x) {
  return x * sigmoid_fast(x);
}

__device__ __forceinline__ float softplus_fast(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}

__device__ __forceinline__ float load_dt_bias(const void* dt_bias, int head,
                                              int dt_bias_type) {
  if (dt_bias_type == kDtBiasBFloat16) {
    return __bfloat162float(static_cast<const __nv_bfloat16*>(dt_bias)[head]);
  }
  if (dt_bias_type == kDtBiasFloat16) {
    return __half2float(static_cast<const __half*>(dt_bias)[head]);
  }
  return static_cast<const float*>(dt_bias)[head];
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

struct Sum2 {
  float x;
  float y;
};

__device__ __forceinline__ Sum2 warp_reduce_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, offset);
    y += __shfl_xor_sync(0xffffffffu, y, offset);
  }
  return {x, y};
}
// --- end of copied helpers ---

constexpr int kRecordTokens = kMaxMtpTokens;  // WS_T: token rows per record

// RecoverSSM verify: the fused GDN spec-decode step above with records instead
// of per-draft-position states. Starts from the request's single state
// (state_indices[request, 0]; the commit replays accepted tokens into it after
// sampling), writes no state, and stores per token t:
//   rec_c[request, t, value_head, :]  delta (the correction), fp32
//   rec_k[request, t, key_head, :]    the normalised key, fp32
//   rec_e[request, t, value_head]     the decay exp(g), fp32
// The state update is h = fma(k, delta, h * decay), the same operations the
// fused kernel's `h *= decay; h += k * delta` compiles to, so the commit's
// replay reproduces the per-token state bit for bit. Norm=false (mixed
// prefill + verify steps) writes the core output without the gated RMSNorm,
// which the caller applies to every token afterwards.
template <typename StateT, int ValueHeadsPerKeyHead, bool SigmoidGate, bool Norm>
__global__ __launch_bounds__(kThreads, 2) void gdn_recover_verify_kernel(
    const __nv_bfloat16* __restrict__ mixed_qkv,
    const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
    const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, const int* __restrict__ cu_seqlens,
    const StateT* __restrict__ state,
    const __nv_bfloat16* __restrict__ output_gate,
    const void* __restrict__ norm_weight, __nv_bfloat16* __restrict__ out,
    float* __restrict__ rec_c, float* __restrict__ rec_k,
    float* __restrict__ rec_e, int rec_rows, int H, int HV,
    int state_indices_width,
    int dt_bias_type, bool norm_weight_is_bf16, float scale, float norm_eps,
    GdnDecodeStrides strides) {
  const int request = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int bos = cu_seqlens[request];
  const int eos = cu_seqlens[request + 1];
  const int num_tokens = eos - bos;
  if (num_tokens <= 0) {
    return;
  }

  const int source_slot = state_indices[request * state_indices_width];
  if (source_slot <= 0 || num_tokens > kMaxMtpTokens) {
    for (int linear = tid; linear < num_tokens * kDimV; linear += kThreads) {
      const int token = bos + linear / kDimV;
      const int value = linear % kDimV;
      const int64_t out_offset =
          (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      out[out_offset] = __float2bfloat16(0.0f);
    }
    return;
  }

  const int key_head = value_head / ValueHeadsPerKeyHead;
  // Records exist for the first rec_rows requests (max_num_seqs). A captured
  // graph's padding rows beyond them have no state slot and returned above.
  const bool records = request < rec_rows;
  const bool writes_key = records && value_head % ValueHeadsPerKeyHead == 0;
  float* const req_c =
      rec_c + static_cast<int64_t>(request) * kRecordTokens * HV * kDimV;
  float* const req_k =
      rec_k + static_cast<int64_t>(request) * kRecordTokens * H * kDimK;
  float* const req_e = rec_e + static_cast<int64_t>(request) * kRecordTokens * HV;
  __shared__ StateT shared_state[2][kChunkV][kDimK];
  __shared__ float shared_q[kMaxMtpTokens][kDimK];
  __shared__ float shared_k[kMaxMtpTokens][kDimK];
  __shared__ __nv_bfloat16 shared_v[kMaxMtpTokens][kDimV];
  __shared__ __nv_bfloat16 shared_out[kMaxMtpTokens][kDimV];
  __shared__ float shared_decay[kMaxMtpTokens];
  __shared__ float shared_beta[kMaxMtpTokens];

  const StateT* source_state =
      state + static_cast<int64_t>(source_slot) * strides.state_slot +
      value_head * kDimV * kDimK;
  copy_state_chunk<StateT, kChunkV, kDimK, 2>(&shared_state[0][0][0],
                                              source_state, 0, tid, kThreads);

  if (warp < num_tokens) {
    const int t = warp;
    const int token = bos + t;
    const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
    float q_values[4];
    float k_values[4];
    float q_square = 0.0f;
    float k_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      q_values[i] =
          __bfloat162float(mixed_qkv[mixed_base + key_head * kDimK + dim]);
      k_values[i] = __bfloat162float(
          mixed_qkv[mixed_base + H * kDimK + key_head * kDimK + dim]);
      shared_v[t][dim] =
          mixed_qkv[mixed_base + 2 * H * kDimK + value_head * kDimV + dim];
      q_square += q_values[i] * q_values[i];
      k_square += k_values[i] * k_values[i];
    }
    const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
    const float q_scale = __shfl_sync(
        0xffffffffu, lane == 0 ? rsqrtf(qk_sums.x + 1.0e-6f) * scale : 0.0f, 0);
    const float k_scale = __shfl_sync(
        0xffffffffu, lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      shared_q[t][dim] = q_values[i] * q_scale;
      const float k_normed = k_values[i] * k_scale;
      shared_k[t][dim] = k_normed;
      if (writes_key) {
        req_k[(t * H + key_head) * kDimK + dim] = k_normed;
      }
    }
    if (lane == 0) {
      const float a_value = __bfloat162float(
          a[static_cast<int64_t>(token) * strides.a_row + value_head]);
      const float b_value = __bfloat162float(
          b[static_cast<int64_t>(token) * strides.b_row + value_head]);
      const float g = -__expf(a_log[value_head]) *
                      softplus_fast(a_value + load_dt_bias(dt_bias, value_head,
                                                           dt_bias_type));
      const float decay = __expf(g);
      shared_decay[t] = decay;
      shared_beta[t] = sigmoid_fast(b_value);
      if (records) {
        req_e[t * HV + value_head] = decay;
      }
    }
  }
  __syncthreads();

  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) {
    rows[row] = warp + row * kWarps;
  }

#pragma unroll
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    cp_async_wait_all();
    __syncthreads();
    if (chunk + 1 < kNumChunks) {
      copy_state_chunk<StateT, kChunkV, kDimK, 2>(
          &shared_state[0][0][0], source_state, chunk + 1, tid, kThreads);
    }

    float h[kRowsPerWarp][4];
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      const float4 state_value =
          load_state4(&shared_state[chunk & 1][rows[row]][k_base]);
      h[row][0] = state_value.x;
      h[row][1] = state_value.y;
      h[row][2] = state_value.z;
      h[row][3] = state_value.w;
    }

    for (int t = 0; t < num_tokens; ++t) {
      const float4 q4 = *reinterpret_cast<const float4*>(&shared_q[t][k_base]);
      const float4 k4 = *reinterpret_cast<const float4*>(&shared_k[t][k_base]);
      const float q_values[4] = {q4.x, q4.y, q4.z, q4.w};
      const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};

      float dot_hk[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] = __fmul_rn(h[row][i], shared_decay[t]);
          dot_hk[row] += h[row][i] * k_values[i];
        }
      }
      const Sum2 dot_hk_01 = warp_reduce_sum_pair(dot_hk[0], dot_hk[1]);
      const Sum2 dot_hk_23 = warp_reduce_sum_pair(dot_hk[2], dot_hk[3]);
      const float reduced_hk[kRowsPerWarp] = {dot_hk_01.x, dot_hk_01.y,
                                              dot_hk_23.x, dot_hk_23.y};

      float dot_hq[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int value = chunk * kChunkV + rows[row];
        const float delta =
            (__bfloat162float(shared_v[t][value]) - reduced_hk[row]) *
            shared_beta[t];
        if (records && lane == 0) {
          req_c[(t * HV + value_head) * kDimV + value] = delta;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] = __fmaf_rn(k_values[i], delta, h[row][i]);
          dot_hq[row] += h[row][i] * q_values[i];
        }
      }
      const Sum2 dot_hq_01 = warp_reduce_sum_pair(dot_hq[0], dot_hq[1]);
      const Sum2 dot_hq_23 = warp_reduce_sum_pair(dot_hq[2], dot_hq[3]);
      if (lane == 0) {
        shared_out[t][chunk * kChunkV + rows[0]] =
            __float2bfloat16(dot_hq_01.x);
        shared_out[t][chunk * kChunkV + rows[1]] =
            __float2bfloat16(dot_hq_01.y);
        shared_out[t][chunk * kChunkV + rows[2]] =
            __float2bfloat16(dot_hq_23.x);
        shared_out[t][chunk * kChunkV + rows[3]] =
            __float2bfloat16(dot_hq_23.y);
      }
    }
  }
  __syncthreads();

  if (warp < num_tokens) {
    const int t = warp;
    const int token = bos + t;
    if constexpr (!Norm) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int value = lane + i * 32;
        out[(static_cast<int64_t>(token) * HV + value_head) * kDimV + value] =
            shared_out[t][value];
      }
      return;
    }
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      output_values[i] = __bfloat162float(shared_out[t][value]);
      sum_square += output_values[i] * output_values[i];
    }
    sum_square = warp_reduce_sum(sum_square);
    const float rstd =
        rsqrtf(sum_square / static_cast<float>(kDimV) + norm_eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      const float gate_input = __bfloat162float(
          output_gate[static_cast<int64_t>(token) * strides.gate_row +
                      value_head * kDimV + value]);
      const float gate =
          SigmoidGate ? sigmoid_fast(gate_input) : silu_fast(gate_input);
      const float weight =
          norm_weight_is_bf16
              ? __bfloat162float(
                    static_cast<const __nv_bfloat16*>(norm_weight)[value])
              : static_cast<const float*>(norm_weight)[value];
      const int64_t out_offset =
          (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      out[out_offset] =
          __float2bfloat16(output_values[i] * rstd * weight * gate);
    }
  }
}

}  // namespace

// Python binding (torch.utils.cpp_extension.load). Shapes as the fused op's,
// plus the layer's records: rec_c [rows, 8, HV, 128], rec_k [rows, 8, H, 128],
// rec_e [rows, 8, HV], all fp32 and contiguous. With norm=false, output_gate
// and norm_weight are not read.
void gdn_recover_verify(torch::Tensor const& mixed_qkv, torch::Tensor const& a,
                        torch::Tensor const& b, torch::Tensor const& a_log,
                        torch::Tensor const& dt_bias,
                        torch::Tensor const& state_indices,
                        torch::Tensor const& cu_seqlens,
                        torch::Tensor const& state,
                        torch::Tensor const& output_gate,
                        torch::Tensor const& norm_weight, torch::Tensor& out,
                        torch::Tensor& rec_c, torch::Tensor& rec_k,
                        torch::Tensor& rec_e, double scale, double norm_eps,
                        std::string const& output_gate_activation, bool norm) {
  TORCH_CHECK(mixed_qkv.is_cuda() && mixed_qkv.scalar_type() == at::kBFloat16 &&
                  mixed_qkv.dim() == 2 && mixed_qkv.stride(1) == 1,
              "mixed_qkv must be CUDA bf16 [L, 2*H*128 + HV*128], channels contiguous");
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16 &&
                  a.stride(1) == 1 && b.stride(1) == 1,
              "a and b must be bf16 with contiguous heads");
  TORCH_CHECK(a_log.scalar_type() == at::kFloat && a_log.is_contiguous(),
              "A_log must be contiguous float32");
  const auto dt_type = dt_bias.scalar_type();
  TORCH_CHECK(dt_bias.is_contiguous() &&
                  (dt_type == at::kFloat || dt_type == at::kBFloat16 ||
                   dt_type == at::kHalf),
              "dt_bias must be contiguous float32, bfloat16 or float16");
  TORCH_CHECK(state_indices.scalar_type() == at::kInt && state_indices.dim() == 2 &&
                  state_indices.is_contiguous() && state_indices.size(0) > 0,
              "state_indices must be contiguous int32 [N, S]");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt && cu_seqlens.is_contiguous() &&
                  cu_seqlens.numel() == state_indices.size(0) + 1,
              "cu_seqlens must be contiguous int32 with N + 1 elements");
  const auto state_type = state.scalar_type();
  TORCH_CHECK((state_type == at::kFloat || state_type == at::kBFloat16) &&
                  state.dim() == 4 && state.size(2) == kDimV &&
                  state.size(3) == kDimK && state.stride(1) == kDimV * kDimK &&
                  state.stride(2) == kDimK && state.stride(3) == 1,
              "state must be float32 or bfloat16 [slots, HV, 128, 128]");
  const int elements_per_copy = state_type == at::kFloat ? 4 : 8;
  TORCH_CHECK(reinterpret_cast<uintptr_t>(state.data_ptr()) % 16 == 0 &&
                  state.stride(0) % elements_per_copy == 0,
              "state slots must preserve 16-byte alignment");
  const int num_value_heads = static_cast<int>(state.size(1));
  const int64_t key_width = mixed_qkv.size(1) - int64_t(num_value_heads) * kDimV;
  TORCH_CHECK(key_width > 0 && key_width % (2 * kDimK) == 0,
              "mixed_qkv width is inconsistent with state");
  const int num_key_heads = static_cast<int>(key_width / (2 * kDimK));
  const int ratio = num_value_heads / num_key_heads;
  TORCH_CHECK(num_value_heads % num_key_heads == 0 &&
                  ((ratio >= 1 && ratio <= 4) || ratio == 8),
              "HV/H must be in {1, 2, 3, 4, 8}");
  const int num_requests = static_cast<int>(state_indices.size(0));
  const int num_tokens = static_cast<int>(mixed_qkv.size(0));
  TORCH_CHECK(a.size(0) == num_tokens && a.size(1) == num_value_heads &&
                  b.size(0) == num_tokens && b.size(1) == num_value_heads,
              "a and b must be [L, HV]");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() &&
                  out.dim() == 3 && out.size(0) >= num_tokens &&
                  out.size(1) == num_value_heads && out.size(2) == kDimV,
              "out must be contiguous bf16 [L, HV, 128]");
  TORCH_CHECK(rec_c.scalar_type() == at::kFloat && rec_c.is_contiguous() &&
                  rec_c.dim() == 4 && rec_c.size(0) > 0 &&
                  rec_c.size(1) == kRecordTokens && rec_c.size(2) == num_value_heads &&
                  rec_c.size(3) == kDimV,
              "rec_c must be contiguous float32 [rows, 8, HV, 128]");
  TORCH_CHECK(rec_k.scalar_type() == at::kFloat && rec_k.is_contiguous() &&
                  rec_k.dim() == 4 && rec_k.size(0) == rec_c.size(0) &&
                  rec_k.size(1) == kRecordTokens && rec_k.size(2) == num_key_heads &&
                  rec_k.size(3) == kDimK,
              "rec_k must be contiguous float32 [rows, 8, H, 128]");
  TORCH_CHECK(rec_e.scalar_type() == at::kFloat && rec_e.is_contiguous() &&
                  rec_e.dim() == 3 && rec_e.size(0) == rec_c.size(0) &&
                  rec_e.size(1) == kRecordTokens && rec_e.size(2) == num_value_heads,
              "rec_e must be contiguous float32 [rows, 8, HV]");
  if (norm) {
    TORCH_CHECK(output_gate.scalar_type() == at::kBFloat16 && output_gate.dim() == 3 &&
                    output_gate.size(0) >= num_tokens &&
                    output_gate.size(1) == num_value_heads &&
                    output_gate.stride(2) == 1 && output_gate.stride(1) == kDimV,
                "output_gate must be bf16 [L, HV, 128] with contiguous head rows");
    TORCH_CHECK(norm_weight.is_contiguous() && norm_weight.numel() == kDimV &&
                    (norm_weight.scalar_type() == at::kFloat ||
                     norm_weight.scalar_type() == at::kBFloat16),
                "norm_weight must be contiguous float32 or bfloat16 with 128 elements");
    TORCH_CHECK(output_gate_activation == "silu" || output_gate_activation == "sigmoid",
                "output_gate_activation must be silu or sigmoid");
  }
  const int dt_bias_type = dt_type == at::kFloat
                               ? kDtBiasFloat32
                               : (dt_type == at::kBFloat16 ? kDtBiasBFloat16
                                                           : kDtBiasFloat16);
  const GdnDecodeStrides strides{mixed_qkv.stride(0), a.stride(0), b.stride(0),
                                 norm ? output_gate.stride(0) : 0, state.stride(0)};
  const at::cuda::CUDAGuard guard(mixed_qkv.device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(num_requests, num_value_heads);
  const bool sigmoid = output_gate_activation == "sigmoid";
  const bool weight_bf16 = norm && norm_weight.scalar_type() == at::kBFloat16;

  const auto launch = [&]<typename StateT, int R, bool Sig, bool N>() {
    gdn_recover_verify_kernel<StateT, R, Sig, N><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(mixed_qkv.data_ptr()),
        static_cast<const __nv_bfloat16*>(a.data_ptr()),
        static_cast<const __nv_bfloat16*>(b.data_ptr()),
        a_log.data_ptr<float>(), dt_bias.data_ptr(),
        state_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(),
        static_cast<const StateT*>(state.data_ptr()),
        norm ? static_cast<const __nv_bfloat16*>(output_gate.data_ptr()) : nullptr,
        norm ? norm_weight.data_ptr() : nullptr,
        static_cast<__nv_bfloat16*>(out.data_ptr()), rec_c.data_ptr<float>(),
        rec_k.data_ptr<float>(), rec_e.data_ptr<float>(),
        static_cast<int>(rec_c.size(0)), num_key_heads,
        num_value_heads, static_cast<int>(state_indices.size(1)), dt_bias_type,
        weight_bf16, static_cast<float>(scale), static_cast<float>(norm_eps),
        strides);
  };
  const auto by_norm = [&]<typename StateT, int R>() {
    if (!norm) {
      launch.template operator()<StateT, R, false, false>();
    } else if (sigmoid) {
      launch.template operator()<StateT, R, true, true>();
    } else {
      launch.template operator()<StateT, R, false, true>();
    }
  };
  const auto by_state = [&]<int R>() {
    if (state_type == at::kFloat) {
      by_norm.template operator()<float, R>();
    } else {
      by_norm.template operator()<__nv_bfloat16, R>();
    }
  };
  switch (ratio) {
    case 1: by_state.template operator()<1>(); break;
    case 2: by_state.template operator()<2>(); break;
    case 3: by_state.template operator()<3>(); break;
    case 4: by_state.template operator()<4>(); break;
    default: by_state.template operator()<8>(); break;
  }
  const cudaError_t error = cudaGetLastError();
  TORCH_CHECK(error == cudaSuccess, "GDN recover verify launch failed: ",
              cudaGetErrorString(error));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_recover_verify", &gdn_recover_verify,
        "GDN spec-decode verify with RecoverSSM records (sm_80)");
}
