#ifndef QWEN36_NATIVE_HYBRID_KERNELS_CUH
#define QWEN36_NATIVE_HYBRID_KERNELS_CUH

#include "../ds4_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static void stub_build_perms(uint32_t n_heads, uint32_t *gg_to_hf, uint32_t *hf_to_gg) {
    uint32_t pos = 0;
    for (uint32_t i = 0; i < n_heads; i += 2) gg_to_hf[pos++] = i;
    for (uint32_t i = 1; i < n_heads; i += 2) gg_to_hf[pos++] = i;
    for (uint32_t i = 0; i < n_heads; ++i) hf_to_gg[gg_to_hf[i]] = i;
}

static void stub_reorder_head_rows_f32(float *dst, const float *src, uint32_t n_heads, uint32_t head_dim) {
    uint32_t *gg_to_hf = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    if (!gg_to_hf || !hf_to_gg) { free(gg_to_hf); free(hf_to_gg); return; }
    stub_build_perms(n_heads, gg_to_hf, hf_to_gg);
    for (uint32_t h = 0; h < n_heads; ++h)
        memcpy(dst + (size_t)h * head_dim, src + (size_t)hf_to_gg[h] * head_dim, (size_t)head_dim * sizeof(float));
    free(gg_to_hf); free(hf_to_gg);
}

static void stub_reorder_head_scalars_f32(float *dst, const float *src, uint32_t n_heads) {
    uint32_t *gg_to_hf = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    if (!gg_to_hf || !hf_to_gg) { free(gg_to_hf); free(hf_to_gg); return; }
    stub_build_perms(n_heads, gg_to_hf, hf_to_gg);
    for (uint32_t h = 0; h < n_heads; ++h) dst[h] = src[hf_to_gg[h]];
    free(gg_to_hf); free(hf_to_gg);
}

static void stub_reorder_qkv_v_gghf(float *dst, const float *src, uint32_t key_dim, uint32_t n_heads, uint32_t head_dim) {
    uint32_t *gg_to_hf = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    const uint32_t value_dim = n_heads * head_dim;
    const uint32_t qkv_dim = key_dim * 2u + value_dim;
    const uint32_t v_off = key_dim * 2u;
    if (!gg_to_hf || !hf_to_gg) { free(gg_to_hf); free(hf_to_gg); return; }
    stub_build_perms(n_heads, gg_to_hf, hf_to_gg);
    memcpy(dst, src, (size_t)(key_dim * 2u) * sizeof(float));
    for (uint32_t h = 0; h < n_heads; ++h)
        memcpy(dst + v_off + (size_t)h * head_dim, src + v_off + (size_t)hf_to_gg[h] * head_dim, (size_t)head_dim * sizeof(float));
    free(gg_to_hf); free(hf_to_gg);
}

static void stub_reorder_out_in_hftogg(float *dst, const float *src, uint32_t n_heads, uint32_t head_dim) {
    uint32_t *gg_to_hf = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    if (!gg_to_hf || !hf_to_gg) { free(gg_to_hf); free(hf_to_gg); return; }
    stub_build_perms(n_heads, gg_to_hf, hf_to_gg);
    for (uint32_t h_gg = 0; h_gg < n_heads; ++h_gg)
        memcpy(dst + (size_t)h_gg * head_dim, src + (size_t)gg_to_hf[h_gg] * head_dim, (size_t)head_dim * sizeof(float));
    free(gg_to_hf); free(hf_to_gg);
}

static inline float stub_silu(float x) { return x / (1.0f + expf(-x)); }
static inline float stub_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float stub_softplus(float x) { return log1pf(expf(x)); }

static int env_flag_enabled_default1(const char *name) {
    const char *v = getenv(name);
    if (!v || !v[0]) return 1;
    if (strcmp(v, "0") == 0) return 0;
    return 1;
}

static void stub_deltanet_head_step(float *state,
                                     uint32_t head_k_dim,
                                     uint32_t head_v_dim,
                                     const float *k_row,
                                     const float *q_row,
                                     const float *v_row,
                                     float gexp,
                                     float beta_t,
                                     float *k_scaled,
                                     float *q_scaled,
                                     float *delta_tmp,
                                     float *out_row) {
    uint32_t hd, vd;
    float qnorm = 0.0f, knorm = 0.0f;
    const float qscale_den = sqrtf((float)head_k_dim);
    for (hd = 0; hd < head_k_dim; ++hd) { qnorm += q_row[hd] * q_row[hd]; knorm += k_row[hd] * k_row[hd]; }
    qnorm = 1.0f / sqrtf(qnorm + 1e-6f);
    knorm = 1.0f / sqrtf(knorm + 1e-6f);
    for (hd = 0; hd < head_k_dim; ++hd) { k_scaled[hd] = k_row[hd] * knorm; q_scaled[hd] = q_row[hd] * qnorm / qscale_den; }
    for (hd = 0; hd < head_k_dim; ++hd) {
        float *row = state + (size_t)hd * head_v_dim;
        for (vd = 0; vd < head_v_dim; ++vd) row[vd] *= gexp;
    }
    for (vd = 0; vd < head_v_dim; ++vd) delta_tmp[vd] = v_row[vd];
    for (hd = 0; hd < head_k_dim; ++hd) {
        const float *row = state + (size_t)hd * head_v_dim;
        const float ks = k_scaled[hd];
        for (vd = 0; vd < head_v_dim; ++vd) delta_tmp[vd] -= row[vd] * ks;
    }
    for (vd = 0; vd < head_v_dim; ++vd) { delta_tmp[vd] *= beta_t; out_row[vd] = 0.0f; }
    for (hd = 0; hd < head_k_dim; ++hd) {
        float *row = state + (size_t)hd * head_v_dim;
        const float ks = k_scaled[hd];
        const float qs = q_scaled[hd];
        for (vd = 0; vd < head_v_dim; ++vd) { row[vd] += ks * delta_tmp[vd]; out_row[vd] += row[vd] * qs; }
    }
}

/* ---- GPU kernels ---- */

/*
 * kernel_reorder_z_inplace — GG→HF reorder for z vector
 *
 * Uses shared memory as temp buffer to avoid allocation.
 * 1 block, value_dim threads.
 */
__global__ static void kernel_reorder_z_inplace(float *z, uint32_t num_v_heads, uint32_t head_v_dim) {
    extern __shared__ float s_buf[];
    uint32_t value_dim = num_v_heads * head_v_dim;
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t half = num_v_heads / 2u;
    if (d < value_dim) s_buf[d] = z[d];
    __syncthreads();
    if (d >= value_dim) return;
    uint32_t hf_head = d / head_v_dim;
    uint32_t hf_off  = d % head_v_dim;
    uint32_t gg_head = (hf_head < half) ? (hf_head * 2u) : ((hf_head - half) * 2u + 1u);
    z[d] = s_buf[gg_head * head_v_dim + hf_off];
}

/*
 * kernel_fused_reorder_conv — fused conv3+SiLU, K head-repeat, V reorder, ring shift
 *
 * 1D grid: (qkv_dim + 255)/256 blocks × 256 threads.
 * Each thread handles one element d of qkv_dim.
 */
__global__ static void kernel_fused_reorder_conv(
        float *conv,           float *q_hf,          float *k_hf,          float *v_hf,
        const float *qkv_raw,
        float *ring0,          float *ring1,          float *ring2,
        const float *conv_w,
        uint32_t key_dim,      uint32_t value_dim,    uint32_t qkv_dim,
        uint32_t num_k_heads,  uint32_t num_v_heads,
        uint32_t head_k_dim,   uint32_t head_v_dim,
        uint32_t ring_count)
{
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= qkv_dim) return;

    uint32_t rep = num_v_heads / num_k_heads;
    uint32_t half_v = num_v_heads / 2u;

    /* read current qkv element, reordering V from GG→HF */
    float qkv_in;
    if (d < key_dim * 2u) {
        qkv_in = qkv_raw[d];
    } else {
        uint32_t d_v = d - key_dim * 2u;
        uint32_t v_head = d_v / head_v_dim;
        uint32_t v_off  = d_v % head_v_dim;
        uint32_t gg_src = (v_head < half_v) ? (v_head * 2u) : ((v_head - half_v) * 2u + 1u);
        qkv_in = qkv_raw[key_dim * 2u + gg_src * head_v_dim + v_off];
    }

    /* conv3 + SiLU over ring buffer */
    float sum = 0.0f;
    if (ring_count >= 3) sum += conv_w[d * 4u + 0u] * ring2[d];
    if (ring_count >= 2) sum += conv_w[d * 4u + 1u] * ring1[d];
    if (ring_count >= 1) sum += conv_w[d * 4u + 2u] * ring0[d];
    sum += conv_w[d * 4u + 3u] * qkv_in;
    float cv = sum / (1.0f + __expf(-sum));
    conv[d] = cv;

    /* K head-repeat for Q and K; V is already HF */
    if (d < key_dim) {
        uint32_t k_head  = d / head_k_dim;
        uint32_t local_d = d % head_k_dim;
        for (uint32_t vh = 0; vh < rep; ++vh)
            q_hf[(k_head * rep + vh) * head_k_dim + local_d] = cv;
    } else if (d < key_dim * 2u) {
        uint32_t d_k     = d - key_dim;
        uint32_t k_head  = d_k / head_k_dim;
        uint32_t local_d = d_k % head_k_dim;
        for (uint32_t vh = 0; vh < rep; ++vh)
            k_hf[(k_head * rep + vh) * head_k_dim + local_d] = cv;
    } else {
        uint32_t d_v = d - key_dim * 2u;
        v_hf[d_v] = cv;
    }

    /* ring shift: rotate buffer, store current qkv_in as newest */
    if (ring_count >= 2) ring2[d] = ring1[d];
    if (ring_count >= 1) ring1[d] = ring0[d];
    ring0[d] = qkv_in;
}

/*
 * kernel_beta_g — per-head beta (sigmoid) and g (A_log * softplus)
 *
 * 1 block, num_v_heads threads.
 * Reads a,b in GG layout; computes on-the-fly reorder.
 */
__global__ static void kernel_beta_g(
        float *beta,        float *g,
        const float *b_gg,  const float *a_gg,
        const float *A_log, const float *dt_bias,
        uint32_t num_v_heads)
{
    uint32_t h = threadIdx.x;
    if (h >= num_v_heads) return;
    uint32_t half = num_v_heads / 2u;
    uint32_t gg = (h < half) ? (h * 2u) : ((h - half) * 2u + 1u);
    beta[h] = 1.0f / (1.0f + __expf(-b_gg[gg]));
    g[h] = A_log[h] * log1pf(__expf(a_gg[gg] + dt_bias[h]));
}

__global__ static void kernel_state_norm_silu_reorder(
        float *out_in_gg,
        const float *core_hf,
        const float *z_hf,
        const float *ssm_norm_w,
        uint32_t num_v_heads,
        uint32_t head_v_dim)
{
    extern __shared__ float s_red[];
    const uint32_t gg_head = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t half = num_v_heads / 2u;
    const uint32_t hf_head = (gg_head < half) ? (gg_head * 2u) : ((gg_head - half) * 2u + 1u);
    const uint32_t src_base = hf_head * head_v_dim;
    const uint32_t dst_base = gg_head * head_v_dim;
    float v = 0.0f;
    if (tid < head_v_dim) {
        const float cv = core_hf[src_base + tid];
        v = cv * cv;
    }
    s_red[tid] = v;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s_red[tid] += s_red[tid + stride];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(s_red[0] / (float)head_v_dim + 1e-6f);
    if (tid < head_v_dim) {
        const float cv = core_hf[src_base + tid];
        out_in_gg[dst_base + tid] = cv * inv_rms * ssm_norm_w[tid] * (z_hf[src_base + tid] / (1.0f + __expf(-z_hf[src_base + tid])));
    }
}

__global__ static void kernel_resid_post_ln(
        float *resid,
        float *post_ln,
        const float *layer_input,
        const float *out_proj,
        const float *post_attn_norm_w,
        uint32_t hidden)
{
    extern __shared__ float s_red[];
    const uint32_t tid = threadIdx.x;
    float sum = 0.0f;
    for (uint32_t idx = tid; idx < hidden; idx += blockDim.x) {
        const float rv = layer_input[idx] + out_proj[idx];
        resid[idx] = rv;
        sum += rv * rv;
    }
    s_red[tid] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) s_red[tid] += s_red[tid + stride];
        __syncthreads();
    }
    if (tid == 0) s_red[0] /= (float)hidden;
    __syncthreads();
    const float inv_rms = rsqrtf(s_red[0] + 1e-6f);
    for (uint32_t idx = tid; idx < hidden; idx += blockDim.x) {
        post_ln[idx] = resid[idx] * inv_rms * post_attn_norm_w[idx];
    }
}

/*
 * ds4_gpu_fused_reorder_conv_tensor — GPU implementation
 *
 * Launches three GPU kernels inside the active command batch:
 *   1. kernel_reorder_z_inplace  – GG→HF reorder for z vector
 *   2. kernel_fused_reorder_conv – conv3+SiLU + K head-repeat + ring shift
 *   3. kernel_beta_g             – sigmoid(b) / A_log * softplus(a + bias)
 *
 * conv_w_gpu, A_log_gpu, dt_bias_gpu must be pre-uploaded before begin_commands.
 */
int ds4_gpu_fused_reorder_conv_tensor(
        ds4_gpu_tensor *conv_gpu,
        ds4_gpu_tensor *q_gpu,
        ds4_gpu_tensor *k_gpu,
        ds4_gpu_tensor *v_gpu,
        ds4_gpu_tensor *beta_gpu,
        ds4_gpu_tensor *g_gpu,
        const ds4_gpu_tensor *z_gpu,
        const ds4_gpu_tensor *a_gpu,
        const ds4_gpu_tensor *b_gpu,
        const ds4_gpu_tensor *qkv_gpu,
        ds4_gpu_tensor *conv_ring_gpu[3],
        uint32_t *conv_ring_count,
        uint32_t num_v_heads,
        uint32_t num_k_heads,
        uint32_t head_k_dim,
        uint32_t head_v_dim,
        uint32_t key_dim,
        const ds4_gpu_tensor *conv_w_gpu,
        const ds4_gpu_tensor *A_log_gpu,
        const ds4_gpu_tensor *dt_bias_gpu)
{
    const uint32_t value_dim = num_v_heads * head_v_dim;
    const uint32_t qkv_dim   = key_dim * 2u + value_dim;

    float *z_ptr   = (float *)ds4_gpu_tensor_contents((ds4_gpu_tensor *)z_gpu);
    float *a_ptr   = (float *)ds4_gpu_tensor_contents((ds4_gpu_tensor *)a_gpu);
    float *b_ptr   = (float *)ds4_gpu_tensor_contents((ds4_gpu_tensor *)b_gpu);
    float *qkv_ptr = (float *)ds4_gpu_tensor_contents((ds4_gpu_tensor *)qkv_gpu);
    float *conv_ptr= (float *)ds4_gpu_tensor_contents(conv_gpu);
    float *q_ptr   = (float *)ds4_gpu_tensor_contents(q_gpu);
    float *k_ptr   = (float *)ds4_gpu_tensor_contents(k_gpu);
    float *v_ptr   = (float *)ds4_gpu_tensor_contents(v_gpu);
    float *beta_ptr= (float *)ds4_gpu_tensor_contents(beta_gpu);
    float *g_ptr   = (float *)ds4_gpu_tensor_contents(g_gpu);
    float *r0_ptr  = (float *)ds4_gpu_tensor_contents(conv_ring_gpu[0]);
    float *r1_ptr  = (float *)ds4_gpu_tensor_contents(conv_ring_gpu[1]);
    float *r2_ptr  = (float *)ds4_gpu_tensor_contents(conv_ring_gpu[2]);
    float *cw_ptr  = (float *)ds4_gpu_tensor_contents((ds4_gpu_tensor *)conv_w_gpu);
    float *al_ptr  = (float *)ds4_gpu_tensor_contents((ds4_gpu_tensor *)A_log_gpu);
    float *db_ptr  = (float *)ds4_gpu_tensor_contents((ds4_gpu_tensor *)dt_bias_gpu);

    if (!z_ptr || !a_ptr || !b_ptr || !qkv_ptr || !conv_ptr || !q_ptr || !k_ptr || !v_ptr ||
        !beta_ptr || !g_ptr || !r0_ptr || !r1_ptr || !r2_ptr || !cw_ptr || !al_ptr || !db_ptr)
        return 0;

    /* 1. reorder z GG→HF in-place */
    kernel_reorder_z_inplace<<<1, value_dim, value_dim * sizeof(float)>>>(z_ptr, num_v_heads, head_v_dim);

    /* 2. fused conv + head-repeat + ring shift */
    uint32_t conv_grid = (qkv_dim + 255u) / 256u;
    kernel_fused_reorder_conv<<<conv_grid, 256u>>>(
        conv_ptr, q_ptr, k_ptr, v_ptr,
        qkv_ptr,
        r0_ptr, r1_ptr, r2_ptr,
        cw_ptr,
        key_dim, value_dim, qkv_dim,
        num_k_heads, num_v_heads,
        head_k_dim, head_v_dim,
        *conv_ring_count);

    /* 3. beta / g */
    kernel_beta_g<<<1, num_v_heads>>>(beta_ptr, g_ptr, b_ptr, a_ptr, al_ptr, db_ptr, num_v_heads);

    if (*conv_ring_count < 3) (*conv_ring_count)++;

    return 1;
}

/*
 * ds4_gpu_deltanet_step_tensor — stub
 *
 * Reads K, Q, V, g_exp, beta from GPU, runs SSM head step on CPU,
 * updates state and writes out_row back to GPU.
 *
 * Later: real GPU kernel.
 */
int ds4_gpu_deltanet_step_tensor(
        ds4_gpu_tensor *state_gpu,
        ds4_gpu_tensor *core_gpu,
        const ds4_gpu_tensor *k_gpu,
        const ds4_gpu_tensor *q_gpu,
        const ds4_gpu_tensor *v_gpu,
        const ds4_gpu_tensor *g_gpu,
        const ds4_gpu_tensor *beta_gpu,
        uint32_t num_v_heads,
        uint32_t head_k_dim,
        uint32_t head_v_dim)
{
    const uint64_t state_bytes = (uint64_t)num_v_heads * head_k_dim * head_v_dim * sizeof(float);
    const uint64_t kqv_bytes = (uint64_t)num_v_heads * head_k_dim * sizeof(float);
    const uint64_t vdim_bytes = (uint64_t)num_v_heads * head_v_dim * sizeof(float);
    const uint64_t scalar_bytes = (uint64_t)num_v_heads * sizeof(float);
    uint32_t h;

    float *state = (float *)malloc(state_bytes);
    float *k_hf = (float *)malloc(kqv_bytes);
    float *q_hf = (float *)malloc(kqv_bytes);
    float *v_hf = (float *)malloc(vdim_bytes);
    float *g_exp_vec = (float *)malloc(scalar_bytes);
    float *beta_vec = (float *)malloc(scalar_bytes);
    float *core = (float *)calloc((size_t)num_v_heads * head_v_dim, sizeof(float));
    float *k_scaled = (float *)malloc((size_t)head_k_dim * sizeof(float));
    float *q_scaled = (float *)malloc((size_t)head_k_dim * sizeof(float));
    float *delta_tmp = (float *)malloc((size_t)head_v_dim * sizeof(float));
    if (!state || !k_hf || !q_hf || !v_hf || !g_exp_vec || !beta_vec || !core || !k_scaled || !q_scaled || !delta_tmp) goto cleanup;

    if (ds4_gpu_tensor_read(state_gpu, 0, state, state_bytes) == 0 ||
        ds4_gpu_tensor_read(k_gpu, 0, k_hf, kqv_bytes) == 0 ||
        ds4_gpu_tensor_read(q_gpu, 0, q_hf, kqv_bytes) == 0 ||
        ds4_gpu_tensor_read(v_gpu, 0, v_hf, vdim_bytes) == 0 ||
        ds4_gpu_tensor_read(g_gpu, 0, g_exp_vec, scalar_bytes) == 0 ||
        ds4_gpu_tensor_read(beta_gpu, 0, beta_vec, scalar_bytes) == 0)
        goto cleanup;

    for (h = 0; h < num_v_heads; ++h) {
        stub_deltanet_head_step(state + (size_t)h * head_k_dim * head_v_dim,
                                 head_k_dim, head_v_dim,
                                 k_hf + (size_t)h * head_k_dim,
                                 q_hf + (size_t)h * head_k_dim,
                                 v_hf + (size_t)h * head_v_dim,
                                 expf(g_exp_vec[h]),
                                 beta_vec[h],
                                 k_scaled, q_scaled, delta_tmp,
                                 core + (size_t)h * head_v_dim);
    }

    if (ds4_gpu_tensor_write(state_gpu, 0, state, state_bytes) == 0 ||
        ds4_gpu_tensor_write(core_gpu, 0, core, (uint64_t)num_v_heads * head_v_dim * sizeof(float)) == 0)
        goto cleanup;

    free(state); free(k_hf); free(q_hf); free(v_hf); free(g_exp_vec); free(beta_vec);
    free(core); free(k_scaled); free(q_scaled); free(delta_tmp);
    return 1;
cleanup:
    free(state); free(k_hf); free(q_hf); free(v_hf); free(g_exp_vec); free(beta_vec);
    free(core); free(k_scaled); free(q_scaled); free(delta_tmp);
    return 0;
}

/*
 * ds4_gpu_state_norm_silu_reorder_tensor — stub
 *
 * Reads core, norm_z from GPU.  Computes per-head RMS norm → silu gate → HF→GG reorder.
 * Writes the GG-layout output back to GPU.
 *
 * Later: real GPU kernel.
 */
int ds4_gpu_state_norm_silu_reorder_tensor(
        ds4_gpu_tensor *out_in_gg_gpu,
        const ds4_gpu_tensor *core_gpu,
        const ds4_gpu_tensor *z_gpu,
        uint32_t num_v_heads,
        uint32_t head_v_dim,
        const float *ssm_norm_w)
{
    const uint64_t vdim_bytes = (uint64_t)num_v_heads * head_v_dim * sizeof(float);
    static int gpu_path_cached = -1;
    if (gpu_path_cached == -1) gpu_path_cached = env_flag_enabled_default1("QWEN36_NATIVE_HYBRID_GPU_STATE_NORM");
    if (!gpu_path_cached) {
        uint32_t h, vd;
        float *core = (float *)malloc(vdim_bytes);
        float *z_hf = (float *)malloc(vdim_bytes);
        float *out_in_hf = (float *)malloc(vdim_bytes);
        float *out_in_gg = (float *)malloc(vdim_bytes);
        if (!core || !z_hf || !out_in_hf || !out_in_gg) goto cpu_cleanup;
        if (ds4_gpu_tensor_read(core_gpu, 0, core, vdim_bytes) == 0 ||
            ds4_gpu_tensor_read(z_gpu, 0, z_hf, vdim_bytes) == 0) goto cpu_cleanup;
        for (h = 0; h < num_v_heads; ++h) {
            size_t base = (size_t)h * head_v_dim;
            double var = 0.0;
            for (vd = 0; vd < head_v_dim; ++vd) {
                double cv = core[base + vd];
                var += cv * cv;
            }
            var /= (double)head_v_dim;
            for (vd = 0; vd < head_v_dim; ++vd) {
                float cv = core[base + vd];
                out_in_hf[base + vd] = (float)(cv / sqrt(var + 1e-6)) * ssm_norm_w[vd] * stub_silu(z_hf[base + vd]);
            }
        }
        stub_reorder_out_in_hftogg(out_in_gg, out_in_hf, num_v_heads, head_v_dim);
        if (ds4_gpu_tensor_write(out_in_gg_gpu, 0, out_in_gg, vdim_bytes) == 0) goto cpu_cleanup;
        free(core); free(z_hf); free(out_in_hf); free(out_in_gg);
        return 1;
cpu_cleanup:
        free(core); free(z_hf); free(out_in_hf); free(out_in_gg);
        return 0;
    }
    ds4_gpu_tensor *ssm_norm_w_gpu = ds4_gpu_tensor_alloc((uint64_t)head_v_dim * sizeof(float));
    float *out_ptr = NULL, *core_ptr = NULL, *z_ptr = NULL, *norm_ptr = NULL;
    if (!ssm_norm_w_gpu) return 0;
    if (ds4_gpu_tensor_write(ssm_norm_w_gpu, 0, ssm_norm_w, (uint64_t)head_v_dim * sizeof(float)) == 0) {
        ds4_gpu_tensor_free(ssm_norm_w_gpu);
        return 0;
    }
    out_ptr = (float *)ds4_gpu_tensor_contents(out_in_gg_gpu);
    core_ptr = (float *)ds4_gpu_tensor_contents((ds4_gpu_tensor *)core_gpu);
    z_ptr = (float *)ds4_gpu_tensor_contents((ds4_gpu_tensor *)z_gpu);
    norm_ptr = (float *)ds4_gpu_tensor_contents(ssm_norm_w_gpu);
    if (!out_ptr || !core_ptr || !z_ptr || !norm_ptr) {
        ds4_gpu_tensor_free(ssm_norm_w_gpu);
        return 0;
    }
    kernel_state_norm_silu_reorder<<<num_v_heads, head_v_dim, head_v_dim * sizeof(float)>>>(
        out_ptr, core_ptr, z_ptr, norm_ptr, num_v_heads, head_v_dim);
    ds4_gpu_tensor_free(ssm_norm_w_gpu);
    (void)vdim_bytes;
    return 1;
}

/*
 * ds4_gpu_resid_post_ln_tensor — stub
 *
 * Reads layer_input and out_proj from GPU, computes residual + RMS norm,
 * writes resid and post_ln back to GPU.
 *
 * Later: real GPU kernel.
 */
int ds4_gpu_resid_post_ln_tensor(
        ds4_gpu_tensor *resid_gpu,
        ds4_gpu_tensor *post_ln_gpu,
        const ds4_gpu_tensor *layer_input_gpu,
        const ds4_gpu_tensor *out_proj_gpu,
        uint32_t hidden,
        const float *post_attn_norm_w)
{
    static int gpu_path_cached = -1;
    if (gpu_path_cached == -1) gpu_path_cached = env_flag_enabled_default1("QWEN36_NATIVE_HYBRID_GPU_RESID_POST");
    if (!gpu_path_cached) {
        const uint64_t hidden_bytes = (uint64_t)hidden * sizeof(float);
        uint32_t d;
        float *layer_input = (float *)malloc(hidden_bytes);
        float *out_proj = (float *)malloc(hidden_bytes);
        float *resid = (float *)malloc(hidden_bytes);
        float *post_ln = (float *)malloc(hidden_bytes);
        if (!layer_input || !out_proj || !resid || !post_ln) goto cpu_cleanup;
        if (ds4_gpu_tensor_read(layer_input_gpu, 0, layer_input, hidden_bytes) == 0 ||
            ds4_gpu_tensor_read(out_proj_gpu, 0, out_proj, hidden_bytes) == 0) goto cpu_cleanup;
        {
            double var = 0.0;
            for (d = 0; d < hidden; ++d) { resid[d] = layer_input[d] + out_proj[d]; var += (double)resid[d] * resid[d]; }
            var /= (double)hidden;
            for (d = 0; d < hidden; ++d) post_ln[d] = (float)(resid[d] / sqrt(var + 1e-6)) * post_attn_norm_w[d];
        }
        if (ds4_gpu_tensor_write(resid_gpu, 0, resid, hidden_bytes) == 0 ||
            ds4_gpu_tensor_write(post_ln_gpu, 0, post_ln, hidden_bytes) == 0) goto cpu_cleanup;
        free(layer_input); free(out_proj); free(resid); free(post_ln);
        return 1;
cpu_cleanup:
        free(layer_input); free(out_proj); free(resid); free(post_ln);
        return 0;
    }
    ds4_gpu_tensor *post_attn_norm_w_gpu = ds4_gpu_tensor_alloc((uint64_t)hidden * sizeof(float));
    float *resid_ptr = NULL, *post_ptr = NULL, *in_ptr = NULL, *proj_ptr = NULL, *w_ptr = NULL;
    const uint32_t block = 256u;
    if (!post_attn_norm_w_gpu) return 0;
    if (ds4_gpu_tensor_write(post_attn_norm_w_gpu, 0, post_attn_norm_w, (uint64_t)hidden * sizeof(float)) == 0) {
        ds4_gpu_tensor_free(post_attn_norm_w_gpu);
        return 0;
    }
    resid_ptr = (float *)ds4_gpu_tensor_contents(resid_gpu);
    post_ptr = (float *)ds4_gpu_tensor_contents(post_ln_gpu);
    in_ptr = (float *)ds4_gpu_tensor_contents((ds4_gpu_tensor *)layer_input_gpu);
    proj_ptr = (float *)ds4_gpu_tensor_contents((ds4_gpu_tensor *)out_proj_gpu);
    w_ptr = (float *)ds4_gpu_tensor_contents(post_attn_norm_w_gpu);
    if (!resid_ptr || !post_ptr || !in_ptr || !proj_ptr || !w_ptr) {
        ds4_gpu_tensor_free(post_attn_norm_w_gpu);
        return 0;
    }
    kernel_resid_post_ln<<<1u, block, block * sizeof(float)>>>(
        resid_ptr, post_ptr, in_ptr, proj_ptr, w_ptr, hidden);
    ds4_gpu_tensor_free(post_attn_norm_w_gpu);
    return 1;
}

/*
 * ds4_gpu_topk_softmax_tensor — stub
 *
 * Reads router logits from GPU, runs topk-softmax on CPU, returns indices + scores.
 * No GPU output tensors — writes directly to caller's CPU arrays.
 *
 * Later: real GPU kernel that writes selected indices + scores to GPU tensors.
 */
int ds4_gpu_topk_softmax_tensor(
        uint32_t *sel_idx,
        float *sel_scores,
        const ds4_gpu_tensor *logits_gpu,
        uint32_t n_logits,
        uint32_t topk)
{
    float *logits = (float *)malloc((size_t)n_logits * sizeof(float));
    if (!logits) return 0;

    if (ds4_gpu_tensor_read(logits_gpu, 0, logits, (uint64_t)n_logits * sizeof(float)) == 0) {
        free(logits);
        return 0;
    }

    {
        uint32_t i, j, m;
        for (i = 0; i < topk; ++i) sel_idx[i] = i;
        for (i = topk; i < n_logits; ++i) {
            m = 0;
            for (j = 1; j < topk; ++j) if (logits[sel_idx[j]] < logits[sel_idx[m]]) m = j;
            if (logits[i] > logits[sel_idx[m]]) sel_idx[m] = i;
        }
        for (i = 0; i < topk; ++i) {
            for (j = i + 1; j < topk; ++j) {
                if (logits[sel_idx[j]] > logits[sel_idx[i]]) {
                    uint32_t tmp = sel_idx[i]; sel_idx[i] = sel_idx[j]; sel_idx[j] = tmp;
                }
            }
        }
        {
            float maxv = logits[sel_idx[0]];
            double sum = 0.0;
            for (i = 0; i < topk; ++i) sum += exp((double)logits[sel_idx[i]] - maxv);
            for (i = 0; i < topk; ++i) sel_scores[i] = (float)(exp((double)logits[sel_idx[i]] - maxv) / sum);
        }
    }

    free(logits);
    return 1;
}

#ifdef __cplusplus
}
#endif

#endif /* QWEN36_NATIVE_HYBRID_KERNELS_CUH */
