#include "ds4_gpu.h"
extern "C" {
#include "qwen36_35a3b_q8.h"
}

#include <hip/hip_runtime.h>

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define ROUTER_COUNT 256

static int decode_q8_rows(const qwen36_gguf_file *gf,
                          const qwen36_gguf_tensor *t,
                          uint32_t row_idx,
                          uint32_t nrows,
                          uint32_t row_elems,
                          float *out,
                          char *err,
                          size_t err_cap);
static void matvec(const float *mat, const float *vec, float *out, uint32_t rows, uint32_t cols);
static inline float sigmoidf_local(float x);
static inline float softplusf_local(float x);
static inline float siluf_local(float x);

typedef struct mapped_file {
    int fd;
    void *map;
    uint64_t size;
} mapped_file;

typedef struct native_hybrid_state {
    float *deltanet_state;
    ds4_gpu_tensor *conv_ring_gpu[3];
    uint32_t conv_ring_count;
} native_hybrid_state;

typedef struct reference_hybrid_state {
    float *deltanet_state;
    float *conv_ring[3];
    uint32_t conv_ring_count;
} reference_hybrid_state;

#include "rocm/qwen36_native_hybrid_kernels.cuh"

static int mapped_file_open(mapped_file *mf, const char *path) {
    struct stat st;
    memset(mf, 0, sizeof(*mf));
    mf->fd = open(path, O_RDONLY);
    if (mf->fd == -1) return 0;
    if (fstat(mf->fd, &st) == -1) { close(mf->fd); mf->fd = -1; return 0; }
    mf->map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, mf->fd, 0);
    if (mf->map == MAP_FAILED) { close(mf->fd); mf->fd = -1; mf->map = NULL; return 0; }
    mf->size = (uint64_t)st.st_size;
    return 1;
}

static void mapped_file_close(mapped_file *mf) {
    if (!mf) return;
    if (mf->map && mf->map != MAP_FAILED && mf->size) munmap(mf->map, (size_t)mf->size);
    if (mf->fd >= 0) close(mf->fd);
    memset(mf, 0, sizeof(*mf));
    mf->fd = -1;
}

static int native_hybrid_state_init(native_hybrid_state *st,
                                    uint32_t num_v_heads, uint32_t head_k_dim, uint32_t head_v_dim,
                                    uint32_t qkv_dim) {
    memset(st, 0, sizeof(*st));
    st->deltanet_state = (float *)calloc((size_t)num_v_heads * head_k_dim * head_v_dim, sizeof(float));
    st->conv_ring_gpu[0] = ds4_gpu_tensor_alloc((uint64_t)qkv_dim * sizeof(float));
    st->conv_ring_gpu[1] = ds4_gpu_tensor_alloc((uint64_t)qkv_dim * sizeof(float));
    st->conv_ring_gpu[2] = ds4_gpu_tensor_alloc((uint64_t)qkv_dim * sizeof(float));
    st->conv_ring_count = 0;
    if (!st->deltanet_state || !st->conv_ring_gpu[0] || !st->conv_ring_gpu[1] || !st->conv_ring_gpu[2]) {
        free(st->deltanet_state);
        ds4_gpu_tensor_free(st->conv_ring_gpu[0]);
        ds4_gpu_tensor_free(st->conv_ring_gpu[1]);
        ds4_gpu_tensor_free(st->conv_ring_gpu[2]);
        memset(st, 0, sizeof(*st));
        return 0;
    }
    /* zero-initialize ring tensors */
    ds4_gpu_tensor_fill_f32(st->conv_ring_gpu[0], 0.0f, qkv_dim);
    ds4_gpu_tensor_fill_f32(st->conv_ring_gpu[1], 0.0f, qkv_dim);
    ds4_gpu_tensor_fill_f32(st->conv_ring_gpu[2], 0.0f, qkv_dim);
    return 1;
}

static void native_hybrid_state_free(native_hybrid_state *st) {
    if (!st) return;
    free(st->deltanet_state);
    ds4_gpu_tensor_free(st->conv_ring_gpu[0]);
    ds4_gpu_tensor_free(st->conv_ring_gpu[1]);
    ds4_gpu_tensor_free(st->conv_ring_gpu[2]);
    memset(st, 0, sizeof(*st));
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1f;
    const uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            uint32_t e = 127 - 15 + 1u, m = mant;
            while ((m & 0x400u) == 0u) { m <<= 1; e--; }
            m &= 0x3ffu;
            bits = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    {
        union { uint32_t u; float f; } v = { bits };
        return v.f;
    }
}

static inline float sigmoidf_local(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float softplusf_local(float x) { return log1pf(expf(x)); }
static inline float siluf_local(float x) { return x / (1.0f + expf(-x)); }

static void matvec(const float *mat, const float *vec, float *out, uint32_t rows, uint32_t cols) {
    for (uint32_t r = 0; r < rows; ++r) {
        const float *row = mat + (size_t)r * cols;
        float sum = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) sum += row[c] * vec[c];
        out[r] = sum;
    }
}

static int decode_q8_rows(const qwen36_gguf_file *gf,
                          const qwen36_gguf_tensor *t,
                          uint32_t row_idx,
                          uint32_t nrows,
                          uint32_t row_elems,
                          float *out,
                          char *err,
                          size_t err_cap) {
    const size_t row_size = 34u * (row_elems / 32u);
    uint8_t *buf = (uint8_t *)malloc((size_t)nrows * row_size);
    if (!buf) {
        snprintf(err, err_cap, "oom decode_q8_rows");
        return 0;
    }
    if (!qwen36_gguf_read_tensor_bytes(gf, t, (uint64_t)row_idx * row_size, buf, (size_t)nrows * row_size, err, err_cap)) {
        free(buf);
        return 0;
    }
    for (uint32_t r = 0; r < nrows; ++r) {
        const uint8_t *row = buf + (size_t)r * row_size;
        float *dst = out + (size_t)r * row_elems;
        for (uint32_t j = 0; j < row_elems / 32u; ++j) {
            const uint8_t *block = row + j * 34u;
            const float dscale = f16_to_f32((uint16_t)(block[0] | (block[1] << 8)));
            const int8_t *qs = (const int8_t *)(block + 2);
            for (uint32_t k = 0; k < 32u; ++k) dst[j * 32u + k] = dscale * (float)qs[k];
        }
    }
    free(buf);
    return 1;
}

static int reference_hybrid_state_init(reference_hybrid_state *st,
                                       uint32_t num_v_heads,
                                       uint32_t head_k_dim,
                                       uint32_t head_v_dim,
                                       uint32_t qkv_dim) {
    memset(st, 0, sizeof(*st));
    st->deltanet_state = (float *)calloc((size_t)num_v_heads * head_k_dim * head_v_dim, sizeof(float));
    st->conv_ring[0] = (float *)calloc(qkv_dim, sizeof(float));
    st->conv_ring[1] = (float *)calloc(qkv_dim, sizeof(float));
    st->conv_ring[2] = (float *)calloc(qkv_dim, sizeof(float));
    if (!st->deltanet_state || !st->conv_ring[0] || !st->conv_ring[1] || !st->conv_ring[2]) {
        free(st->deltanet_state);
        free(st->conv_ring[0]);
        free(st->conv_ring[1]);
        free(st->conv_ring[2]);
        memset(st, 0, sizeof(*st));
        return 0;
    }
    return 1;
}

static void reference_hybrid_state_free(reference_hybrid_state *st) {
    if (!st) return;
    free(st->deltanet_state);
    free(st->conv_ring[0]);
    free(st->conv_ring[1]);
    free(st->conv_ring[2]);
    memset(st, 0, sizeof(*st));
}

static int read_f32_tensor_all(const qwen36_gguf_file *gf,
                               const qwen36_gguf_tensor *t,
                               float **out,
                               size_t elems,
                               char *err,
                               size_t err_cap) {
    float *buf = (float *)malloc(elems * sizeof(float));
    if (!buf) {
        snprintf(err, err_cap, "oom read_f32_tensor_all");
        return 0;
    }
    if (!qwen36_gguf_read_tensor_bytes(gf, t, 0, buf, elems * sizeof(float), err, err_cap)) {
        free(buf);
        return 0;
    }
    *out = buf;
    return 1;
}

static void ref_build_perms(uint32_t n_heads, uint32_t *gg_to_hf, uint32_t *hf_to_gg) {
    uint32_t pos = 0;
    for (uint32_t i = 0; i < n_heads; i += 2) gg_to_hf[pos++] = i;
    for (uint32_t i = 1; i < n_heads; i += 2) gg_to_hf[pos++] = i;
    for (uint32_t i = 0; i < n_heads; ++i) hf_to_gg[gg_to_hf[i]] = i;
}

static void ref_reorder_head_rows_f32(float *dst, const float *src, uint32_t n_heads, uint32_t head_dim) {
    uint32_t *gg_to_hf = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    if (!gg_to_hf || !hf_to_gg) {
        free(gg_to_hf);
        free(hf_to_gg);
        return;
    }
    ref_build_perms(n_heads, gg_to_hf, hf_to_gg);
    for (uint32_t h = 0; h < n_heads; ++h) {
        memcpy(dst + (size_t)h * head_dim,
               src + (size_t)hf_to_gg[h] * head_dim,
               (size_t)head_dim * sizeof(float));
    }
    free(gg_to_hf);
    free(hf_to_gg);
}

static void ref_reorder_head_scalars_f32(float *dst, const float *src, uint32_t n_heads) {
    uint32_t *gg_to_hf = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    if (!gg_to_hf || !hf_to_gg) {
        free(gg_to_hf);
        free(hf_to_gg);
        return;
    }
    ref_build_perms(n_heads, gg_to_hf, hf_to_gg);
    for (uint32_t h = 0; h < n_heads; ++h) dst[h] = src[hf_to_gg[h]];
    free(gg_to_hf);
    free(hf_to_gg);
}

static void ref_reorder_qkv_v_gghf(float *dst,
                                   const float *src,
                                   uint32_t key_dim,
                                   uint32_t n_heads,
                                   uint32_t head_dim) {
    uint32_t *gg_to_hf = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    const uint32_t value_dim = n_heads * head_dim;
    const uint32_t qkv_dim = key_dim * 2u + value_dim;
    const uint32_t v_off = key_dim * 2u;
    if (!gg_to_hf || !hf_to_gg) {
        free(gg_to_hf);
        free(hf_to_gg);
        return;
    }
    ref_build_perms(n_heads, gg_to_hf, hf_to_gg);
    memcpy(dst, src, (size_t)(key_dim * 2u) * sizeof(float));
    for (uint32_t h = 0; h < n_heads; ++h) {
        memcpy(dst + v_off + (size_t)h * head_dim,
               src + v_off + (size_t)hf_to_gg[h] * head_dim,
               (size_t)head_dim * sizeof(float));
    }
    free(gg_to_hf);
    free(hf_to_gg);
}

static void ref_reorder_out_in_hftogg(float *dst, const float *src, uint32_t n_heads, uint32_t head_dim) {
    uint32_t *gg_to_hf = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    if (!gg_to_hf || !hf_to_gg) {
        free(gg_to_hf);
        free(hf_to_gg);
        return;
    }
    ref_build_perms(n_heads, gg_to_hf, hf_to_gg);
    for (uint32_t h_gg = 0; h_gg < n_heads; ++h_gg) {
        memcpy(dst + (size_t)h_gg * head_dim,
               src + (size_t)gg_to_hf[h_gg] * head_dim,
               (size_t)head_dim * sizeof(float));
    }
    free(gg_to_hf);
    free(hf_to_gg);
}

static int matvec_q8_tensor_all_rows(const qwen36_gguf_file *gf,
                                     const qwen36_gguf_tensor *t,
                                     uint32_t row_idx,
                                     uint32_t rows,
                                     uint32_t cols,
                                     const float *vec,
                                     float *out,
                                     char *err,
                                     size_t err_cap) {
    float *mat = (float *)malloc((size_t)rows * cols * sizeof(float));
    if (!mat) {
        snprintf(err, err_cap, "oom matvec_q8_tensor_all_rows");
        return 0;
    }
    if (!decode_q8_rows(gf, t, row_idx, rows, cols, mat, err, err_cap)) {
        free(mat);
        return 0;
    }
    matvec(mat, vec, out, rows, cols);
    free(mat);
    return 1;
}

static int matvec_f32_tensor_all_rows(const qwen36_gguf_file *gf,
                                      const qwen36_gguf_tensor *t,
                                      uint32_t rows,
                                      uint32_t cols,
                                      const float *vec,
                                      float *out,
                                      char *err,
                                      size_t err_cap) {
    float *mat = NULL;
    if (!read_f32_tensor_all(gf, t, &mat, (size_t)rows * cols, err, err_cap)) return 0;
    matvec(mat, vec, out, rows, cols);
    free(mat);
    return 1;
}

static void ref_rms_norm_weight(float *out, const float *in, const float *w, uint32_t dim) {
    double var = 0.0;
    for (uint32_t d = 0; d < dim; ++d) var += (double)in[d] * in[d];
    var /= (double)dim;
    for (uint32_t d = 0; d < dim; ++d) out[d] = (float)(in[d] / sqrt(var + 1e-6)) * w[d];
}

static void ref_topk_softmax256(const float *logits, uint32_t k, uint32_t *idx, float *scores) {
    uint32_t i, j, m;
    for (i = 0; i < k; ++i) idx[i] = i;
    for (i = k; i < ROUTER_COUNT; ++i) {
        m = 0;
        for (j = 1; j < k; ++j) if (logits[idx[j]] < logits[idx[m]]) m = j;
        if (logits[i] > logits[idx[m]]) idx[m] = i;
    }
    for (i = 0; i < k; ++i) {
        for (j = i + 1; j < k; ++j) {
            if (logits[idx[j]] > logits[idx[i]]) {
                uint32_t tmp = idx[i];
                idx[i] = idx[j];
                idx[j] = tmp;
            }
        }
    }
    {
        float maxv = logits[idx[0]];
        double sum = 0.0;
        for (i = 0; i < k; ++i) sum += exp((double)logits[idx[i]] - maxv);
        for (i = 0; i < k; ++i) scores[i] = (float)(exp((double)logits[idx[i]] - maxv) / sum);
    }
}

static void reference_fused_reorder_conv(float *z_hf,
                                         float *conv,
                                         float *q_hf,
                                         float *k_hf,
                                         float *v_hf,
                                         float *beta,
                                         float *g,
                                         float *ring0,
                                         float *ring1,
                                         float *ring2,
                                         uint32_t *ring_count,
                                         const float *z_gg,
                                         const float *a_gg,
                                         const float *b_gg,
                                         const float *qkv_raw,
                                         const float *conv_w,
                                         const float *A_log,
                                         const float *dt_bias,
                                         uint32_t num_v_heads,
                                         uint32_t num_k_heads,
                                         uint32_t head_k_dim,
                                         uint32_t head_v_dim,
                                         uint32_t key_dim) {
    const uint32_t value_dim = num_v_heads * head_v_dim;
    const uint32_t qkv_dim = key_dim * 2u + value_dim;
    const uint32_t rep = num_v_heads / num_k_heads;
    float *qkv_hf = (float *)malloc((size_t)qkv_dim * sizeof(float));
    uint32_t h, d;

    ref_reorder_head_rows_f32(z_hf, z_gg, num_v_heads, head_v_dim);
    ref_reorder_head_scalars_f32(beta, b_gg, num_v_heads);
    ref_reorder_head_scalars_f32(g, a_gg, num_v_heads);
    ref_reorder_qkv_v_gghf(qkv_hf, qkv_raw, key_dim, num_v_heads, head_v_dim);

    for (d = 0; d < qkv_dim; ++d) {
        double sum = 0.0;
        if (*ring_count >= 3) sum += (double)conv_w[(size_t)d * 4u + 0u] * ring2[d];
        if (*ring_count >= 2) sum += (double)conv_w[(size_t)d * 4u + 1u] * ring1[d];
        if (*ring_count >= 1) sum += (double)conv_w[(size_t)d * 4u + 2u] * ring0[d];
        sum += (double)conv_w[(size_t)d * 4u + 3u] * qkv_hf[d];
        conv[d] = siluf_local((float)sum);
    }
    {
        const float *cq = conv;
        const float *ck = cq + key_dim;
        const float *cv = ck + key_dim;
        for (h = 0; h < num_k_heads; ++h) {
            for (d = 0; d < head_k_dim; ++d) {
                const float qv = cq[h * head_k_dim + d];
                const float kv = ck[h * head_k_dim + d];
                for (uint32_t vh = 0; vh < rep; ++vh) {
                    const uint32_t dst_h = h * rep + vh;
                    q_hf[(size_t)dst_h * head_k_dim + d] = qv;
                    k_hf[(size_t)dst_h * head_k_dim + d] = kv;
                }
            }
        }
        for (h = 0; h < num_v_heads; ++h) {
            memcpy(v_hf + (size_t)h * head_v_dim,
                   cv + (size_t)h * head_v_dim,
                   (size_t)head_v_dim * sizeof(float));
            beta[h] = sigmoidf_local(beta[h]);
            g[h] = A_log[h] * softplusf_local(g[h] + dt_bias[h]);
        }
    }
    if (*ring_count >= 2) memcpy(ring2, ring1, (size_t)qkv_dim * sizeof(float));
    if (*ring_count >= 1) memcpy(ring1, ring0, (size_t)qkv_dim * sizeof(float));
    memcpy(ring0, qkv_hf, (size_t)qkv_dim * sizeof(float));
    if (*ring_count < 3) (*ring_count)++;
    free(qkv_hf);
}

static void compare_arrays_brief(const char *label,
                                 const float *a,
                                 const float *b,
                                 uint32_t n) {
    double mean_abs = 0.0;
    float max_abs = 0.0f;
    uint32_t max_idx = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const float diff = fabsf(a[i] - b[i]);
        mean_abs += diff;
        if (diff > max_abs) {
            max_abs = diff;
            max_idx = i;
        }
    }
    mean_abs /= (double)n;
    printf("%s max_abs=%.8f max_idx=%u mean_abs=%.8f %s\n",
           label, max_abs, max_idx, (float)mean_abs,
           max_abs <= 1e-5f ? "PASS" : "FAIL");
}

static void print_f32_prefix(const char *label, const float *x, uint32_t n) {
    const uint32_t limit = n < 8u ? n : 8u;
    printf("%s", label);
    for (uint32_t i = 0; i < limit; ++i) printf("%s%.6f", i == 0 ? "" : ",", x[i]);
    printf("\n");
}

static void print_topk_summary(const char *label, const uint32_t *idx, const float *scores, uint32_t k) {
    const uint32_t limit = k < 8u ? k : 8u;
    printf("%s", label);
    for (uint32_t i = 0; i < limit; ++i) {
        printf("%s%u:%.6f", i == 0 ? "" : ",", idx[i], scores[i]);
    }
    printf("\n");
}

static int validate_fused_reorder_conv_kernel(const qwen36_gguf_file *gf,
                                              const qwen36_35a3b_q8_layer *layer,
                                              uint32_t num_v_heads,
                                              uint32_t num_k_heads,
                                              uint32_t head_k_dim,
                                              uint32_t head_v_dim,
                                              uint32_t key_dim,
                                              char *err,
                                              size_t err_cap) {
    const uint32_t value_dim = num_v_heads * head_v_dim;
    const uint32_t qkv_dim = key_dim * 2u + value_dim;
    const uint32_t ab_dim = num_v_heads;
    const uint64_t z_bytes = (uint64_t)value_dim * sizeof(float);
    const uint64_t ab_bytes = (uint64_t)ab_dim * sizeof(float);
    const uint64_t qkv_bytes = (uint64_t)qkv_dim * sizeof(float);
    float *conv_w = NULL, *A_log = NULL, *dt_bias = NULL;
    float *z_gg = NULL, *a_gg = NULL, *b_gg = NULL, *qkv_raw = NULL;
    float *cpu_z = NULL, *cpu_conv = NULL, *cpu_q = NULL, *cpu_k = NULL, *cpu_v = NULL, *cpu_beta = NULL, *cpu_g = NULL;
    float *gpu_z = NULL, *gpu_conv = NULL, *gpu_q = NULL, *gpu_k = NULL, *gpu_v = NULL, *gpu_beta = NULL, *gpu_g = NULL;
    float *cpu_ring[3] = {NULL, NULL, NULL};
    float *gpu_ring_cpu[3] = {NULL, NULL, NULL};
    uint32_t cpu_ring_count = 2u;
    uint32_t gpu_ring_count = 2u;
    ds4_gpu_tensor *z_gpu = NULL, *a_gpu = NULL, *b_gpu = NULL, *qkv_gpu = NULL;
    ds4_gpu_tensor *conv_gpu = NULL, *q_gpu = NULL, *k_gpu = NULL, *v_gpu = NULL, *beta_gpu = NULL, *g_gpu = NULL;
    ds4_gpu_tensor *conv_w_gpu = NULL, *A_log_gpu = NULL, *dt_bias_gpu = NULL;
    ds4_gpu_tensor *ring_gpu[3] = {NULL, NULL, NULL};
    int ok = 0;

    if (!read_f32_tensor_all(gf, layer->ssm_conv1d, &conv_w, (size_t)qkv_dim * 4u, err, err_cap) ||
        !read_f32_tensor_all(gf, layer->ssm_a, &A_log, num_v_heads, err, err_cap) ||
        !read_f32_tensor_all(gf, layer->ssm_dt_bias, &dt_bias, num_v_heads, err, err_cap)) {
        goto cleanup;
    }
    z_gg = (float *)malloc(z_bytes);
    a_gg = (float *)malloc(ab_bytes);
    b_gg = (float *)malloc(ab_bytes);
    qkv_raw = (float *)malloc(qkv_bytes);
    cpu_z = (float *)malloc(z_bytes);
    cpu_conv = (float *)malloc(qkv_bytes);
    cpu_q = (float *)malloc((uint64_t)num_v_heads * head_k_dim * sizeof(float));
    cpu_k = (float *)malloc((uint64_t)num_v_heads * head_k_dim * sizeof(float));
    cpu_v = (float *)malloc(z_bytes);
    cpu_beta = (float *)malloc(ab_bytes);
    cpu_g = (float *)malloc(ab_bytes);
    gpu_z = (float *)malloc(z_bytes);
    gpu_conv = (float *)malloc(qkv_bytes);
    gpu_q = (float *)malloc((uint64_t)num_v_heads * head_k_dim * sizeof(float));
    gpu_k = (float *)malloc((uint64_t)num_v_heads * head_k_dim * sizeof(float));
    gpu_v = (float *)malloc(z_bytes);
    gpu_beta = (float *)malloc(ab_bytes);
    gpu_g = (float *)malloc(ab_bytes);
    cpu_ring[0] = (float *)malloc(qkv_bytes);
    cpu_ring[1] = (float *)malloc(qkv_bytes);
    cpu_ring[2] = (float *)malloc(qkv_bytes);
    gpu_ring_cpu[0] = (float *)malloc(qkv_bytes);
    gpu_ring_cpu[1] = (float *)malloc(qkv_bytes);
    gpu_ring_cpu[2] = (float *)malloc(qkv_bytes);
    if (!z_gg || !a_gg || !b_gg || !qkv_raw || !cpu_z || !cpu_conv || !cpu_q || !cpu_k || !cpu_v || !cpu_beta || !cpu_g ||
        !gpu_z || !gpu_conv || !gpu_q || !gpu_k || !gpu_v || !gpu_beta || !gpu_g ||
        !cpu_ring[0] || !cpu_ring[1] || !cpu_ring[2] || !gpu_ring_cpu[0] || !gpu_ring_cpu[1] || !gpu_ring_cpu[2]) {
        snprintf(err, err_cap, "oom validate_fused_reorder_conv_kernel");
        goto cleanup;
    }

    for (uint32_t i = 0; i < value_dim; ++i) z_gg[i] = 0.2f * sinf(0.11f * (float)(i + 1)) + 0.1f * cosf(0.07f * (float)(i + 3));
    for (uint32_t i = 0; i < ab_dim; ++i) {
        a_gg[i] = -0.35f + 0.17f * (float)i;
        b_gg[i] = 0.21f - 0.09f * (float)i;
    }
    for (uint32_t i = 0; i < qkv_dim; ++i) qkv_raw[i] = 0.3f * sinf(0.013f * (float)(i + 1)) + 0.08f * cosf(0.021f * (float)(i + 9));
    for (uint32_t i = 0; i < qkv_dim; ++i) {
        cpu_ring[0][i] = 0.05f * sinf(0.019f * (float)(i + 1));
        cpu_ring[1][i] = 0.04f * cosf(0.023f * (float)(i + 5));
        cpu_ring[2][i] = 0.0f;
    }
    memcpy(gpu_ring_cpu[0], cpu_ring[0], qkv_bytes);
    memcpy(gpu_ring_cpu[1], cpu_ring[1], qkv_bytes);
    memcpy(gpu_ring_cpu[2], cpu_ring[2], qkv_bytes);

    reference_fused_reorder_conv(cpu_z, cpu_conv, cpu_q, cpu_k, cpu_v, cpu_beta, cpu_g,
                                 cpu_ring[0], cpu_ring[1], cpu_ring[2], &cpu_ring_count,
                                 z_gg, a_gg, b_gg, qkv_raw, conv_w, A_log, dt_bias,
                                 num_v_heads, num_k_heads, head_k_dim, head_v_dim, key_dim);

    z_gpu = ds4_gpu_tensor_alloc(z_bytes);
    a_gpu = ds4_gpu_tensor_alloc(ab_bytes);
    b_gpu = ds4_gpu_tensor_alloc(ab_bytes);
    qkv_gpu = ds4_gpu_tensor_alloc(qkv_bytes);
    conv_gpu = ds4_gpu_tensor_alloc(qkv_bytes);
    q_gpu = ds4_gpu_tensor_alloc((uint64_t)num_v_heads * head_k_dim * sizeof(float));
    k_gpu = ds4_gpu_tensor_alloc((uint64_t)num_v_heads * head_k_dim * sizeof(float));
    v_gpu = ds4_gpu_tensor_alloc(z_bytes);
    beta_gpu = ds4_gpu_tensor_alloc(ab_bytes);
    g_gpu = ds4_gpu_tensor_alloc(ab_bytes);
    conv_w_gpu = ds4_gpu_tensor_alloc((uint64_t)qkv_dim * 4u * sizeof(float));
    A_log_gpu = ds4_gpu_tensor_alloc(ab_bytes);
    dt_bias_gpu = ds4_gpu_tensor_alloc(ab_bytes);
    ring_gpu[0] = ds4_gpu_tensor_alloc(qkv_bytes);
    ring_gpu[1] = ds4_gpu_tensor_alloc(qkv_bytes);
    ring_gpu[2] = ds4_gpu_tensor_alloc(qkv_bytes);
    if (!z_gpu || !a_gpu || !b_gpu || !qkv_gpu || !conv_gpu || !q_gpu || !k_gpu || !v_gpu || !beta_gpu || !g_gpu ||
        !conv_w_gpu || !A_log_gpu || !dt_bias_gpu || !ring_gpu[0] || !ring_gpu[1] || !ring_gpu[2]) {
        snprintf(err, err_cap, "gpu alloc failed validate_fused_reorder_conv_kernel");
        goto cleanup;
    }
    if (ds4_gpu_tensor_write(z_gpu, 0, z_gg, z_bytes) == 0 ||
        ds4_gpu_tensor_write(a_gpu, 0, a_gg, ab_bytes) == 0 ||
        ds4_gpu_tensor_write(b_gpu, 0, b_gg, ab_bytes) == 0 ||
        ds4_gpu_tensor_write(qkv_gpu, 0, qkv_raw, qkv_bytes) == 0 ||
        ds4_gpu_tensor_write(conv_w_gpu, 0, conv_w, (uint64_t)qkv_dim * 4u * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(A_log_gpu, 0, A_log, ab_bytes) == 0 ||
        ds4_gpu_tensor_write(dt_bias_gpu, 0, dt_bias, ab_bytes) == 0 ||
        ds4_gpu_tensor_write(ring_gpu[0], 0, gpu_ring_cpu[0], qkv_bytes) == 0 ||
        ds4_gpu_tensor_write(ring_gpu[1], 0, gpu_ring_cpu[1], qkv_bytes) == 0 ||
        ds4_gpu_tensor_write(ring_gpu[2], 0, gpu_ring_cpu[2], qkv_bytes) == 0) {
        snprintf(err, err_cap, "gpu write failed validate_fused_reorder_conv_kernel");
        goto cleanup;
    }
    if (ds4_gpu_begin_commands() == 0) {
        snprintf(err, err_cap, "gpu begin failed validate_fused_reorder_conv_kernel");
        goto cleanup;
    }
    if (ds4_gpu_fused_reorder_conv_tensor(conv_gpu, q_gpu, k_gpu, v_gpu, beta_gpu, g_gpu,
                                          z_gpu, a_gpu, b_gpu, qkv_gpu,
                                          ring_gpu, &gpu_ring_count,
                                          num_v_heads, num_k_heads, head_k_dim, head_v_dim, key_dim,
                                          conv_w_gpu, A_log_gpu, dt_bias_gpu) == 0) {
        snprintf(err, err_cap, "gpu fused reorder conv failed");
        goto cleanup;
    }
    if (ds4_gpu_end_commands() == 0) {
        snprintf(err, err_cap, "gpu end failed validate_fused_reorder_conv_kernel");
        goto cleanup;
    }
    if (ds4_gpu_tensor_read(z_gpu, 0, gpu_z, z_bytes) == 0 ||
        ds4_gpu_tensor_read(conv_gpu, 0, gpu_conv, qkv_bytes) == 0 ||
        ds4_gpu_tensor_read(q_gpu, 0, gpu_q, (uint64_t)num_v_heads * head_k_dim * sizeof(float)) == 0 ||
        ds4_gpu_tensor_read(k_gpu, 0, gpu_k, (uint64_t)num_v_heads * head_k_dim * sizeof(float)) == 0 ||
        ds4_gpu_tensor_read(v_gpu, 0, gpu_v, z_bytes) == 0 ||
        ds4_gpu_tensor_read(beta_gpu, 0, gpu_beta, ab_bytes) == 0 ||
        ds4_gpu_tensor_read(g_gpu, 0, gpu_g, ab_bytes) == 0 ||
        ds4_gpu_tensor_read(ring_gpu[0], 0, gpu_ring_cpu[0], qkv_bytes) == 0 ||
        ds4_gpu_tensor_read(ring_gpu[1], 0, gpu_ring_cpu[1], qkv_bytes) == 0 ||
        ds4_gpu_tensor_read(ring_gpu[2], 0, gpu_ring_cpu[2], qkv_bytes) == 0) {
        snprintf(err, err_cap, "gpu read failed validate_fused_reorder_conv_kernel");
        goto cleanup;
    }

    printf("fused_reorder_conv ring_count cpu=%u gpu=%u %s\n",
           cpu_ring_count, gpu_ring_count, cpu_ring_count == gpu_ring_count ? "PASS" : "FAIL");
    compare_arrays_brief("fused/z", cpu_z, gpu_z, value_dim);
    compare_arrays_brief("fused/conv", cpu_conv, gpu_conv, qkv_dim);
    compare_arrays_brief("fused/q", cpu_q, gpu_q, num_v_heads * head_k_dim);
    compare_arrays_brief("fused/k", cpu_k, gpu_k, num_v_heads * head_k_dim);
    compare_arrays_brief("fused/v", cpu_v, gpu_v, value_dim);
    compare_arrays_brief("fused/beta", cpu_beta, gpu_beta, num_v_heads);
    compare_arrays_brief("fused/g", cpu_g, gpu_g, num_v_heads);
    compare_arrays_brief("fused/ring0", cpu_ring[0], gpu_ring_cpu[0], qkv_dim);
    compare_arrays_brief("fused/ring1", cpu_ring[1], gpu_ring_cpu[1], qkv_dim);
    compare_arrays_brief("fused/ring2", cpu_ring[2], gpu_ring_cpu[2], qkv_dim);
    ok = 1;

cleanup:
    free(conv_w);
    free(A_log);
    free(dt_bias);
    free(z_gg);
    free(a_gg);
    free(b_gg);
    free(qkv_raw);
    free(cpu_z);
    free(cpu_conv);
    free(cpu_q);
    free(cpu_k);
    free(cpu_v);
    free(cpu_beta);
    free(cpu_g);
    free(gpu_z);
    free(gpu_conv);
    free(gpu_q);
    free(gpu_k);
    free(gpu_v);
    free(gpu_beta);
    free(gpu_g);
    free(cpu_ring[0]);
    free(cpu_ring[1]);
    free(cpu_ring[2]);
    free(gpu_ring_cpu[0]);
    free(gpu_ring_cpu[1]);
    free(gpu_ring_cpu[2]);
    ds4_gpu_tensor_free(z_gpu);
    ds4_gpu_tensor_free(a_gpu);
    ds4_gpu_tensor_free(b_gpu);
    ds4_gpu_tensor_free(qkv_gpu);
    ds4_gpu_tensor_free(conv_gpu);
    ds4_gpu_tensor_free(q_gpu);
    ds4_gpu_tensor_free(k_gpu);
    ds4_gpu_tensor_free(v_gpu);
    ds4_gpu_tensor_free(beta_gpu);
    ds4_gpu_tensor_free(g_gpu);
    ds4_gpu_tensor_free(conv_w_gpu);
    ds4_gpu_tensor_free(A_log_gpu);
    ds4_gpu_tensor_free(dt_bias_gpu);
    ds4_gpu_tensor_free(ring_gpu[0]);
    ds4_gpu_tensor_free(ring_gpu[1]);
    ds4_gpu_tensor_free(ring_gpu[2]);
    return ok;
}

static int validate_deltanet_norm_kernel(const qwen36_gguf_file *gf,
                                         const qwen36_35a3b_q8_layer *layer,
                                         uint32_t num_v_heads,
                                         uint32_t head_k_dim,
                                         uint32_t head_v_dim,
                                         char *err,
                                         size_t err_cap) {
    const uint32_t value_dim = num_v_heads * head_v_dim;
    const uint64_t state_bytes = (uint64_t)num_v_heads * head_k_dim * head_v_dim * sizeof(float);
    const uint64_t kq_bytes = (uint64_t)num_v_heads * head_k_dim * sizeof(float);
    const uint64_t v_bytes = (uint64_t)value_dim * sizeof(float);
    const uint64_t scalar_bytes = (uint64_t)num_v_heads * sizeof(float);
    float *ssm_norm_w = NULL;
    float *state_init = NULL, *k_hf = NULL, *q_hf = NULL, *v_hf = NULL, *g_in = NULL, *beta_in = NULL, *z_hf = NULL;
    float *cpu_state = NULL, *cpu_core = NULL, *cpu_out_in_hf = NULL, *cpu_out_in_gg = NULL;
    float *gpu_state = NULL, *gpu_core = NULL, *gpu_out_in_gg = NULL;
    float *k_scaled = NULL, *q_scaled = NULL, *delta_tmp = NULL;
    ds4_gpu_tensor *state_gpu = NULL, *core_gpu = NULL, *k_gpu = NULL, *q_gpu = NULL, *v_gpu = NULL, *g_gpu = NULL, *beta_gpu = NULL;
    ds4_gpu_tensor *z_gpu = NULL, *out_in_gg_gpu = NULL;
    int ok = 0;

    if (!read_f32_tensor_all(gf, layer->ssm_norm, &ssm_norm_w, head_v_dim, err, err_cap)) goto cleanup;

    state_init = (float *)malloc(state_bytes);
    k_hf = (float *)malloc(kq_bytes);
    q_hf = (float *)malloc(kq_bytes);
    v_hf = (float *)malloc(v_bytes);
    g_in = (float *)malloc(scalar_bytes);
    beta_in = (float *)malloc(scalar_bytes);
    z_hf = (float *)malloc(v_bytes);
    cpu_state = (float *)malloc(state_bytes);
    cpu_core = (float *)calloc(value_dim, sizeof(float));
    cpu_out_in_hf = (float *)malloc(v_bytes);
    cpu_out_in_gg = (float *)malloc(v_bytes);
    gpu_state = (float *)malloc(state_bytes);
    gpu_core = (float *)malloc(v_bytes);
    gpu_out_in_gg = (float *)malloc(v_bytes);
    k_scaled = (float *)malloc((size_t)head_k_dim * sizeof(float));
    q_scaled = (float *)malloc((size_t)head_k_dim * sizeof(float));
    delta_tmp = (float *)malloc((size_t)head_v_dim * sizeof(float));
    if (!state_init || !k_hf || !q_hf || !v_hf || !g_in || !beta_in || !z_hf ||
        !cpu_state || !cpu_core || !cpu_out_in_hf || !cpu_out_in_gg ||
        !gpu_state || !gpu_core || !gpu_out_in_gg || !k_scaled || !q_scaled || !delta_tmp) {
        snprintf(err, err_cap, "oom validate_deltanet_norm_kernel");
        goto cleanup;
    }

    for (uint32_t i = 0; i < num_v_heads * head_k_dim * head_v_dim; ++i) {
        state_init[i] = 0.03f * sinf(0.017f * (float)(i + 1)) + 0.02f * cosf(0.011f * (float)(i + 7));
    }
    for (uint32_t i = 0; i < num_v_heads * head_k_dim; ++i) {
        k_hf[i] = 0.25f * sinf(0.023f * (float)(i + 1));
        q_hf[i] = 0.18f * cosf(0.019f * (float)(i + 3));
    }
    for (uint32_t i = 0; i < value_dim; ++i) {
        v_hf[i] = 0.14f * sinf(0.031f * (float)(i + 5));
        z_hf[i] = 0.22f * cosf(0.013f * (float)(i + 9));
    }
    for (uint32_t h = 0; h < num_v_heads; ++h) {
        g_in[h] = -0.08f + 0.11f * (float)h;
        beta_in[h] = 0.17f + 0.07f * (float)h;
    }

    memcpy(cpu_state, state_init, state_bytes);
    for (uint32_t h = 0; h < num_v_heads; ++h) {
        stub_deltanet_head_step(cpu_state + (size_t)h * head_k_dim * head_v_dim,
                                head_k_dim, head_v_dim,
                                k_hf + (size_t)h * head_k_dim,
                                q_hf + (size_t)h * head_k_dim,
                                v_hf + (size_t)h * head_v_dim,
                                expf(g_in[h]),
                                beta_in[h],
                                k_scaled,
                                q_scaled,
                                delta_tmp,
                                cpu_core + (size_t)h * head_v_dim);
    }
    for (uint32_t h = 0; h < num_v_heads; ++h) {
        size_t base = (size_t)h * head_v_dim;
        double var = 0.0;
        for (uint32_t vd = 0; vd < head_v_dim; ++vd) {
            double cv = cpu_core[base + vd];
            var += cv * cv;
        }
        var /= (double)head_v_dim;
        for (uint32_t vd = 0; vd < head_v_dim; ++vd) {
            float cv = cpu_core[base + vd];
            cpu_out_in_hf[base + vd] = (float)(cv / sqrt(var + 1e-6)) * ssm_norm_w[vd] * siluf_local(z_hf[base + vd]);
        }
    }
    ref_reorder_out_in_hftogg(cpu_out_in_gg, cpu_out_in_hf, num_v_heads, head_v_dim);

    state_gpu = ds4_gpu_tensor_alloc(state_bytes);
    core_gpu = ds4_gpu_tensor_alloc(v_bytes);
    k_gpu = ds4_gpu_tensor_alloc(kq_bytes);
    q_gpu = ds4_gpu_tensor_alloc(kq_bytes);
    v_gpu = ds4_gpu_tensor_alloc(v_bytes);
    g_gpu = ds4_gpu_tensor_alloc(scalar_bytes);
    beta_gpu = ds4_gpu_tensor_alloc(scalar_bytes);
    z_gpu = ds4_gpu_tensor_alloc(v_bytes);
    out_in_gg_gpu = ds4_gpu_tensor_alloc(v_bytes);
    if (!state_gpu || !core_gpu || !k_gpu || !q_gpu || !v_gpu || !g_gpu || !beta_gpu || !z_gpu || !out_in_gg_gpu) {
        snprintf(err, err_cap, "gpu alloc failed validate_deltanet_norm_kernel");
        goto cleanup;
    }
    if (ds4_gpu_tensor_write(state_gpu, 0, state_init, state_bytes) == 0 ||
        ds4_gpu_tensor_write(k_gpu, 0, k_hf, kq_bytes) == 0 ||
        ds4_gpu_tensor_write(q_gpu, 0, q_hf, kq_bytes) == 0 ||
        ds4_gpu_tensor_write(v_gpu, 0, v_hf, v_bytes) == 0 ||
        ds4_gpu_tensor_write(g_gpu, 0, g_in, scalar_bytes) == 0 ||
        ds4_gpu_tensor_write(beta_gpu, 0, beta_in, scalar_bytes) == 0 ||
        ds4_gpu_tensor_write(z_gpu, 0, z_hf, v_bytes) == 0) {
        snprintf(err, err_cap, "gpu write failed validate_deltanet_norm_kernel");
        goto cleanup;
    }
    if (!ds4_gpu_deltanet_step_tensor(state_gpu, core_gpu, k_gpu, q_gpu, v_gpu, g_gpu, beta_gpu,
                                      num_v_heads, head_k_dim, head_v_dim)) {
        snprintf(err, err_cap, "gpu deltanet stub failed");
        goto cleanup;
    }
    if (!ds4_gpu_state_norm_silu_reorder_tensor(out_in_gg_gpu, core_gpu, z_gpu,
                                                num_v_heads, head_v_dim, ssm_norm_w)) {
        snprintf(err, err_cap, "gpu norm/reorder stub failed");
        goto cleanup;
    }
    if (ds4_gpu_tensor_read(state_gpu, 0, gpu_state, state_bytes) == 0 ||
        ds4_gpu_tensor_read(core_gpu, 0, gpu_core, v_bytes) == 0 ||
        ds4_gpu_tensor_read(out_in_gg_gpu, 0, gpu_out_in_gg, v_bytes) == 0) {
        snprintf(err, err_cap, "gpu read failed validate_deltanet_norm_kernel");
        goto cleanup;
    }

    compare_arrays_brief("deltanet/state", cpu_state, gpu_state, num_v_heads * head_k_dim * head_v_dim);
    compare_arrays_brief("deltanet/core", cpu_core, gpu_core, value_dim);
    compare_arrays_brief("deltanet/out_in_gg", cpu_out_in_gg, gpu_out_in_gg, value_dim);
    ok = 1;

cleanup:
    free(ssm_norm_w);
    free(state_init);
    free(k_hf);
    free(q_hf);
    free(v_hf);
    free(g_in);
    free(beta_in);
    free(z_hf);
    free(cpu_state);
    free(cpu_core);
    free(cpu_out_in_hf);
    free(cpu_out_in_gg);
    free(gpu_state);
    free(gpu_core);
    free(gpu_out_in_gg);
    free(k_scaled);
    free(q_scaled);
    free(delta_tmp);
    ds4_gpu_tensor_free(state_gpu);
    ds4_gpu_tensor_free(core_gpu);
    ds4_gpu_tensor_free(k_gpu);
    ds4_gpu_tensor_free(q_gpu);
    ds4_gpu_tensor_free(v_gpu);
    ds4_gpu_tensor_free(g_gpu);
    ds4_gpu_tensor_free(beta_gpu);
    ds4_gpu_tensor_free(z_gpu);
    ds4_gpu_tensor_free(out_in_gg_gpu);
    return ok;
}

static int validate_tail_path(const qwen36_gguf_file *gf,
                              const mapped_file *mf,
                              const qwen36_35a3b_q8_layer *layer,
                              uint32_t hidden,
                              uint32_t num_v_heads,
                              uint32_t head_v_dim,
                              uint32_t topk,
                              uint32_t inter,
                              char *err,
                              size_t err_cap) {
    const uint32_t value_dim = num_v_heads * head_v_dim;
    const uint64_t hidden_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t value_bytes = (uint64_t)value_dim * sizeof(float);
    const uint64_t logits_bytes = (uint64_t)ROUTER_COUNT * sizeof(float);
    float *out_in_gg = NULL, *layer_input = NULL;
    float *post_attn_norm_w = NULL, *gate_inp_shexp_w = NULL;
    float *cpu_out_proj = NULL, *cpu_resid = NULL, *cpu_post_ln = NULL, *cpu_router = NULL;
    float *cpu_shared_gate = NULL, *cpu_shared_up = NULL, *cpu_shared_mid = NULL, *cpu_shared_out = NULL;
    float *cpu_gate = NULL, *cpu_up = NULL, *cpu_act = NULL, *cpu_down = NULL, *cpu_final = NULL;
    uint32_t *cpu_idx = NULL;
    float *cpu_scores = NULL;
    float *gpu_out_proj = NULL, *gpu_resid = NULL, *gpu_post_ln = NULL, *gpu_router = NULL, *gpu_shared_out = NULL, *gpu_final = NULL;
    uint32_t *gpu_idx = NULL;
    float *gpu_scores = NULL;
    ds4_gpu_tensor *out_in_gpu = NULL, *out_proj_gpu = NULL, *input_gpu = NULL, *resid_gpu = NULL, *post_ln_gpu = NULL;
    ds4_gpu_tensor *router_gpu = NULL, *shared_gate_gpu = NULL, *shared_up_gpu = NULL, *shared_mid_gpu = NULL, *shared_out_gpu = NULL;
    ds4_gpu_tensor *expert_gate_gpu = NULL, *expert_up_gpu = NULL, *expert_mid_gpu = NULL, *expert_down_gpu = NULL;
    int ok = 0;

    out_in_gg = (float *)malloc(value_bytes);
    layer_input = (float *)malloc(hidden_bytes);
    cpu_out_proj = (float *)malloc(hidden_bytes);
    cpu_resid = (float *)malloc(hidden_bytes);
    cpu_post_ln = (float *)malloc(hidden_bytes);
    cpu_router = (float *)malloc(logits_bytes);
    cpu_shared_gate = (float *)malloc((size_t)inter * sizeof(float));
    cpu_shared_up = (float *)malloc((size_t)inter * sizeof(float));
    cpu_shared_mid = (float *)malloc((size_t)inter * sizeof(float));
    cpu_shared_out = (float *)malloc(hidden_bytes);
    cpu_gate = (float *)malloc((size_t)inter * sizeof(float));
    cpu_up = (float *)malloc((size_t)inter * sizeof(float));
    cpu_act = (float *)malloc((size_t)inter * sizeof(float));
    cpu_down = (float *)malloc(hidden_bytes);
    cpu_final = (float *)calloc(hidden, sizeof(float));
    cpu_idx = (uint32_t *)malloc((size_t)topk * sizeof(uint32_t));
    cpu_scores = (float *)malloc((size_t)topk * sizeof(float));
    gpu_out_proj = (float *)malloc(hidden_bytes);
    gpu_resid = (float *)malloc(hidden_bytes);
    gpu_post_ln = (float *)malloc(hidden_bytes);
    gpu_router = (float *)malloc(logits_bytes);
    gpu_shared_out = (float *)malloc(hidden_bytes);
    gpu_final = (float *)calloc(hidden, sizeof(float));
    gpu_idx = (uint32_t *)malloc((size_t)topk * sizeof(uint32_t));
    gpu_scores = (float *)malloc((size_t)topk * sizeof(float));
    if (!out_in_gg || !layer_input || !cpu_out_proj || !cpu_resid || !cpu_post_ln || !cpu_router ||
        !cpu_shared_gate || !cpu_shared_up || !cpu_shared_mid || !cpu_shared_out ||
        !cpu_gate || !cpu_up || !cpu_act || !cpu_down || !cpu_final || !cpu_idx || !cpu_scores ||
        !gpu_out_proj || !gpu_resid || !gpu_post_ln || !gpu_router || !gpu_shared_out || !gpu_final || !gpu_idx || !gpu_scores) {
        snprintf(err, err_cap, "oom validate_tail_path");
        goto cleanup;
    }
    if (!read_f32_tensor_all(gf, layer->post_attn_norm, &post_attn_norm_w, hidden, err, err_cap) ||
        !read_f32_tensor_all(gf, layer->ffn_gate_inp_shexp, &gate_inp_shexp_w, hidden, err, err_cap)) {
        goto cleanup;
    }
    printf("tail/router tensor type=%u dims=[%llu,%llu,%llu]\n",
           (unsigned)layer->ffn_gate_inp->type,
           (unsigned long long)layer->ffn_gate_inp->dims[0],
           (unsigned long long)layer->ffn_gate_inp->dims[1],
           (unsigned long long)layer->ffn_gate_inp->dims[2]);
    for (uint32_t i = 0; i < value_dim; ++i) out_in_gg[i] = 0.19f * sinf(0.029f * (float)(i + 1)) + 0.07f * cosf(0.017f * (float)(i + 11));
    for (uint32_t i = 0; i < hidden; ++i) layer_input[i] = 0.16f * cosf(0.009f * (float)(i + 3)) - 0.05f * sinf(0.015f * (float)(i + 7));

    if (!matvec_q8_tensor_all_rows(gf, layer->ssm_out, 0u, hidden, value_dim, out_in_gg, cpu_out_proj, err, err_cap)) goto cleanup;
    {
        double var = 0.0;
        float scale_in = 0.0f;
        for (uint32_t d = 0; d < hidden; ++d) {
            cpu_resid[d] = layer_input[d] + cpu_out_proj[d];
            var += (double)cpu_resid[d] * cpu_resid[d];
        }
        var /= (double)hidden;
        for (uint32_t d = 0; d < hidden; ++d) {
            cpu_post_ln[d] = (float)(cpu_resid[d] / sqrt(var + 1e-6)) * post_attn_norm_w[d];
            scale_in += cpu_post_ln[d] * gate_inp_shexp_w[d];
        }
        if (!matvec_f32_tensor_all_rows(gf, layer->ffn_gate_inp, ROUTER_COUNT, hidden, cpu_post_ln, cpu_router, err, err_cap) ||
            !matvec_q8_tensor_all_rows(gf, layer->ffn_gate_shexp, 0u, inter, hidden, cpu_post_ln, cpu_shared_gate, err, err_cap) ||
            !matvec_q8_tensor_all_rows(gf, layer->ffn_up_shexp, 0u, inter, hidden, cpu_post_ln, cpu_shared_up, err, err_cap)) goto cleanup;
        for (uint32_t i = 0; i < inter; ++i) cpu_shared_mid[i] = siluf_local(cpu_shared_gate[i]) * cpu_shared_up[i];
        if (!matvec_q8_tensor_all_rows(gf, layer->ffn_down_shexp, 0u, hidden, inter, cpu_shared_mid, cpu_shared_out, err, err_cap)) goto cleanup;
        ref_topk_softmax256(cpu_router, topk, cpu_idx, cpu_scores);
        for (uint32_t i = 0; i < topk; ++i) {
            uint32_t expert_id = cpu_idx[i];
            if (!matvec_q8_tensor_all_rows(gf, layer->ffn_gate_exps, expert_id * inter, inter, hidden, cpu_post_ln, cpu_gate, err, err_cap) ||
                !matvec_q8_tensor_all_rows(gf, layer->ffn_up_exps, expert_id * inter, inter, hidden, cpu_post_ln, cpu_up, err, err_cap)) goto cleanup;
            for (uint32_t j = 0; j < inter; ++j) cpu_act[j] = siluf_local(cpu_gate[j]) * cpu_up[j];
            if (!matvec_q8_tensor_all_rows(gf, layer->ffn_down_exps, expert_id * hidden, hidden, inter, cpu_act, cpu_down, err, err_cap)) goto cleanup;
            for (uint32_t d = 0; d < hidden; ++d) cpu_final[d] += cpu_down[d] * cpu_scores[i];
        }
        {
            float s = sigmoidf_local(scale_in);
            for (uint32_t d = 0; d < hidden; ++d) cpu_final[d] += cpu_resid[d] + cpu_shared_out[d] * s;
        }
    }

    out_in_gpu = ds4_gpu_tensor_alloc(value_bytes);
    out_proj_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    input_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    resid_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    post_ln_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    router_gpu = ds4_gpu_tensor_alloc(logits_bytes);
    shared_gate_gpu = ds4_gpu_tensor_alloc((uint64_t)inter * sizeof(float));
    shared_up_gpu = ds4_gpu_tensor_alloc((uint64_t)inter * sizeof(float));
    shared_mid_gpu = ds4_gpu_tensor_alloc((uint64_t)inter * sizeof(float));
    shared_out_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    expert_gate_gpu = ds4_gpu_tensor_alloc((uint64_t)inter * sizeof(float));
    expert_up_gpu = ds4_gpu_tensor_alloc((uint64_t)inter * sizeof(float));
    expert_mid_gpu = ds4_gpu_tensor_alloc((uint64_t)inter * sizeof(float));
    expert_down_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    if (!out_in_gpu || !out_proj_gpu || !input_gpu || !resid_gpu || !post_ln_gpu || !router_gpu || !shared_gate_gpu ||
        !shared_up_gpu || !shared_mid_gpu || !shared_out_gpu || !expert_gate_gpu || !expert_up_gpu || !expert_mid_gpu || !expert_down_gpu) {
        snprintf(err, err_cap, "gpu alloc failed validate_tail_path");
        goto cleanup;
    }
    if (ds4_gpu_tensor_write(out_in_gpu, 0, out_in_gg, value_bytes) == 0 ||
        ds4_gpu_tensor_write(input_gpu, 0, layer_input, hidden_bytes) == 0) {
        snprintf(err, err_cap, "gpu write failed validate_tail_path");
        goto cleanup;
    }
    if (ds4_gpu_begin_commands() == 0) { snprintf(err, err_cap, "gpu begin out_proj validate_tail_path"); goto cleanup; }
    if (ds4_gpu_matmul_q8_0_tensor(out_proj_gpu, mf->map, mf->size, layer->ssm_out->abs_offset, value_dim, hidden, out_in_gpu, 1u) == 0) {
        snprintf(err, err_cap, "gpu out_proj failed validate_tail_path");
        goto cleanup;
    }
    if (ds4_gpu_end_commands() == 0) { snprintf(err, err_cap, "gpu end out_proj validate_tail_path"); goto cleanup; }
    if (ds4_gpu_tensor_read(out_proj_gpu, 0, gpu_out_proj, hidden_bytes) == 0) { snprintf(err, err_cap, "gpu out_proj read"); goto cleanup; }
    compare_arrays_brief("tail/out_proj", cpu_out_proj, gpu_out_proj, hidden);

    if (!ds4_gpu_resid_post_ln_tensor(resid_gpu, post_ln_gpu, input_gpu, out_proj_gpu, hidden, post_attn_norm_w)) {
        snprintf(err, err_cap, "gpu resid_post_ln failed validate_tail_path");
        goto cleanup;
    }
    if (ds4_gpu_tensor_read(resid_gpu, 0, gpu_resid, hidden_bytes) == 0 ||
        ds4_gpu_tensor_read(post_ln_gpu, 0, gpu_post_ln, hidden_bytes) == 0) {
        snprintf(err, err_cap, "gpu resid/post read");
        goto cleanup;
    }
    compare_arrays_brief("tail/resid", cpu_resid, gpu_resid, hidden);
    compare_arrays_brief("tail/post_ln", cpu_post_ln, gpu_post_ln, hidden);

    if (ds4_gpu_begin_commands() == 0) { snprintf(err, err_cap, "gpu begin moe tail"); goto cleanup; }
    if (ds4_gpu_matmul_f32_tensor(router_gpu, mf->map, mf->size, layer->ffn_gate_inp->abs_offset, hidden, ROUTER_COUNT, post_ln_gpu, 1u) == 0 ||
        ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(shared_gate_gpu, shared_up_gpu, shared_mid_gpu,
                                                  mf->map, mf->size,
                                                  layer->ffn_gate_shexp->abs_offset,
                                                  layer->ffn_up_shexp->abs_offset,
                                                  hidden, inter, post_ln_gpu, 80.0f) == 0 ||
        ds4_gpu_matmul_q8_0_tensor(shared_out_gpu, mf->map, mf->size,
                                   layer->ffn_down_shexp->abs_offset,
                                   inter, hidden, shared_mid_gpu, 1u) == 0) {
        snprintf(err, err_cap, "gpu shared/router failed validate_tail_path");
        goto cleanup;
    }
    if (ds4_gpu_end_commands() == 0) { snprintf(err, err_cap, "gpu end shared/router"); goto cleanup; }
    if (ds4_gpu_tensor_read(router_gpu, 0, gpu_router, logits_bytes) == 0 ||
        ds4_gpu_tensor_read(shared_out_gpu, 0, gpu_shared_out, hidden_bytes) == 0) {
        snprintf(err, err_cap, "gpu router/shared read");
        goto cleanup;
    }
    compare_arrays_brief("tail/router", cpu_router, gpu_router, ROUTER_COUNT);
    compare_arrays_brief("tail/shared_out", cpu_shared_out, gpu_shared_out, hidden);
    print_f32_prefix("tail/router cpu[0:8]=", cpu_router, ROUTER_COUNT);
    print_f32_prefix("tail/router gpu[0:8]=", gpu_router, ROUTER_COUNT);

    ref_topk_softmax256(gpu_router, topk, gpu_idx, gpu_scores);
    print_topk_summary("tail/topk cpu=", cpu_idx, cpu_scores, topk);
    print_topk_summary("tail/topk gpu=", gpu_idx, gpu_scores, topk);
    for (uint32_t i = 0; i < topk; ++i) {
        uint32_t expert_id = gpu_idx[i];
        if (ds4_gpu_begin_commands() == 0) { snprintf(err, err_cap, "gpu begin expert tail"); goto cleanup; }
        if (ds4_gpu_matmul_q8_0_pair_tensor(expert_gate_gpu, expert_up_gpu,
                                            mf->map, mf->size,
                                            layer->ffn_gate_exps->abs_offset + (uint64_t)((uint64_t)inter * hidden / 32u) * 34u * expert_id,
                                            layer->ffn_up_exps->abs_offset + (uint64_t)((uint64_t)inter * hidden / 32u) * 34u * expert_id,
                                            hidden, inter, inter, post_ln_gpu, 1u) == 0 ||
            ds4_gpu_swiglu_tensor(expert_mid_gpu, expert_gate_gpu, expert_up_gpu, inter, 80.0f, 1.0f) == 0 ||
            ds4_gpu_matmul_q8_0_tensor(expert_down_gpu, mf->map, mf->size,
                                       layer->ffn_down_exps->abs_offset + (uint64_t)((uint64_t)hidden * inter / 32u) * 34u * expert_id,
                                       inter, hidden, expert_mid_gpu, 1u) == 0) {
            snprintf(err, err_cap, "gpu expert tail failed");
            goto cleanup;
        }
        if (ds4_gpu_end_commands() == 0) { snprintf(err, err_cap, "gpu end expert tail"); goto cleanup; }
        if (ds4_gpu_tensor_read(expert_down_gpu, 0, cpu_down, hidden_bytes) == 0) { snprintf(err, err_cap, "gpu expert down read"); goto cleanup; }
        for (uint32_t d = 0; d < hidden; ++d) gpu_final[d] += cpu_down[d] * gpu_scores[i];
    }
    {
        float scale_in = 0.0f;
        for (uint32_t d = 0; d < hidden; ++d) scale_in += gpu_post_ln[d] * gate_inp_shexp_w[d];
        float s = sigmoidf_local(scale_in);
        for (uint32_t d = 0; d < hidden; ++d) gpu_final[d] += gpu_resid[d] + gpu_shared_out[d] * s;
    }
    compare_arrays_brief("tail/final", cpu_final, gpu_final, hidden);
    ok = 1;

cleanup:
    free(out_in_gg); free(layer_input); free(post_attn_norm_w); free(gate_inp_shexp_w);
    free(cpu_out_proj); free(cpu_resid); free(cpu_post_ln); free(cpu_router);
    free(cpu_shared_gate); free(cpu_shared_up); free(cpu_shared_mid); free(cpu_shared_out);
    free(cpu_gate); free(cpu_up); free(cpu_act); free(cpu_down); free(cpu_final); free(cpu_idx); free(cpu_scores);
    free(gpu_out_proj); free(gpu_resid); free(gpu_post_ln); free(gpu_router); free(gpu_shared_out); free(gpu_final); free(gpu_idx); free(gpu_scores);
    ds4_gpu_tensor_free(out_in_gpu); ds4_gpu_tensor_free(out_proj_gpu); ds4_gpu_tensor_free(input_gpu); ds4_gpu_tensor_free(resid_gpu); ds4_gpu_tensor_free(post_ln_gpu);
    ds4_gpu_tensor_free(router_gpu); ds4_gpu_tensor_free(shared_gate_gpu); ds4_gpu_tensor_free(shared_up_gpu); ds4_gpu_tensor_free(shared_mid_gpu); ds4_gpu_tensor_free(shared_out_gpu);
    ds4_gpu_tensor_free(expert_gate_gpu); ds4_gpu_tensor_free(expert_up_gpu); ds4_gpu_tensor_free(expert_mid_gpu); ds4_gpu_tensor_free(expert_down_gpu);
    return ok;
}

static int run_reference_hybrid_layer_cpu(const qwen36_gguf_file *gf,
                                          const qwen36_35a3b_q8_layer *layer,
                                          reference_hybrid_state *st,
                                          uint32_t hidden,
                                          uint32_t num_v_heads,
                                          uint32_t num_k_heads,
                                          uint32_t head_k_dim,
                                          uint32_t head_v_dim,
                                          uint32_t key_dim,
                                          uint32_t value_dim,
                                          uint32_t topk,
                                          uint32_t inter,
                                          const float *layer_input,
                                          float *out_row,
                                          char *err,
                                          size_t err_cap) {
    const uint32_t qkv_dim = key_dim * 2u + value_dim;
    const uint32_t rep = num_v_heads / num_k_heads;
    float *attn_norm_w = NULL, *conv_w = NULL, *A_log = NULL, *dt_bias = NULL;
    float *ssm_norm_w = NULL, *post_attn_norm_w = NULL, *gate_inp_shexp_w = NULL;
    float *input_ln = NULL, *z_raw = NULL, *z = NULL, *a_raw = NULL, *a = NULL, *b_raw = NULL, *b = NULL;
    float *qkv_raw = NULL, *qkv = NULL, *conv = NULL, *q = NULL, *k = NULL, *v = NULL;
    float *beta = NULL, *g = NULL, *core = NULL, *out_in = NULL, *out_in_gg = NULL, *out_proj = NULL;
    float *resid = NULL, *post_ln = NULL, *mlp = NULL, *shared = NULL;
    float *gate = NULL, *up = NULL, *act = NULL, *down = NULL;
    float *shared_gate = NULL, *shared_up = NULL, *shared_act = NULL;
    float *router_logits = NULL, *k_scaled = NULL, *q_scaled = NULL, *delta_tmp = NULL;
    uint32_t *router_idx = NULL;
    float *router_scores = NULL;
    int ok = 0;

    if (!read_f32_tensor_all(gf, layer->attn_norm, &attn_norm_w, hidden, err, err_cap) ||
        !read_f32_tensor_all(gf, layer->ssm_conv1d, &conv_w, (size_t)qkv_dim * 4u, err, err_cap) ||
        !read_f32_tensor_all(gf, layer->ssm_a, &A_log, num_v_heads, err, err_cap) ||
        !read_f32_tensor_all(gf, layer->ssm_dt_bias, &dt_bias, num_v_heads, err, err_cap) ||
        !read_f32_tensor_all(gf, layer->ssm_norm, &ssm_norm_w, head_v_dim, err, err_cap) ||
        !read_f32_tensor_all(gf, layer->post_attn_norm, &post_attn_norm_w, hidden, err, err_cap) ||
        !read_f32_tensor_all(gf, layer->ffn_gate_inp_shexp, &gate_inp_shexp_w, hidden, err, err_cap)) {
        goto cleanup;
    }

    input_ln = (float *)malloc((size_t)hidden * sizeof(float));
    z_raw = (float *)malloc((size_t)value_dim * sizeof(float));
    z = (float *)malloc((size_t)value_dim * sizeof(float));
    a_raw = (float *)malloc((size_t)num_v_heads * sizeof(float));
    a = (float *)malloc((size_t)num_v_heads * sizeof(float));
    b_raw = (float *)malloc((size_t)num_v_heads * sizeof(float));
    b = (float *)malloc((size_t)num_v_heads * sizeof(float));
    qkv_raw = (float *)malloc((size_t)qkv_dim * sizeof(float));
    qkv = (float *)malloc((size_t)qkv_dim * sizeof(float));
    conv = (float *)malloc((size_t)qkv_dim * sizeof(float));
    q = (float *)malloc((size_t)num_v_heads * head_k_dim * sizeof(float));
    k = (float *)malloc((size_t)num_v_heads * head_k_dim * sizeof(float));
    v = (float *)malloc((size_t)num_v_heads * head_v_dim * sizeof(float));
    beta = (float *)malloc((size_t)num_v_heads * sizeof(float));
    g = (float *)malloc((size_t)num_v_heads * sizeof(float));
    core = (float *)calloc((size_t)num_v_heads * head_v_dim, sizeof(float));
    out_in = (float *)malloc((size_t)value_dim * sizeof(float));
    out_in_gg = (float *)malloc((size_t)value_dim * sizeof(float));
    out_proj = (float *)malloc((size_t)hidden * sizeof(float));
    resid = (float *)malloc((size_t)hidden * sizeof(float));
    post_ln = (float *)malloc((size_t)hidden * sizeof(float));
    mlp = (float *)calloc((size_t)hidden, sizeof(float));
    shared = (float *)malloc((size_t)hidden * sizeof(float));
    gate = (float *)malloc((size_t)inter * sizeof(float));
    up = (float *)malloc((size_t)inter * sizeof(float));
    act = (float *)malloc((size_t)inter * sizeof(float));
    down = (float *)malloc((size_t)hidden * sizeof(float));
    shared_gate = (float *)malloc((size_t)inter * sizeof(float));
    shared_up = (float *)malloc((size_t)inter * sizeof(float));
    shared_act = (float *)malloc((size_t)inter * sizeof(float));
    router_logits = (float *)malloc((size_t)ROUTER_COUNT * sizeof(float));
    router_idx = (uint32_t *)malloc((size_t)ROUTER_COUNT * sizeof(uint32_t));
    router_scores = (float *)malloc((size_t)ROUTER_COUNT * sizeof(float));
    k_scaled = (float *)malloc((size_t)head_k_dim * sizeof(float));
    q_scaled = (float *)malloc((size_t)head_k_dim * sizeof(float));
    delta_tmp = (float *)malloc((size_t)head_v_dim * sizeof(float));
    if (!input_ln || !z_raw || !z || !a_raw || !a || !b_raw || !b || !qkv_raw || !qkv || !conv ||
        !q || !k || !v || !beta || !g || !core || !out_in || !out_in_gg || !out_proj || !resid ||
        !post_ln || !mlp || !shared || !gate || !up || !act || !down || !shared_gate || !shared_up ||
        !shared_act || !router_logits || !router_idx || !router_scores || !k_scaled || !q_scaled || !delta_tmp) {
        snprintf(err, err_cap, "oom run_reference_hybrid_layer_cpu");
        goto cleanup;
    }

    ref_rms_norm_weight(input_ln, layer_input, attn_norm_w, hidden);
    if (!matvec_q8_tensor_all_rows(gf, layer->attn_gate, 0u, value_dim, hidden, input_ln, z_raw, err, err_cap) ||
        !matvec_q8_tensor_all_rows(gf, layer->ssm_alpha, 0u, num_v_heads, hidden, input_ln, a_raw, err, err_cap) ||
        !matvec_q8_tensor_all_rows(gf, layer->ssm_beta, 0u, num_v_heads, hidden, input_ln, b_raw, err, err_cap) ||
        !matvec_q8_tensor_all_rows(gf, layer->attn_qkv, 0u, qkv_dim, hidden, input_ln, qkv_raw, err, err_cap)) {
        goto cleanup;
    }

    ref_reorder_head_rows_f32(z, z_raw, num_v_heads, head_v_dim);
    ref_reorder_head_scalars_f32(a, a_raw, num_v_heads);
    ref_reorder_head_scalars_f32(b, b_raw, num_v_heads);
    ref_reorder_qkv_v_gghf(qkv, qkv_raw, key_dim, num_v_heads, head_v_dim);

    for (uint32_t d = 0; d < qkv_dim; ++d) {
        double sum = 0.0;
        if (st->conv_ring_count >= 3) sum += (double)conv_w[(size_t)d * 4u + 0u] * st->conv_ring[2][d];
        if (st->conv_ring_count >= 2) sum += (double)conv_w[(size_t)d * 4u + 1u] * st->conv_ring[1][d];
        if (st->conv_ring_count >= 1) sum += (double)conv_w[(size_t)d * 4u + 2u] * st->conv_ring[0][d];
        sum += (double)conv_w[(size_t)d * 4u + 3u] * qkv[d];
        conv[d] = siluf_local((float)sum);
    }
    {
        const float *cq = conv;
        const float *ck = cq + key_dim;
        const float *cv = ck + key_dim;
        for (uint32_t h = 0; h < num_k_heads; ++h) {
            for (uint32_t d = 0; d < head_k_dim; ++d) {
                float qv = cq[h * head_k_dim + d];
                float kv = ck[h * head_k_dim + d];
                for (uint32_t vh = 0; vh < rep; ++vh) {
                    uint32_t dst_h = h * rep + vh;
                    q[(size_t)dst_h * head_k_dim + d] = qv;
                    k[(size_t)dst_h * head_k_dim + d] = kv;
                }
            }
        }
        for (uint32_t h = 0; h < num_v_heads; ++h) {
            memcpy(v + (size_t)h * head_v_dim, cv + (size_t)h * head_v_dim, (size_t)head_v_dim * sizeof(float));
            beta[h] = sigmoidf_local(b[h]);
            g[h] = A_log[h] * softplusf_local(a[h] + dt_bias[h]);
        }
    }
    for (uint32_t h = 0; h < num_v_heads; ++h) {
        stub_deltanet_head_step(st->deltanet_state + (size_t)h * head_k_dim * head_v_dim,
                                head_k_dim, head_v_dim,
                                k + (size_t)h * head_k_dim,
                                q + (size_t)h * head_k_dim,
                                v + (size_t)h * head_v_dim,
                                expf(g[h]),
                                beta[h],
                                k_scaled,
                                q_scaled,
                                delta_tmp,
                                core + (size_t)h * head_v_dim);
    }
    for (uint32_t h = 0; h < num_v_heads; ++h) {
        size_t base = (size_t)h * head_v_dim;
        double var = 0.0;
        for (uint32_t vd = 0; vd < head_v_dim; ++vd) {
            double cv = core[base + vd];
            var += cv * cv;
        }
        var /= (double)head_v_dim;
        for (uint32_t vd = 0; vd < head_v_dim; ++vd) {
            float cv = core[base + vd];
            out_in[base + vd] = (float)(cv / sqrt(var + 1e-6)) * ssm_norm_w[vd] * siluf_local(z[base + vd]);
        }
    }
    ref_reorder_out_in_hftogg(out_in_gg, out_in, num_v_heads, head_v_dim);
    if (!matvec_q8_tensor_all_rows(gf, layer->ssm_out, 0u, hidden, value_dim, out_in_gg, out_proj, err, err_cap)) {
        goto cleanup;
    }

    {
        double var = 0.0;
        float scale_in = 0.0f;
        for (uint32_t d = 0; d < hidden; ++d) {
            resid[d] = layer_input[d] + out_proj[d];
            var += (double)resid[d] * resid[d];
        }
        var /= (double)hidden;
        for (uint32_t d = 0; d < hidden; ++d) {
            post_ln[d] = (float)(resid[d] / sqrt(var + 1e-6)) * post_attn_norm_w[d];
            scale_in += post_ln[d] * gate_inp_shexp_w[d];
        }
        if (!matvec_f32_tensor_all_rows(gf, layer->ffn_gate_inp, ROUTER_COUNT, hidden, post_ln, router_logits, err, err_cap) ||
            !matvec_q8_tensor_all_rows(gf, layer->ffn_gate_shexp, 0u, inter, hidden, post_ln, shared_gate, err, err_cap) ||
            !matvec_q8_tensor_all_rows(gf, layer->ffn_up_shexp, 0u, inter, hidden, post_ln, shared_up, err, err_cap)) {
            goto cleanup;
        }
        for (uint32_t vd = 0; vd < inter; ++vd) shared_act[vd] = siluf_local(shared_gate[vd]) * shared_up[vd];
        if (!matvec_q8_tensor_all_rows(gf, layer->ffn_down_shexp, 0u, hidden, inter, shared_act, shared, err, err_cap)) {
            goto cleanup;
        }
        ref_topk_softmax256(router_logits, topk, router_idx, router_scores);
        for (uint32_t i = 0; i < topk; ++i) {
            uint32_t expert_id = router_idx[i];
            if (!matvec_q8_tensor_all_rows(gf, layer->ffn_gate_exps, expert_id * inter, inter, hidden, post_ln, gate, err, err_cap) ||
                !matvec_q8_tensor_all_rows(gf, layer->ffn_up_exps, expert_id * inter, inter, hidden, post_ln, up, err, err_cap)) {
                goto cleanup;
            }
            for (uint32_t vd = 0; vd < inter; ++vd) act[vd] = siluf_local(gate[vd]) * up[vd];
            if (!matvec_q8_tensor_all_rows(gf, layer->ffn_down_exps, expert_id * hidden, hidden, inter, act, down, err, err_cap)) {
                goto cleanup;
            }
            for (uint32_t d = 0; d < hidden; ++d) mlp[d] += down[d] * router_scores[i];
        }
        {
            float s = sigmoidf_local(scale_in);
            for (uint32_t d = 0; d < hidden; ++d) out_row[d] = resid[d] + mlp[d] + shared[d] * s;
        }
    }

    if (st->conv_ring_count >= 2) memcpy(st->conv_ring[2], st->conv_ring[1], (size_t)qkv_dim * sizeof(float));
    if (st->conv_ring_count >= 1) memcpy(st->conv_ring[1], st->conv_ring[0], (size_t)qkv_dim * sizeof(float));
    memcpy(st->conv_ring[0], qkv, (size_t)qkv_dim * sizeof(float));
    if (st->conv_ring_count < 3) st->conv_ring_count++;
    ok = 1;

cleanup:
    free(attn_norm_w);
    free(conv_w);
    free(A_log);
    free(dt_bias);
    free(ssm_norm_w);
    free(post_attn_norm_w);
    free(gate_inp_shexp_w);
    free(input_ln);
    free(z_raw);
    free(z);
    free(a_raw);
    free(a);
    free(b_raw);
    free(b);
    free(qkv_raw);
    free(qkv);
    free(conv);
    free(q);
    free(k);
    free(v);
    free(beta);
    free(g);
    free(core);
    free(out_in);
    free(out_in_gg);
    free(out_proj);
    free(resid);
    free(post_ln);
    free(mlp);
    free(shared);
    free(gate);
    free(up);
    free(act);
    free(down);
    free(shared_gate);
    free(shared_up);
    free(shared_act);
    free(router_logits);
    free(router_idx);
    free(router_scores);
    free(k_scaled);
    free(q_scaled);
    free(delta_tmp);
    return ok;
}

/*
 * run_native_hybrid_layer_gpu
 *
 * Implements a single decode-step for a Qwen3.6-35B-A3B hybrid (SSM) layer
 * using GPU projections wherever possible.  ALL GPU kernel launches are
 * funneled through ONE ds4_gpu_begin_commands … ds4_gpu_end_commands pair.
 *
 * CPU-only logic (reorder, conv-3, deltanet-step, rms-norm, silu-gate,
 * topk-softmax) is bridged via stub functions in qwen36_native_hybrid_kernels.cuh
 * that read/write GPU tensors inside the batch.  Those stubs will be replaced
 * by native GPU kernels later.
 *
 * Parameters:
 *   mf          – mmap'd GGUF model file (map + size)
 *   layer       – tensor bindings for this layer (abs_offsets)
 *   st          – persistent state (deltanet SSM state + conv ring)
 *   hidden      – 2048
 *   num_v_heads – 4
 *   num_k_heads – 1
 *   head_k_dim  – 128
 *   head_v_dim  – 64
 *   key_dim     – num_k_heads * head_k_dim = 128
 *   value_dim   – num_v_heads * head_v_dim = 256
 *   topk        – number of routed experts (e.g. 8)
 *   inter       – FFN intermediate dim (e.g. 512)
 *   layer_input – [hidden] input row (CPU)
 *   out_row     – [hidden] output row (CPU, written by this function)
 *   err         – error buffer
 *   err_cap     – error buffer capacity
 *
 * Returns: 1 on success, 0 on failure (err filled).
 */
static int run_native_hybrid_layer_gpu(const mapped_file *mf,
                                       const qwen36_35a3b_q8_layer *layer,
                                       native_hybrid_state *st,
                                       uint32_t hidden,
                                       uint32_t num_v_heads,
                                       uint32_t num_k_heads,
                                       uint32_t head_k_dim,
                                       uint32_t head_v_dim,
                                       uint32_t key_dim,
                                       uint32_t value_dim,
                                       uint32_t topk,
                                       uint32_t inter,
                                       const float *layer_input,
                                       float *out_row,
                                       char *err,
                                       size_t err_cap) {
    const uint32_t qkv_dim = key_dim * 2u + value_dim;
    const uint64_t hidden_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t value_dim_bytes = (uint64_t)value_dim * sizeof(float);
    const uint64_t key_dim_bytes = (uint64_t)key_dim * sizeof(float);
    const uint64_t qkv_dim_bytes = (uint64_t)qkv_dim * sizeof(float);
    const uint64_t ab_bytes = (uint64_t)num_v_heads * sizeof(float);
    const uint64_t inter_bytes = (uint64_t)inter * sizeof(float);
    const uint64_t router_logits_bytes = (uint64_t)ROUTER_COUNT * sizeof(float);
    const uint64_t gate_expert_bytes = (uint64_t)((uint64_t)inter * hidden / 32u) * 34u;
    const uint64_t down_expert_bytes = (uint64_t)((uint64_t)hidden * inter / 32u) * 34u;
    const uint64_t qkv_row_bytes = (uint64_t)(hidden / 32u) * 34u;
    const uint64_t q_off = layer->attn_qkv->abs_offset;
    const uint64_t k_off = q_off + (uint64_t)key_dim * qkv_row_bytes;
    const uint64_t v_off = k_off + (uint64_t)key_dim * qkv_row_bytes;

    ds4_gpu_tensor *input_gpu = NULL, *input_ln_gpu = NULL;
    ds4_gpu_tensor *z_gpu = NULL, *a_gpu = NULL, *b_gpu = NULL;
    ds4_gpu_tensor *qkv_gpu = NULL, *q_gpu = NULL, *k_gpu = NULL, *v_gpu = NULL;
    ds4_gpu_tensor *conv_gpu = NULL, *core_gpu = NULL;
    ds4_gpu_tensor *beta_gpu = NULL, *g_gpu = NULL;
    ds4_gpu_tensor *out_in_gpu = NULL, *out_proj_gpu = NULL;
    ds4_gpu_tensor *resid_gpu = NULL, *post_ln_gpu = NULL;
    ds4_gpu_tensor *router_logits_gpu = NULL;
    ds4_gpu_tensor *shared_gate_gpu = NULL, *shared_up_gpu = NULL, *shared_mid_gpu = NULL, *shared_out_gpu = NULL;
    ds4_gpu_tensor *state_gpu = NULL;
    ds4_gpu_tensor *expert_gate_gpu[8] = {NULL};
    ds4_gpu_tensor *expert_up_gpu[8] = {NULL};
    ds4_gpu_tensor *expert_mid_gpu[8] = {NULL};
    ds4_gpu_tensor *expert_down_gpu[8] = {NULL};
    ds4_gpu_tensor *conv_w_gpu = NULL, *A_log_gpu = NULL, *dt_bias_gpu = NULL;

    float *resid = NULL, *post_ln = NULL, *shared = NULL;
    float *conv_w = NULL, *A_log = NULL, *dt_bias = NULL;
    float *ssm_norm_w = NULL, *post_attn_norm_w = NULL, *gate_inp_shexp_w = NULL;
    uint32_t router_idx[256];
    float router_scores[256];
    uint32_t i, d;
    int ok = 0;
    int ex_flags[8] = {0};
    const uint8_t *base_map = (const uint8_t *)mf->map;

    if (topk > 8) { snprintf(err, err_cap, "topk %u > max 8", topk); goto cleanup; }

    conv_w = (float *)malloc((size_t)qkv_dim * 4u * sizeof(float));
    A_log = (float *)malloc((size_t)num_v_heads * sizeof(float));
    dt_bias = (float *)malloc((size_t)num_v_heads * sizeof(float));
    ssm_norm_w = (float *)malloc((size_t)head_v_dim * sizeof(float));
    post_attn_norm_w = (float *)malloc(hidden_bytes);
    gate_inp_shexp_w = (float *)malloc(hidden_bytes);
    if (!conv_w || !A_log || !dt_bias || !ssm_norm_w || !post_attn_norm_w || !gate_inp_shexp_w) {
        snprintf(err, err_cap, "weight malloc failed"); goto cleanup;
    }
    memcpy(conv_w, base_map + layer->ssm_conv1d->abs_offset, (size_t)qkv_dim * 4u * sizeof(float));
    memcpy(A_log, base_map + layer->ssm_a->abs_offset, (size_t)num_v_heads * sizeof(float));
    memcpy(dt_bias, base_map + layer->ssm_dt_bias->abs_offset, (size_t)num_v_heads * sizeof(float));
    memcpy(ssm_norm_w, base_map + layer->ssm_norm->abs_offset, (size_t)head_v_dim * sizeof(float));
    memcpy(post_attn_norm_w, base_map + layer->post_attn_norm->abs_offset, hidden_bytes);
    memcpy(gate_inp_shexp_w, base_map + layer->ffn_gate_inp_shexp->abs_offset, hidden_bytes);

    /* upload conv_w, A_log, dt_bias to GPU tensors before begin_commands */
    conv_w_gpu = ds4_gpu_tensor_alloc((uint64_t)qkv_dim * 4u * sizeof(float));
    A_log_gpu = ds4_gpu_tensor_alloc((uint64_t)num_v_heads * sizeof(float));
    dt_bias_gpu = ds4_gpu_tensor_alloc((uint64_t)num_v_heads * sizeof(float));
    if (!conv_w_gpu || !A_log_gpu || !dt_bias_gpu) {
        snprintf(err, err_cap, "gpu weight tensor alloc failed"); goto cleanup;
    }
    if (ds4_gpu_tensor_write(conv_w_gpu, 0, conv_w, (uint64_t)qkv_dim * 4u * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(A_log_gpu, 0, A_log, (uint64_t)num_v_heads * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(dt_bias_gpu, 0, dt_bias, (uint64_t)num_v_heads * sizeof(float)) == 0) {
        snprintf(err, err_cap, "gpu weight write failed"); goto cleanup;
    }

    resid = (float *)malloc(hidden_bytes);
    post_ln = (float *)malloc(hidden_bytes);
    shared = (float *)malloc(hidden_bytes);
    if (!resid || !post_ln || !shared) { snprintf(err, err_cap, "cpu malloc failed"); goto cleanup; }

    input_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    input_ln_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    z_gpu = ds4_gpu_tensor_alloc(value_dim_bytes);
    a_gpu = ds4_gpu_tensor_alloc(ab_bytes);
    b_gpu = ds4_gpu_tensor_alloc(ab_bytes);
    qkv_gpu = ds4_gpu_tensor_alloc(qkv_dim_bytes);
    q_gpu = ds4_gpu_tensor_alloc((uint64_t)num_v_heads * head_k_dim * sizeof(float));
    k_gpu = ds4_gpu_tensor_alloc((uint64_t)num_v_heads * head_k_dim * sizeof(float));
    v_gpu = ds4_gpu_tensor_alloc(value_dim_bytes);
    conv_gpu = ds4_gpu_tensor_alloc(qkv_dim_bytes);
    core_gpu = ds4_gpu_tensor_alloc((uint64_t)num_v_heads * head_v_dim * sizeof(float));
    beta_gpu = ds4_gpu_tensor_alloc(ab_bytes);
    g_gpu = ds4_gpu_tensor_alloc(ab_bytes);
    out_in_gpu = ds4_gpu_tensor_alloc(value_dim_bytes);
    out_proj_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    resid_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    post_ln_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    router_logits_gpu = ds4_gpu_tensor_alloc(router_logits_bytes);
    shared_gate_gpu = ds4_gpu_tensor_alloc(inter_bytes);
    shared_up_gpu = ds4_gpu_tensor_alloc(inter_bytes);
    shared_mid_gpu = ds4_gpu_tensor_alloc(inter_bytes);
    shared_out_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    state_gpu = ds4_gpu_tensor_alloc((uint64_t)num_v_heads * head_k_dim * head_v_dim * sizeof(float));

    if (!input_gpu || !input_ln_gpu || !z_gpu || !a_gpu || !b_gpu ||
        !qkv_gpu || !q_gpu || !k_gpu || !v_gpu || !conv_gpu || !core_gpu ||
        !beta_gpu || !g_gpu || !out_in_gpu || !out_proj_gpu ||
        !resid_gpu || !post_ln_gpu || !router_logits_gpu ||
        !shared_gate_gpu || !shared_up_gpu || !shared_mid_gpu || !shared_out_gpu ||
        !state_gpu) {
        snprintf(err, err_cap, "gpu tensor alloc failed");
        goto cleanup;
    }

    if (ds4_gpu_tensor_write(state_gpu, 0, st->deltanet_state,
                             (uint64_t)num_v_heads * head_k_dim * head_v_dim * sizeof(float)) == 0) {
        snprintf(err, err_cap, "gpu state write failed");
        goto cleanup;
    }

    /* ---- SINGLE COMMAND BATCH ---- */
    if (ds4_gpu_begin_commands() == 0) {
        snprintf(err, err_cap, "gpu begin failed");
        goto cleanup;
    }

    if (ds4_gpu_tensor_write(input_gpu, 0, layer_input, hidden_bytes) == 0) {
        snprintf(err, err_cap, "gpu input write failed");
        goto cleanup;
    }

    /* 1. RMS norm */
    if (ds4_gpu_rms_norm_weight_rows_tensor(input_ln_gpu, input_gpu,
                                            mf->map, mf->size,
                                            layer->attn_norm->abs_offset,
                                            hidden, 1u, 1e-6f) == 0) {
        snprintf(err, err_cap, "gpu rms norm failed");
        goto cleanup;
    }

    /* 2. z, a, b projections */
    if (ds4_gpu_matmul_q8_0_tensor(z_gpu, mf->map, mf->size,
                                    layer->attn_gate->abs_offset,
                                    hidden, value_dim, input_ln_gpu, 1u) == 0) {
        snprintf(err, err_cap, "gpu z matmul failed");
        goto cleanup;
    }
    if (ds4_gpu_matmul_q8_0_tensor(a_gpu, mf->map, mf->size,
                                    layer->ssm_alpha->abs_offset,
                                    hidden, num_v_heads, input_ln_gpu, 1u) == 0) {
        snprintf(err, err_cap, "gpu a matmul failed");
        goto cleanup;
    }
    if (ds4_gpu_matmul_q8_0_tensor(b_gpu, mf->map, mf->size,
                                    layer->ssm_beta->abs_offset,
                                    hidden, num_v_heads, input_ln_gpu, 1u) == 0) {
        snprintf(err, err_cap, "gpu b matmul failed");
        goto cleanup;
    }

    /* 3. q, k, v from merged QKV weight */
    if (ds4_gpu_matmul_q8_0_tensor(qkv_gpu, mf->map, mf->size, q_off,
                                    hidden, qkv_dim, input_ln_gpu, 1u) == 0) {
        snprintf(err, err_cap, "gpu qkv matmul failed");
        goto cleanup;
    }

    /* 4. GPU: reorder → conv3 → head-split → beta/g   (reads z,a,b,qkv; writes conv,q,k,v,beta,g) */
    if (ds4_gpu_fused_reorder_conv_tensor(conv_gpu, q_gpu, k_gpu, v_gpu, beta_gpu, g_gpu,
                                          z_gpu, a_gpu, b_gpu, qkv_gpu,
                                          st->conv_ring_gpu, &st->conv_ring_count,
                                          num_v_heads, num_k_heads, head_k_dim, head_v_dim, key_dim,
                                          conv_w_gpu, A_log_gpu, dt_bias_gpu) == 0) {
        snprintf(err, err_cap, "gpu reorder_conv failed");
        goto cleanup;
    }

    /* 5. stub: deltanet SSM step   (reads k,q,v,beta,g,state; writes state,core) */
    if (ds4_gpu_deltanet_step_tensor(state_gpu, core_gpu,
                                     k_gpu, q_gpu, v_gpu, g_gpu, beta_gpu,
                                     num_v_heads, head_k_dim, head_v_dim) == 0) {
        snprintf(err, err_cap, "stub deltanet step failed");
        goto cleanup;
    }

    /* 6. stub: per-head rms-norm → silu-gate → HF→GG reorder */
    if (ds4_gpu_state_norm_silu_reorder_tensor(out_in_gpu, core_gpu, z_gpu,
                                               num_v_heads, head_v_dim,
                                               ssm_norm_w) == 0) {
        snprintf(err, err_cap, "stub norm_silu_reorder failed");
        goto cleanup;
    }

    /* 7. out_proj matmul */
    if (ds4_gpu_matmul_q8_0_tensor(out_proj_gpu, mf->map, mf->size,
                                    layer->ssm_out->abs_offset,
                                    value_dim, hidden, out_in_gpu, 1u) == 0) {
        snprintf(err, err_cap, "gpu out_proj failed");
        goto cleanup;
    }

    /* 8. stub: residual + post_ln norm */
    if (ds4_gpu_resid_post_ln_tensor(resid_gpu, post_ln_gpu,
                                     input_gpu, out_proj_gpu,
                                     hidden, post_attn_norm_w) == 0) {
        snprintf(err, err_cap, "stub resid_post_ln failed");
        goto cleanup;
    }

    /* 9. router matmul (f32 weight) */
    if (ds4_gpu_matmul_f32_tensor(router_logits_gpu, mf->map, mf->size,
                                  layer->ffn_gate_inp->abs_offset,
                                  hidden, ROUTER_COUNT, post_ln_gpu, 1u) == 0) {
        snprintf(err, err_cap, "gpu router failed");
        goto cleanup;
    }

    /* 10. shared gate/up + swiglu (fused) */
    if (ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(shared_gate_gpu, shared_up_gpu, shared_mid_gpu,
                                                   mf->map, mf->size,
                                                   layer->ffn_gate_shexp->abs_offset,
                                                   layer->ffn_up_shexp->abs_offset,
                                                   hidden, inter, post_ln_gpu,
                                                   80.0f) == 0) {
        snprintf(err, err_cap, "gpu shared gate/up/swiglu failed");
        goto cleanup;
    }

    /* 11. shared down */
    if (ds4_gpu_matmul_q8_0_tensor(shared_out_gpu, mf->map, mf->size,
                                    layer->ffn_down_shexp->abs_offset,
                                    inter, hidden, shared_mid_gpu, 1u) == 0) {
        snprintf(err, err_cap, "gpu shared down failed");
        goto cleanup;
    }

    /* 12. stub: topk-softmax on router logits   (reads router_logits → CPU → sel idx + scores) */
    if (ds4_gpu_topk_softmax_tensor(router_idx, router_scores,
                                    router_logits_gpu, ROUTER_COUNT, topk) == 0) {
        snprintf(err, err_cap, "stub topk_softmax failed");
        goto cleanup;
    }

    /* 13. expert dispatch: all selected experts in the same batch */
    for (i = 0; i < topk; ++i) {
        const uint32_t eid = router_idx[i];
        expert_gate_gpu[i] = ds4_gpu_tensor_alloc(inter_bytes);
        expert_up_gpu[i] = ds4_gpu_tensor_alloc(inter_bytes);
        expert_mid_gpu[i] = ds4_gpu_tensor_alloc(inter_bytes);
        expert_down_gpu[i] = ds4_gpu_tensor_alloc(hidden_bytes);
        if (!expert_gate_gpu[i] || !expert_up_gpu[i] || !expert_mid_gpu[i] || !expert_down_gpu[i]) {
            snprintf(err, err_cap, "gpu expert tensor alloc failed for eid=%u", eid);
            goto cleanup;
        }
        ex_flags[i] = 1;
        if (ds4_gpu_matmul_q8_0_pair_tensor(expert_gate_gpu[i], expert_up_gpu[i],
                                             mf->map, mf->size,
                                             layer->ffn_gate_exps->abs_offset + gate_expert_bytes * eid,
                                             layer->ffn_up_exps->abs_offset + gate_expert_bytes * eid,
                                             hidden, inter, inter, post_ln_gpu, 1u) == 0) {
            snprintf(err, err_cap, "gpu expert gate/up failed eid=%u", eid);
            goto cleanup;
        }
        if (ds4_gpu_swiglu_tensor(expert_mid_gpu[i], expert_gate_gpu[i], expert_up_gpu[i],
                                   inter, 80.0f, 1.0f) == 0) {
            snprintf(err, err_cap, "gpu expert swiglu failed eid=%u", eid);
            goto cleanup;
        }
        if (ds4_gpu_matmul_q8_0_tensor(expert_down_gpu[i], mf->map, mf->size,
                                        layer->ffn_down_exps->abs_offset + down_expert_bytes * eid,
                                        inter, hidden, expert_mid_gpu[i], 1u) == 0) {
            snprintf(err, err_cap, "gpu expert down failed eid=%u", eid);
            goto cleanup;
        }
    }

    /* ---- END OF SINGLE COMMAND BATCH ---- */
    if (ds4_gpu_end_commands() == 0) {
        snprintf(err, err_cap, "gpu end failed");
        goto cleanup;
    }

    /* Readbacks */
    if (ds4_gpu_tensor_read(resid_gpu, 0, resid, hidden_bytes) == 0 ||
        ds4_gpu_tensor_read(post_ln_gpu, 0, post_ln, hidden_bytes) == 0 ||
        ds4_gpu_tensor_read(shared_out_gpu, 0, shared, hidden_bytes) == 0) {
        snprintf(err, err_cap, "gpu readback failed");
        goto cleanup;
    }

    {
        const uint64_t state_bytes = (uint64_t)num_v_heads * head_k_dim * head_v_dim * sizeof(float);
        float *state_cpu = (float *)malloc(state_bytes);
        if (state_cpu && ds4_gpu_tensor_read(state_gpu, 0, state_cpu, state_bytes) != 0) {
            memcpy(st->deltanet_state, state_cpu, state_bytes);
        }
        free(state_cpu);
    }

    /* CPU: final combine */
    {
        float scale_in = 0.0f;
        for (d = 0; d < hidden; ++d) scale_in += post_ln[d] * gate_inp_shexp_w[d];

        for (d = 0; d < hidden; ++d) out_row[d] = resid[d];
        for (i = 0; i < topk; ++i) {
            float *expert_down = (float *)malloc(hidden_bytes);
            if (expert_down && ds4_gpu_tensor_read(expert_down_gpu[i], 0, expert_down, hidden_bytes) != 0) {
                for (d = 0; d < hidden; ++d) out_row[d] += expert_down[d] * router_scores[i];
            }
            free(expert_down);
        }
        {
            float s = 1.0f / (1.0f + expf(-scale_in));
            for (d = 0; d < hidden; ++d) out_row[d] += shared[d] * s;
        }
    }

    ok = 1;

cleanup:
    ds4_gpu_tensor_free(input_gpu);
    ds4_gpu_tensor_free(input_ln_gpu);
    ds4_gpu_tensor_free(z_gpu);
    ds4_gpu_tensor_free(a_gpu);
    ds4_gpu_tensor_free(b_gpu);
    ds4_gpu_tensor_free(qkv_gpu);
    ds4_gpu_tensor_free(q_gpu);
    ds4_gpu_tensor_free(k_gpu);
    ds4_gpu_tensor_free(v_gpu);
    ds4_gpu_tensor_free(conv_gpu);
    ds4_gpu_tensor_free(core_gpu);
    ds4_gpu_tensor_free(beta_gpu);
    ds4_gpu_tensor_free(g_gpu);
    ds4_gpu_tensor_free(out_in_gpu);
    ds4_gpu_tensor_free(out_proj_gpu);
    ds4_gpu_tensor_free(resid_gpu);
    ds4_gpu_tensor_free(post_ln_gpu);
    ds4_gpu_tensor_free(router_logits_gpu);
    ds4_gpu_tensor_free(shared_gate_gpu);
    ds4_gpu_tensor_free(shared_up_gpu);
    ds4_gpu_tensor_free(shared_mid_gpu);
    ds4_gpu_tensor_free(shared_out_gpu);
    ds4_gpu_tensor_free(state_gpu);
    ds4_gpu_tensor_free(conv_w_gpu);
    ds4_gpu_tensor_free(A_log_gpu);
    ds4_gpu_tensor_free(dt_bias_gpu);
    for (i = 0; i < 8; ++i) {
        if (ex_flags[i]) {
            ds4_gpu_tensor_free(expert_gate_gpu[i]);
            ds4_gpu_tensor_free(expert_up_gpu[i]);
            ds4_gpu_tensor_free(expert_mid_gpu[i]);
            ds4_gpu_tensor_free(expert_down_gpu[i]);
        }
    }
    free(resid);
    free(post_ln);
    free(shared);
    free(conv_w);
    free(A_log);
    free(dt_bias);
    free(ssm_norm_w);
    free(post_attn_norm_w);
    free(gate_inp_shexp_w);
    return ok;
}

/* ---- standalone test ---- */
int main(int argc, char **argv) {
    const char *gguf_path;
    int layer_idx = 0;
    int repeat = 1;
    qwen36_gguf_file gf;
    qwen36_35a3b_q8_model q8;
    mapped_file mf;
    native_hybrid_state st;
    reference_hybrid_state ref_st;
    char err[512];
    float *layer_input;
    float *out_row;
    float *ref_out_row;
    uint32_t hidden, topk, inter;
    uint32_t num_k_heads, num_v_heads, head_k_dim, head_v_dim, key_dim, value_dim;
    int i;

    memset(&gf, 0, sizeof(gf));
    memset(&q8, 0, sizeof(q8));
    memset(&mf, 0, sizeof(mf));
    mf.fd = -1;
    memset(&st, 0, sizeof(st));
    memset(&ref_st, 0, sizeof(ref_st));

    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [layer_idx=0] [repeat=1]\n", argv[0]);
        return 1;
    }
    gguf_path = argv[1];
    if (argc > 2) layer_idx = atoi(argv[2]);
    if (argc > 3) repeat = atoi(argv[3]);
    if (repeat < 1) repeat = 1;

    if (!qwen36_gguf_open(&gf, gguf_path, err, sizeof(err))) {
        fprintf(stderr, "gguf_open: %s\n", err);
        return 1;
    }
    if (!qwen36_35a3b_q8_bind(&gf, &q8, err, sizeof(err))) {
        fprintf(stderr, "bind: %s\n", err);
        qwen36_gguf_close(&gf);
        return 1;
    }
    if (layer_idx < 0 || (uint32_t)layer_idx >= QWEN36_35A3B_Q8_BLOCK_COUNT) {
        fprintf(stderr, "layer_idx %d out of range [0,%u)\n", layer_idx, QWEN36_35A3B_Q8_BLOCK_COUNT);
        qwen36_gguf_close(&gf);
        return 1;
    }
    if (q8.layers[layer_idx].kind != QWEN36_LAYER_KIND_HYBRID_SSM) {
        fprintf(stderr, "layer %d is not a hybrid SSM layer\n", layer_idx);
        qwen36_gguf_close(&gf);
        return 1;
    }

    if (!mapped_file_open(&mf, gguf_path)) {
        fprintf(stderr, "mmap: %s\n", strerror(errno));
        qwen36_gguf_close(&gf);
        return 1;
    }
    if (ds4_gpu_init() == 0) {
        fprintf(stderr, "ds4_gpu_init failed\n");
        mapped_file_close(&mf);
        qwen36_gguf_close(&gf);
        return 1;
    }
    (void)ds4_gpu_set_model_map(mf.map, mf.size);
    (void)ds4_gpu_set_model_fd_for_map(mf.fd, mf.map);

    hidden = 2048;
    topk = 8;
    inter = 512;
    num_k_heads = 1;
    num_v_heads = 4;
    head_k_dim = 128;
    head_v_dim = 64;
    key_dim = num_k_heads * head_k_dim;
    value_dim = num_v_heads * head_v_dim;

    layer_input = (float *)calloc(hidden, sizeof(float));
    out_row = (float *)calloc(hidden, sizeof(float));
    ref_out_row = (float *)calloc(hidden, sizeof(float));
    if (!layer_input || !out_row || !ref_out_row) {
        fprintf(stderr, "malloc failed\n");
        goto cleanup;
    }

    if (!native_hybrid_state_init(&st, num_v_heads, head_k_dim, head_v_dim, key_dim * 2u + value_dim)) {
        fprintf(stderr, "state init failed\n");
        goto cleanup;
    }
    if (!reference_hybrid_state_init(&ref_st, num_v_heads, head_k_dim, head_v_dim, key_dim * 2u + value_dim)) {
        fprintf(stderr, "reference state init failed\n");
        goto cleanup;
    }

    printf("qwen36_native_hybrid_test: layer=%d hidden=%u repeat=%d\n", layer_idx, hidden, repeat);

    if (!validate_fused_reorder_conv_kernel(&gf, &q8.layers[layer_idx],
                                            num_v_heads, num_k_heads,
                                            head_k_dim, head_v_dim,
                                            key_dim, err, sizeof(err))) {
        fprintf(stderr, "fused validator failed: %s\n", err);
        goto cleanup;
    }
    if (!validate_deltanet_norm_kernel(&gf, &q8.layers[layer_idx],
                                       num_v_heads, head_k_dim, head_v_dim,
                                       err, sizeof(err))) {
        fprintf(stderr, "deltanet validator failed: %s\n", err);
        goto cleanup;
    }
    if (!validate_tail_path(&gf, &mf, &q8.layers[layer_idx],
                            hidden, num_v_heads, head_v_dim, topk, inter,
                            err, sizeof(err))) {
        fprintf(stderr, "tail validator failed: %s\n", err);
        goto cleanup;
    }

    for (i = 0; i < repeat; ++i) {
        for (uint32_t d = 0; d < hidden; ++d) {
            float x = (float)(d + 1);
            layer_input[d] = 0.35f * sinf(0.0013f * x * (float)(i + 1)) +
                             0.20f * cosf(0.0007f * x * (float)(i + 2));
        }
        if (!run_native_hybrid_layer_gpu(&mf, &q8.layers[layer_idx], &st,
                                         hidden, num_v_heads, num_k_heads,
                                         head_k_dim, head_v_dim,
                                         key_dim, value_dim,
                                         topk, inter,
                                         layer_input, out_row,
                                         err, sizeof(err))) {
            fprintf(stderr, "Step %d: %s\n", i, err);
            goto cleanup;
        }
        if (!run_reference_hybrid_layer_cpu(&gf, &q8.layers[layer_idx], &ref_st,
                                            hidden, num_v_heads, num_k_heads,
                                            head_k_dim, head_v_dim,
                                            key_dim, value_dim,
                                            topk, inter,
                                            layer_input, ref_out_row,
                                            err, sizeof(err))) {
            fprintf(stderr, "Reference step %d: %s\n", i, err);
            goto cleanup;
        }
        {
            double sum_native = 0.0;
            double sum_ref = 0.0;
            double mae = 0.0;
            float max_abs = 0.0f;
            uint32_t max_idx = 0;
            for (uint32_t d = 0; d < hidden; ++d) {
                float diff = fabsf(out_row[d] - ref_out_row[d]);
                sum_native += out_row[d];
                sum_ref += ref_out_row[d];
                mae += diff;
                if (diff > max_abs) {
                    max_abs = diff;
                    max_idx = d;
                }
            }
            mae /= (double)hidden;
            printf("step=%d native_sum=%.8f ref_sum=%.8f max_abs=%.8f max_idx=%u mean_abs=%.8f %s\n",
                   i,
                   (float)sum_native,
                   (float)sum_ref,
                   max_abs,
                   max_idx,
                   (float)mae,
                   max_abs <= 1e-4f ? "PASS" : "FAIL");
        }
    }

    {
        double sum = 0.0;
        for (uint32_t d = 0; d < hidden; ++d) sum += (double)out_row[d];
        printf("output_sum: %.8f\n", (float)sum);
        printf("ok: true\n");
    }

cleanup:
    reference_hybrid_state_free(&ref_st);
    native_hybrid_state_free(&st);
    free(layer_input);
    free(out_row);
    free(ref_out_row);
    ds4_gpu_cleanup();
    mapped_file_close(&mf);
    qwen36_gguf_close(&gf);
    return 0;
}
