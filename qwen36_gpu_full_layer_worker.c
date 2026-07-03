#include "ds4_gpu.h"
#include "qwen36_35a3b_q4xl.h"
#include "qwen36_35a3b_q8.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define ROUTER_COUNT 256
#define INTER 512
#define ROTARY_DIM 64
#define NUM_HEADS 16
#define NUM_KV_HEADS 2
#define HEAD_DIM 256
#define HIDDEN 2048
#define ATTN_GATE_DIM (NUM_HEADS * HEAD_DIM)
#define QWEN_RMS_EPS 1e-6f
#define QWEN_SWIGLU_CLAMP 80.0f
#define QK_K 256
#define K_SCALE_SIZE 12

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qs[QK_K / 2];
} qwen36_block_q4_K;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
} qwen36_block_q5_K;

typedef struct {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    uint16_t d;
} qwen36_block_q6_K;

typedef struct qwen36_contract_layer {
    int kind;
    const qwen36_gguf_tensor *attn_norm;
    const qwen36_gguf_tensor *post_attn_norm;
    const qwen36_gguf_tensor *attn_gate;
    const qwen36_gguf_tensor *attn_qkv;
    const qwen36_gguf_tensor *attn_q;
    const qwen36_gguf_tensor *attn_q_norm;
    const qwen36_gguf_tensor *attn_k;
    const qwen36_gguf_tensor *attn_k_norm;
    const qwen36_gguf_tensor *attn_v;
    const qwen36_gguf_tensor *attn_output;
    const qwen36_gguf_tensor *ssm_a;
    const qwen36_gguf_tensor *ssm_alpha;
    const qwen36_gguf_tensor *ssm_beta;
    const qwen36_gguf_tensor *ssm_conv1d;
    const qwen36_gguf_tensor *ssm_dt_bias;
    const qwen36_gguf_tensor *ssm_norm;
    const qwen36_gguf_tensor *ssm_out;
    const qwen36_gguf_tensor *ffn_gate_inp;
    const qwen36_gguf_tensor *ffn_gate_inp_shexp;
    const qwen36_gguf_tensor *ffn_gate_exps;
    const qwen36_gguf_tensor *ffn_up_exps;
    const qwen36_gguf_tensor *ffn_down_exps;
    const qwen36_gguf_tensor *ffn_gate_shexp;
    const qwen36_gguf_tensor *ffn_up_shexp;
    const qwen36_gguf_tensor *ffn_down_shexp;
} qwen36_contract_layer;

#define QWEN36_CONTRACT_LAYER_KIND_FULL_ATTENTION 2

typedef struct qwen36_contract_model {
    const qwen36_gguf_file *gf;
    const qwen36_gguf_tensor *token_embd;
    const qwen36_gguf_tensor *output_norm;
    const qwen36_gguf_tensor *output;
    qwen36_contract_layer layers[QWEN36_35A3B_Q8_BLOCK_COUNT];
    uint32_t hybrid_layer_count;
    uint32_t full_attn_layer_count;
} qwen36_contract_model;

typedef struct mapped_file {
    int fd;
    void *map;
    uint64_t size;
} mapped_file;

typedef struct full_layer_small_cache {
    float *q_norm_w;
    float *k_norm_w;
    float *shared_gate_inp_w;
} full_layer_small_cache;

typedef struct worker_state {
    qwen36_gguf_file gf;
    qwen36_contract_model model;
    const qwen36_contract_layer *layer;
    uint32_t layer_idx;
    mapped_file mf;
    full_layer_small_cache cache;
    uint32_t seq_len;
    int prefilled;
    float *k_all;
    float *v_all;
    float *output_seq;
} worker_state;

typedef struct full_ffn_timing {
    double alloc_ms;
    double shared_gpu_ms;
    double shared_readback_ms;
    double route_ms;
    double expert_gpu_ms;
    double expert_readback_ms;
    double combine_ms;
    uint32_t n_union;
} full_ffn_timing;

static void q4_k_get_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static void map_q8_contract_layer(qwen36_contract_layer *dst, const qwen36_35a3b_q8_layer *src) {
    memcpy(dst, src, sizeof(*dst));
}

static void map_q4xl_contract_layer(qwen36_contract_layer *dst, const qwen36_35a3b_q4xl_layer *src) {
    memcpy(dst, src, sizeof(*dst));
}

static int bind_contract_model(const qwen36_gguf_file *gf, qwen36_contract_model *out, char *err, size_t err_cap) {
    qwen36_35a3b_q8_model q8;
    qwen36_35a3b_q4xl_model q4;
    uint32_t i;
    memset(out, 0, sizeof(*out));
    if (qwen36_35a3b_q8_bind(gf, &q8, err, err_cap)) {
        out->gf = q8.gf;
        out->token_embd = q8.token_embd;
        out->output_norm = q8.output_norm;
        out->output = q8.output;
        out->hybrid_layer_count = q8.hybrid_layer_count;
        out->full_attn_layer_count = q8.full_attn_layer_count;
        for (i = 0; i < QWEN36_35A3B_Q8_BLOCK_COUNT; ++i) map_q8_contract_layer(&out->layers[i], &q8.layers[i]);
        return 1;
    }
    if (qwen36_35a3b_q4xl_bind(gf, &q4, err, err_cap)) {
        out->gf = q4.gf;
        out->token_embd = q4.token_embd;
        out->output_norm = q4.output_norm;
        out->output = q4.output;
        out->hybrid_layer_count = q4.hybrid_layer_count;
        out->full_attn_layer_count = q4.full_attn_layer_count;
        for (i = 0; i < QWEN36_35A3B_Q4XL_BLOCK_COUNT; ++i) map_q4xl_contract_layer(&out->layers[i], &q4.layers[i]);
        return 1;
    }
    return 0;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int mapped_file_open(mapped_file *mf, const char *path) {
    struct stat st;
    memset(mf, 0, sizeof(*mf));
    mf->fd = open(path, O_RDONLY);
    if (mf->fd == -1) return 0;
    if (fstat(mf->fd, &st) == -1) {
        close(mf->fd);
        mf->fd = -1;
        return 0;
    }
    mf->map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, mf->fd, 0);
    if (mf->map == MAP_FAILED) {
        close(mf->fd);
        mf->fd = -1;
        mf->map = NULL;
        return 0;
    }
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

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1f;
    const uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            uint32_t e = 127 - 15 + 1, m = mant;
            while ((m & 0x400) == 0) { m <<= 1; e--; }
            m &= 0x3ff;
            bits = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    { union { uint32_t u; float f; } v = { bits }; return v.f; }
}

static int read_f32_tensor(const qwen36_gguf_file *gf, const qwen36_gguf_tensor *t, float **out, size_t elems, char *err, size_t err_cap) {
    float *buf = (float *)malloc(elems * sizeof(float));
    if (!buf) return 0;
    if (!qwen36_gguf_read_tensor_bytes(gf, t, 0, buf, elems * sizeof(float), err, err_cap)) {
        free(buf);
        return 0;
    }
    *out = buf;
    return 1;
}

static void free_small_cache(full_layer_small_cache *cache) {
    if (!cache) return;
    free(cache->q_norm_w);
    free(cache->k_norm_w);
    free(cache->shared_gate_inp_w);
    memset(cache, 0, sizeof(*cache));
}

static int load_small_cache(const qwen36_gguf_file *gf, const qwen36_contract_layer *layer, full_layer_small_cache *cache, char *err, size_t err_cap) {
    memset(cache, 0, sizeof(*cache));
    if (!read_f32_tensor(gf, layer->attn_q_norm, &cache->q_norm_w, HEAD_DIM, err, err_cap)) return 0;
    if (!read_f32_tensor(gf, layer->attn_k_norm, &cache->k_norm_w, HEAD_DIM, err, err_cap)) return 0;
    if (!read_f32_tensor(gf, layer->ffn_gate_inp_shexp, &cache->shared_gate_inp_w, HIDDEN, err, err_cap)) return 0;
    return 1;
}

static inline float siluf_local(float x) { return x / (1.0f + expf(-x)); }
static inline float sigmoidf_local(float x) {
    if (x >= 0.0f) {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        const float z = expf(x);
        return z / (1.0f + z);
    }
}

static uint64_t tensor_row_bytes(const qwen36_gguf_tensor *t, uint32_t cols) {
    switch (t->type) {
    case QWEN36_GGUF_TYPE_Q8_0:
        return (uint64_t)(cols / 32u) * 34u;
    case QWEN36_GGUF_TYPE_Q4_K:
        return (uint64_t)(cols / QK_K) * sizeof(qwen36_block_q4_K);
    case QWEN36_GGUF_TYPE_Q5_K:
        return (uint64_t)(cols / QK_K) * sizeof(qwen36_block_q5_K);
    case QWEN36_GGUF_TYPE_Q6_K:
        return (uint64_t)(cols / QK_K) * sizeof(qwen36_block_q6_K);
    default:
        return 0;
    }
}

static void topk_softmax256(const float *logits, uint32_t k, int32_t *idx, float *scores) {
    uint32_t i, j, m;
    for (i = 0; i < k; ++i) idx[i] = (int32_t)i;
    for (i = k; i < ROUTER_COUNT; ++i) {
        m = 0;
        for (j = 1; j < k; ++j) if (logits[idx[j]] < logits[idx[m]]) m = j;
        if (logits[i] > logits[idx[m]]) idx[m] = (int32_t)i;
    }
    for (i = 0; i < k; ++i) {
        for (j = i + 1; j < k; ++j) {
            if (logits[idx[j]] > logits[idx[i]]) {
                const int32_t tmp = idx[i];
                idx[i] = idx[j];
                idx[j] = tmp;
            }
        }
    }
    {
        const float maxv = logits[idx[0]];
        double sum = 0.0;
        for (i = 0; i < k; ++i) sum += exp((double)logits[idx[i]] - maxv);
        for (i = 0; i < k; ++i) scores[i] = (float)(exp((double)logits[idx[i]] - maxv) / sum);
    }
}

static void rotate_half(const float *x, float *out, uint32_t dim) {
    const uint32_t half = dim / 2;
    for (uint32_t i = 0; i < half; ++i) {
        out[i] = -x[half + i];
        out[half + i] = x[i];
    }
}

static void apply_rope_inplace(float *q, float *k, uint32_t pos) {
    static int init = 0;
    static float inv_freq[ROTARY_DIM / 2];
    float freqs[ROTARY_DIM / 2];
    float cosv[ROTARY_DIM];
    float sinv[ROTARY_DIM];
    float tmp_q[ROTARY_DIM];
    float tmp_k[ROTARY_DIM];
    if (!init) {
        for (uint32_t i = 0; i < ROTARY_DIM / 2; ++i) inv_freq[i] = powf(10000000.0f, -(float)(2 * i) / (float)ROTARY_DIM);
        init = 1;
    }
    for (uint32_t i = 0; i < ROTARY_DIM / 2; ++i) freqs[i] = (float)pos * inv_freq[i];
    for (uint32_t i = 0; i < ROTARY_DIM / 2; ++i) {
        cosv[i] = cosf(freqs[i]);
        cosv[i + ROTARY_DIM / 2] = cosv[i];
        sinv[i] = sinf(freqs[i]);
        sinv[i + ROTARY_DIM / 2] = sinv[i];
    }
    rotate_half(q, tmp_q, ROTARY_DIM);
    rotate_half(k, tmp_k, ROTARY_DIM);
    for (uint32_t i = 0; i < ROTARY_DIM; ++i) {
        q[i] = q[i] * cosv[i] + tmp_q[i] * sinv[i];
        k[i] = k[i] * cosv[i] + tmp_k[i] * sinv[i];
    }
}

static void apply_rope_one_inplace(float *x, uint32_t pos) {
    float tmp[ROTARY_DIM];
    apply_rope_inplace(x, tmp, pos);
}

static void rmsnorm_weight_plus1(const float *in, float *out, const float *w, uint32_t dim) {
    double var = 0.0;
    for (uint32_t i = 0; i < dim; ++i) var += (double)in[i] * in[i];
    var /= (double)dim;
    for (uint32_t i = 0; i < dim; ++i) out[i] = (float)(in[i] / sqrt(var + 1e-6)) * (1.0f + w[i]);
}

static void rmsnorm_weight_raw(const float *in, float *out, const float *w, uint32_t dim) {
    double var = 0.0;
    for (uint32_t i = 0; i < dim; ++i) var += (double)in[i] * in[i];
    var /= (double)dim;
    for (uint32_t i = 0; i < dim; ++i) out[i] = (float)(in[i] / sqrt(var + 1e-6)) * w[i];
}

static int run_gpu_ffn_from_residual(
        const mapped_file *mf,
        const qwen36_contract_layer *layer,
        const float *shared_gate_inp_w,
        uint32_t n_tokens,
        const float *residual_in,
        float *out_seq,
        full_ffn_timing *timing) {
    const uint32_t hidden = HIDDEN;
    const uint32_t topk = 8u;
    const uint32_t shared_dim = INTER;
    const uint64_t seq_hidden_bytes = (uint64_t)n_tokens * hidden * sizeof(float);
    const uint64_t router_logits_bytes = (uint64_t)n_tokens * ROUTER_COUNT * sizeof(float);
    const uint64_t shared_bytes = (uint64_t)n_tokens * shared_dim * sizeof(float);
    const uint64_t gate_row_bytes = tensor_row_bytes(layer->ffn_gate_exps, hidden);
    const uint64_t gate_expert_bytes = gate_row_bytes * INTER;
    const uint64_t down_row_bytes = tensor_row_bytes(layer->ffn_down_exps, INTER);
    const uint64_t down_expert_bytes = down_row_bytes * hidden;
    ds4_gpu_tensor *residual = NULL, *post = NULL, *router_logits = NULL;
    ds4_gpu_tensor *shared_gate = NULL, *shared_up = NULL, *shared_mid = NULL, *shared_out = NULL;
    ds4_gpu_tensor *expert_gate = NULL, *expert_up = NULL, *expert_mid = NULL, *expert_down = NULL;
    float *router_logits_cpu = NULL, *shared_out_cpu = NULL, *expert_down_cpu = NULL, *routed_out_cpu = NULL, *post_cpu = NULL;
    int32_t *router_selected_cpu = NULL;
    float *router_weights_cpu = NULL;
    int32_t union_ids[ROUTER_COUNT];
    int32_t union_pos[ROUTER_COUNT];
    uint32_t n_union = 0;
    int ok = 0;
    double t0 = now_ms();

    residual = ds4_gpu_tensor_alloc(seq_hidden_bytes);
    post = ds4_gpu_tensor_alloc(seq_hidden_bytes);
    router_logits = ds4_gpu_tensor_alloc(router_logits_bytes);
    shared_gate = ds4_gpu_tensor_alloc(shared_bytes);
    shared_up = ds4_gpu_tensor_alloc(shared_bytes);
    shared_mid = ds4_gpu_tensor_alloc(shared_bytes);
    shared_out = ds4_gpu_tensor_alloc(seq_hidden_bytes);
    expert_gate = ds4_gpu_tensor_alloc(shared_bytes);
    expert_up = ds4_gpu_tensor_alloc(shared_bytes);
    expert_mid = ds4_gpu_tensor_alloc(shared_bytes);
    expert_down = ds4_gpu_tensor_alloc(seq_hidden_bytes);
    if (!residual || !post || !router_logits || !shared_gate || !shared_up || !shared_mid || !shared_out ||
        !expert_gate || !expert_up || !expert_mid || !expert_down) goto cleanup;
    if (timing) timing->alloc_ms = now_ms() - t0;

    t0 = now_ms();
    if (ds4_gpu_tensor_write(residual, 0, residual_in, seq_hidden_bytes) == 0) goto cleanup;
    if (ds4_gpu_begin_commands() == 0) goto cleanup;
    if (ds4_gpu_rms_norm_weight_rows_tensor(post, residual, mf->map, mf->size,
                                            layer->post_attn_norm->abs_offset,
                                            hidden, n_tokens, QWEN_RMS_EPS) == 0) goto cleanup;
    if (ds4_gpu_matmul_f32_tensor(router_logits, mf->map, mf->size,
                                  layer->ffn_gate_inp->abs_offset,
                                  hidden, ROUTER_COUNT, post, n_tokens) == 0) goto cleanup;
    if (ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(shared_gate, shared_up, shared_mid,
                                                  mf->map, mf->size,
                                                  layer->ffn_gate_shexp->abs_offset,
                                                  layer->ffn_up_shexp->abs_offset,
                                                  hidden, shared_dim, post,
                                                  QWEN_SWIGLU_CLAMP) == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(shared_out, mf->map, mf->size,
                                   layer->ffn_down_shexp->abs_offset,
                                   shared_dim, hidden, shared_mid, n_tokens) == 0) goto cleanup;
    if (ds4_gpu_end_commands() == 0) goto cleanup;
    if (timing) timing->shared_gpu_ms = now_ms() - t0;

    t0 = now_ms();
    router_logits_cpu = (float *)malloc(router_logits_bytes);
    shared_out_cpu = (float *)malloc(seq_hidden_bytes);
    expert_down_cpu = (float *)malloc(seq_hidden_bytes);
    routed_out_cpu = (float *)calloc((size_t)n_tokens * hidden, sizeof(float));
    post_cpu = (float *)malloc(seq_hidden_bytes);
    router_selected_cpu = (int32_t *)malloc((size_t)n_tokens * topk * sizeof(int32_t));
    router_weights_cpu = (float *)malloc((size_t)n_tokens * topk * sizeof(float));
    if (!router_logits_cpu || !shared_out_cpu || !expert_down_cpu || !routed_out_cpu || !post_cpu ||
        !router_selected_cpu || !router_weights_cpu) goto cleanup;
    memset(union_pos, 0xff, sizeof(union_pos));
    if (ds4_gpu_tensor_read(router_logits, 0, router_logits_cpu, router_logits_bytes) == 0) goto cleanup;
    if (ds4_gpu_tensor_read(shared_out, 0, shared_out_cpu, seq_hidden_bytes) == 0) goto cleanup;
    if (ds4_gpu_tensor_read(post, 0, post_cpu, seq_hidden_bytes) == 0) goto cleanup;
    if (timing) timing->shared_readback_ms = now_ms() - t0;

    t0 = now_ms();
    for (uint32_t t = 0; t < n_tokens; ++t) {
        topk_softmax256(router_logits_cpu + (size_t)t * ROUTER_COUNT, topk,
                        router_selected_cpu + (size_t)t * topk,
                        router_weights_cpu + (size_t)t * topk);
        for (uint32_t i = 0; i < topk; ++i) {
            const int32_t expert_id = router_selected_cpu[(size_t)t * topk + i];
            if (union_pos[expert_id] < 0) {
                union_pos[expert_id] = (int32_t)n_union;
                union_ids[n_union++] = expert_id;
            }
        }
    }
    if (timing) {
        timing->route_ms = now_ms() - t0;
        timing->n_union = n_union;
    }

    t0 = now_ms();
    for (uint32_t u = 0; u < n_union; ++u) {
        const uint32_t expert_id = (uint32_t)union_ids[u];
        double expert_t0 = now_ms();
        if (ds4_gpu_begin_commands() == 0) goto cleanup;
        if (ds4_gpu_matmul_q8_0_pair_tensor(expert_gate, expert_up,
                                            mf->map, mf->size,
                                            layer->ffn_gate_exps->abs_offset + gate_expert_bytes * expert_id,
                                            layer->ffn_up_exps->abs_offset + gate_expert_bytes * expert_id,
                                            hidden, shared_dim, shared_dim, post, n_tokens) == 0) goto cleanup;
        if (ds4_gpu_swiglu_tensor(expert_mid, expert_gate, expert_up,
                                  n_tokens * shared_dim, QWEN_SWIGLU_CLAMP, 1.0f) == 0) goto cleanup;
        if (ds4_gpu_matmul_q8_0_tensor(expert_down, mf->map, mf->size,
                                       layer->ffn_down_exps->abs_offset + down_expert_bytes * expert_id,
                                       shared_dim, hidden, expert_mid, n_tokens) == 0) goto cleanup;
        if (ds4_gpu_end_commands() == 0) goto cleanup;
        if (timing) timing->expert_gpu_ms += now_ms() - expert_t0;
        expert_t0 = now_ms();
        if (ds4_gpu_tensor_read(expert_down, 0, expert_down_cpu, seq_hidden_bytes) == 0) goto cleanup;
        if (timing) timing->expert_readback_ms += now_ms() - expert_t0;
        for (uint32_t t = 0; t < n_tokens; ++t) {
            float weight = 0.0f;
            for (uint32_t i = 0; i < topk; ++i) {
                const size_t base = (size_t)t * topk + i;
                if ((uint32_t)router_selected_cpu[base] == expert_id) {
                    weight = router_weights_cpu[base];
                    break;
                }
            }
            if (weight == 0.0f) continue;
            for (uint32_t d = 0; d < hidden; ++d) {
                const size_t idx = (size_t)t * hidden + d;
                routed_out_cpu[idx] += expert_down_cpu[idx] * weight;
            }
        }
    }

    t0 = now_ms();
    for (uint32_t t = 0; t < n_tokens; ++t) {
        float scale_in = 0.0f;
        for (uint32_t d = 0; d < hidden; ++d) scale_in += post_cpu[(size_t)t * hidden + d] * shared_gate_inp_w[d];
        {
            const float s = sigmoidf_local(scale_in);
            for (uint32_t d = 0; d < hidden; ++d) {
                const size_t idx = (size_t)t * hidden + d;
                out_seq[idx] = residual_in[idx] + routed_out_cpu[idx] + shared_out_cpu[idx] * s;
            }
        }
    }
    if (timing) timing->combine_ms = now_ms() - t0;
    ok = 1;

cleanup:
    free(router_logits_cpu);
    free(shared_out_cpu);
    free(expert_down_cpu);
    free(routed_out_cpu);
    free(post_cpu);
    free(router_selected_cpu);
    free(router_weights_cpu);
    ds4_gpu_tensor_free(expert_down);
    ds4_gpu_tensor_free(expert_mid);
    ds4_gpu_tensor_free(expert_up);
    ds4_gpu_tensor_free(expert_gate);
    ds4_gpu_tensor_free(shared_out);
    ds4_gpu_tensor_free(shared_mid);
    ds4_gpu_tensor_free(shared_up);
    ds4_gpu_tensor_free(shared_gate);
    ds4_gpu_tensor_free(router_logits);
    ds4_gpu_tensor_free(post);
    ds4_gpu_tensor_free(residual);
    return ok;
}

static void worker_reset(worker_state *ws) {
    ws->prefilled = 0;
    ws->seq_len = 0;
    free(ws->k_all); ws->k_all = NULL;
    free(ws->v_all); ws->v_all = NULL;
    free(ws->output_seq); ws->output_seq = NULL;
}

static int load_seq_file(const char *path, float **out, uint32_t *seq_len, char *err, size_t err_cap) {
    FILE *fp = fopen(path, "rb");
    long sz;
    float *buf = NULL;
    if (!fp) {
        snprintf(err, err_cap, "open seq failed: %s", strerror(errno));
        return 0;
    }
    if (fseek(fp, 0, SEEK_END) != 0 || (sz = ftell(fp)) < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        snprintf(err, err_cap, "stat seq failed");
        fclose(fp);
        return 0;
    }
    *seq_len = (uint32_t)((size_t)sz / (HIDDEN * sizeof(float)));
    if ((size_t)sz != (size_t)(*seq_len) * HIDDEN * sizeof(float)) {
        snprintf(err, err_cap, "bad seq size");
        fclose(fp);
        return 0;
    }
    buf = (float *)malloc((size_t)(*seq_len) * HIDDEN * sizeof(float));
    if (!buf) {
        snprintf(err, err_cap, "oom seq");
        fclose(fp);
        return 0;
    }
    if (fread(buf, sizeof(float), (size_t)(*seq_len) * HIDDEN, fp) != (size_t)(*seq_len) * HIDDEN) {
        snprintf(err, err_cap, "read seq failed");
        free(buf);
        fclose(fp);
        return 0;
    }
    fclose(fp);
    *out = buf;
    return 1;
}

static int load_row_file(const char *path, float *out, char *err, size_t err_cap) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        snprintf(err, err_cap, "open row failed: %s", strerror(errno));
        return 0;
    }
    if (fread(out, sizeof(float), HIDDEN, fp) != HIDDEN) {
        snprintf(err, err_cap, "read row failed");
        fclose(fp);
        return 0;
    }
    fclose(fp);
    return 1;
}

static int load_seq_bin(float **out, uint32_t seq_len, char *err, size_t err_cap) {
    float *buf = (float *)malloc((size_t)seq_len * HIDDEN * sizeof(float));
    if (!buf) {
        snprintf(err, err_cap, "oom seq bin");
        return 0;
    }
    if (fread(buf, sizeof(float), (size_t)seq_len * HIDDEN, stdin) != (size_t)seq_len * HIDDEN) {
        snprintf(err, err_cap, "read seq bin failed");
        free(buf);
        return 0;
    }
    *out = buf;
    return 1;
}

static int load_row_bin(float *out, char *err, size_t err_cap) {
    if (fread(out, sizeof(float), HIDDEN, stdin) != HIDDEN) {
        snprintf(err, err_cap, "read row bin failed");
        return 0;
    }
    return 1;
}

static int run_prefill_full_layer(worker_state *ws, const float *input_seq, uint32_t seq_len) {
    ds4_gpu_tensor *input_gpu = NULL, *attn_in_gpu = NULL, *qg_gpu = NULL, *k_gpu = NULL, *v_gpu = NULL;
    ds4_gpu_tensor *attn_out_gpu = NULL, *proj_out_gpu = NULL;
    float *attn_in = NULL, *qg = NULL, *kk = NULL, *vv = NULL;
    float *q_all = NULL, *k_all = NULL, *v_all = NULL, *gate_all = NULL;
    float *attn_out_flat = NULL, *proj_out = NULL, *residual = NULL, *output_seq = NULL;
    int ok = 0;

    attn_in = (float *)malloc((size_t)seq_len * HIDDEN * sizeof(float));
    qg = (float *)malloc((size_t)seq_len * ATTN_GATE_DIM * 2u * sizeof(float));
    kk = (float *)malloc((size_t)seq_len * NUM_KV_HEADS * HEAD_DIM * sizeof(float));
    vv = (float *)malloc((size_t)seq_len * NUM_KV_HEADS * HEAD_DIM * sizeof(float));
    q_all = (float *)calloc((size_t)seq_len * NUM_HEADS * HEAD_DIM, sizeof(float));
    k_all = (float *)calloc((size_t)seq_len * NUM_KV_HEADS * HEAD_DIM, sizeof(float));
    v_all = (float *)calloc((size_t)seq_len * NUM_KV_HEADS * HEAD_DIM, sizeof(float));
    gate_all = (float *)calloc((size_t)seq_len * ATTN_GATE_DIM, sizeof(float));
    attn_out_flat = (float *)calloc((size_t)seq_len * ATTN_GATE_DIM, sizeof(float));
    proj_out = (float *)calloc((size_t)seq_len * HIDDEN, sizeof(float));
    residual = (float *)calloc((size_t)seq_len * HIDDEN, sizeof(float));
    output_seq = (float *)calloc((size_t)seq_len * HIDDEN, sizeof(float));
    if (!attn_in || !qg || !kk || !vv || !q_all || !k_all || !v_all || !gate_all || !attn_out_flat || !proj_out || !residual || !output_seq) goto cleanup;

    input_gpu = ds4_gpu_tensor_alloc((uint64_t)seq_len * HIDDEN * sizeof(float));
    attn_in_gpu = ds4_gpu_tensor_alloc((uint64_t)seq_len * HIDDEN * sizeof(float));
    qg_gpu = ds4_gpu_tensor_alloc((uint64_t)seq_len * ATTN_GATE_DIM * 2u * sizeof(float));
    k_gpu = ds4_gpu_tensor_alloc((uint64_t)seq_len * NUM_KV_HEADS * HEAD_DIM * sizeof(float));
    v_gpu = ds4_gpu_tensor_alloc((uint64_t)seq_len * NUM_KV_HEADS * HEAD_DIM * sizeof(float));
    attn_out_gpu = ds4_gpu_tensor_alloc((uint64_t)seq_len * ATTN_GATE_DIM * sizeof(float));
    proj_out_gpu = ds4_gpu_tensor_alloc((uint64_t)seq_len * HIDDEN * sizeof(float));
    if (!input_gpu || !attn_in_gpu || !qg_gpu || !k_gpu || !v_gpu || !attn_out_gpu || !proj_out_gpu) goto cleanup;

    if (ds4_gpu_tensor_write(input_gpu, 0, input_seq, (uint64_t)seq_len * HIDDEN * sizeof(float)) == 0) goto cleanup;
    if (ds4_gpu_begin_commands() == 0) goto cleanup;
    if (ds4_gpu_rms_norm_weight_rows_tensor(attn_in_gpu, input_gpu, ws->mf.map, ws->mf.size,
                                            ws->layer->attn_norm->abs_offset, HIDDEN, seq_len, QWEN_RMS_EPS) == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(qg_gpu, ws->mf.map, ws->mf.size,
                                   ws->layer->attn_q->abs_offset, HIDDEN, ATTN_GATE_DIM * 2u, attn_in_gpu, seq_len) == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(k_gpu, ws->mf.map, ws->mf.size,
                                   ws->layer->attn_k->abs_offset, HIDDEN, NUM_KV_HEADS * HEAD_DIM, attn_in_gpu, seq_len) == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(v_gpu, ws->mf.map, ws->mf.size,
                                   ws->layer->attn_v->abs_offset, HIDDEN, NUM_KV_HEADS * HEAD_DIM, attn_in_gpu, seq_len) == 0) goto cleanup;
    if (ds4_gpu_end_commands() == 0) goto cleanup;

    if (ds4_gpu_tensor_read(attn_in_gpu, 0, attn_in, (uint64_t)seq_len * HIDDEN * sizeof(float)) == 0) goto cleanup;
    if (ds4_gpu_tensor_read(qg_gpu, 0, qg, (uint64_t)seq_len * ATTN_GATE_DIM * 2u * sizeof(float)) == 0) goto cleanup;
    if (ds4_gpu_tensor_read(k_gpu, 0, kk, (uint64_t)seq_len * NUM_KV_HEADS * HEAD_DIM * sizeof(float)) == 0) goto cleanup;
    if (ds4_gpu_tensor_read(v_gpu, 0, vv, (uint64_t)seq_len * NUM_KV_HEADS * HEAD_DIM * sizeof(float)) == 0) goto cleanup;

    for (uint32_t t = 0; t < seq_len; ++t) {
        for (uint32_t h = 0; h < NUM_HEADS; ++h) {
            const float *src = qg + (size_t)t * (ATTN_GATE_DIM * 2u) + (size_t)h * (HEAD_DIM * 2u);
            float *qdst = q_all + ((size_t)t * NUM_HEADS + h) * HEAD_DIM;
            rmsnorm_weight_raw(src, qdst, ws->cache.q_norm_w, HEAD_DIM);
            memcpy(gate_all + (size_t)t * ATTN_GATE_DIM + (size_t)h * HEAD_DIM, src + HEAD_DIM, HEAD_DIM * sizeof(float));
        }
        for (uint32_t h = 0; h < NUM_KV_HEADS; ++h) {
            rmsnorm_weight_raw(kk + (size_t)t * NUM_KV_HEADS * HEAD_DIM + (size_t)h * HEAD_DIM,
                               k_all + ((size_t)t * NUM_KV_HEADS + h) * HEAD_DIM, ws->cache.k_norm_w, HEAD_DIM);
            memcpy(v_all + ((size_t)t * NUM_KV_HEADS + h) * HEAD_DIM,
                   vv + (size_t)t * NUM_KV_HEADS * HEAD_DIM + (size_t)h * HEAD_DIM, HEAD_DIM * sizeof(float));
        }
        for (uint32_t h = 0; h < NUM_KV_HEADS; ++h) apply_rope_one_inplace(k_all + ((size_t)t * NUM_KV_HEADS + h) * HEAD_DIM, t);
        for (uint32_t h = 0; h < NUM_HEADS; ++h) apply_rope_one_inplace(q_all + ((size_t)t * NUM_HEADS + h) * HEAD_DIM, t);
    }
    for (uint32_t t = 0; t < seq_len; ++t) {
        for (uint32_t h = 0; h < NUM_HEADS; ++h) {
            const uint32_t kvh = h / (NUM_HEADS / NUM_KV_HEADS);
            float scores[4096];
            float maxv = -1e30f, sum = 0.0f;
            float *out = attn_out_flat + (size_t)t * ATTN_GATE_DIM + (size_t)h * HEAD_DIM;
            const float *qv = q_all + ((size_t)t * NUM_HEADS + h) * HEAD_DIM;
            memset(out, 0, HEAD_DIM * sizeof(float));
            for (uint32_t s = 0; s <= t; ++s) {
                double dot = 0.0;
                const float *kv = k_all + ((size_t)s * NUM_KV_HEADS + kvh) * HEAD_DIM;
                for (uint32_t d = 0; d < HEAD_DIM; ++d) dot += (double)qv[d] * kv[d];
                scores[s] = (float)(dot / sqrt((double)HEAD_DIM));
                if (scores[s] > maxv) maxv = scores[s];
            }
            for (uint32_t s = 0; s <= t; ++s) {
                scores[s] = expf(scores[s] - maxv);
                sum += scores[s];
            }
            for (uint32_t s = 0; s <= t; ++s) {
                const float w = scores[s] / sum;
                const float *vvh = v_all + ((size_t)s * NUM_KV_HEADS + kvh) * HEAD_DIM;
                for (uint32_t d = 0; d < HEAD_DIM; ++d) out[d] += w * vvh[d];
            }
            for (uint32_t d = 0; d < HEAD_DIM; ++d) out[d] *= sigmoidf_local(gate_all[(size_t)t * ATTN_GATE_DIM + (size_t)h * HEAD_DIM + d]);
        }
    }

    if (ds4_gpu_tensor_write(attn_out_gpu, 0, attn_out_flat, (uint64_t)seq_len * ATTN_GATE_DIM * sizeof(float)) == 0) goto cleanup;
    if (ds4_gpu_begin_commands() == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(proj_out_gpu, ws->mf.map, ws->mf.size,
                                   ws->layer->attn_output->abs_offset, ATTN_GATE_DIM, HIDDEN, attn_out_gpu, seq_len) == 0) goto cleanup;
    if (ds4_gpu_end_commands() == 0) goto cleanup;
    if (ds4_gpu_tensor_read(proj_out_gpu, 0, proj_out, (uint64_t)seq_len * HIDDEN * sizeof(float)) == 0) goto cleanup;

    for (uint32_t t = 0; t < seq_len; ++t) {
        for (uint32_t d = 0; d < HIDDEN; ++d) {
            const size_t idx = (size_t)t * HIDDEN + d;
            residual[idx] = input_seq[idx] + proj_out[idx];
        }
    }
    if (!run_gpu_ffn_from_residual(&ws->mf, ws->layer, ws->cache.shared_gate_inp_w, seq_len, residual, output_seq, NULL)) goto cleanup;

    free(ws->k_all); ws->k_all = k_all; k_all = NULL;
    free(ws->v_all); ws->v_all = v_all; v_all = NULL;
    free(ws->output_seq); ws->output_seq = output_seq; output_seq = NULL;
    ws->seq_len = seq_len;
    ws->prefilled = 1;
    ok = 1;

cleanup:
    ds4_gpu_tensor_free(proj_out_gpu);
    ds4_gpu_tensor_free(attn_out_gpu);
    ds4_gpu_tensor_free(v_gpu);
    ds4_gpu_tensor_free(k_gpu);
    ds4_gpu_tensor_free(qg_gpu);
    ds4_gpu_tensor_free(attn_in_gpu);
    ds4_gpu_tensor_free(input_gpu);
    free(attn_in); free(qg); free(kk); free(vv); free(q_all); free(k_all); free(v_all); free(gate_all); free(attn_out_flat); free(proj_out); free(residual); free(output_seq);
    return ok;
}

static int run_step_full_layer(worker_state *ws, const float *input_row) {
    ds4_gpu_tensor *input_gpu = NULL, *attn_in_gpu = NULL, *qg_gpu = NULL, *k_gpu = NULL, *v_gpu = NULL;
    ds4_gpu_tensor *attn_out_gpu = NULL, *proj_out_gpu = NULL;
    float *attn_in = NULL, *qg = NULL, *kk = NULL, *vv = NULL;
    float q_cur[HEAD_DIM * NUM_HEADS];
    float k_cur[HEAD_DIM * NUM_KV_HEADS];
    float v_cur[HEAD_DIM * NUM_KV_HEADS];
    float gate_cur[ATTN_GATE_DIM];
    float attn_out_flat[ATTN_GATE_DIM];
    float proj_out[HIDDEN];
    float residual[HIDDEN];
    float output_row[HIDDEN];
    float *new_k = NULL, *new_v = NULL, *new_out = NULL;
    uint32_t pos = ws->seq_len;
    int ok = 0;
    full_ffn_timing ffn_timing;
    double step_t0 = now_ms();
    double alloc_ms = 0.0, qkv_gpu_ms = 0.0, qkv_readback_ms = 0.0;
    double attn_cpu_ms = 0.0, out_proj_ms = 0.0, kv_append_ms = 0.0;

    memset(&ffn_timing, 0, sizeof(ffn_timing));

    step_t0 = now_ms();
    attn_in = (float *)malloc((size_t)HIDDEN * sizeof(float));
    qg = (float *)malloc((size_t)ATTN_GATE_DIM * 2u * sizeof(float));
    kk = (float *)malloc((size_t)NUM_KV_HEADS * HEAD_DIM * sizeof(float));
    vv = (float *)malloc((size_t)NUM_KV_HEADS * HEAD_DIM * sizeof(float));
    if (!attn_in || !qg || !kk || !vv) goto cleanup;

    input_gpu = ds4_gpu_tensor_alloc((uint64_t)HIDDEN * sizeof(float));
    attn_in_gpu = ds4_gpu_tensor_alloc((uint64_t)HIDDEN * sizeof(float));
    qg_gpu = ds4_gpu_tensor_alloc((uint64_t)ATTN_GATE_DIM * 2u * sizeof(float));
    k_gpu = ds4_gpu_tensor_alloc((uint64_t)NUM_KV_HEADS * HEAD_DIM * sizeof(float));
    v_gpu = ds4_gpu_tensor_alloc((uint64_t)NUM_KV_HEADS * HEAD_DIM * sizeof(float));
    attn_out_gpu = ds4_gpu_tensor_alloc((uint64_t)ATTN_GATE_DIM * sizeof(float));
    proj_out_gpu = ds4_gpu_tensor_alloc((uint64_t)HIDDEN * sizeof(float));
    if (!input_gpu || !attn_in_gpu || !qg_gpu || !k_gpu || !v_gpu || !attn_out_gpu || !proj_out_gpu) goto cleanup;
    alloc_ms = now_ms() - step_t0;

    step_t0 = now_ms();
    if (ds4_gpu_tensor_write(input_gpu, 0, input_row, (uint64_t)HIDDEN * sizeof(float)) == 0) goto cleanup;
    if (ds4_gpu_begin_commands() == 0) goto cleanup;
    if (ds4_gpu_rms_norm_weight_rows_tensor(attn_in_gpu, input_gpu, ws->mf.map, ws->mf.size,
                                            ws->layer->attn_norm->abs_offset, HIDDEN, 1u, QWEN_RMS_EPS) == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(qg_gpu, ws->mf.map, ws->mf.size,
                                   ws->layer->attn_q->abs_offset, HIDDEN, ATTN_GATE_DIM * 2u, attn_in_gpu, 1u) == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(k_gpu, ws->mf.map, ws->mf.size,
                                   ws->layer->attn_k->abs_offset, HIDDEN, NUM_KV_HEADS * HEAD_DIM, attn_in_gpu, 1u) == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(v_gpu, ws->mf.map, ws->mf.size,
                                   ws->layer->attn_v->abs_offset, HIDDEN, NUM_KV_HEADS * HEAD_DIM, attn_in_gpu, 1u) == 0) goto cleanup;
    if (ds4_gpu_end_commands() == 0) goto cleanup;
    qkv_gpu_ms = now_ms() - step_t0;

    step_t0 = now_ms();
    if (ds4_gpu_tensor_read(attn_in_gpu, 0, attn_in, (uint64_t)HIDDEN * sizeof(float)) == 0) goto cleanup;
    if (ds4_gpu_tensor_read(qg_gpu, 0, qg, (uint64_t)ATTN_GATE_DIM * 2u * sizeof(float)) == 0) goto cleanup;
    if (ds4_gpu_tensor_read(k_gpu, 0, kk, (uint64_t)NUM_KV_HEADS * HEAD_DIM * sizeof(float)) == 0) goto cleanup;
    if (ds4_gpu_tensor_read(v_gpu, 0, vv, (uint64_t)NUM_KV_HEADS * HEAD_DIM * sizeof(float)) == 0) goto cleanup;
    qkv_readback_ms = now_ms() - step_t0;

    step_t0 = now_ms();
    for (uint32_t h = 0; h < NUM_HEADS; ++h) {
        const float *src = qg + (size_t)h * (HEAD_DIM * 2u);
        float *qdst = q_cur + (size_t)h * HEAD_DIM;
        rmsnorm_weight_raw(src, qdst, ws->cache.q_norm_w, HEAD_DIM);
        memcpy(gate_cur + (size_t)h * HEAD_DIM, src + HEAD_DIM, HEAD_DIM * sizeof(float));
    }
    for (uint32_t h = 0; h < NUM_KV_HEADS; ++h) {
        rmsnorm_weight_raw(kk + (size_t)h * HEAD_DIM, k_cur + (size_t)h * HEAD_DIM, ws->cache.k_norm_w, HEAD_DIM);
        memcpy(v_cur + (size_t)h * HEAD_DIM, vv + (size_t)h * HEAD_DIM, HEAD_DIM * sizeof(float));
    }
    for (uint32_t h = 0; h < NUM_KV_HEADS; ++h) apply_rope_one_inplace(k_cur + (size_t)h * HEAD_DIM, pos);
    for (uint32_t h = 0; h < NUM_HEADS; ++h) apply_rope_one_inplace(q_cur + (size_t)h * HEAD_DIM, pos);

    memset(attn_out_flat, 0, sizeof(attn_out_flat));
    for (uint32_t h = 0; h < NUM_HEADS; ++h) {
        const uint32_t kvh = h / (NUM_HEADS / NUM_KV_HEADS);
        float *out = attn_out_flat + (size_t)h * HEAD_DIM;
        const float *qv = q_cur + (size_t)h * HEAD_DIM;
        double scores[8192];
        double maxv = -1e300, sum = 0.0;
        for (uint32_t s = 0; s <= pos; ++s) {
            const float *kv = (s == pos) ? (k_cur + (size_t)kvh * HEAD_DIM) : (ws->k_all + ((size_t)s * NUM_KV_HEADS + kvh) * HEAD_DIM);
            double dot = 0.0;
            for (uint32_t d = 0; d < HEAD_DIM; ++d) dot += (double)qv[d] * kv[d];
            scores[s] = dot / sqrt((double)HEAD_DIM);
            if (scores[s] > maxv) maxv = scores[s];
        }
        for (uint32_t s = 0; s <= pos; ++s) {
            scores[s] = exp(scores[s] - maxv);
            sum += scores[s];
        }
        for (uint32_t s = 0; s <= pos; ++s) {
            const float w = (float)(scores[s] / sum);
            const float *vvh = (s == pos) ? (v_cur + (size_t)kvh * HEAD_DIM) : (ws->v_all + ((size_t)s * NUM_KV_HEADS + kvh) * HEAD_DIM);
            for (uint32_t d = 0; d < HEAD_DIM; ++d) out[d] += w * vvh[d];
        }
        for (uint32_t d = 0; d < HEAD_DIM; ++d) out[d] *= sigmoidf_local(gate_cur[(size_t)h * HEAD_DIM + d]);
    }
    attn_cpu_ms = now_ms() - step_t0;

    step_t0 = now_ms();
    if (ds4_gpu_tensor_write(attn_out_gpu, 0, attn_out_flat, (uint64_t)ATTN_GATE_DIM * sizeof(float)) == 0) goto cleanup;
    if (ds4_gpu_begin_commands() == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(proj_out_gpu, ws->mf.map, ws->mf.size,
                                   ws->layer->attn_output->abs_offset, ATTN_GATE_DIM, HIDDEN, attn_out_gpu, 1u) == 0) goto cleanup;
    if (ds4_gpu_end_commands() == 0) goto cleanup;
    if (ds4_gpu_tensor_read(proj_out_gpu, 0, proj_out, (uint64_t)HIDDEN * sizeof(float)) == 0) goto cleanup;
    out_proj_ms = now_ms() - step_t0;

    for (uint32_t d = 0; d < HIDDEN; ++d) residual[d] = input_row[d] + proj_out[d];
    if (!run_gpu_ffn_from_residual(&ws->mf, ws->layer, ws->cache.shared_gate_inp_w, 1u, residual, output_row, &ffn_timing)) goto cleanup;

    step_t0 = now_ms();
    new_k = (float *)realloc(ws->k_all, (size_t)(pos + 1u) * NUM_KV_HEADS * HEAD_DIM * sizeof(float));
    new_v = (float *)realloc(ws->v_all, (size_t)(pos + 1u) * NUM_KV_HEADS * HEAD_DIM * sizeof(float));
    new_out = (float *)realloc(ws->output_seq, (size_t)(pos + 1u) * HIDDEN * sizeof(float));
    if (!new_k || !new_v || !new_out) goto cleanup;
    ws->k_all = new_k;
    ws->v_all = new_v;
    ws->output_seq = new_out;
    memcpy(ws->k_all + (size_t)pos * NUM_KV_HEADS * HEAD_DIM, k_cur, sizeof(k_cur));
    memcpy(ws->v_all + (size_t)pos * NUM_KV_HEADS * HEAD_DIM, v_cur, sizeof(v_cur));
    memcpy(ws->output_seq + (size_t)pos * HIDDEN, output_row, sizeof(output_row));
    ws->seq_len = pos + 1u;
    kv_append_ms = now_ms() - step_t0;
    fprintf(stderr,
            "qwen36_full layer=%u seq=%u alloc_ms=%.2f qkv_gpu_ms=%.2f qkv_readback_ms=%.2f "
            "attn_cpu_ms=%.2f out_proj_ms=%.2f ffn_alloc_ms=%.2f ffn_shared_gpu_ms=%.2f "
            "ffn_shared_readback_ms=%.2f ffn_route_ms=%.2f ffn_expert_gpu_ms=%.2f "
            "ffn_expert_readback_ms=%.2f ffn_combine_ms=%.2f ffn_union=%u kv_append_ms=%.2f total_ms=%.2f\n",
            ws->layer_idx, ws->seq_len,
            alloc_ms, qkv_gpu_ms, qkv_readback_ms,
            attn_cpu_ms, out_proj_ms, ffn_timing.alloc_ms, ffn_timing.shared_gpu_ms,
            ffn_timing.shared_readback_ms, ffn_timing.route_ms, ffn_timing.expert_gpu_ms,
            ffn_timing.expert_readback_ms, ffn_timing.combine_ms, ffn_timing.n_union,
            kv_append_ms,
            alloc_ms + qkv_gpu_ms + qkv_readback_ms + attn_cpu_ms + out_proj_ms +
            ffn_timing.alloc_ms + ffn_timing.shared_gpu_ms + ffn_timing.shared_readback_ms +
            ffn_timing.route_ms + ffn_timing.expert_gpu_ms + ffn_timing.expert_readback_ms +
            ffn_timing.combine_ms + kv_append_ms);
    ds4_gpu_print_memory_report("full_step_end");
    ok = 1;

cleanup:
    ds4_gpu_tensor_free(proj_out_gpu);
    ds4_gpu_tensor_free(attn_out_gpu);
    ds4_gpu_tensor_free(v_gpu);
    ds4_gpu_tensor_free(k_gpu);
    ds4_gpu_tensor_free(qg_gpu);
    ds4_gpu_tensor_free(attn_in_gpu);
    ds4_gpu_tensor_free(input_gpu);
    free(attn_in); free(qg); free(kk); free(vv);
    return ok;
}

static void handle_prefill_seq(worker_state *ws, const char *path) {
    char err[512] = {0};
    float *input_seq = NULL;
    uint32_t seq_len = 0;
    if (!load_seq_file(path, &input_seq, &seq_len, err, sizeof(err))) {
        printf("ERROR %s\n", err);
        fflush(stdout);
        return;
    }
    worker_reset(ws);
    if (!run_prefill_full_layer(ws, input_seq, seq_len)) {
        printf("ERROR prefill failed\n");
        fflush(stdout);
        free(input_seq);
        return;
    }
    printf("PREFILL_OK %u %u\n", ws->seq_len, (uint32_t)HIDDEN);
    fflush(stdout);
    free(input_seq);
}

static void handle_prefill_seq_bin(worker_state *ws, char *rest) {
    char err[512] = {0};
    float *input_seq = NULL;
    uint32_t seq_len = 0, hidden = 0;
    if (sscanf(rest ? rest : "", "%u %u", &seq_len, &hidden) != 2 || hidden != HIDDEN) {
        printf("ERROR bad PREFILL_SEQ_BIN args\n");
        fflush(stdout);
        return;
    }
    if (!load_seq_bin(&input_seq, seq_len, err, sizeof(err))) {
        printf("ERROR %s\n", err);
        fflush(stdout);
        return;
    }
    worker_reset(ws);
    if (!run_prefill_full_layer(ws, input_seq, seq_len)) {
        printf("ERROR prefill failed\n");
        fflush(stdout);
        free(input_seq);
        return;
    }
    printf("PREFILL_OK %u %u\n", ws->seq_len, (uint32_t)HIDDEN);
    fflush(stdout);
    free(input_seq);
}

static void handle_step_row(worker_state *ws, const char *path) {
    char err[512] = {0};
    float input_row[HIDDEN];
    if (!ws->prefilled) {
        printf("ERROR not prefilled\n");
        fflush(stdout);
        return;
    }
    if (!load_row_file(path, input_row, err, sizeof(err))) {
        printf("ERROR %s\n", err);
        fflush(stdout);
        return;
    }
    if (!run_step_full_layer(ws, input_row)) {
        printf("ERROR step failed\n");
        fflush(stdout);
        return;
    }
    printf("STEP_OK %u %u\n", ws->seq_len, (uint32_t)HIDDEN);
    fflush(stdout);
}

static void handle_step_row_bin(worker_state *ws, char *rest) {
    char err[512] = {0};
    float input_row[HIDDEN];
    uint32_t hidden = 0;
    if (!ws->prefilled) {
        printf("ERROR not prefilled\n");
        fflush(stdout);
        return;
    }
    if (sscanf(rest ? rest : "", "%u", &hidden) != 1 || hidden != HIDDEN) {
        printf("ERROR bad STEP_ROW_BIN args\n");
        fflush(stdout);
        return;
    }
    if (!load_row_bin(input_row, err, sizeof(err))) {
        printf("ERROR %s\n", err);
        fflush(stdout);
        return;
    }
    if (!run_step_full_layer(ws, input_row)) {
        printf("ERROR step failed\n");
        fflush(stdout);
        return;
    }
    printf("STEP_OK %u %u\n", ws->seq_len, (uint32_t)HIDDEN);
    fflush(stdout);
}

static void handle_dump_hidden(worker_state *ws) {
    size_t n, bytes;
    if (!ws->prefilled) {
        printf("ERROR not prefilled\n");
        fflush(stdout);
        return;
    }
    n = (size_t)ws->seq_len * HIDDEN;
    bytes = n * sizeof(float);
    printf("HIDDEN %zu %zu\n", n, bytes);
    fflush(stdout);
    fwrite(ws->output_seq, sizeof(float), n, stdout);
    fflush(stdout);
}

static void handle_dump_last(worker_state *ws) {
    if (!ws->prefilled || ws->seq_len == 0) {
        printf("ERROR not prefilled\n");
        fflush(stdout);
        return;
    }
    printf("LAST %u %zu\n", HIDDEN, (size_t)HIDDEN * sizeof(float));
    fflush(stdout);
    fwrite(ws->output_seq + (size_t)(ws->seq_len - 1u) * HIDDEN, sizeof(float), HIDDEN, stdout);
    fflush(stdout);
}

int main(int argc, char **argv) {
    worker_state ws;
    char err[512];
    char line[4096];
    int argi;

    memset(&ws, 0, sizeof(ws));
    ws.mf.fd = -1;
    ws.layer_idx = 3;

    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [--layer N]\n", argv[0]);
        return 1;
    }
    for (argi = 2; argi < argc; ++argi) {
        if (strcmp(argv[argi], "--layer") == 0 && argi + 1 < argc) ws.layer_idx = (uint32_t)strtoul(argv[++argi], NULL, 10);
    }
    if (!qwen36_gguf_open(&ws.gf, argv[1], err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    if (!bind_contract_model(&ws.gf, &ws.model, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        qwen36_gguf_close(&ws.gf);
        return 1;
    }
    if (ws.layer_idx >= QWEN36_35A3B_Q8_BLOCK_COUNT) {
        fprintf(stderr, "bad layer index\n");
        qwen36_gguf_close(&ws.gf);
        return 1;
    }
    ws.layer = &ws.model.layers[ws.layer_idx];
    if (ws.layer->kind != QWEN36_CONTRACT_LAYER_KIND_FULL_ATTENTION) {
        fprintf(stderr, "layer %u is not full attention\n", ws.layer_idx);
        qwen36_gguf_close(&ws.gf);
        return 1;
    }
    if (!load_small_cache(&ws.gf, ws.layer, &ws.cache, err, sizeof(err))) {
        fprintf(stderr, "cache load failed: %s\n", err);
        qwen36_gguf_close(&ws.gf);
        return 1;
    }
    if (!mapped_file_open(&ws.mf, argv[1])) {
        fprintf(stderr, "failed to mmap gguf: %s\n", strerror(errno));
        free_small_cache(&ws.cache);
        qwen36_gguf_close(&ws.gf);
        return 1;
    }
    if (ds4_gpu_init() == 0) {
        fprintf(stderr, "ds4_gpu_init failed\n");
        mapped_file_close(&ws.mf);
        free_small_cache(&ws.cache);
        qwen36_gguf_close(&ws.gf);
        return 1;
    }
    (void)ds4_gpu_set_model_map(ws.mf.map, ws.mf.size);
    (void)ds4_gpu_set_model_fd_for_map(ws.mf.fd, ws.mf.map);
    fprintf(stderr, "qwen36_full layer=%u startup\n", ws.layer_idx);
    ds4_gpu_print_memory_report("full_worker_ready");

    printf("READY\n");
    fflush(stdout);

    while (fgets(line, sizeof(line), stdin)) {
        char *cmd;
        char *rest;
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        if (len == 0) continue;
        cmd = strtok(line, " ");
        if (!cmd) continue;
        rest = cmd + strlen(cmd) + 1;
        if (rest >= line + len) rest = line + len;

        if (strcmp(cmd, "PREFILL_SEQ") == 0) {
            handle_prefill_seq(&ws, rest);
        } else if (strcmp(cmd, "PREFILL_SEQ_BIN") == 0) {
            handle_prefill_seq_bin(&ws, rest);
        } else if (strcmp(cmd, "STEP_ROW") == 0) {
            handle_step_row(&ws, rest);
        } else if (strcmp(cmd, "STEP_ROW_BIN") == 0) {
            handle_step_row_bin(&ws, rest);
        } else if (strcmp(cmd, "DUMP_HIDDEN") == 0) {
            handle_dump_hidden(&ws);
        } else if (strcmp(cmd, "DUMP_LAST") == 0) {
            handle_dump_last(&ws);
        } else if (strcmp(cmd, "RESET") == 0) {
            worker_reset(&ws);
            printf("OK\n");
            fflush(stdout);
        } else if (strcmp(cmd, "QUIT") == 0) {
            printf("OK\n");
            fflush(stdout);
            break;
        } else {
            printf("ERROR unknown command\n");
            fflush(stdout);
        }
    }

    worker_reset(&ws);
    free_small_cache(&ws.cache);
    mapped_file_close(&ws.mf);
    qwen36_gguf_close(&ws.gf);
    return 0;
}
