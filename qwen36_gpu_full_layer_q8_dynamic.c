#include "ds4_gpu.h"
#include "qwen36_35a3b_q8.h"

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
#define INTER 512
#define ROTARY_DIM 64
#define NUM_HEADS 16
#define NUM_KV_HEADS 2
#define HEAD_DIM 256
#define HIDDEN 2048
#define ATTN_GATE_DIM (NUM_HEADS * HEAD_DIM)
#define QWEN_RMS_EPS 1e-6f
#define QWEN_SWIGLU_CLAMP 80.0f

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

static int load_small_cache(const qwen36_gguf_file *gf, const qwen36_35a3b_q8_layer *layer, full_layer_small_cache *cache, char *err, size_t err_cap) {
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

static uint64_t q8_row_bytes(uint32_t cols) {
    return (uint64_t)(cols / 32u) * 34u;
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
        for (uint32_t i = 0; i < ROTARY_DIM / 2; ++i) {
            inv_freq[i] = powf(10000000.0f, -(float)(2 * i) / (float)ROTARY_DIM);
        }
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

static int run_gpu_ffn_from_residual(
        const mapped_file *mf,
        const qwen36_35a3b_q8_layer *layer,
        const float *shared_gate_inp_w,
        uint32_t n_tokens,
        const float *residual_in,
        float *out_seq) {
    const uint32_t hidden = HIDDEN;
    const uint32_t topk = 8u;
    const uint32_t shared_dim = INTER;
    const uint64_t seq_hidden_bytes = (uint64_t)n_tokens * hidden * sizeof(float);
    const uint64_t router_logits_bytes = (uint64_t)n_tokens * ROUTER_COUNT * sizeof(float);
    const uint64_t shared_bytes = (uint64_t)n_tokens * shared_dim * sizeof(float);
    const uint64_t gate_row_bytes = q8_row_bytes(hidden);
    const uint64_t gate_expert_bytes = gate_row_bytes * INTER;
    const uint64_t down_row_bytes = q8_row_bytes(INTER);
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

    for (uint32_t t = 0; t < n_tokens; ++t) {
        topk_softmax256(router_logits_cpu + (size_t)t * ROUTER_COUNT,
                        topk,
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

    for (uint32_t u = 0; u < n_union; ++u) {
        const uint32_t expert_id = (uint32_t)union_ids[u];
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
        if (ds4_gpu_tensor_read(expert_down, 0, expert_down_cpu, seq_hidden_bytes) == 0) goto cleanup;
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
    ok = 1;

cleanup:
    if (!ok) ds4_gpu_print_memory_report("qwen36_gpu_full_layer_q8_dynamic_ffn_failure");
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

int main(int argc, char **argv) {
    const char *gguf_path, *in_seq_path, *out_seq_path;
    uint32_t layer_idx = 3;
    qwen36_gguf_file gf;
    qwen36_35a3b_q8_model q8;
    const qwen36_35a3b_q8_layer *layer;
    full_layer_small_cache cache;
    mapped_file mf;
    char err[512];
    FILE *fp = NULL;
    float *input_seq = NULL, *output_seq = NULL;
    float *attn_in = NULL, *qg = NULL, *kk = NULL, *vv = NULL;
    float *q_all = NULL, *k_all = NULL, *v_all = NULL, *gate_all = NULL;
    float *attn_out_flat = NULL, *proj_out = NULL, *residual = NULL;
    long sz = 0;
    uint32_t seq_len;
    int argi;
    int ok = 0;
    ds4_gpu_tensor *input_gpu = NULL, *attn_in_gpu = NULL, *qg_gpu = NULL, *k_gpu = NULL, *v_gpu = NULL;
    ds4_gpu_tensor *attn_out_gpu = NULL, *proj_out_gpu = NULL;

    memset(&gf, 0, sizeof(gf));
    memset(&cache, 0, sizeof(cache));
    memset(&mf, 0, sizeof(mf));
    mf.fd = -1;

    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL.gguf INPUT_SEQ.f32 OUTPUT_SEQ.f32 [--layer N]\n", argv[0]);
        return 1;
    }
    gguf_path = argv[1];
    in_seq_path = argv[2];
    out_seq_path = argv[3];
    for (argi = 4; argi < argc; ++argi) {
        if (strcmp(argv[argi], "--layer") == 0 && argi + 1 < argc) layer_idx = (uint32_t)strtoul(argv[++argi], NULL, 10);
    }
    if (!qwen36_gguf_open(&gf, gguf_path, err, sizeof(err))) { fprintf(stderr, "%s\n", err); goto cleanup; }
    if (!qwen36_35a3b_q8_bind(&gf, &q8, err, sizeof(err))) { fprintf(stderr, "%s\n", err); goto cleanup; }
    if (layer_idx >= QWEN36_35A3B_Q8_BLOCK_COUNT) { fprintf(stderr, "bad layer index\n"); goto cleanup; }
    layer = &q8.layers[layer_idx];
    if (layer->kind != QWEN36_LAYER_KIND_FULL_ATTENTION) {
        fprintf(stderr, "layer %u is not full attention\n", layer_idx);
        goto cleanup;
    }
    if (!load_small_cache(&gf, layer, &cache, err, sizeof(err))) {
        fprintf(stderr, "cache load failed: %s\n", err);
        goto cleanup;
    }
    if (!mapped_file_open(&mf, gguf_path)) {
        fprintf(stderr, "failed to mmap gguf: %s\n", strerror(errno));
        goto cleanup;
    }
    if (ds4_gpu_init() == 0) {
        fprintf(stderr, "ds4_gpu_init failed\n");
        goto cleanup;
    }
    (void)ds4_gpu_set_model_map(mf.map, mf.size);
    (void)ds4_gpu_set_model_fd_for_map(mf.fd, mf.map);

    fp = fopen(in_seq_path, "rb");
    if (!fp) { perror("open input"); goto cleanup; }
    if (fseek(fp, 0, SEEK_END) != 0 || (sz = ftell(fp)) < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        perror("stat input");
        goto cleanup;
    }
    seq_len = (uint32_t)((size_t)sz / (HIDDEN * sizeof(float)));
    if ((size_t)sz != (size_t)seq_len * HIDDEN * sizeof(float)) {
        fprintf(stderr, "bad input size\n");
        goto cleanup;
    }

    input_seq = (float *)malloc((size_t)seq_len * HIDDEN * sizeof(float));
    output_seq = (float *)calloc((size_t)seq_len * HIDDEN, sizeof(float));
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
    if (!input_seq || !output_seq || !attn_in || !qg || !kk || !vv || !q_all || !k_all || !v_all ||
        !gate_all || !attn_out_flat || !proj_out || !residual) {
        fprintf(stderr, "oom\n");
        goto cleanup;
    }
    if (fread(input_seq, sizeof(float), (size_t)seq_len * HIDDEN, fp) != (size_t)seq_len * HIDDEN) {
        perror("read input");
        goto cleanup;
    }
    fclose(fp);
    fp = NULL;

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
    if (ds4_gpu_rms_norm_weight_rows_tensor(attn_in_gpu, input_gpu, mf.map, mf.size,
                                            layer->attn_norm->abs_offset,
                                            HIDDEN, seq_len, QWEN_RMS_EPS) == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(qg_gpu, mf.map, mf.size,
                                   layer->attn_q->abs_offset,
                                   HIDDEN, ATTN_GATE_DIM * 2u, attn_in_gpu, seq_len) == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(k_gpu, mf.map, mf.size,
                                   layer->attn_k->abs_offset,
                                   HIDDEN, NUM_KV_HEADS * HEAD_DIM, attn_in_gpu, seq_len) == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(v_gpu, mf.map, mf.size,
                                   layer->attn_v->abs_offset,
                                   HIDDEN, NUM_KV_HEADS * HEAD_DIM, attn_in_gpu, seq_len) == 0) goto cleanup;
    if (ds4_gpu_end_commands() == 0) goto cleanup;

    if (ds4_gpu_tensor_read(attn_in_gpu, 0, attn_in, (uint64_t)seq_len * HIDDEN * sizeof(float)) == 0) goto cleanup;
    if (ds4_gpu_tensor_read(qg_gpu, 0, qg, (uint64_t)seq_len * ATTN_GATE_DIM * 2u * sizeof(float)) == 0) goto cleanup;
    if (ds4_gpu_tensor_read(k_gpu, 0, kk, (uint64_t)seq_len * NUM_KV_HEADS * HEAD_DIM * sizeof(float)) == 0) goto cleanup;
    if (ds4_gpu_tensor_read(v_gpu, 0, vv, (uint64_t)seq_len * NUM_KV_HEADS * HEAD_DIM * sizeof(float)) == 0) goto cleanup;

    for (uint32_t t = 0; t < seq_len; ++t) {
        for (uint32_t h = 0; h < NUM_HEADS; ++h) {
            const float *src = qg + (size_t)t * (ATTN_GATE_DIM * 2u) + (size_t)h * (HEAD_DIM * 2u);
            float *qdst = q_all + ((size_t)t * NUM_HEADS + h) * HEAD_DIM;
            rmsnorm_weight_plus1(src, qdst, cache.q_norm_w, HEAD_DIM);
            memcpy(gate_all + (size_t)t * ATTN_GATE_DIM + (size_t)h * HEAD_DIM,
                   src + HEAD_DIM,
                   HEAD_DIM * sizeof(float));
        }
        for (uint32_t h = 0; h < NUM_KV_HEADS; ++h) {
            rmsnorm_weight_plus1(kk + (size_t)t * NUM_KV_HEADS * HEAD_DIM + (size_t)h * HEAD_DIM,
                                 k_all + ((size_t)t * NUM_KV_HEADS + h) * HEAD_DIM,
                                 cache.k_norm_w,
                                 HEAD_DIM);
            memcpy(v_all + ((size_t)t * NUM_KV_HEADS + h) * HEAD_DIM,
                   vv + (size_t)t * NUM_KV_HEADS * HEAD_DIM + (size_t)h * HEAD_DIM,
                   HEAD_DIM * sizeof(float));
        }
        for (uint32_t h = 0; h < NUM_KV_HEADS; ++h) {
            apply_rope_one_inplace(k_all + ((size_t)t * NUM_KV_HEADS + h) * HEAD_DIM, t);
        }
        for (uint32_t h = 0; h < NUM_HEADS; ++h) {
            apply_rope_one_inplace(q_all + ((size_t)t * NUM_HEADS + h) * HEAD_DIM, t);
        }
    }

    for (uint32_t t = 0; t < seq_len; ++t) {
        for (uint32_t h = 0; h < NUM_HEADS; ++h) {
            const uint32_t kvh = h / (NUM_HEADS / NUM_KV_HEADS);
            float scores[4096];
            float maxv = -1e30f;
            float sum = 0.0f;
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
            for (uint32_t d = 0; d < HEAD_DIM; ++d) {
                out[d] *= sigmoidf_local(gate_all[(size_t)t * ATTN_GATE_DIM + (size_t)h * HEAD_DIM + d]);
            }
        }
    }

    if (ds4_gpu_tensor_write(attn_out_gpu, 0, attn_out_flat, (uint64_t)seq_len * ATTN_GATE_DIM * sizeof(float)) == 0) goto cleanup;
    if (ds4_gpu_begin_commands() == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(proj_out_gpu, mf.map, mf.size,
                                   layer->attn_output->abs_offset,
                                   ATTN_GATE_DIM, HIDDEN, attn_out_gpu, seq_len) == 0) goto cleanup;
    if (ds4_gpu_end_commands() == 0) goto cleanup;
    if (ds4_gpu_tensor_read(proj_out_gpu, 0, proj_out, (uint64_t)seq_len * HIDDEN * sizeof(float)) == 0) goto cleanup;

    for (uint32_t t = 0; t < seq_len; ++t) {
        for (uint32_t d = 0; d < HIDDEN; ++d) {
            const size_t idx = (size_t)t * HIDDEN + d;
            residual[idx] = input_seq[idx] + proj_out[idx];
        }
    }

    if (!run_gpu_ffn_from_residual(&mf, layer, cache.shared_gate_inp_w, seq_len, residual, output_seq)) goto cleanup;

    fp = fopen(out_seq_path, "wb");
    if (!fp) { perror("open output"); goto cleanup; }
    fwrite(output_seq, sizeof(float), (size_t)seq_len * HIDDEN, fp);
    fclose(fp);
    fp = NULL;
    printf("layer: %u\n", layer_idx);
    printf("seq_len: %u\n", seq_len);
    printf("wrote_seq_hidden: %s\n", out_seq_path);
    ok = 1;

cleanup:
    if (fp) fclose(fp);
    ds4_gpu_tensor_free(proj_out_gpu);
    ds4_gpu_tensor_free(attn_out_gpu);
    ds4_gpu_tensor_free(v_gpu);
    ds4_gpu_tensor_free(k_gpu);
    ds4_gpu_tensor_free(qg_gpu);
    ds4_gpu_tensor_free(attn_in_gpu);
    ds4_gpu_tensor_free(input_gpu);
    free(input_seq);
    free(output_seq);
    free(attn_in);
    free(qg);
    free(kk);
    free(vv);
    free(q_all);
    free(k_all);
    free(v_all);
    free(gate_all);
    free(attn_out_flat);
    free(proj_out);
    free(residual);
    free_small_cache(&cache);
    mapped_file_close(&mf);
    qwen36_gguf_close(&gf);
    return ok ? 0 : 1;
}
