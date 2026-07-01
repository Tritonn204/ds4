#include "qwen36_35a3b_q8.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROUTER_COUNT 256
#define INTER 512
#define ROTARY_DIM 64
#define NUM_HEADS 16
#define NUM_KV_HEADS 2
#define HEAD_DIM 256
#define HIDDEN 2048
#define ATTN_GATE_DIM (NUM_HEADS * HEAD_DIM)

typedef struct expert_cache_entry {
    int loaded;
    float *gate;
    float *up;
    float *down;
} expert_cache_entry;

typedef struct full_layer_cache {
    int loaded;
    float *attn_norm_w;
    float *post_attn_norm_w;
    float *q_proj_w;
    float *k_proj_w;
    float *v_proj_w;
    float *o_proj_w;
    float *q_norm_w;
    float *k_norm_w;
    float *router_w;
    float *gate_shexp_w;
    float *up_shexp_w;
    float *down_shexp_w;
    float *gate_inp_shexp_w;
    expert_cache_entry experts[ROUTER_COUNT];
} full_layer_cache;

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
    } else if (exp == 31) bits = sign | 0x7f800000u | (mant << 13);
    else bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    { union { uint32_t u; float f; } v = { bits }; return v.f; }
}

static inline float siluf_local(float x) { return x / (1.0f + expf(-x)); }
static inline float sigmoidf_local(float x) { return 1.0f / (1.0f + expf(-x)); }

static void matvec(const float *mat, const float *vec, float *out, uint32_t rows, uint32_t cols) {
    for (uint32_t r = 0; r < rows; ++r) {
        const float *row = mat + (size_t)r * cols;
        double sum = 0.0;
        for (uint32_t c = 0; c < cols; ++c) sum += (double)row[c] * vec[c];
        out[r] = (float)sum;
    }
}

static void topk_softmax256(const float *logits, uint32_t k, uint32_t *idx, float *scores) {
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

static int decode_q8_rows(const qwen36_gguf_file *gf, const qwen36_gguf_tensor *t, uint32_t row_idx, uint32_t nrows, uint32_t row_elems, float *out, char *err, size_t err_cap) {
    const size_t row_size = 34u * (row_elems / 32u);
    uint8_t *buf = (uint8_t *)malloc((size_t)nrows * row_size);
    if (!buf) return 0;
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

static void rotate_half(const float *x, float *out, uint32_t dim) {
    uint32_t half = dim / 2;
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

static int ensure_expert_loaded(const qwen36_gguf_file *gf, const qwen36_35a3b_q8_layer *layer, full_layer_cache *cache, uint32_t expert_id, char *err, size_t err_cap) {
    expert_cache_entry *e = &cache->experts[expert_id];
    if (e->loaded) return 1;
    e->gate = (float *)malloc((size_t)INTER * HIDDEN * sizeof(float));
    e->up = (float *)malloc((size_t)INTER * HIDDEN * sizeof(float));
    e->down = (float *)malloc((size_t)HIDDEN * INTER * sizeof(float));
    if (!e->gate || !e->up || !e->down) return 0;
    if (!decode_q8_rows(gf, layer->ffn_gate_exps, expert_id * INTER, INTER, HIDDEN, e->gate, err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->ffn_up_exps, expert_id * INTER, INTER, HIDDEN, e->up, err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->ffn_down_exps, expert_id * HIDDEN, HIDDEN, INTER, e->down, err, err_cap)) return 0;
    e->loaded = 1;
    return 1;
}

static int ensure_layer_loaded(const qwen36_gguf_file *gf, const qwen36_35a3b_q8_layer *layer, full_layer_cache *cache, char *err, size_t err_cap) {
    if (cache->loaded) return 1;
    if (!read_f32_tensor(gf, layer->attn_norm, &cache->attn_norm_w, HIDDEN, err, err_cap)) return 0;
    if (!read_f32_tensor(gf, layer->post_attn_norm, &cache->post_attn_norm_w, HIDDEN, err, err_cap)) return 0;
    if (!read_f32_tensor(gf, layer->attn_q_norm, &cache->q_norm_w, HEAD_DIM, err, err_cap)) return 0;
    if (!read_f32_tensor(gf, layer->attn_k_norm, &cache->k_norm_w, HEAD_DIM, err, err_cap)) return 0;
    if (!read_f32_tensor(gf, layer->ffn_gate_inp, &cache->router_w, (size_t)ROUTER_COUNT * HIDDEN, err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->ffn_gate_shexp, 0, INTER, HIDDEN, (cache->gate_shexp_w = (float *)malloc((size_t)INTER * HIDDEN * sizeof(float))), err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->ffn_up_shexp, 0, INTER, HIDDEN, (cache->up_shexp_w = (float *)malloc((size_t)INTER * HIDDEN * sizeof(float))), err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->ffn_down_shexp, 0, HIDDEN, INTER, (cache->down_shexp_w = (float *)malloc((size_t)HIDDEN * INTER * sizeof(float))), err, err_cap)) return 0;
    if (!read_f32_tensor(gf, layer->ffn_gate_inp_shexp, &cache->gate_inp_shexp_w, HIDDEN, err, err_cap)) return 0;
    cache->q_proj_w = (float *)malloc((size_t)ATTN_GATE_DIM * 2u * HIDDEN * sizeof(float));
    cache->k_proj_w = (float *)malloc((size_t)NUM_KV_HEADS * HEAD_DIM * HIDDEN * sizeof(float));
    cache->v_proj_w = (float *)malloc((size_t)NUM_KV_HEADS * HEAD_DIM * HIDDEN * sizeof(float));
    cache->o_proj_w = (float *)malloc((size_t)HIDDEN * ATTN_GATE_DIM * sizeof(float));
    if (!cache->q_proj_w || !cache->k_proj_w || !cache->v_proj_w || !cache->o_proj_w) return 0;
    if (!decode_q8_rows(gf, layer->attn_q, 0, ATTN_GATE_DIM * 2u, HIDDEN, cache->q_proj_w, err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->attn_k, 0, NUM_KV_HEADS * HEAD_DIM, HIDDEN, cache->k_proj_w, err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->attn_v, 0, NUM_KV_HEADS * HEAD_DIM, HIDDEN, cache->v_proj_w, err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->attn_output, 0, HIDDEN, ATTN_GATE_DIM, cache->o_proj_w, err, err_cap)) return 0;
    cache->loaded = 1;
    return 1;
}

static void free_layer_cache(full_layer_cache *cache) {
    if (!cache) return;
    free(cache->attn_norm_w);
    free(cache->post_attn_norm_w);
    free(cache->q_proj_w);
    free(cache->k_proj_w);
    free(cache->v_proj_w);
    free(cache->o_proj_w);
    free(cache->q_norm_w);
    free(cache->k_norm_w);
    free(cache->router_w);
    free(cache->gate_shexp_w);
    free(cache->up_shexp_w);
    free(cache->down_shexp_w);
    free(cache->gate_inp_shexp_w);
    for (uint32_t i = 0; i < ROUTER_COUNT; ++i) {
        free(cache->experts[i].gate);
        free(cache->experts[i].up);
        free(cache->experts[i].down);
    }
    memset(cache, 0, sizeof(*cache));
}

int main(int argc, char **argv) {
    const char *gguf_path, *in_seq_path, *out_seq_path;
    uint32_t layer_idx = 3;
    qwen36_gguf_file gf;
    qwen36_35a3b_q8_model q8;
    const qwen36_35a3b_q8_layer *layer;
    full_layer_cache cache = {0};
    char err[512];
    FILE *fp = NULL;
    float *input_seq = NULL, *output_seq = NULL;
    long sz = 0;
    uint32_t seq_len;
    int argi;
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
    if (!qwen36_gguf_open(&gf, gguf_path, err, sizeof(err))) { fprintf(stderr, "%s\n", err); return 1; }
    if (!qwen36_35a3b_q8_bind(&gf, &q8, err, sizeof(err))) { fprintf(stderr, "%s\n", err); qwen36_gguf_close(&gf); return 1; }
    if (layer_idx >= QWEN36_35A3B_Q8_BLOCK_COUNT) { fprintf(stderr, "bad layer index\n"); qwen36_gguf_close(&gf); return 1; }
    layer = &q8.layers[layer_idx];
    if (layer->kind != QWEN36_LAYER_KIND_FULL_ATTENTION) { fprintf(stderr, "layer %u is not full attention\n", layer_idx); qwen36_gguf_close(&gf); return 1; }
    if (!ensure_layer_loaded(&gf, layer, &cache, err, sizeof(err))) { fprintf(stderr, "load layer failed: %s\n", err); free_layer_cache(&cache); qwen36_gguf_close(&gf); return 1; }
    fp = fopen(in_seq_path, "rb");
    if (!fp) { perror("open input"); free_layer_cache(&cache); qwen36_gguf_close(&gf); return 1; }
    if (fseek(fp, 0, SEEK_END) != 0 || (sz = ftell(fp)) < 0 || fseek(fp, 0, SEEK_SET) != 0) { perror("stat input"); fclose(fp); free_layer_cache(&cache); qwen36_gguf_close(&gf); return 1; }
    seq_len = (uint32_t)((size_t)sz / (HIDDEN * sizeof(float)));
    if ((size_t)sz != (size_t)seq_len * HIDDEN * sizeof(float)) { fprintf(stderr, "bad input size\n"); fclose(fp); free_layer_cache(&cache); qwen36_gguf_close(&gf); return 1; }
    input_seq = (float *)malloc((size_t)seq_len * HIDDEN * sizeof(float));
    output_seq = (float *)calloc((size_t)seq_len * HIDDEN, sizeof(float));
    if (!input_seq || !output_seq) { fclose(fp); free(input_seq); free(output_seq); free_layer_cache(&cache); qwen36_gguf_close(&gf); return 1; }
    if (fread(input_seq, sizeof(float), (size_t)seq_len * HIDDEN, fp) != (size_t)seq_len * HIDDEN) { perror("read input"); fclose(fp); free(input_seq); free(output_seq); free_layer_cache(&cache); qwen36_gguf_close(&gf); return 1; }
    fclose(fp);

    {
        float *attn_in = (float *)calloc((size_t)seq_len * HIDDEN, sizeof(float));
        float *q_all = (float *)calloc((size_t)seq_len * NUM_HEADS * HEAD_DIM, sizeof(float));
        float *k_all = (float *)calloc((size_t)seq_len * NUM_KV_HEADS * HEAD_DIM, sizeof(float));
        float *v_all = (float *)calloc((size_t)seq_len * NUM_KV_HEADS * HEAD_DIM, sizeof(float));
        float *gate_all = (float *)calloc((size_t)seq_len * ATTN_GATE_DIM, sizeof(float));
        float *attn_out_flat = (float *)calloc((size_t)seq_len * ATTN_GATE_DIM, sizeof(float));
        float *proj_out = (float *)calloc((size_t)seq_len * HIDDEN, sizeof(float));
        float *post_ln = (float *)calloc((size_t)seq_len * HIDDEN, sizeof(float));
        float router_logits[ROUTER_COUNT];
        uint32_t router_idx[8];
        float router_scores[8];
        float *gate = (float *)calloc(INTER, sizeof(float));
        float *up = (float *)calloc(INTER, sizeof(float));
        float *act = (float *)calloc(INTER, sizeof(float));
        float *down = (float *)calloc(HIDDEN, sizeof(float));
        float *shared_gate = (float *)calloc(INTER, sizeof(float));
        float *shared_up = (float *)calloc(INTER, sizeof(float));
        float *shared_act = (float *)calloc(INTER, sizeof(float));
        float *shared = (float *)calloc(HIDDEN, sizeof(float));
        float qg[ATTN_GATE_DIM * 2u];
        float kk[NUM_KV_HEADS * HEAD_DIM];
        float vv[NUM_KV_HEADS * HEAD_DIM];
        if (!attn_in || !q_all || !k_all || !v_all || !gate_all || !attn_out_flat || !proj_out || !post_ln ||
            !gate || !up || !act || !down || !shared_gate || !shared_up || !shared_act || !shared) {
            fprintf(stderr, "oom\n");
            free(attn_in); free(q_all); free(k_all); free(v_all); free(gate_all); free(attn_out_flat); free(proj_out); free(post_ln);
            free(gate); free(up); free(act); free(down); free(shared_gate); free(shared_up); free(shared_act); free(shared);
            free(input_seq); free(output_seq); free_layer_cache(&cache); qwen36_gguf_close(&gf);
            return 1;
        }
        for (uint32_t t = 0; t < seq_len; ++t) {
            rmsnorm_weight_raw(input_seq + (size_t)t * HIDDEN, attn_in + (size_t)t * HIDDEN, cache.attn_norm_w, HIDDEN);
            matvec(cache.q_proj_w, attn_in + (size_t)t * HIDDEN, qg, ATTN_GATE_DIM * 2u, HIDDEN);
            matvec(cache.k_proj_w, attn_in + (size_t)t * HIDDEN, kk, NUM_KV_HEADS * HEAD_DIM, HIDDEN);
            matvec(cache.v_proj_w, attn_in + (size_t)t * HIDDEN, vv, NUM_KV_HEADS * HEAD_DIM, HIDDEN);
            for (uint32_t h = 0; h < NUM_HEADS; ++h) {
                rmsnorm_weight_plus1(qg + (size_t)h * (HEAD_DIM * 2u), q_all + ((size_t)t * NUM_HEADS + h) * HEAD_DIM, cache.q_norm_w, HEAD_DIM);
                memcpy(gate_all + (size_t)t * ATTN_GATE_DIM + (size_t)h * HEAD_DIM, qg + (size_t)h * (HEAD_DIM * 2u) + HEAD_DIM, HEAD_DIM * sizeof(float));
            }
            for (uint32_t h = 0; h < NUM_KV_HEADS; ++h) {
                rmsnorm_weight_plus1(kk + (size_t)h * HEAD_DIM, k_all + ((size_t)t * NUM_KV_HEADS + h) * HEAD_DIM, cache.k_norm_w, HEAD_DIM);
                memcpy(v_all + ((size_t)t * NUM_KV_HEADS + h) * HEAD_DIM, vv + (size_t)h * HEAD_DIM, HEAD_DIM * sizeof(float));
            }
            for (uint32_t h = 0; h < NUM_HEADS; ++h) {
                apply_rope_inplace(q_all + ((size_t)t * NUM_HEADS + h) * HEAD_DIM,
                                   k_all + ((size_t)t * NUM_KV_HEADS + (h / (NUM_HEADS / NUM_KV_HEADS))) * HEAD_DIM,
                                   t);
            }
        }
        for (uint32_t t = 0; t < seq_len; ++t) {
            for (uint32_t h = 0; h < NUM_HEADS; ++h) {
                const uint32_t kvh = h / (NUM_HEADS / NUM_KV_HEADS);
                float scores[4096];
                float maxv = -1e30f;
                float sum = 0.0f;
                float *out = attn_out_flat + (size_t)t * ATTN_GATE_DIM + (size_t)h * HEAD_DIM;
                memset(out, 0, HEAD_DIM * sizeof(float));
                for (uint32_t s = 0; s <= t; ++s) {
                    double dot = 0.0;
                    const float *qv = q_all + ((size_t)t * NUM_HEADS + h) * HEAD_DIM;
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
            matvec(cache.o_proj_w, attn_out_flat + (size_t)t * ATTN_GATE_DIM, proj_out + (size_t)t * HIDDEN, HIDDEN, ATTN_GATE_DIM);
            for (uint32_t d = 0; d < HIDDEN; ++d) output_seq[(size_t)t * HIDDEN + d] = input_seq[(size_t)t * HIDDEN + d] + proj_out[(size_t)t * HIDDEN + d];
            rmsnorm_weight_raw(output_seq + (size_t)t * HIDDEN, post_ln + (size_t)t * HIDDEN, cache.post_attn_norm_w, HIDDEN);
            memset(output_seq + (size_t)t * HIDDEN, 0, HIDDEN * sizeof(float));
            matvec(cache.router_w, post_ln + (size_t)t * HIDDEN, router_logits, ROUTER_COUNT, HIDDEN);
            topk_softmax256(router_logits, 8, router_idx, router_scores);
            for (uint32_t i = 0; i < 8; ++i) {
                if (!ensure_expert_loaded(&gf, layer, &cache, router_idx[i], err, sizeof(err))) {
                    fprintf(stderr, "expert load failed: %s\n", err);
                    free(attn_in); free(q_all); free(k_all); free(v_all); free(gate_all); free(attn_out_flat); free(proj_out); free(post_ln);
                    free(gate); free(up); free(act); free(down); free(shared_gate); free(shared_up); free(shared_act); free(shared);
                    free(input_seq); free(output_seq); free_layer_cache(&cache); qwen36_gguf_close(&gf);
                    return 1;
                }
                matvec(cache.experts[router_idx[i]].gate, post_ln + (size_t)t * HIDDEN, gate, INTER, HIDDEN);
                matvec(cache.experts[router_idx[i]].up, post_ln + (size_t)t * HIDDEN, up, INTER, HIDDEN);
                for (uint32_t vd = 0; vd < INTER; ++vd) act[vd] = siluf_local(gate[vd]) * up[vd];
                matvec(cache.experts[router_idx[i]].down, act, down, HIDDEN, INTER);
                for (uint32_t d = 0; d < HIDDEN; ++d) output_seq[(size_t)t * HIDDEN + d] += down[d] * router_scores[i];
            }
            matvec(cache.gate_shexp_w, post_ln + (size_t)t * HIDDEN, shared_gate, INTER, HIDDEN);
            matvec(cache.up_shexp_w, post_ln + (size_t)t * HIDDEN, shared_up, INTER, HIDDEN);
            for (uint32_t vd = 0; vd < INTER; ++vd) shared_act[vd] = siluf_local(shared_gate[vd]) * shared_up[vd];
            matvec(cache.down_shexp_w, shared_act, shared, HIDDEN, INTER);
            {
                float s = 0.0f;
                for (uint32_t d = 0; d < HIDDEN; ++d) s += post_ln[(size_t)t * HIDDEN + d] * cache.gate_inp_shexp_w[d];
                s = sigmoidf_local(s);
                for (uint32_t d = 0; d < HIDDEN; ++d) output_seq[(size_t)t * HIDDEN + d] = input_seq[(size_t)t * HIDDEN + d] + proj_out[(size_t)t * HIDDEN + d] + output_seq[(size_t)t * HIDDEN + d] + shared[d] * s;
            }
        }
        free(attn_in); free(q_all); free(k_all); free(v_all); free(gate_all); free(attn_out_flat); free(proj_out); free(post_ln);
        free(gate); free(up); free(act); free(down); free(shared_gate); free(shared_up); free(shared_act); free(shared);
    }
    fp = fopen(out_seq_path, "wb");
    if (!fp) { perror("open output"); free(input_seq); free(output_seq); free_layer_cache(&cache); qwen36_gguf_close(&gf); return 1; }
    fwrite(output_seq, sizeof(float), (size_t)seq_len * HIDDEN, fp);
    fclose(fp);
    printf("layer: %u\n", layer_idx);
    printf("seq_len: %u\n", seq_len);
    printf("wrote_seq_hidden: %s\n", out_seq_path);
    free(input_seq); free(output_seq); free_layer_cache(&cache); qwen36_gguf_close(&gf);
    return 0;
}
