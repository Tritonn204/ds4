#include "qwen36_35a3b_q8.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "Q36LCF01"
#define MAGIC_LEN 8
#define PREFIX_MAGIC "Q36PFX01"
#define PREFIX_MAGIC_LEN 8
#define ROUTER_COUNT 256

typedef struct prefix_fixture {
    uint32_t seq_len;
    uint32_t hidden;
    uint32_t *token_ids;
    float *input_seq_ref;
} prefix_fixture;

typedef struct live_fixture {
    uint32_t layer, hidden, num_v_heads, num_k_heads, head_k_dim, head_v_dim, key_dim, value_dim, topk, inter;
    float *attn_norm_w, *post_attn_norm_w, *w_qkv, *w_z, *w_a, *w_b, *conv_w, *A_log, *dt_bias, *ssm_norm_w, *w_out;
    float *router_w, *gate_shexp, *up_shexp, *down_shexp, *gate_inp_shexp;
} live_fixture;

typedef struct expert_cache_entry {
    int loaded;
    float *gate;
    float *up;
    float *down;
} expert_cache_entry;

static int read_exact(FILE *fp, void *buf, size_t n) { return fread(buf, 1, n, fp) == n; }
static int alloc_read_f32(FILE *fp, float **out, size_t count) {
    float *p = (float *)malloc(count * sizeof(float));
    if (!p) return 0;
    if (!read_exact(fp, p, count * sizeof(float))) { free(p); return 0; }
    *out = p;
    return 1;
}
static int alloc_read_u32(FILE *fp, uint32_t **out, size_t count) {
    uint32_t *p = (uint32_t *)malloc(count * sizeof(uint32_t));
    if (!p) return 0;
    if (!read_exact(fp, p, count * sizeof(uint32_t))) { free(p); return 0; }
    *out = p;
    return 1;
}

static void prefix_free(prefix_fixture *fx) {
    if (!fx) return;
    free(fx->token_ids);
    free(fx->input_seq_ref);
    memset(fx, 0, sizeof(*fx));
}

static int prefix_load(const char *path, prefix_fixture *fx) {
    FILE *fp = fopen(path, "rb");
    char magic[PREFIX_MAGIC_LEN];
    memset(fx, 0, sizeof(*fx));
    if (!fp) return 0;
    if (!read_exact(fp, magic, PREFIX_MAGIC_LEN) || memcmp(magic, PREFIX_MAGIC, PREFIX_MAGIC_LEN) != 0) { fclose(fp); return 0; }
    if (!read_exact(fp, &fx->seq_len, sizeof(uint32_t)) || !read_exact(fp, &fx->hidden, sizeof(uint32_t))) { fclose(fp); return 0; }
    if (!alloc_read_u32(fp, &fx->token_ids, fx->seq_len) ||
        !alloc_read_f32(fp, &fx->input_seq_ref, (size_t)fx->seq_len * fx->hidden)) {
        fclose(fp); prefix_free(fx); return 0;
    }
    fclose(fp);
    return 1;
}

static void live_fixture_free(live_fixture *fx) {
    if (!fx) return;
    free(fx->attn_norm_w); free(fx->post_attn_norm_w); free(fx->w_qkv); free(fx->w_z); free(fx->w_a); free(fx->w_b);
    free(fx->conv_w); free(fx->A_log); free(fx->dt_bias); free(fx->ssm_norm_w); free(fx->w_out);
    free(fx->router_w); free(fx->gate_shexp); free(fx->up_shexp); free(fx->down_shexp); free(fx->gate_inp_shexp);
    memset(fx, 0, sizeof(*fx));
}

static int live_fixture_load(const char *path, live_fixture *fx) {
    FILE *fp = fopen(path, "rb");
    char magic[MAGIC_LEN];
    memset(fx, 0, sizeof(*fx));
    if (!fp) return 0;
    if (!read_exact(fp, magic, MAGIC_LEN) || memcmp(magic, MAGIC, MAGIC_LEN) != 0) { fclose(fp); return 0; }
    if (!read_exact(fp, &fx->layer, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->hidden, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->num_v_heads, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->num_k_heads, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->head_k_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->head_v_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->key_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->value_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->topk, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->inter, sizeof(uint32_t))) {
        fclose(fp);
        return 0;
    }
#define R(name,count) if (!alloc_read_f32(fp, &fx->name, (count))) { fclose(fp); live_fixture_free(fx); return 0; }
    R(attn_norm_w, fx->hidden);
    R(post_attn_norm_w, fx->hidden);
    R(w_qkv, (size_t)(fx->key_dim * 2 + fx->value_dim) * fx->hidden);
    R(w_z, (size_t)fx->value_dim * fx->hidden);
    R(w_a, (size_t)fx->num_v_heads * fx->hidden);
    R(w_b, (size_t)fx->num_v_heads * fx->hidden);
    R(conv_w, (size_t)(fx->key_dim * 2 + fx->value_dim) * 4);
    R(A_log, fx->num_v_heads);
    R(dt_bias, fx->num_v_heads);
    R(ssm_norm_w, fx->head_v_dim);
    R(w_out, (size_t)fx->hidden * fx->value_dim);
    R(router_w, (size_t)ROUTER_COUNT * fx->hidden);
    R(gate_shexp, (size_t)fx->inter * fx->hidden);
    R(up_shexp, (size_t)fx->inter * fx->hidden);
    R(down_shexp, (size_t)fx->hidden * fx->inter);
    R(gate_inp_shexp, fx->hidden);
#undef R
    fclose(fp);
    return 1;
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
    } else if (exp == 31) bits = sign | 0x7f800000u | (mant << 13);
    else bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    { union { uint32_t u; float f; } v = { bits }; return v.f; }
}

static inline float sigmoidf_local(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float softplusf_local(float x) { return log1pf(expf(x)); }
static inline float siluf_local(float x) { return x / (1.0f + expf(-x)); }

static void matvec(const float *mat, const float *vec, float *out, uint32_t rows, uint32_t cols) {
    uint32_t r, c;
    for (r = 0; r < rows; ++r) {
        const float *row = mat + (size_t)r * cols;
        double sum = 0.0;
        for (c = 0; c < cols; ++c) sum += (double)row[c] * vec[c];
        out[r] = (float)sum;
    }
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

static int decode_q8_row(const qwen36_gguf_file *gf, const qwen36_gguf_tensor *t, uint32_t row_idx, uint32_t hidden, float *out, char *err, size_t err_cap) {
    return decode_q8_rows(gf, t, row_idx, 1u, hidden, out, err, err_cap);
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

static void free_expert_cache(expert_cache_entry *cache) {
    for (uint32_t i = 0; i < ROUTER_COUNT; ++i) {
        free(cache[i].gate);
        free(cache[i].up);
        free(cache[i].down);
    }
}

static int ensure_expert_loaded(const qwen36_gguf_file *gf, const qwen36_35a3b_q8_layer *layer, expert_cache_entry *cache, uint32_t expert_id, uint32_t hidden, uint32_t inter, char *err, size_t err_cap) {
    expert_cache_entry *e = &cache[expert_id];
    if (e->loaded) return 1;
    e->gate = (float *)malloc((size_t)inter * hidden * sizeof(float));
    e->up = (float *)malloc((size_t)inter * hidden * sizeof(float));
    e->down = (float *)malloc((size_t)hidden * inter * sizeof(float));
    if (!e->gate || !e->up || !e->down) return 0;
    if (!decode_q8_rows(gf, layer->ffn_gate_exps, expert_id * inter, inter, hidden, e->gate, err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->ffn_up_exps, expert_id * inter, inter, hidden, e->up, err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->ffn_down_exps, expert_id * hidden, hidden, inter, e->down, err, err_cap)) return 0;
    e->loaded = 1;
    return 1;
}

static int run_layer_live(const live_fixture *fx, const qwen36_gguf_file *gf, const qwen36_35a3b_q8_layer *layer, expert_cache_entry *cache,
                          uint32_t seq_len, const float *layer_input_seq, float *out_final_seq, char *err, size_t err_cap) {
    uint32_t qkv_dim = fx->key_dim * 2 + fx->value_dim;
    uint32_t rep = fx->num_v_heads / fx->num_k_heads;
    size_t seq_hidden = (size_t)seq_len * fx->hidden;
    float *input_ln = (float *)calloc(seq_hidden, sizeof(float));
    float *qkv = (float *)calloc((size_t)seq_len * qkv_dim, sizeof(float));
    float *z = (float *)calloc((size_t)seq_len * fx->value_dim, sizeof(float));
    float *a = (float *)calloc((size_t)seq_len * fx->num_v_heads, sizeof(float));
    float *b = (float *)calloc((size_t)seq_len * fx->num_v_heads, sizeof(float));
    float *conv = (float *)calloc((size_t)seq_len * qkv_dim, sizeof(float));
    float *q = (float *)calloc((size_t)seq_len * fx->num_v_heads * fx->head_k_dim, sizeof(float));
    float *k = (float *)calloc((size_t)seq_len * fx->num_v_heads * fx->head_k_dim, sizeof(float));
    float *v = (float *)calloc((size_t)seq_len * fx->num_v_heads * fx->head_v_dim, sizeof(float));
    float *beta = (float *)calloc((size_t)seq_len * fx->num_v_heads, sizeof(float));
    float *g = (float *)calloc((size_t)seq_len * fx->num_v_heads, sizeof(float));
    float *core = (float *)calloc((size_t)seq_len * fx->num_v_heads * fx->head_v_dim, sizeof(float));
    float *state = (float *)calloc((size_t)fx->num_v_heads * fx->head_k_dim * fx->head_v_dim, sizeof(float));
    float *out_in = (float *)calloc((size_t)seq_len * fx->value_dim, sizeof(float));
    float *out_proj = (float *)calloc(seq_hidden, sizeof(float));
    float *resid_seq = (float *)calloc(seq_hidden, sizeof(float));
    float *post_attn_ln_seq = (float *)calloc(seq_hidden, sizeof(float));
    float *mlp_seq = (float *)calloc(seq_hidden, sizeof(float));
    float *gate = (float *)calloc(fx->inter, sizeof(float));
    float *up = (float *)calloc(fx->inter, sizeof(float));
    float *act = (float *)calloc(fx->inter, sizeof(float));
    float *down = (float *)calloc(fx->hidden, sizeof(float));
    float *shared_gate = (float *)calloc(fx->inter, sizeof(float));
    float *shared_up = (float *)calloc(fx->inter, sizeof(float));
    float *shared_act = (float *)calloc(fx->inter, sizeof(float));
    float *shared = (float *)calloc(fx->hidden, sizeof(float));
    float router_logits[ROUTER_COUNT];
    uint32_t router_idx[ROUTER_COUNT];
    float router_scores[ROUTER_COUNT];
    uint32_t t, h, d, hd, vd, i;
    if (!input_ln || !qkv || !z || !a || !b || !conv || !q || !k || !v || !beta || !g || !core || !state || !out_in ||
        !out_proj || !resid_seq || !post_attn_ln_seq || !mlp_seq || !gate || !up || !act || !down || !shared_gate || !shared_up ||
        !shared_act || !shared) goto oom;

    for (t = 0; t < seq_len; ++t) {
        const float *in = layer_input_seq + (size_t)t * fx->hidden;
        float *out = input_ln + (size_t)t * fx->hidden;
        double var = 0.0;
        for (d = 0; d < fx->hidden; ++d) var += (double)in[d] * in[d];
        var /= (double)fx->hidden;
        for (d = 0; d < fx->hidden; ++d) out[d] = (float)(in[d] / sqrt(var + 1e-6)) * fx->attn_norm_w[d];
        matvec(fx->w_qkv, out, qkv + (size_t)t * qkv_dim, qkv_dim, fx->hidden);
        matvec(fx->w_z, out, z + (size_t)t * fx->value_dim, fx->value_dim, fx->hidden);
        matvec(fx->w_a, out, a + (size_t)t * fx->num_v_heads, fx->num_v_heads, fx->hidden);
        matvec(fx->w_b, out, b + (size_t)t * fx->num_v_heads, fx->num_v_heads, fx->hidden);
    }
    for (t = 0; t < seq_len; ++t) {
        for (d = 0; d < qkv_dim; ++d) {
            double sum = 0.0;
            for (uint32_t kk = 0; kk < 4; ++kk) {
                int src_t = (int)t - 3 + (int)kk;
                if (src_t >= 0 && src_t < (int)seq_len) sum += (double)fx->conv_w[(size_t)d * 4 + kk] * qkv[(size_t)src_t * qkv_dim + d];
            }
            conv[(size_t)t * qkv_dim + d] = siluf_local((float)sum);
        }
    }
    for (t = 0; t < seq_len; ++t) {
        const float *cq = conv + (size_t)t * qkv_dim;
        const float *ck = cq + fx->key_dim;
        const float *cv = ck + fx->key_dim;
        for (h = 0; h < fx->num_k_heads; ++h) {
            for (d = 0; d < fx->head_k_dim; ++d) {
                float qv = cq[h * fx->head_k_dim + d], kv = ck[h * fx->head_k_dim + d];
                for (uint32_t vh = 0; vh < rep; ++vh) {
                    uint32_t dst_h = h * rep + vh;
                    q[((size_t)t * fx->num_v_heads + dst_h) * fx->head_k_dim + d] = qv;
                    k[((size_t)t * fx->num_v_heads + dst_h) * fx->head_k_dim + d] = kv;
                }
            }
        }
        for (h = 0; h < fx->num_v_heads; ++h) {
            memcpy(v + ((size_t)t * fx->num_v_heads + h) * fx->head_v_dim, cv + (size_t)h * fx->head_v_dim, fx->head_v_dim * sizeof(float));
            beta[t * fx->num_v_heads + h] = sigmoidf_local(b[(size_t)t * fx->num_v_heads + h]);
            g[t * fx->num_v_heads + h] = fx->A_log[h] * softplusf_local(a[(size_t)t * fx->num_v_heads + h] + fx->dt_bias[h]);
        }
    }
    for (t = 0; t < seq_len; ++t) {
        for (h = 0; h < fx->num_v_heads; ++h) {
            float qnorm = 0.0f, knorm = 0.0f, gexp = expf(g[t * fx->num_v_heads + h]), beta_t = beta[t * fx->num_v_heads + h];
            size_t qbase = ((size_t)t * fx->num_v_heads + h) * fx->head_k_dim;
            size_t vbase = ((size_t)t * fx->num_v_heads + h) * fx->head_v_dim;
            size_t sbase = (size_t)h * fx->head_k_dim * fx->head_v_dim;
            for (hd = 0; hd < fx->head_k_dim; ++hd) { qnorm += q[qbase + hd] * q[qbase + hd]; knorm += k[qbase + hd] * k[qbase + hd]; }
            qnorm = 1.0f / sqrtf(qnorm + 1e-6f);
            knorm = 1.0f / sqrtf(knorm + 1e-6f);
            for (hd = 0; hd < fx->head_k_dim * fx->head_v_dim; ++hd) state[sbase + hd] *= gexp;
            {
                float delta[128];
                for (vd = 0; vd < fx->head_v_dim; ++vd) {
                    float kv_mem = 0.0f;
                    for (hd = 0; hd < fx->head_k_dim; ++hd) kv_mem += state[sbase + (size_t)hd * fx->head_v_dim + vd] * (k[qbase + hd] * knorm);
                    delta[vd] = (v[vbase + vd] - kv_mem) * beta_t;
                }
                for (hd = 0; hd < fx->head_k_dim; ++hd) {
                    float kval = k[qbase + hd] * knorm;
                    for (vd = 0; vd < fx->head_v_dim; ++vd) state[sbase + (size_t)hd * fx->head_v_dim + vd] += kval * delta[vd];
                }
                for (vd = 0; vd < fx->head_v_dim; ++vd) {
                    float outv = 0.0f;
                    for (hd = 0; hd < fx->head_k_dim; ++hd) outv += state[sbase + (size_t)hd * fx->head_v_dim + vd] * (q[qbase + hd] * qnorm / sqrtf((float)fx->head_k_dim));
                    core[vbase + vd] = outv;
                }
            }
        }
    }
    for (t = 0; t < seq_len; ++t) {
        for (h = 0; h < fx->num_v_heads; ++h) {
            size_t base = (size_t)t * fx->value_dim + (size_t)h * fx->head_v_dim;
            double var = 0.0;
            for (vd = 0; vd < fx->head_v_dim; ++vd) {
                double cv = core[((size_t)t * fx->num_v_heads + h) * fx->head_v_dim + vd];
                var += cv * cv;
            }
            var /= (double)fx->head_v_dim;
            for (vd = 0; vd < fx->head_v_dim; ++vd) {
                float cv = core[((size_t)t * fx->num_v_heads + h) * fx->head_v_dim + vd];
                out_in[base + vd] = (float)(cv / sqrt(var + 1e-6)) * fx->ssm_norm_w[vd] * siluf_local(z[(size_t)t * fx->value_dim + (size_t)h * fx->head_v_dim + vd]);
            }
        }
        matvec(fx->w_out, out_in + (size_t)t * fx->value_dim, out_proj + (size_t)t * fx->hidden, fx->hidden, fx->value_dim);
    }
    for (t = 0; t < seq_len; ++t) {
        const float *in_row = layer_input_seq + (size_t)t * fx->hidden;
        float *resid_row = resid_seq + (size_t)t * fx->hidden;
        float *post_row = post_attn_ln_seq + (size_t)t * fx->hidden;
        float *mlp_row = mlp_seq + (size_t)t * fx->hidden;
        float *out_row = out_final_seq + (size_t)t * fx->hidden;
        double var = 0.0;
        float scale_in = 0.0f;
        memset(mlp_row, 0, (size_t)fx->hidden * sizeof(float));
        for (d = 0; d < fx->hidden; ++d) { resid_row[d] = in_row[d] + out_proj[(size_t)t * fx->hidden + d]; var += (double)resid_row[d] * resid_row[d]; }
        var /= (double)fx->hidden;
        for (d = 0; d < fx->hidden; ++d) {
            post_row[d] = (float)(resid_row[d] / sqrt(var + 1e-6)) * fx->post_attn_norm_w[d];
            scale_in += post_row[d] * fx->gate_inp_shexp[d];
        }
        matvec(fx->router_w, post_row, router_logits, ROUTER_COUNT, fx->hidden);
        topk_softmax256(router_logits, fx->topk, router_idx, router_scores);
        for (i = 0; i < fx->topk; ++i) {
            if (!ensure_expert_loaded(gf, layer, cache, router_idx[i], fx->hidden, fx->inter, err, err_cap)) goto oom;
            matvec(cache[router_idx[i]].gate, post_row, gate, fx->inter, fx->hidden);
            matvec(cache[router_idx[i]].up, post_row, up, fx->inter, fx->hidden);
            for (vd = 0; vd < fx->inter; ++vd) act[vd] = siluf_local(gate[vd]) * up[vd];
            matvec(cache[router_idx[i]].down, act, down, fx->hidden, fx->inter);
            for (d = 0; d < fx->hidden; ++d) mlp_row[d] += down[d] * router_scores[i];
        }
        matvec(fx->gate_shexp, post_row, shared_gate, fx->inter, fx->hidden);
        matvec(fx->up_shexp, post_row, shared_up, fx->inter, fx->hidden);
        for (vd = 0; vd < fx->inter; ++vd) shared_act[vd] = siluf_local(shared_gate[vd]) * shared_up[vd];
        matvec(fx->down_shexp, shared_act, shared, fx->hidden, fx->inter);
        {
            float s = sigmoidf_local(scale_in);
            for (d = 0; d < fx->hidden; ++d) out_row[d] = resid_row[d] + mlp_row[d] + shared[d] * s;
        }
    }

    free(input_ln); free(qkv); free(z); free(a); free(b); free(conv); free(q); free(k); free(v); free(beta); free(g); free(core); free(state); free(out_in); free(out_proj);
    free(resid_seq); free(post_attn_ln_seq); free(mlp_seq); free(gate); free(up); free(act); free(down); free(shared_gate); free(shared_up); free(shared_act); free(shared);
    return 1;
oom:
    free(input_ln); free(qkv); free(z); free(a); free(b); free(conv); free(q); free(k); free(v); free(beta); free(g); free(core); free(state); free(out_in); free(out_proj);
    free(resid_seq); free(post_attn_ln_seq); free(mlp_seq); free(gate); free(up); free(act); free(down); free(shared_gate); free(shared_up); free(shared_act); free(shared);
    return 0;
}

int main(int argc, char **argv) {
    const char *gguf_path, *prefix_path, *dump_last_path = NULL, *dump_seq_path = NULL;
    int n_fx = 0, argi;
    qwen36_gguf_file gf;
    qwen36_35a3b_q8_model q8;
    prefix_fixture pfx;
    live_fixture *fxs = NULL;
    expert_cache_entry *caches = NULL;
    float *input_seq = NULL, *state_seq = NULL;
    char err[512];
    int use_prefix_input_seq = 0;
    if (argc < 4) { fprintf(stderr, "usage: %s MODEL.gguf PREFIX.bin LAYER0.live.bin [LAYER1.live.bin ...] [--use-prefix-input-seq] [--dump-last PATH] [--dump-seq PATH]\n", argv[0]); return 1; }
    gguf_path = argv[1];
    prefix_path = argv[2];
    for (argi = 3; argi < argc; ++argi) {
        if (strcmp(argv[argi], "--use-prefix-input-seq") == 0) { use_prefix_input_seq = 1; continue; }
        if (strcmp(argv[argi], "--dump-last") == 0 && argi + 1 < argc) { dump_last_path = argv[++argi]; continue; }
        if (strcmp(argv[argi], "--dump-seq") == 0 && argi + 1 < argc) { dump_seq_path = argv[++argi]; continue; }
        ++n_fx;
    }
    if (!prefix_load(prefix_path, &pfx)) { fprintf(stderr, "failed to load prefix fixture\n"); return 1; }
    if (!qwen36_gguf_open(&gf, gguf_path, err, sizeof(err))) { fprintf(stderr, "%s\n", err); prefix_free(&pfx); return 1; }
    if (!qwen36_35a3b_q8_bind(&gf, &q8, err, sizeof(err))) { fprintf(stderr, "%s\n", err); qwen36_gguf_close(&gf); prefix_free(&pfx); return 1; }
    fxs = (live_fixture *)calloc((size_t)n_fx, sizeof(live_fixture));
    caches = (expert_cache_entry *)calloc((size_t)n_fx * ROUTER_COUNT, sizeof(expert_cache_entry));
    if (!fxs || !caches) { qwen36_gguf_close(&gf); prefix_free(&pfx); free(fxs); free(caches); return 1; }
    {
        int fxi = 0;
        for (argi = 3; argi < argc; ++argi) {
            if (strcmp(argv[argi], "--use-prefix-input-seq") == 0) continue;
            if (strcmp(argv[argi], "--dump-last") == 0 || strcmp(argv[argi], "--dump-seq") == 0) { ++argi; continue; }
            if (!live_fixture_load(argv[argi], &fxs[fxi])) { fprintf(stderr, "failed to load live fixture: %s\n", argv[argi]); goto fail; }
            if (fxs[fxi].hidden != pfx.hidden) { fprintf(stderr, "fixture hidden mismatch: %s\n", argv[argi]); goto fail; }
            if (q8.layers[fxs[fxi].layer].kind != QWEN36_LAYER_KIND_HYBRID_SSM) { fprintf(stderr, "only hybrid SSM layers are supported: blk.%u\n", fxs[fxi].layer); goto fail; }
            ++fxi;
        }
    }
    input_seq = (float *)calloc((size_t)pfx.seq_len * pfx.hidden, sizeof(float));
    if (!input_seq) goto fail;
    if (use_prefix_input_seq) {
        memcpy(input_seq, pfx.input_seq_ref, (size_t)pfx.seq_len * pfx.hidden * sizeof(float));
        printf("prefix_source: input_seq_ref\n");
    } else {
        for (uint32_t t = 0; t < pfx.seq_len; ++t) {
            if (!decode_q8_row(&gf, q8.token_embd, pfx.token_ids[t], pfx.hidden, input_seq + (size_t)t * pfx.hidden, err, sizeof(err))) {
                fprintf(stderr, "%s\n", err);
                goto fail;
            }
        }
        printf("prefix_source: token_embd\n");
    }
    printf("prompt_tokens: %u\n", pfx.seq_len);
    state_seq = input_seq;
    for (int i = 0; i < n_fx; ++i) {
        float *next_seq = (float *)calloc((size_t)pfx.seq_len * pfx.hidden, sizeof(float));
        if (!next_seq) goto fail;
        printf("run_layer_live: blk.%u seq_len=%u\n", fxs[i].layer, pfx.seq_len);
        if (!run_layer_live(&fxs[i], &gf, &q8.layers[fxs[i].layer], caches + (size_t)i * ROUTER_COUNT, pfx.seq_len, state_seq, next_seq, err, sizeof(err))) {
            fprintf(stderr, "run_layer_live failed: blk.%u: %s\n", fxs[i].layer, err);
            free(next_seq);
            goto fail;
        }
        if (state_seq != input_seq) free(state_seq);
        state_seq = next_seq;
    }
    if (dump_last_path) {
        FILE *fp = fopen(dump_last_path, "wb");
        if (!fp) goto fail;
        fwrite(state_seq + (size_t)(pfx.seq_len - 1) * pfx.hidden, sizeof(float), pfx.hidden, fp);
        fclose(fp);
        printf("wrote_last_hidden: %s\n", dump_last_path);
    }
    if (dump_seq_path) {
        FILE *fp = fopen(dump_seq_path, "wb");
        if (!fp) goto fail;
        fwrite(state_seq, sizeof(float), (size_t)pfx.seq_len * pfx.hidden, fp);
        fclose(fp);
        printf("wrote_seq_hidden: %s\n", dump_seq_path);
    }
    for (int i = 0; i < n_fx; ++i) {
        free_expert_cache(caches + (size_t)i * ROUTER_COUNT);
        live_fixture_free(&fxs[i]);
    }
    free(caches); free(fxs); if (state_seq != input_seq) free(state_seq); free(input_seq); prefix_free(&pfx); qwen36_gguf_close(&gf);
    return 0;
fail:
    if (caches) {
        for (int i = 0; i < n_fx; ++i) free_expert_cache(caches + (size_t)i * ROUTER_COUNT);
    }
    if (fxs) {
        for (int i = 0; i < n_fx; ++i) live_fixture_free(&fxs[i]);
    }
    free(caches); free(fxs); if (state_seq && state_seq != input_seq) free(state_seq); free(input_seq); prefix_free(&pfx); qwen36_gguf_close(&gf);
    return 1;
}
