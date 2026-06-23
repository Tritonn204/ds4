#include "qwen36_35a3b_q8.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "Q36DWF02"
#define MAGIC_LEN 8
#define PREFIX_MAGIC "Q36PFX01"
#define PREFIX_MAGIC_LEN 8
#define ROUTER_COUNT 256
#define INTER 512

typedef struct prefix_fixture {
    uint32_t seq_len;
    uint32_t hidden;
    uint32_t *token_ids;
    float *input_seq_ref;
} prefix_fixture;

typedef struct dwf_fixture {
    uint32_t layer, seq_len, hidden, num_v_heads, num_k_heads, head_k_dim, head_v_dim, key_dim, value_dim, topk, union_experts;
    float *layer_input_seq, *input_ln_seq, *qkv_seq, *z_seq, *a_seq, *b_seq, *conv_raw;
    float *q_ref, *k_ref, *v_ref, *beta_ref, *g_ref, *core_ref, *out_in_seq, *out_proj_out_seq;
    float *mixer_out_ref, *layer_input_last_ref, *residual_after_mixer_ref, *post_attn_ln_ref, *mlp_out_ref, *layer_output_ref;
    float *router_logits_ref, *router_indices_ref_f32, *router_scores_ref, *shared_gate_pre_ref;
    float *layer_input_seq_ref, *residual_after_mixer_seq_ref, *post_attn_ln_seq_ref, *mlp_out_seq_ref, *layer_output_seq_ref;
    float *router_logits_seq_ref, *router_indices_seq_ref_f32, *router_scores_seq_ref, *shared_gate_pre_seq_ref;
    float *attn_norm_w, *post_attn_norm_w, *w_qkv, *w_z, *w_a, *w_b, *conv_w, *A_log, *dt_bias, *ssm_norm_w, *w_out;
    float *router_w;
    float *union_expert_ids_f32, *router_union_pos_seq_f32;
    float *gate_sel, *up_sel, *down_sel, *gate_shexp, *up_shexp, *down_shexp, *gate_inp_shexp;
} dwf_fixture;

typedef struct expert_cache_entry {
    int loaded;
    float *gate;
    float *up;
    float *down;
} expert_cache_entry;

typedef struct layer_runtime {
    float *deltanet_state;
    float *conv_ring[3];
    uint32_t conv_ring_count;
    expert_cache_entry experts[ROUTER_COUNT];
} layer_runtime;

typedef struct worker_state {
    qwen36_35a3b_q8_model model;
    dwf_fixture *fxs;
    uint32_t n_fx;
    int prefilled;
    uint32_t seq_len;
    uint32_t hidden;
    float *output_seq;
    layer_runtime *layers;
} worker_state;

static int run_step_chain_row(worker_state *ws, const qwen36_gguf_file *gf, const float *input_row, char *err, size_t err_cap);

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

static int prefix_alloc_seq(prefix_fixture *fx, uint32_t seq_len, uint32_t hidden, int want_token_ids, int want_input_seq) {
    memset(fx, 0, sizeof(*fx));
    fx->seq_len = seq_len;
    fx->hidden = hidden;
    if (want_token_ids) {
        fx->token_ids = (uint32_t *)malloc((size_t)seq_len * sizeof(uint32_t));
        if (!fx->token_ids) return 0;
    }
    if (want_input_seq) {
        fx->input_seq_ref = (float *)malloc((size_t)seq_len * hidden * sizeof(float));
        if (!fx->input_seq_ref) {
            free(fx->token_ids);
            fx->token_ids = NULL;
            return 0;
        }
    }
    return 1;
}

static void fixture_free(dwf_fixture *fx) {
    if (!fx) return;
    free(fx->layer_input_seq); free(fx->input_ln_seq); free(fx->qkv_seq); free(fx->z_seq); free(fx->a_seq); free(fx->b_seq); free(fx->conv_raw);
    free(fx->q_ref); free(fx->k_ref); free(fx->v_ref); free(fx->beta_ref); free(fx->g_ref); free(fx->core_ref); free(fx->out_in_seq); free(fx->out_proj_out_seq);
    free(fx->mixer_out_ref); free(fx->layer_input_last_ref); free(fx->residual_after_mixer_ref); free(fx->post_attn_ln_ref); free(fx->mlp_out_ref); free(fx->layer_output_ref);
    free(fx->router_logits_ref); free(fx->router_indices_ref_f32); free(fx->router_scores_ref); free(fx->shared_gate_pre_ref);
    free(fx->layer_input_seq_ref); free(fx->residual_after_mixer_seq_ref); free(fx->post_attn_ln_seq_ref); free(fx->mlp_out_seq_ref); free(fx->layer_output_seq_ref);
    free(fx->router_logits_seq_ref); free(fx->router_indices_seq_ref_f32); free(fx->router_scores_seq_ref); free(fx->shared_gate_pre_seq_ref);
    free(fx->attn_norm_w); free(fx->post_attn_norm_w); free(fx->w_qkv); free(fx->w_z); free(fx->w_a); free(fx->w_b); free(fx->conv_w); free(fx->A_log); free(fx->dt_bias); free(fx->ssm_norm_w); free(fx->w_out);
    free(fx->router_w); free(fx->union_expert_ids_f32); free(fx->router_union_pos_seq_f32);
    free(fx->gate_sel); free(fx->up_sel); free(fx->down_sel); free(fx->gate_shexp); free(fx->up_shexp); free(fx->down_shexp); free(fx->gate_inp_shexp);
    memset(fx, 0, sizeof(*fx));
}

static int fixture_load(const char *path, dwf_fixture *fx) {
    FILE *fp = fopen(path, "rb");
    char magic[MAGIC_LEN];
    memset(fx, 0, sizeof(*fx));
    if (!fp) return 0;
    if (!read_exact(fp, magic, MAGIC_LEN) || memcmp(magic, MAGIC, MAGIC_LEN) != 0) { fclose(fp); return 0; }
    if (!read_exact(fp, &fx->layer, sizeof(uint32_t)) || !read_exact(fp, &fx->seq_len, sizeof(uint32_t)) || !read_exact(fp, &fx->hidden, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->num_v_heads, sizeof(uint32_t)) || !read_exact(fp, &fx->num_k_heads, sizeof(uint32_t)) || !read_exact(fp, &fx->head_k_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->head_v_dim, sizeof(uint32_t)) || !read_exact(fp, &fx->key_dim, sizeof(uint32_t)) || !read_exact(fp, &fx->value_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->topk, sizeof(uint32_t)) || !read_exact(fp, &fx->union_experts, sizeof(uint32_t))) { fclose(fp); return 0; }
#define R(name,count) if (!alloc_read_f32(fp, &fx->name, (count))) { fclose(fp); fixture_free(fx); return 0; }
    R(layer_input_seq, (size_t)fx->seq_len * fx->hidden); R(input_ln_seq, (size_t)fx->seq_len * fx->hidden); R(qkv_seq, (size_t)fx->seq_len * (fx->key_dim * 2 + fx->value_dim));
    R(z_seq, (size_t)fx->seq_len * fx->value_dim); R(a_seq, (size_t)fx->seq_len * fx->num_v_heads); R(b_seq, (size_t)fx->seq_len * fx->num_v_heads); R(conv_raw, (size_t)(fx->key_dim * 2 + fx->value_dim) * (fx->seq_len + 3));
    R(q_ref, (size_t)fx->seq_len * fx->num_v_heads * fx->head_k_dim); R(k_ref, (size_t)fx->seq_len * fx->num_v_heads * fx->head_k_dim); R(v_ref, (size_t)fx->seq_len * fx->num_v_heads * fx->head_v_dim);
    R(beta_ref, (size_t)fx->seq_len * fx->num_v_heads); R(g_ref, (size_t)fx->seq_len * fx->num_v_heads); R(core_ref, (size_t)fx->seq_len * fx->num_v_heads * fx->head_v_dim);
    R(out_in_seq, (size_t)fx->seq_len * fx->value_dim); R(out_proj_out_seq, (size_t)fx->seq_len * fx->hidden);
    R(mixer_out_ref, fx->hidden); R(layer_input_last_ref, fx->hidden); R(residual_after_mixer_ref, fx->hidden); R(post_attn_ln_ref, fx->hidden); R(mlp_out_ref, fx->hidden); R(layer_output_ref, fx->hidden);
    R(router_logits_ref, ROUTER_COUNT); R(router_indices_ref_f32, fx->topk); R(router_scores_ref, fx->topk); R(shared_gate_pre_ref, 1);
    R(layer_input_seq_ref, (size_t)fx->seq_len * fx->hidden); R(residual_after_mixer_seq_ref, (size_t)fx->seq_len * fx->hidden); R(post_attn_ln_seq_ref, (size_t)fx->seq_len * fx->hidden);
    R(mlp_out_seq_ref, (size_t)fx->seq_len * fx->hidden); R(layer_output_seq_ref, (size_t)fx->seq_len * fx->hidden);
    R(router_logits_seq_ref, (size_t)fx->seq_len * ROUTER_COUNT); R(router_indices_seq_ref_f32, (size_t)fx->seq_len * fx->topk); R(router_scores_seq_ref, (size_t)fx->seq_len * fx->topk); R(shared_gate_pre_seq_ref, fx->seq_len);
    R(attn_norm_w, fx->hidden); R(post_attn_norm_w, fx->hidden); R(w_qkv, (size_t)(fx->key_dim * 2 + fx->value_dim) * fx->hidden); R(w_z, (size_t)fx->value_dim * fx->hidden);
    R(w_a, (size_t)fx->num_v_heads * fx->hidden); R(w_b, (size_t)fx->num_v_heads * fx->hidden); R(conv_w, (size_t)(fx->key_dim * 2 + fx->value_dim) * 4); R(A_log, fx->num_v_heads); R(dt_bias, fx->num_v_heads);
    R(ssm_norm_w, fx->head_v_dim); R(w_out, (size_t)fx->hidden * fx->value_dim); R(router_w, (size_t)ROUTER_COUNT * fx->hidden);
    R(union_expert_ids_f32, fx->union_experts); R(router_union_pos_seq_f32, (size_t)fx->seq_len * fx->topk);
    R(gate_sel, (size_t)fx->union_experts * INTER * fx->hidden); R(up_sel, (size_t)fx->union_experts * INTER * fx->hidden); R(down_sel, (size_t)fx->union_experts * fx->hidden * INTER);
    R(gate_shexp, (size_t)INTER * fx->hidden); R(up_shexp, (size_t)INTER * fx->hidden); R(down_shexp, (size_t)fx->hidden * INTER); R(gate_inp_shexp, fx->hidden);
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
    for (uint32_t r = 0; r < rows; ++r) {
        const float *row = mat + (size_t)r * cols;
        double sum = 0.0;
        for (uint32_t c = 0; c < cols; ++c) sum += (double)row[c] * vec[c];
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
                uint32_t tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
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
        memset(&cache[i], 0, sizeof(cache[i]));
    }
}

static int ensure_expert_loaded(const qwen36_gguf_file *gf, const qwen36_35a3b_q8_layer *layer, expert_cache_entry *cache, uint32_t expert_id, uint32_t hidden, char *err, size_t err_cap) {
    expert_cache_entry *e = &cache[expert_id];
    if (e->loaded) return 1;
    e->gate = (float *)malloc((size_t)INTER * hidden * sizeof(float));
    e->up = (float *)malloc((size_t)INTER * hidden * sizeof(float));
    e->down = (float *)malloc((size_t)hidden * INTER * sizeof(float));
    if (!e->gate || !e->up || !e->down) return 0;
    if (!decode_q8_rows(gf, layer->ffn_gate_exps, expert_id * INTER, INTER, hidden, e->gate, err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->ffn_up_exps, expert_id * INTER, INTER, hidden, e->up, err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->ffn_down_exps, expert_id * hidden, hidden, INTER, e->down, err, err_cap)) return 0;
    e->loaded = 1;
    return 1;
}

static int run_prefill_layer_dynamic(const dwf_fixture *fx,
                                     const qwen36_gguf_file *gf,
                                     const qwen36_35a3b_q8_layer *layer,
                                     layer_runtime *ls,
                                     uint32_t seq_len,
                                     const float *layer_input_seq,
                                     float *out_final_seq,
                                     char *err,
                                     size_t err_cap) {
    const uint32_t qkv_dim = fx->key_dim * 2u + fx->value_dim;
    const uint32_t rep = fx->num_v_heads / fx->num_k_heads;
    const size_t seq_hidden = (size_t)seq_len * fx->hidden;
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
    float *out_in = (float *)calloc((size_t)seq_len * fx->value_dim, sizeof(float));
    float *out_proj = (float *)calloc(seq_hidden, sizeof(float));
    float *resid_seq = (float *)calloc(seq_hidden, sizeof(float));
    float *post_attn_ln_seq = (float *)calloc(seq_hidden, sizeof(float));
    float *mlp_seq = (float *)calloc(seq_hidden, sizeof(float));
    float *gate = (float *)calloc(INTER, sizeof(float));
    float *up = (float *)calloc(INTER, sizeof(float));
    float *act = (float *)calloc(INTER, sizeof(float));
    float *down = (float *)calloc(fx->hidden, sizeof(float));
    float *shared_gate = (float *)calloc(INTER, sizeof(float));
    float *shared_up = (float *)calloc(INTER, sizeof(float));
    float *shared_act = (float *)calloc(INTER, sizeof(float));
    float *shared = (float *)calloc(fx->hidden, sizeof(float));
    float router_logits[ROUTER_COUNT];
    uint32_t router_idx[ROUTER_COUNT];
    float router_scores[ROUTER_COUNT];
    uint32_t t, h, d, hd, vd, i;

    if (!input_ln || !qkv || !z || !a || !b || !conv || !q || !k || !v || !beta || !g || !core || !out_in ||
        !out_proj || !resid_seq || !post_attn_ln_seq || !mlp_seq || !gate || !up || !act || !down ||
        !shared_gate || !shared_up || !shared_act || !shared) {
        snprintf(err, err_cap, "oom in prefill layer blk.%u", fx->layer);
        goto done;
    }

    memset(ls->deltanet_state, 0, (size_t)fx->num_v_heads * fx->head_k_dim * fx->head_v_dim * sizeof(float));
    ls->conv_ring_count = 0;
    for (i = 0; i < 3; ++i) memset(ls->conv_ring[i], 0, (size_t)qkv_dim * sizeof(float));

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
            for (hd = 0; hd < fx->head_k_dim * fx->head_v_dim; ++hd) ls->deltanet_state[sbase + hd] *= gexp;
            {
                float delta[128];
                for (vd = 0; vd < fx->head_v_dim; ++vd) {
                    float kv_mem = 0.0f;
                    for (hd = 0; hd < fx->head_k_dim; ++hd) kv_mem += ls->deltanet_state[sbase + (size_t)hd * fx->head_v_dim + vd] * (k[qbase + hd] * knorm);
                    delta[vd] = (v[vbase + vd] - kv_mem) * beta_t;
                }
                for (hd = 0; hd < fx->head_k_dim; ++hd) {
                    float kval = k[qbase + hd] * knorm;
                    for (vd = 0; vd < fx->head_v_dim; ++vd) ls->deltanet_state[sbase + (size_t)hd * fx->head_v_dim + vd] += kval * delta[vd];
                }
                for (vd = 0; vd < fx->head_v_dim; ++vd) {
                    float outv = 0.0f;
                    for (hd = 0; hd < fx->head_k_dim; ++hd) outv += ls->deltanet_state[sbase + (size_t)hd * fx->head_v_dim + vd] * (q[qbase + hd] * qnorm / sqrtf((float)fx->head_k_dim));
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
            if (!ensure_expert_loaded(gf, layer, ls->experts, router_idx[i], fx->hidden, err, err_cap)) goto done;
            matvec(ls->experts[router_idx[i]].gate, post_row, gate, INTER, fx->hidden);
            matvec(ls->experts[router_idx[i]].up, post_row, up, INTER, fx->hidden);
            for (vd = 0; vd < INTER; ++vd) act[vd] = siluf_local(gate[vd]) * up[vd];
            matvec(ls->experts[router_idx[i]].down, act, down, fx->hidden, INTER);
            for (d = 0; d < fx->hidden; ++d) mlp_row[d] += down[d] * router_scores[i];
        }
        matvec(fx->gate_shexp, post_row, shared_gate, INTER, fx->hidden);
        matvec(fx->up_shexp, post_row, shared_up, INTER, fx->hidden);
        for (vd = 0; vd < INTER; ++vd) shared_act[vd] = siluf_local(shared_gate[vd]) * shared_up[vd];
        matvec(fx->down_shexp, shared_act, shared, fx->hidden, INTER);
        {
            float s = sigmoidf_local(scale_in);
            for (d = 0; d < fx->hidden; ++d) out_row[d] = resid_row[d] + mlp_row[d] + shared[d] * s;
        }
    }
    for (i = 0; i < 3u && i < seq_len; ++i) {
        int src_t = (int)seq_len - 1 - (int)i;
        if (src_t >= 0) memcpy(ls->conv_ring[i], qkv + (size_t)src_t * qkv_dim, (size_t)qkv_dim * sizeof(float));
    }
    ls->conv_ring_count = seq_len < 3 ? seq_len : 3;

done:
    free(input_ln); free(qkv); free(z); free(a); free(b); free(conv); free(q); free(k); free(v); free(beta); free(g); free(core);
    free(out_in); free(out_proj); free(resid_seq); free(post_attn_ln_seq); free(mlp_seq); free(gate); free(up); free(act); free(down);
    free(shared_gate); free(shared_up); free(shared_act); free(shared);
    return err[0] == '\0';
}

static int run_step_layer_dynamic(const dwf_fixture *fx,
                                  const qwen36_gguf_file *gf,
                                  const qwen36_35a3b_q8_layer *layer,
                                  layer_runtime *ls,
                                  const float *layer_input,
                                  float *out_row,
                                  char *err,
                                  size_t err_cap) {
    const uint32_t qkv_dim = fx->key_dim * 2u + fx->value_dim;
    const uint32_t rep = fx->num_v_heads / fx->num_k_heads;
    float *input_ln = NULL, *qkv = NULL, *z = NULL, *a = NULL, *b = NULL;
    float *conv = NULL, *q = NULL, *k = NULL, *v = NULL, *beta = NULL, *g = NULL, *core = NULL, *out_in = NULL, *out_proj = NULL;
    float *resid = NULL, *post_ln = NULL, *mlp = NULL;
    float *gate = NULL, *up = NULL, *act = NULL, *down = NULL;
    float *shared_gate = NULL, *shared_up = NULL, *shared_act = NULL, *shared = NULL;
    float router_logits[ROUTER_COUNT];
    uint32_t router_idx[ROUTER_COUNT];
    float router_scores[ROUTER_COUNT];
    uint32_t h, d, hd, vd, i;

    input_ln = (float *)malloc((size_t)fx->hidden * sizeof(float));
    qkv = (float *)malloc((size_t)qkv_dim * sizeof(float));
    z = (float *)malloc((size_t)fx->value_dim * sizeof(float));
    a = (float *)malloc((size_t)fx->num_v_heads * sizeof(float));
    b = (float *)malloc((size_t)fx->num_v_heads * sizeof(float));
    conv = (float *)malloc((size_t)qkv_dim * sizeof(float));
    q = (float *)malloc((size_t)fx->num_v_heads * fx->head_k_dim * sizeof(float));
    k = (float *)malloc((size_t)fx->num_v_heads * fx->head_k_dim * sizeof(float));
    v = (float *)malloc((size_t)fx->num_v_heads * fx->head_v_dim * sizeof(float));
    beta = (float *)malloc((size_t)fx->num_v_heads * sizeof(float));
    g = (float *)malloc((size_t)fx->num_v_heads * sizeof(float));
    core = (float *)malloc((size_t)fx->num_v_heads * fx->head_v_dim * sizeof(float));
    out_in = (float *)malloc((size_t)fx->value_dim * sizeof(float));
    out_proj = (float *)malloc((size_t)fx->hidden * sizeof(float));
    resid = (float *)malloc((size_t)fx->hidden * sizeof(float));
    post_ln = (float *)malloc((size_t)fx->hidden * sizeof(float));
    mlp = (float *)malloc((size_t)fx->hidden * sizeof(float));
    gate = (float *)malloc((size_t)INTER * sizeof(float));
    up = (float *)malloc((size_t)INTER * sizeof(float));
    act = (float *)malloc((size_t)INTER * sizeof(float));
    down = (float *)malloc((size_t)fx->hidden * sizeof(float));
    shared_gate = (float *)malloc((size_t)INTER * sizeof(float));
    shared_up = (float *)malloc((size_t)INTER * sizeof(float));
    shared_act = (float *)malloc((size_t)INTER * sizeof(float));
    shared = (float *)malloc((size_t)fx->hidden * sizeof(float));
    if (!input_ln || !qkv || !z || !a || !b || !conv || !q || !k || !v || !beta || !g || !core ||
        !out_in || !out_proj || !resid || !post_ln || !mlp || !gate || !up || !act || !down || !shared_gate || !shared_up || !shared_act || !shared) {
        snprintf(err, err_cap, "oom in step layer blk.%u", fx->layer);
        goto done;
    }

    {
        double var = 0.0;
        for (d = 0; d < fx->hidden; ++d) var += (double)layer_input[d] * layer_input[d];
        var /= (double)fx->hidden;
        for (d = 0; d < fx->hidden; ++d) input_ln[d] = (float)(layer_input[d] / sqrt(var + 1e-6)) * fx->attn_norm_w[d];
    }
    matvec(fx->w_qkv, input_ln, qkv, qkv_dim, fx->hidden);
    matvec(fx->w_z, input_ln, z, fx->value_dim, fx->hidden);
    matvec(fx->w_a, input_ln, a, fx->num_v_heads, fx->hidden);
    matvec(fx->w_b, input_ln, b, fx->num_v_heads, fx->hidden);

    for (d = 0; d < qkv_dim; ++d) {
        double sum = 0.0;
        if (ls->conv_ring_count >= 3) sum += (double)fx->conv_w[(size_t)d * 4u + 0u] * ls->conv_ring[2][d];
        if (ls->conv_ring_count >= 2) sum += (double)fx->conv_w[(size_t)d * 4u + 1u] * ls->conv_ring[1][d];
        if (ls->conv_ring_count >= 1) sum += (double)fx->conv_w[(size_t)d * 4u + 2u] * ls->conv_ring[0][d];
        sum += (double)fx->conv_w[(size_t)d * 4u + 3u] * qkv[d];
        conv[d] = siluf_local((float)sum);
    }
    {
        const float *cq = conv;
        const float *ck = cq + fx->key_dim;
        const float *cv = ck + fx->key_dim;
        for (h = 0; h < fx->num_k_heads; ++h) {
            for (d = 0; d < fx->head_k_dim; ++d) {
                float qv = cq[h * fx->head_k_dim + d], kv = ck[h * fx->head_k_dim + d];
                for (uint32_t vh = 0; vh < rep; ++vh) {
                    uint32_t dst_h = h * rep + vh;
                    q[(size_t)dst_h * fx->head_k_dim + d] = qv;
                    k[(size_t)dst_h * fx->head_k_dim + d] = kv;
                }
            }
        }
        for (h = 0; h < fx->num_v_heads; ++h) {
            memcpy(v + (size_t)h * fx->head_v_dim, cv + (size_t)h * fx->head_v_dim, fx->head_v_dim * sizeof(float));
            beta[h] = sigmoidf_local(b[h]);
            g[h] = fx->A_log[h] * softplusf_local(a[h] + fx->dt_bias[h]);
        }
    }
    for (h = 0; h < fx->num_v_heads; ++h) {
        float qnorm = 0.0f, knorm = 0.0f, gexp = expf(g[h]), beta_t = beta[h];
        size_t qbase = (size_t)h * fx->head_k_dim;
        size_t vbase = (size_t)h * fx->head_v_dim;
        size_t sbase = (size_t)h * fx->head_k_dim * fx->head_v_dim;
        for (hd = 0; hd < fx->head_k_dim; ++hd) { qnorm += q[qbase + hd] * q[qbase + hd]; knorm += k[qbase + hd] * k[qbase + hd]; }
        qnorm = 1.0f / sqrtf(qnorm + 1e-6f);
        knorm = 1.0f / sqrtf(knorm + 1e-6f);
        for (hd = 0; hd < fx->head_k_dim * fx->head_v_dim; ++hd) ls->deltanet_state[sbase + hd] *= gexp;
        {
            float delta[128];
            for (vd = 0; vd < fx->head_v_dim; ++vd) {
                float kv_mem = 0.0f;
                for (hd = 0; hd < fx->head_k_dim; ++hd) kv_mem += ls->deltanet_state[sbase + (size_t)hd * fx->head_v_dim + vd] * (k[qbase + hd] * knorm);
                delta[vd] = (v[vbase + vd] - kv_mem) * beta_t;
            }
            for (hd = 0; hd < fx->head_k_dim; ++hd) {
                float kval = k[qbase + hd] * knorm;
                for (vd = 0; vd < fx->head_v_dim; ++vd) ls->deltanet_state[sbase + (size_t)hd * fx->head_v_dim + vd] += kval * delta[vd];
            }
            for (vd = 0; vd < fx->head_v_dim; ++vd) {
                float outv = 0.0f;
                for (hd = 0; hd < fx->head_k_dim; ++hd) outv += ls->deltanet_state[sbase + (size_t)hd * fx->head_v_dim + vd] * (q[qbase + hd] * qnorm / sqrtf((float)fx->head_k_dim));
                core[vbase + vd] = outv;
            }
        }
    }
    for (h = 0; h < fx->num_v_heads; ++h) {
        size_t base = (size_t)h * fx->head_v_dim;
        double var = 0.0;
        for (vd = 0; vd < fx->head_v_dim; ++vd) {
            double cv = core[(size_t)h * fx->head_v_dim + vd];
            var += cv * cv;
        }
        var /= (double)fx->head_v_dim;
        for (vd = 0; vd < fx->head_v_dim; ++vd) {
            float cv = core[(size_t)h * fx->head_v_dim + vd];
            out_in[base + vd] = (float)(cv / sqrt(var + 1e-6)) * fx->ssm_norm_w[vd] * siluf_local(z[(size_t)h * fx->head_v_dim + vd]);
        }
    }
    matvec(fx->w_out, out_in, out_proj, fx->hidden, fx->value_dim);

    {
        double var = 0.0;
        float scale_in = 0.0f;
        for (d = 0; d < fx->hidden; ++d) { resid[d] = layer_input[d] + out_proj[d]; var += (double)resid[d] * resid[d]; }
        var /= (double)fx->hidden;
        for (d = 0; d < fx->hidden; ++d) {
            post_ln[d] = (float)(resid[d] / sqrt(var + 1e-6)) * fx->post_attn_norm_w[d];
            scale_in += post_ln[d] * fx->gate_inp_shexp[d];
            mlp[d] = 0.0f;
        }
        matvec(fx->router_w, post_ln, router_logits, ROUTER_COUNT, fx->hidden);
        topk_softmax256(router_logits, fx->topk, router_idx, router_scores);
        for (i = 0; i < fx->topk; ++i) {
            if (!ensure_expert_loaded(gf, layer, ls->experts, router_idx[i], fx->hidden, err, err_cap)) goto done;
            matvec(ls->experts[router_idx[i]].gate, post_ln, gate, INTER, fx->hidden);
            matvec(ls->experts[router_idx[i]].up, post_ln, up, INTER, fx->hidden);
            for (vd = 0; vd < INTER; ++vd) act[vd] = siluf_local(gate[vd]) * up[vd];
            matvec(ls->experts[router_idx[i]].down, act, down, fx->hidden, INTER);
            for (d = 0; d < fx->hidden; ++d) mlp[d] += down[d] * router_scores[i];
        }
        matvec(fx->gate_shexp, post_ln, shared_gate, INTER, fx->hidden);
        matvec(fx->up_shexp, post_ln, shared_up, INTER, fx->hidden);
        for (vd = 0; vd < INTER; ++vd) shared_act[vd] = siluf_local(shared_gate[vd]) * shared_up[vd];
        matvec(fx->down_shexp, shared_act, shared, fx->hidden, INTER);
        {
            float s = sigmoidf_local(scale_in);
            for (d = 0; d < fx->hidden; ++d) out_row[d] = resid[d] + mlp[d] + shared[d] * s;
        }
    }

    if (ls->conv_ring_count >= 2) memcpy(ls->conv_ring[2], ls->conv_ring[1], (size_t)qkv_dim * sizeof(float));
    if (ls->conv_ring_count >= 1) memcpy(ls->conv_ring[1], ls->conv_ring[0], (size_t)qkv_dim * sizeof(float));
    memcpy(ls->conv_ring[0], qkv, (size_t)qkv_dim * sizeof(float));
    if (ls->conv_ring_count < 3) ls->conv_ring_count++;

done:
    free(input_ln); free(qkv); free(z); free(a); free(b); free(conv); free(q); free(k); free(v); free(beta); free(g); free(core);
    free(out_in); free(out_proj); free(resid); free(post_ln); free(mlp); free(gate); free(up); free(act); free(down); free(shared_gate); free(shared_up); free(shared_act); free(shared);
    return err[0] == '\0';
}

#if 0
static int run_prefill_blk0_unused(void) { return 0; }
static int run_step_blk0_unused(void) { return 0; }
#endif

static void worker_reset(worker_state *ws) {
    ws->prefilled = 0;
    ws->seq_len = 0;
    free(ws->output_seq);
    ws->output_seq = NULL;
    for (uint32_t li = 0; li < ws->n_fx; ++li) {
        const dwf_fixture *fx = &ws->fxs[li];
        layer_runtime *ls = &ws->layers[li];
        if (ls->deltanet_state) memset(ls->deltanet_state, 0, (size_t)fx->num_v_heads * fx->head_k_dim * fx->head_v_dim * sizeof(float));
        for (int i = 0; i < 3; ++i) {
            if (ls->conv_ring[i]) memset(ls->conv_ring[i], 0, (size_t)(fx->key_dim * 2u + fx->value_dim) * sizeof(float));
        }
        ls->conv_ring_count = 0;
    }
}

static int run_prefill_chain(worker_state *ws, const qwen36_gguf_file *gf, const prefix_fixture *pfx, char *err, size_t err_cap) {
    float *state_seq = NULL;
    float *next_seq = NULL;
    size_t seq_hidden = (size_t)pfx->seq_len * ws->hidden;
    int ok = 0;

    if (ws->n_fx == 0) {
        snprintf(err, err_cap, "no fixtures loaded");
        return 0;
    }
    state_seq = (float *)calloc(seq_hidden, sizeof(float));
    if (!state_seq) {
        snprintf(err, err_cap, "oom initial seq");
        return 0;
    }
    if (ws->fxs[0].layer == 0) {
        for (uint32_t t = 0; t < pfx->seq_len; ++t) {
            if (!decode_q8_row(gf, ws->model.token_embd, pfx->token_ids[t], ws->hidden, state_seq + (size_t)t * ws->hidden, err, err_cap)) goto done;
        }
    } else {
        memcpy(state_seq, pfx->input_seq_ref, seq_hidden * sizeof(float));
    }
    for (uint32_t li = 0; li < ws->n_fx; ++li) {
        next_seq = (float *)calloc(seq_hidden, sizeof(float));
        if (!next_seq) {
            snprintf(err, err_cap, "oom layer seq blk.%u", ws->fxs[li].layer);
            goto done;
        }
        if (!run_prefill_layer_dynamic(&ws->fxs[li], gf, &ws->model.layers[ws->fxs[li].layer], &ws->layers[li], pfx->seq_len, state_seq, next_seq, err, err_cap)) {
            goto done;
        }
        free(state_seq);
        state_seq = next_seq;
        next_seq = NULL;
    }
    ws->output_seq = state_seq;
    state_seq = NULL;
    ws->seq_len = pfx->seq_len;
    ws->prefilled = 1;
    ok = 1;
done:
    free(state_seq);
    free(next_seq);
    return ok;
}

static int run_step_chain_token(worker_state *ws, const qwen36_gguf_file *gf, uint32_t token_id, char *err, size_t err_cap) {
    float *state_row = NULL;
    float *next_row = NULL;
    float *new_seq = NULL;
    int ok = 0;

    if (ws->n_fx == 0 || ws->fxs[0].layer != 0) {
        snprintf(err, err_cap, "STEP token requires blk.0 first fixture");
        return 0;
    }
    state_row = (float *)malloc((size_t)ws->hidden * sizeof(float));
    if (!state_row) {
        snprintf(err, err_cap, "oom state row");
        return 0;
    }
    if (!decode_q8_row(gf, ws->model.token_embd, token_id, ws->hidden, state_row, err, err_cap)) goto done;
    for (uint32_t li = 0; li < ws->n_fx; ++li) {
        next_row = (float *)malloc((size_t)ws->hidden * sizeof(float));
        if (!next_row) {
            snprintf(err, err_cap, "oom next row blk.%u", ws->fxs[li].layer);
            goto done;
        }
        if (!run_step_layer_dynamic(&ws->fxs[li], gf, &ws->model.layers[ws->fxs[li].layer], &ws->layers[li], state_row, next_row, err, err_cap)) {
            goto done;
        }
        free(state_row);
        state_row = next_row;
        next_row = NULL;
    }
    new_seq = (float *)realloc(ws->output_seq, (size_t)(ws->seq_len + 1u) * ws->hidden * sizeof(float));
    if (!new_seq) {
        snprintf(err, err_cap, "oom append seq");
        goto done;
    }
    ws->output_seq = new_seq;
    memcpy(ws->output_seq + (size_t)ws->seq_len * ws->hidden, state_row, (size_t)ws->hidden * sizeof(float));
    ws->seq_len += 1u;
    ok = 1;
done:
    free(state_row);
    free(next_row);
    return ok;
}

static int run_step_chain_prefix(worker_state *ws, const qwen36_gguf_file *gf, const prefix_fixture *pfx, char *err, size_t err_cap) {
    if (pfx->hidden != ws->hidden || pfx->seq_len == 0) {
        snprintf(err, err_cap, "prefix mismatch");
        return 0;
    }
    return run_step_chain_row(ws, gf, pfx->input_seq_ref + (size_t)(pfx->seq_len - 1u) * ws->hidden, err, err_cap);
}

static int run_step_chain_row(worker_state *ws, const qwen36_gguf_file *gf, const float *input_row, char *err, size_t err_cap) {
    float *state_row = NULL;
    float *next_row = NULL;
    float *new_seq = NULL;
    int ok = 0;

    state_row = (float *)malloc((size_t)ws->hidden * sizeof(float));
    if (!state_row) {
        snprintf(err, err_cap, "oom state row");
        return 0;
    }
    memcpy(state_row, input_row, (size_t)ws->hidden * sizeof(float));
    for (uint32_t li = 0; li < ws->n_fx; ++li) {
        next_row = (float *)malloc((size_t)ws->hidden * sizeof(float));
        if (!next_row) {
            snprintf(err, err_cap, "oom next row blk.%u", ws->fxs[li].layer);
            goto done;
        }
        if (!run_step_layer_dynamic(&ws->fxs[li], gf, &ws->model.layers[ws->fxs[li].layer], &ws->layers[li], state_row, next_row, err, err_cap)) {
            goto done;
        }
        free(state_row);
        state_row = next_row;
        next_row = NULL;
    }
    new_seq = (float *)realloc(ws->output_seq, (size_t)(ws->seq_len + 1u) * ws->hidden * sizeof(float));
    if (!new_seq) {
        snprintf(err, err_cap, "oom append seq");
        goto done;
    }
    ws->output_seq = new_seq;
    memcpy(ws->output_seq + (size_t)ws->seq_len * ws->hidden, state_row, (size_t)ws->hidden * sizeof(float));
    ws->seq_len += 1u;
    ok = 1;
done:
    free(state_row);
    free(next_row);
    return ok;
}

static void handle_prefill_prefix(worker_state *ws, const qwen36_gguf_file *gf, const char *path) {
    prefix_fixture pfx;
    char err[512] = {0};
    if (!prefix_load(path, &pfx)) {
        printf("ERROR failed to load prefix fixture\n");
        fflush(stdout);
        return;
    }
    if (pfx.hidden != ws->hidden) {
        printf("ERROR prefix hidden mismatch\n");
        fflush(stdout);
        prefix_free(&pfx);
        return;
    }
    worker_reset(ws);
    if (!run_prefill_chain(ws, gf, &pfx, err, sizeof(err))) {
        printf("ERROR %s\n", err[0] ? err : "prefill failed");
        fflush(stdout);
        prefix_free(&pfx);
        return;
    }
    printf("PREFILL_OK %u %u\n", ws->seq_len, ws->hidden);
    fflush(stdout);
    prefix_free(&pfx);
}

static void handle_prefill_prefix_bin(worker_state *ws, const qwen36_gguf_file *gf, char *rest) {
    prefix_fixture pfx;
    char err[512] = {0};
    uint32_t seq_len = 0, hidden = 0;
    if (sscanf(rest ? rest : "", "%u %u", &seq_len, &hidden) != 2) {
        printf("ERROR bad PREFILL_PREFIX_BIN args\n");
        fflush(stdout);
        return;
    }
    if (hidden != ws->hidden) {
        printf("ERROR prefix hidden mismatch\n");
        fflush(stdout);
        return;
    }
    if (!prefix_alloc_seq(&pfx, seq_len, hidden, 1, 1)) {
        printf("ERROR oom prefix bin\n");
        fflush(stdout);
        return;
    }
    if (!read_exact(stdin, pfx.token_ids, (size_t)seq_len * sizeof(uint32_t)) ||
        !read_exact(stdin, pfx.input_seq_ref, (size_t)seq_len * hidden * sizeof(float))) {
        printf("ERROR short PREFILL_PREFIX_BIN payload\n");
        fflush(stdout);
        prefix_free(&pfx);
        return;
    }
    worker_reset(ws);
    if (!run_prefill_chain(ws, gf, &pfx, err, sizeof(err))) {
        printf("ERROR %s\n", err[0] ? err : "prefill failed");
        fflush(stdout);
        prefix_free(&pfx);
        return;
    }
    printf("PREFILL_OK %u %u\n", ws->seq_len, ws->hidden);
    fflush(stdout);
    prefix_free(&pfx);
}

static void handle_prefill_seq_bin(worker_state *ws, const qwen36_gguf_file *gf, char *rest) {
    prefix_fixture pfx;
    char err[512] = {0};
    uint32_t seq_len = 0, hidden = 0;
    if (sscanf(rest ? rest : "", "%u %u", &seq_len, &hidden) != 2) {
        printf("ERROR bad PREFILL_SEQ_BIN args\n");
        fflush(stdout);
        return;
    }
    if (hidden != ws->hidden) {
        printf("ERROR prefix hidden mismatch\n");
        fflush(stdout);
        return;
    }
    if (ws->fxs[0].layer == 0) {
        printf("ERROR PREFILL_SEQ_BIN requires non-blk.0 first fixture\n");
        fflush(stdout);
        return;
    }
    if (!prefix_alloc_seq(&pfx, seq_len, hidden, 0, 1)) {
        printf("ERROR oom seq bin\n");
        fflush(stdout);
        return;
    }
    if (!read_exact(stdin, pfx.input_seq_ref, (size_t)seq_len * hidden * sizeof(float))) {
        printf("ERROR short PREFILL_SEQ_BIN payload\n");
        fflush(stdout);
        prefix_free(&pfx);
        return;
    }
    worker_reset(ws);
    if (!run_prefill_chain(ws, gf, &pfx, err, sizeof(err))) {
        printf("ERROR %s\n", err[0] ? err : "prefill failed");
        fflush(stdout);
        prefix_free(&pfx);
        return;
    }
    printf("PREFILL_OK %u %u\n", ws->seq_len, ws->hidden);
    fflush(stdout);
    prefix_free(&pfx);
}

static void handle_step(worker_state *ws, const qwen36_gguf_file *gf, char *rest) {
    char err[512] = {0};
    uint32_t token_id;
    if (!ws->prefilled) {
        printf("ERROR not prefilled\n");
        fflush(stdout);
        return;
    }
    if (!rest || !*rest) {
        printf("ERROR missing token\n");
        fflush(stdout);
        return;
    }
    token_id = (uint32_t)strtoul(rest, NULL, 10);
    if (!run_step_chain_token(ws, gf, token_id, err, sizeof(err))) {
        printf("ERROR %s\n", err[0] ? err : "step failed");
        fflush(stdout);
        return;
    }
    printf("STEP_OK %u %u\n", ws->seq_len, ws->hidden);
    fflush(stdout);
}

static void handle_step_prefix(worker_state *ws, const qwen36_gguf_file *gf, const char *path) {
    prefix_fixture pfx;
    char err[512] = {0};
    if (!ws->prefilled) {
        printf("ERROR not prefilled\n");
        fflush(stdout);
        return;
    }
    if (!prefix_load(path, &pfx)) {
        printf("ERROR failed to load prefix fixture\n");
        fflush(stdout);
        return;
    }
    if (!run_step_chain_prefix(ws, gf, &pfx, err, sizeof(err))) {
        printf("ERROR %s\n", err[0] ? err : "step prefix failed");
        fflush(stdout);
        prefix_free(&pfx);
        return;
    }
    printf("STEP_OK %u %u\n", ws->seq_len, ws->hidden);
    fflush(stdout);
    prefix_free(&pfx);
}

static void handle_step_row_bin(worker_state *ws, const qwen36_gguf_file *gf, char *rest) {
    char err[512] = {0};
    float *row = NULL;
    uint32_t hidden = 0;
    if (!ws->prefilled) {
        printf("ERROR not prefilled\n");
        fflush(stdout);
        return;
    }
    if (sscanf(rest ? rest : "", "%u", &hidden) != 1 || hidden != ws->hidden) {
        printf("ERROR bad STEP_ROW_BIN args\n");
        fflush(stdout);
        return;
    }
    row = (float *)malloc((size_t)hidden * sizeof(float));
    if (!row) {
        printf("ERROR oom step row\n");
        fflush(stdout);
        return;
    }
    if (!read_exact(stdin, row, (size_t)hidden * sizeof(float))) {
        printf("ERROR short STEP_ROW_BIN payload\n");
        fflush(stdout);
        free(row);
        return;
    }
    if (!run_step_chain_row(ws, gf, row, err, sizeof(err))) {
        printf("ERROR %s\n", err[0] ? err : "step row failed");
        fflush(stdout);
        free(row);
        return;
    }
    printf("STEP_OK %u %u\n", ws->seq_len, ws->hidden);
    fflush(stdout);
    free(row);
}

static void handle_dump_hidden(worker_state *ws) {
    size_t n, bytes;
    if (!ws->prefilled) {
        printf("ERROR not prefilled\n");
        fflush(stdout);
        return;
    }
    n = (size_t)ws->seq_len * ws->hidden;
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
    printf("LAST %u %zu\n", ws->hidden, (size_t)ws->hidden * sizeof(float));
    fflush(stdout);
    fwrite(ws->output_seq + (size_t)(ws->seq_len - 1u) * ws->hidden, sizeof(float), ws->hidden, stdout);
    fflush(stdout);
}

int main(int argc, char **argv) {
    qwen36_gguf_file gf;
    worker_state ws;
    char err[512];
    char line[4096];
    uint32_t li;

    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf LAYER0.bin [LAYER1.bin ...]\n", argv[0]);
        return 1;
    }
    memset(&gf, 0, sizeof(gf));
    memset(&ws, 0, sizeof(ws));

    if (!qwen36_gguf_open(&gf, argv[1], err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    if (!qwen36_35a3b_q8_bind(&gf, &ws.model, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        qwen36_gguf_close(&gf);
        return 1;
    }
    ws.n_fx = (uint32_t)(argc - 2);
    ws.fxs = (dwf_fixture *)calloc(ws.n_fx, sizeof(dwf_fixture));
    ws.layers = (layer_runtime *)calloc(ws.n_fx, sizeof(layer_runtime));
    if (!ws.fxs || !ws.layers) {
        fprintf(stderr, "oom init worker arrays\n");
        qwen36_gguf_close(&gf);
        return 1;
    }
    for (li = 0; li < ws.n_fx; ++li) {
        if (!fixture_load(argv[2 + li], &ws.fxs[li])) {
            fprintf(stderr, "failed to load fixture: %s\n", argv[2 + li]);
            qwen36_gguf_close(&gf);
            return 1;
        }
        if (li == 0) ws.hidden = ws.fxs[li].hidden;
        if (ws.fxs[li].hidden != ws.hidden) {
            fprintf(stderr, "hidden mismatch across fixtures\n");
            qwen36_gguf_close(&gf);
            return 1;
        }
        ws.layers[li].deltanet_state = (float *)calloc((size_t)ws.fxs[li].num_v_heads * ws.fxs[li].head_k_dim * ws.fxs[li].head_v_dim, sizeof(float));
        for (int i = 0; i < 3; ++i) {
            ws.layers[li].conv_ring[i] = (float *)calloc((size_t)(ws.fxs[li].key_dim * 2u + ws.fxs[li].value_dim), sizeof(float));
        }
        if (!ws.layers[li].deltanet_state || !ws.layers[li].conv_ring[0] || !ws.layers[li].conv_ring[1] || !ws.layers[li].conv_ring[2]) {
            fprintf(stderr, "oom init worker layer %u\n", li);
            qwen36_gguf_close(&gf);
            return 1;
        }
    }

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

        if (strcmp(cmd, "PREFILL_PREFIX") == 0) {
            handle_prefill_prefix(&ws, &gf, rest);
        } else if (strcmp(cmd, "PREFILL_PREFIX_BIN") == 0) {
            handle_prefill_prefix_bin(&ws, &gf, rest);
        } else if (strcmp(cmd, "PREFILL_SEQ_BIN") == 0) {
            handle_prefill_seq_bin(&ws, &gf, rest);
        } else if (strcmp(cmd, "STEP") == 0) {
            handle_step(&ws, &gf, rest);
        } else if (strcmp(cmd, "STEP_PREFIX") == 0) {
            handle_step_prefix(&ws, &gf, rest);
        } else if (strcmp(cmd, "STEP_ROW_BIN") == 0) {
            handle_step_row_bin(&ws, &gf, rest);
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
    for (li = 0; li < ws.n_fx; ++li) {
        free_expert_cache(ws.layers[li].experts);
        for (int i = 0; i < 3; ++i) free(ws.layers[li].conv_ring[i]);
        free(ws.layers[li].deltanet_state);
        fixture_free(&ws.fxs[li]);
    }
    free(ws.layers);
    free(ws.fxs);
    qwen36_gguf_close(&gf);
    return 0;
}
