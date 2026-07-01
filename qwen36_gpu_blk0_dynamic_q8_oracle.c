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

#define MAGIC "Q36DWF02"
#define MAGIC_LEN 8
#define PREFIX_MAGIC "Q36PFX01"
#define PREFIX_MAGIC_LEN 8
#define ROUTER_COUNT 256
#define INTER 512
#define QWEN_RMS_EPS 1e-6f
#define QWEN_SWIGLU_CLAMP 80.0f

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

typedef struct mapped_file {
    int fd;
    void *map;
    uint64_t size;
} mapped_file;

extern int ds4_gpu_embed_tokens_hc_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens_t,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_vocab,
        uint32_t              n_tokens,
        uint32_t              n_embd,
        uint32_t              n_hc);

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
        fclose(fp);
        prefix_free(fx);
        return 0;
    }
    fclose(fp);
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

static inline float sigmoidf_local(float x) {
    if (x >= 0.0f) {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        const float z = expf(x);
        return z / (1.0f + z);
    }
}

static inline float softplusf_local(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}
static inline float siluf_local(float x) { return x / (1.0f + expf(-x)); }

static float cosine(const float *a, const float *b, size_t n) {
    double dot = 0.0, an = 0.0, bn = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += (double)a[i] * b[i];
        an += (double)a[i] * a[i];
        bn += (double)b[i] * b[i];
    }
    if (an == 0.0 || bn == 0.0) return 0.0f;
    return (float)(dot / sqrt(an * bn));
}

static float rmse(const float *a, const float *b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double)a[i] - b[i];
        s += d * d;
    }
    return (float)sqrt(s / (double)n);
}

static void matmul_cpu_rows(const float *mat, const float *x, float *out, uint32_t rows, uint32_t cols) {
    for (uint32_t r = 0; r < rows; ++r) {
        const float *row = mat + (size_t)r * cols;
        double sum = 0.0;
        for (uint32_t c = 0; c < cols; ++c) sum += (double)row[c] * x[c];
        out[r] = (float)sum;
    }
}

static int all_finite_f32(const float *x, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(x[i])) return 0;
    }
    return 1;
}

static void stats_f32(const char *label, const float *x, size_t n) {
    size_t finite = 0, non_finite = 0;
    float minv = 0.0f, maxv = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            ++non_finite;
            continue;
        }
        if (finite == 0) {
            minv = maxv = x[i];
        } else {
            if (x[i] < minv) minv = x[i];
            if (x[i] > maxv) maxv = x[i];
        }
        ++finite;
    }
    printf("%s_stats: finite=%zu non_finite=%zu", label, finite, non_finite);
    if (finite) printf(" min=%.8f max=%.8f", minv, maxv);
    printf("\n");
}

static void print_first_non_finite(const char *label, const float *x, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(x[i])) {
            printf("%s_first_non_finite: index=%zu value=%f\n", label, i, x[i]);
            return;
        }
    }
}

static uint64_t q8_row_bytes(uint32_t cols) {
    return ((uint64_t)cols / 32u) * 34u;
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

static void build_perms(uint32_t n_heads, uint32_t *gg_to_hf, uint32_t *hf_to_gg) {
    uint32_t pos = 0;
    for (uint32_t i = 0; i < n_heads; i += 2) gg_to_hf[pos++] = i;
    for (uint32_t i = 1; i < n_heads; i += 2) gg_to_hf[pos++] = i;
    for (uint32_t i = 0; i < n_heads; ++i) hf_to_gg[gg_to_hf[i]] = i;
}

static void reorder_head_rows_seq_f32(float *dst, const float *src, uint32_t seq_len, uint32_t n_heads, uint32_t head_dim) {
    uint32_t *gg_to_hf = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    if (!gg_to_hf || !hf_to_gg) {
        free(gg_to_hf); free(hf_to_gg);
        return;
    }
    build_perms(n_heads, gg_to_hf, hf_to_gg);
    for (uint32_t t = 0; t < seq_len; ++t) {
        for (uint32_t h = 0; h < n_heads; ++h) {
            memcpy(dst + ((size_t)t * n_heads + h) * head_dim,
                   src + ((size_t)t * n_heads + hf_to_gg[h]) * head_dim,
                   (size_t)head_dim * sizeof(float));
        }
    }
    free(gg_to_hf);
    free(hf_to_gg);
}

static void reorder_head_scalars_seq_f32(float *dst, const float *src, uint32_t seq_len, uint32_t n_heads) {
    uint32_t *gg_to_hf = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    if (!gg_to_hf || !hf_to_gg) {
        free(gg_to_hf); free(hf_to_gg);
        return;
    }
    build_perms(n_heads, gg_to_hf, hf_to_gg);
    for (uint32_t t = 0; t < seq_len; ++t) {
        for (uint32_t h = 0; h < n_heads; ++h) {
            dst[(size_t)t * n_heads + h] = src[(size_t)t * n_heads + hf_to_gg[h]];
        }
    }
    free(gg_to_hf);
    free(hf_to_gg);
}

static void reorder_qkv_v_tokenmajor_gg_to_hf(
        float *dst,
        const float *src,
        uint32_t seq_len,
        uint32_t key_dim,
        uint32_t num_v_heads,
        uint32_t head_v_dim) {
    uint32_t *gg_to_hf_v = (uint32_t *)malloc((size_t)num_v_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg_v = (uint32_t *)malloc((size_t)num_v_heads * sizeof(uint32_t));
    const uint32_t value_dim = num_v_heads * head_v_dim;
    const uint32_t qkv_dim = key_dim * 2u + value_dim;
    const uint32_t v_off = key_dim * 2u;
    if (!gg_to_hf_v || !hf_to_gg_v) {
        free(gg_to_hf_v);
        free(hf_to_gg_v);
        return;
    }
    build_perms(num_v_heads, gg_to_hf_v, hf_to_gg_v);
    for (uint32_t t = 0; t < seq_len; ++t) {
        const float *src_row = src + (size_t)t * qkv_dim;
        float *dst_row = dst + (size_t)t * qkv_dim;
        memcpy(dst_row, src_row, (size_t)(key_dim * 2u) * sizeof(float));
        for (uint32_t h = 0; h < num_v_heads; ++h) {
            memcpy(dst_row + v_off + (size_t)h * head_v_dim,
                   src_row + v_off + (size_t)hf_to_gg_v[h] * head_v_dim,
                   (size_t)head_v_dim * sizeof(float));
        }
    }
    free(gg_to_hf_v);
    free(hf_to_gg_v);
}

static void reorder_out_in_hf_to_gg(float *dst, const float *src, uint32_t seq_len, uint32_t n_heads, uint32_t head_dim) {
    uint32_t *gg_to_hf = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    if (!gg_to_hf || !hf_to_gg) {
        free(gg_to_hf); free(hf_to_gg);
        return;
    }
    build_perms(n_heads, gg_to_hf, hf_to_gg);
    for (uint32_t t = 0; t < seq_len; ++t) {
        for (uint32_t h_gg = 0; h_gg < n_heads; ++h_gg) {
            memcpy(dst + ((size_t)t * n_heads + h_gg) * head_dim,
                   src + ((size_t)t * n_heads + gg_to_hf[h_gg]) * head_dim,
                   (size_t)head_dim * sizeof(float));
        }
    }
    free(gg_to_hf);
    free(hf_to_gg);
}

static int run_gpu_q8_embed(const mapped_file *mf, const qwen36_35a3b_q8_model *q8, const prefix_fixture *pfx, float *embed_seq_out) {
    ds4_gpu_tensor *tokens = NULL, *embed = NULL;
    const uint64_t tokens_bytes = (uint64_t)pfx->seq_len * sizeof(uint32_t);
    const uint64_t embed_bytes = (uint64_t)pfx->seq_len * pfx->hidden * sizeof(float);
    int ok = 0;
    tokens = ds4_gpu_tensor_alloc(tokens_bytes);
    embed = ds4_gpu_tensor_alloc(embed_bytes);
    if (!tokens || !embed) goto cleanup;
    if (ds4_gpu_tensor_write(tokens, 0, pfx->token_ids, tokens_bytes) == 0) goto cleanup;
    if (ds4_gpu_begin_commands() == 0) goto cleanup;
    if (ds4_gpu_embed_tokens_hc_q8_0_tensor(embed, tokens, mf->map, mf->size,
                                            q8->token_embd->abs_offset,
                                            (uint32_t)q8->token_embd->dims[1],
                                            pfx->seq_len, pfx->hidden, 1) == 0) goto cleanup;
    if (ds4_gpu_end_commands() == 0) goto cleanup;
    if (ds4_gpu_tensor_read(embed, 0, embed_seq_out, embed_bytes) == 0) goto cleanup;
    ok = 1;
cleanup:
    ds4_gpu_tensor_free(embed);
    ds4_gpu_tensor_free(tokens);
    return ok;
}

static int run_gpu_q8_matmul_rowwise(
        const mapped_file *mf,
        uint64_t weight_offset,
        uint32_t in_dim,
        uint32_t out_dim,
        const ds4_gpu_tensor *input_seq,
        uint32_t n_tokens,
        float *out_seq_cpu) {
    ds4_gpu_tensor *in_row = NULL, *out_row = NULL;
    const uint64_t in_row_bytes = (uint64_t)in_dim * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)out_dim * sizeof(float);
    int ok = 0;

    out_row = ds4_gpu_tensor_alloc(out_row_bytes);
    if (!out_row) goto cleanup;
    for (uint32_t t = 0; t < n_tokens; ++t) {
        in_row = ds4_gpu_tensor_view(input_seq, (uint64_t)t * in_row_bytes, in_row_bytes);
        if (!in_row) goto cleanup;
        if (ds4_gpu_begin_commands() == 0) goto cleanup;
        if (ds4_gpu_matmul_q8_0_tensor(out_row, mf->map, mf->size,
                                       weight_offset,
                                       in_dim, out_dim, in_row, 1) == 0) goto cleanup;
        if (ds4_gpu_end_commands() == 0) goto cleanup;
        if (ds4_gpu_tensor_read(out_row, 0, out_seq_cpu + (size_t)t * out_dim, out_row_bytes) == 0) goto cleanup;
        ds4_gpu_tensor_free(in_row);
        in_row = NULL;
    }
    ok = 1;

cleanup:
    ds4_gpu_tensor_free(in_row);
    ds4_gpu_tensor_free(out_row);
    return ok;
}

static int run_gpu_q8_qkv_split_rowwise(
        const mapped_file *mf,
        const qwen36_35a3b_q8_layer *layer,
        uint32_t in_dim,
        uint32_t key_dim,
        uint32_t value_dim,
        const ds4_gpu_tensor *input_seq,
        uint32_t n_tokens,
        float *qkv_out_cpu) {
    const uint64_t row_bytes = q8_row_bytes(in_dim);
    const uint64_t q_rows = key_dim;
    const uint64_t k_rows = key_dim;
    const uint64_t q_off = layer->attn_qkv->abs_offset;
    const uint64_t k_off = q_off + q_rows * row_bytes;
    const uint64_t v_off = k_off + k_rows * row_bytes;
    const uint32_t qkv_dim = key_dim * 2u + value_dim;
    float *q_cpu = (float *)malloc((size_t)n_tokens * key_dim * sizeof(float));
    float *k_cpu = (float *)malloc((size_t)n_tokens * key_dim * sizeof(float));
    float *v_cpu = (float *)malloc((size_t)n_tokens * value_dim * sizeof(float));
    if (!q_cpu || !k_cpu || !v_cpu) {
        free(q_cpu);
        free(k_cpu);
        free(v_cpu);
        return 0;
    }
    if (!run_gpu_q8_matmul_rowwise(mf, q_off, in_dim, key_dim, input_seq, n_tokens, q_cpu)) goto fail;
    if (!run_gpu_q8_matmul_rowwise(mf, k_off, in_dim, key_dim, input_seq, n_tokens, k_cpu)) goto fail;
    if (!run_gpu_q8_matmul_rowwise(mf, v_off, in_dim, value_dim, input_seq, n_tokens, v_cpu)) goto fail;
    for (uint32_t t = 0; t < n_tokens; ++t) {
        float *dst = qkv_out_cpu + (size_t)t * qkv_dim;
        memcpy(dst, q_cpu + (size_t)t * key_dim, (size_t)key_dim * sizeof(float));
        memcpy(dst + key_dim, k_cpu + (size_t)t * key_dim, (size_t)key_dim * sizeof(float));
        memcpy(dst + key_dim * 2u, v_cpu + (size_t)t * value_dim, (size_t)value_dim * sizeof(float));
    }
    free(q_cpu);
    free(k_cpu);
    free(v_cpu);
    return 1;
fail:
    free(q_cpu);
    free(k_cpu);
    free(v_cpu);
    return 0;
}

static int run_gpu_ffn_from_residual(
        const mapped_file *mf,
        const qwen36_35a3b_q8_layer *layer,
        const dwf_fixture *fx,
        uint32_t n_tokens,
        const float *residual_in,
        float *out_seq,
        int32_t *router_selected_cpu,
        float *router_weights_cpu) {
    const uint32_t hidden = fx->hidden;
    const uint32_t topk = fx->topk;
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
    if (!router_logits_cpu || !shared_out_cpu || !expert_down_cpu || !routed_out_cpu || !post_cpu) goto cleanup;
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
        for (uint32_t d = 0; d < hidden; ++d) scale_in += post_cpu[(size_t)t * hidden + d] * fx->gate_inp_shexp[d];
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
    if (!ok) ds4_gpu_print_memory_report("qwen36_gpu_hybrid_dynamic_q8_runner_ffn_failure");
    free(router_logits_cpu);
    free(shared_out_cpu);
    free(expert_down_cpu);
    free(routed_out_cpu);
    free(post_cpu);
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

static int run_gpu_hybrid_dynamic(
        const mapped_file *mf,
        const qwen36_35a3b_q8_model *q8,
        const qwen36_35a3b_q8_layer *layer,
        const prefix_fixture *pfx,
        const dwf_fixture *fx,
        const float *input_seq_in,
        float *out_seq,
        int32_t *router_selected_cpu,
        float *router_weights_cpu) {
    const uint32_t n_tokens = pfx->seq_len;
    const uint32_t hidden = pfx->hidden;
    const int have_trace_seq = (n_tokens == fx->seq_len);
    const uint32_t qkv_dim = fx->key_dim * 2u + fx->value_dim;
    const uint32_t rep = fx->num_v_heads / fx->num_k_heads;
    const uint64_t seq_hidden_bytes = (uint64_t)n_tokens * hidden * sizeof(float);
    const uint64_t qkv_bytes = (uint64_t)n_tokens * qkv_dim * sizeof(float);
    const uint64_t z_bytes = (uint64_t)n_tokens * fx->value_dim * sizeof(float);
    const uint64_t ab_bytes = (uint64_t)n_tokens * fx->num_v_heads * sizeof(float);
    const uint64_t out_in_bytes = (uint64_t)n_tokens * fx->value_dim * sizeof(float);
    ds4_gpu_tensor *input = NULL, *input_ln = NULL, *qkv_gpu = NULL, *z_gpu = NULL, *a_gpu = NULL, *b_gpu = NULL, *out_in_gpu = NULL, *out_proj_gpu = NULL;
    float *layer_input_seq = NULL, *qkv_raw = NULL, *qkv = NULL, *z_raw = NULL, *z = NULL, *a_raw = NULL, *a = NULL, *b_raw = NULL, *b = NULL;
    float *conv = NULL, *q = NULL, *k = NULL, *v = NULL, *beta = NULL, *g = NULL, *core = NULL, *state = NULL;
    float *out_in = NULL, *out_in_gg = NULL, *out_proj = NULL, *residual = NULL, *qkv_cpu_contract = NULL;
    int ok = 0;

    layer_input_seq = (float *)malloc(seq_hidden_bytes);
    qkv_raw = (float *)malloc(qkv_bytes);
    qkv = (float *)malloc(qkv_bytes);
    z_raw = (float *)malloc(z_bytes);
    z = (float *)malloc(z_bytes);
    a_raw = (float *)malloc(ab_bytes);
    a = (float *)malloc(ab_bytes);
    b_raw = (float *)malloc(ab_bytes);
    b = (float *)malloc(ab_bytes);
    conv = (float *)malloc(qkv_bytes);
    q = (float *)malloc((uint64_t)n_tokens * fx->num_v_heads * fx->head_k_dim * sizeof(float));
    k = (float *)malloc((uint64_t)n_tokens * fx->num_v_heads * fx->head_k_dim * sizeof(float));
    v = (float *)malloc((uint64_t)n_tokens * fx->num_v_heads * fx->head_v_dim * sizeof(float));
    beta = (float *)malloc(ab_bytes);
    g = (float *)malloc(ab_bytes);
    core = (float *)calloc((uint64_t)n_tokens * fx->num_v_heads * fx->head_v_dim, sizeof(float));
    state = (float *)calloc((uint64_t)fx->num_v_heads * fx->head_k_dim * fx->head_v_dim, sizeof(float));
    out_in = (float *)malloc(out_in_bytes);
    out_in_gg = (float *)malloc(out_in_bytes);
    out_proj = (float *)malloc(seq_hidden_bytes);
    residual = (float *)malloc(seq_hidden_bytes);
    qkv_cpu_contract = (float *)malloc(qkv_bytes);
    if (!layer_input_seq || !qkv_raw || !qkv || !z_raw || !z || !a_raw || !a || !b_raw || !b || !conv || !q || !k || !v ||
        !beta || !g || !core || !state || !out_in || !out_in_gg || !out_proj || !residual || !qkv_cpu_contract) goto cleanup;

    if (input_seq_in) {
        memcpy(layer_input_seq, input_seq_in, seq_hidden_bytes);
    } else {
        if (!run_gpu_q8_embed(mf, q8, pfx, layer_input_seq)) goto cleanup;
    }

    input = ds4_gpu_tensor_alloc(seq_hidden_bytes);
    input_ln = ds4_gpu_tensor_alloc(seq_hidden_bytes);
    qkv_gpu = ds4_gpu_tensor_alloc(qkv_bytes);
    z_gpu = ds4_gpu_tensor_alloc(z_bytes);
    a_gpu = ds4_gpu_tensor_alloc(ab_bytes);
    b_gpu = ds4_gpu_tensor_alloc(ab_bytes);
    out_in_gpu = ds4_gpu_tensor_alloc(out_in_bytes);
    out_proj_gpu = ds4_gpu_tensor_alloc(seq_hidden_bytes);
    if (!input || !input_ln || !qkv_gpu || !z_gpu || !a_gpu || !b_gpu || !out_in_gpu || !out_proj_gpu) goto cleanup;

    if (ds4_gpu_tensor_write(input, 0, layer_input_seq, seq_hidden_bytes) == 0) goto cleanup;
    if (ds4_gpu_begin_commands() == 0) goto cleanup;
    if (ds4_gpu_rms_norm_weight_rows_tensor(input_ln, input, mf->map, mf->size,
                                            layer->attn_norm->abs_offset,
                                            hidden, n_tokens, QWEN_RMS_EPS) == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(z_gpu, mf->map, mf->size,
                                   layer->attn_gate->abs_offset,
                                   hidden, fx->value_dim, input_ln, n_tokens) == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(a_gpu, mf->map, mf->size,
                                   layer->ssm_alpha->abs_offset,
                                   hidden, fx->num_v_heads, input_ln, n_tokens) == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(b_gpu, mf->map, mf->size,
                                   layer->ssm_beta->abs_offset,
                                   hidden, fx->num_v_heads, input_ln, n_tokens) == 0) goto cleanup;
    if (ds4_gpu_end_commands() == 0) goto cleanup;

    if (!run_gpu_q8_qkv_split_rowwise(mf, layer, hidden, fx->key_dim, fx->value_dim, input_ln, n_tokens, qkv_raw)) goto cleanup;
    if (ds4_gpu_tensor_read(input_ln, 0, residual, seq_hidden_bytes) == 0) goto cleanup;
    if (ds4_gpu_tensor_read(z_gpu, 0, z_raw, z_bytes) == 0) goto cleanup;
    if (ds4_gpu_tensor_read(a_gpu, 0, a_raw, ab_bytes) == 0) goto cleanup;
    if (ds4_gpu_tensor_read(b_gpu, 0, b_raw, ab_bytes) == 0) goto cleanup;

    if (have_trace_seq) {
        printf("input_ln_gpu_vs_trace_rmse: %.8f\n", rmse(residual, fx->input_ln_seq, (size_t)n_tokens * hidden));
        printf("input_ln_gpu_vs_trace_cosine: %.8f\n", cosine(residual, fx->input_ln_seq, (size_t)n_tokens * hidden));
        printf("qkv_raw_rmse: %.8f\n", rmse(qkv_raw, fx->qkv_seq, (size_t)n_tokens * qkv_dim));
        printf("qkv_raw_cosine: %.8f\n", cosine(qkv_raw, fx->qkv_seq, (size_t)n_tokens * qkv_dim));
        printf("z_raw_rmse: %.8f\n", rmse(z_raw, fx->z_seq, (size_t)n_tokens * fx->value_dim));
        printf("z_raw_cosine: %.8f\n", cosine(z_raw, fx->z_seq, (size_t)n_tokens * fx->value_dim));
        printf("a_raw_rmse: %.8f\n", rmse(a_raw, fx->a_seq, (size_t)n_tokens * fx->num_v_heads));
        printf("a_raw_cosine: %.8f\n", cosine(a_raw, fx->a_seq, (size_t)n_tokens * fx->num_v_heads));
        printf("b_raw_rmse: %.8f\n", rmse(b_raw, fx->b_seq, (size_t)n_tokens * fx->num_v_heads));
        printf("b_raw_cosine: %.8f\n", cosine(b_raw, fx->b_seq, (size_t)n_tokens * fx->num_v_heads));
    } else {
        printf("dynamic_seq_len: %u fixture_seq_len: %u\n", n_tokens, fx->seq_len);
    }
    stats_f32("qkv_raw", qkv_raw, (size_t)n_tokens * qkv_dim);
    stats_f32("z_raw", z_raw, (size_t)n_tokens * fx->value_dim);
    stats_f32("a_raw", a_raw, (size_t)n_tokens * fx->num_v_heads);
    stats_f32("b_raw", b_raw, (size_t)n_tokens * fx->num_v_heads);
    if (!all_finite_f32(qkv_raw, (size_t)n_tokens * qkv_dim)) print_first_non_finite("qkv_raw", qkv_raw, (size_t)n_tokens * qkv_dim);
    if (!all_finite_f32(z_raw, (size_t)n_tokens * fx->value_dim)) print_first_non_finite("z_raw", z_raw, (size_t)n_tokens * fx->value_dim);
    if (!all_finite_f32(a_raw, (size_t)n_tokens * fx->num_v_heads)) print_first_non_finite("a_raw", a_raw, (size_t)n_tokens * fx->num_v_heads);
    if (!all_finite_f32(b_raw, (size_t)n_tokens * fx->num_v_heads)) print_first_non_finite("b_raw", b_raw, (size_t)n_tokens * fx->num_v_heads);

    memcpy(qkv, qkv_raw, qkv_bytes);
    reorder_head_rows_seq_f32(z, z_raw, n_tokens, fx->num_v_heads, fx->head_v_dim);
    reorder_head_scalars_seq_f32(a, a_raw, n_tokens, fx->num_v_heads);
    reorder_head_scalars_seq_f32(b, b_raw, n_tokens, fx->num_v_heads);
    reorder_qkv_v_tokenmajor_gg_to_hf(qkv, qkv_raw, n_tokens,
                                      fx->key_dim,
                                      fx->num_v_heads, fx->head_v_dim);
    if (have_trace_seq) {
        printf("qkv_reordered_rmse: %.8f\n", rmse(qkv, fx->qkv_seq, (size_t)n_tokens * qkv_dim));
        printf("qkv_reordered_cosine: %.8f\n", cosine(qkv, fx->qkv_seq, (size_t)n_tokens * qkv_dim));
        for (uint32_t t = 0; t < n_tokens; ++t) {
            matmul_cpu_rows(fx->w_qkv,
                            fx->input_ln_seq + (size_t)t * hidden,
                            qkv_cpu_contract + (size_t)t * qkv_dim,
                            qkv_dim,
                            hidden);
        }
        printf("qkv_cpu_contract_vs_trace_rmse: %.8f\n", rmse(qkv_cpu_contract, fx->qkv_seq, (size_t)n_tokens * qkv_dim));
        printf("qkv_cpu_contract_vs_trace_cosine: %.8f\n", cosine(qkv_cpu_contract, fx->qkv_seq, (size_t)n_tokens * qkv_dim));
        printf("qkv_gpu_vs_cpu_contract_rmse: %.8f\n", rmse(qkv, qkv_cpu_contract, (size_t)n_tokens * qkv_dim));
        printf("qkv_gpu_vs_cpu_contract_cosine: %.8f\n", cosine(qkv, qkv_cpu_contract, (size_t)n_tokens * qkv_dim));
        {
            const size_t q_sz = (size_t)n_tokens * fx->key_dim;
            const size_t k_sz = (size_t)n_tokens * fx->key_dim;
            const size_t v_sz = (size_t)n_tokens * fx->value_dim;
            printf("q_proj_gpu_vs_cpu_contract_rmse: %.8f\n", rmse(qkv, qkv_cpu_contract, q_sz));
            printf("q_proj_gpu_vs_cpu_contract_cosine: %.8f\n", cosine(qkv, qkv_cpu_contract, q_sz));
            printf("k_proj_gpu_vs_cpu_contract_rmse: %.8f\n", rmse(qkv + q_sz, qkv_cpu_contract + q_sz, k_sz));
            printf("k_proj_gpu_vs_cpu_contract_cosine: %.8f\n", cosine(qkv + q_sz, qkv_cpu_contract + q_sz, k_sz));
            printf("v_proj_gpu_vs_cpu_contract_rmse: %.8f\n", rmse(qkv + q_sz + k_sz, qkv_cpu_contract + q_sz + k_sz, v_sz));
            printf("v_proj_gpu_vs_cpu_contract_cosine: %.8f\n", cosine(qkv + q_sz + k_sz, qkv_cpu_contract + q_sz + k_sz, v_sz));
        }
        printf("z_reordered_rmse: %.8f\n", rmse(z, fx->z_seq, (size_t)n_tokens * fx->value_dim));
        printf("z_reordered_cosine: %.8f\n", cosine(z, fx->z_seq, (size_t)n_tokens * fx->value_dim));
        printf("a_reordered_rmse: %.8f\n", rmse(a, fx->a_seq, (size_t)n_tokens * fx->num_v_heads));
        printf("a_reordered_cosine: %.8f\n", cosine(a, fx->a_seq, (size_t)n_tokens * fx->num_v_heads));
        printf("b_reordered_rmse: %.8f\n", rmse(b, fx->b_seq, (size_t)n_tokens * fx->num_v_heads));
        printf("b_reordered_cosine: %.8f\n", cosine(b, fx->b_seq, (size_t)n_tokens * fx->num_v_heads));
    }
    stats_f32("a_reordered", a, (size_t)n_tokens * fx->num_v_heads);
    stats_f32("b_reordered", b, (size_t)n_tokens * fx->num_v_heads);

    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t d = 0; d < qkv_dim; ++d) {
            double sum = 0.0;
            for (uint32_t kk = 0; kk < 4; ++kk) {
                int src_t = (int)t - 3 + (int)kk;
                if (src_t >= 0 && src_t < (int)n_tokens) {
                    sum += (double)fx->conv_w[(size_t)d * 4u + kk] * qkv[(size_t)src_t * qkv_dim + d];
                }
            }
            conv[(size_t)t * qkv_dim + d] = siluf_local((float)sum);
        }
    }
    if (have_trace_seq) {
        float *conv_ref = (float *)malloc((size_t)n_tokens * qkv_dim * sizeof(float));
        if (conv_ref) {
            for (uint32_t t = 0; t < n_tokens; ++t) {
                for (uint32_t d = 0; d < qkv_dim; ++d) {
                    conv_ref[(size_t)t * qkv_dim + d] = siluf_local(fx->conv_raw[(size_t)d * (n_tokens + 3u) + t]);
                }
            }
            printf("conv_post_rmse: %.8f\n", rmse(conv, conv_ref, (size_t)n_tokens * qkv_dim));
            printf("conv_post_cosine: %.8f\n", cosine(conv, conv_ref, (size_t)n_tokens * qkv_dim));
            free(conv_ref);
        }
    }
    for (uint32_t t = 0; t < n_tokens; ++t) {
        const float *cq = conv + (size_t)t * qkv_dim;
        const float *ck = cq + fx->key_dim;
        const float *cv = ck + fx->key_dim;
        for (uint32_t h = 0; h < fx->num_k_heads; ++h) {
            for (uint32_t d = 0; d < fx->head_k_dim; ++d) {
                const float qv = cq[h * fx->head_k_dim + d];
                const float kv = ck[h * fx->head_k_dim + d];
                for (uint32_t vh = 0; vh < rep; ++vh) {
                    const uint32_t dst_h = h * rep + vh;
                    q[((size_t)t * fx->num_v_heads + dst_h) * fx->head_k_dim + d] = qv;
                    k[((size_t)t * fx->num_v_heads + dst_h) * fx->head_k_dim + d] = kv;
                }
            }
        }
        for (uint32_t h = 0; h < fx->num_v_heads; ++h) {
            memcpy(v + ((size_t)t * fx->num_v_heads + h) * fx->head_v_dim,
                   cv + (size_t)h * fx->head_v_dim,
                   (size_t)fx->head_v_dim * sizeof(float));
            beta[(size_t)t * fx->num_v_heads + h] = sigmoidf_local(b[(size_t)t * fx->num_v_heads + h]);
            g[(size_t)t * fx->num_v_heads + h] = fx->A_log[h] * softplusf_local(a[(size_t)t * fx->num_v_heads + h] + fx->dt_bias[h]);
        }
    }
    if (have_trace_seq) {
        printf("query_rmse: %.8f\n", rmse(q, fx->q_ref, (size_t)n_tokens * fx->num_v_heads * fx->head_k_dim));
        printf("query_cosine: %.8f\n", cosine(q, fx->q_ref, (size_t)n_tokens * fx->num_v_heads * fx->head_k_dim));
        printf("key_rmse: %.8f\n", rmse(k, fx->k_ref, (size_t)n_tokens * fx->num_v_heads * fx->head_k_dim));
        printf("key_cosine: %.8f\n", cosine(k, fx->k_ref, (size_t)n_tokens * fx->num_v_heads * fx->head_k_dim));
        printf("value_rmse: %.8f\n", rmse(v, fx->v_ref, (size_t)n_tokens * fx->num_v_heads * fx->head_v_dim));
        printf("value_cosine: %.8f\n", cosine(v, fx->v_ref, (size_t)n_tokens * fx->num_v_heads * fx->head_v_dim));
    }
    if (!all_finite_f32(beta, (size_t)n_tokens * fx->num_v_heads) ||
        !all_finite_f32(g, (size_t)n_tokens * fx->num_v_heads)) {
        stats_f32("beta", beta, (size_t)n_tokens * fx->num_v_heads);
        stats_f32("g", g, (size_t)n_tokens * fx->num_v_heads);
        print_first_non_finite("beta", beta, (size_t)n_tokens * fx->num_v_heads);
        print_first_non_finite("g", g, (size_t)n_tokens * fx->num_v_heads);
        fprintf(stderr, "qwen36_gpu_hybrid_dynamic_q8_runner: non-finite beta/g\n");
        goto cleanup;
    }
    if (have_trace_seq) {
        printf("beta_rmse: %.8f\n", rmse(beta, fx->beta_ref, (size_t)n_tokens * fx->num_v_heads));
        printf("beta_cosine: %.8f\n", cosine(beta, fx->beta_ref, (size_t)n_tokens * fx->num_v_heads));
        printf("g_rmse: %.8f\n", rmse(g, fx->g_ref, (size_t)n_tokens * fx->num_v_heads));
        printf("g_cosine: %.8f\n", cosine(g, fx->g_ref, (size_t)n_tokens * fx->num_v_heads));
    }
    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t h = 0; h < fx->num_v_heads; ++h) {
            float qnorm = 0.0f, knorm = 0.0f;
            const float gexp = expf(g[(size_t)t * fx->num_v_heads + h]);
            const float beta_t = beta[(size_t)t * fx->num_v_heads + h];
            const size_t qbase = ((size_t)t * fx->num_v_heads + h) * fx->head_k_dim;
            const size_t vbase = ((size_t)t * fx->num_v_heads + h) * fx->head_v_dim;
            const size_t sbase = (size_t)h * fx->head_k_dim * fx->head_v_dim;
            for (uint32_t hd = 0; hd < fx->head_k_dim; ++hd) {
                qnorm += q[qbase + hd] * q[qbase + hd];
                knorm += k[qbase + hd] * k[qbase + hd];
            }
            qnorm = 1.0f / sqrtf(qnorm + 1e-6f);
            knorm = 1.0f / sqrtf(knorm + 1e-6f);
            for (uint32_t hd = 0; hd < fx->head_k_dim * fx->head_v_dim; ++hd) state[sbase + hd] *= gexp;
            {
                float delta[128];
                for (uint32_t vd = 0; vd < fx->head_v_dim; ++vd) {
                    float kv_mem = 0.0f;
                    for (uint32_t hd = 0; hd < fx->head_k_dim; ++hd) {
                        kv_mem += state[sbase + (size_t)hd * fx->head_v_dim + vd] * (k[qbase + hd] * knorm);
                    }
                    delta[vd] = (v[vbase + vd] - kv_mem) * beta_t;
                }
                for (uint32_t hd = 0; hd < fx->head_k_dim; ++hd) {
                    const float kval = k[qbase + hd] * knorm;
                    for (uint32_t vd = 0; vd < fx->head_v_dim; ++vd) {
                        state[sbase + (size_t)hd * fx->head_v_dim + vd] += kval * delta[vd];
                    }
                }
                for (uint32_t vd = 0; vd < fx->head_v_dim; ++vd) {
                    float outv = 0.0f;
                    for (uint32_t hd = 0; hd < fx->head_k_dim; ++hd) {
                        outv += state[sbase + (size_t)hd * fx->head_v_dim + vd] *
                                (q[qbase + hd] * qnorm / sqrtf((float)fx->head_k_dim));
                    }
                    core[vbase + vd] = outv;
                }
            }
        }
    }
    if (!all_finite_f32(core, (size_t)n_tokens * fx->num_v_heads * fx->head_v_dim)) {
        fprintf(stderr, "qwen36_gpu_hybrid_dynamic_q8_runner: non-finite core\n");
        goto cleanup;
    }
    if (have_trace_seq) {
        printf("core_rmse: %.8f\n", rmse(core, fx->core_ref, (size_t)n_tokens * fx->num_v_heads * fx->head_v_dim));
        printf("core_cosine: %.8f\n", cosine(core, fx->core_ref, (size_t)n_tokens * fx->num_v_heads * fx->head_v_dim));
    }
    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t h = 0; h < fx->num_v_heads; ++h) {
            const size_t base = (size_t)t * fx->value_dim + (size_t)h * fx->head_v_dim;
            double var = 0.0;
            for (uint32_t vd = 0; vd < fx->head_v_dim; ++vd) {
                const double cv = core[((size_t)t * fx->num_v_heads + h) * fx->head_v_dim + vd];
                var += cv * cv;
            }
            var /= (double)fx->head_v_dim;
            for (uint32_t vd = 0; vd < fx->head_v_dim; ++vd) {
                const float cv = core[((size_t)t * fx->num_v_heads + h) * fx->head_v_dim + vd];
                out_in[base + vd] = (float)(cv / sqrt(var + 1e-6)) *
                                    fx->ssm_norm_w[vd] *
                                    siluf_local(z[(size_t)t * fx->value_dim + (size_t)h * fx->head_v_dim + vd]);
            }
        }
    }
    if (!all_finite_f32(out_in, (size_t)n_tokens * fx->value_dim)) {
        fprintf(stderr, "qwen36_gpu_hybrid_dynamic_q8_runner: non-finite out_in\n");
        goto cleanup;
    }
    if (have_trace_seq) {
        printf("out_in_rmse: %.8f\n", rmse(out_in, fx->out_in_seq, (size_t)n_tokens * fx->value_dim));
        printf("out_in_cosine: %.8f\n", cosine(out_in, fx->out_in_seq, (size_t)n_tokens * fx->value_dim));
    }

    reorder_out_in_hf_to_gg(out_in_gg, out_in, n_tokens, fx->num_v_heads, fx->head_v_dim);
    if (ds4_gpu_tensor_write(out_in_gpu, 0, out_in_gg, out_in_bytes) == 0) goto cleanup;
    if (ds4_gpu_begin_commands() == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(out_proj_gpu, mf->map, mf->size,
                                   layer->ssm_out->abs_offset,
                                   fx->value_dim, hidden, out_in_gpu, n_tokens) == 0) goto cleanup;
    if (ds4_gpu_end_commands() == 0) goto cleanup;
    if (ds4_gpu_tensor_read(out_proj_gpu, 0, out_proj, seq_hidden_bytes) == 0) goto cleanup;
    if (!all_finite_f32(out_proj, (size_t)n_tokens * hidden)) {
        fprintf(stderr, "qwen36_gpu_hybrid_dynamic_q8_runner: non-finite out_proj\n");
        goto cleanup;
    }
    if (have_trace_seq) {
        printf("out_proj_rmse: %.8f\n", rmse(out_proj, fx->out_proj_out_seq, (size_t)n_tokens * hidden));
        printf("out_proj_cosine: %.8f\n", cosine(out_proj, fx->out_proj_out_seq, (size_t)n_tokens * hidden));
    }

    for (uint32_t t = 0; t < n_tokens; ++t) {
        for (uint32_t d = 0; d < hidden; ++d) {
            const size_t idx = (size_t)t * hidden + d;
            residual[idx] = layer_input_seq[idx] + out_proj[idx];
        }
    }
    if (!all_finite_f32(residual, (size_t)n_tokens * hidden)) {
        fprintf(stderr, "qwen36_gpu_hybrid_dynamic_q8_runner: non-finite residual\n");
        goto cleanup;
    }
    if (have_trace_seq) {
        printf("residual_after_mixer_seq_rmse: %.8f\n", rmse(residual, fx->residual_after_mixer_seq_ref, (size_t)n_tokens * hidden));
        printf("residual_after_mixer_seq_cosine: %.8f\n", cosine(residual, fx->residual_after_mixer_seq_ref, (size_t)n_tokens * hidden));
    }

    if (!run_gpu_ffn_from_residual(mf, layer, fx, n_tokens, residual, out_seq, router_selected_cpu, router_weights_cpu)) goto cleanup;
    ok = 1;

cleanup:
    ds4_gpu_tensor_free(out_proj_gpu);
    ds4_gpu_tensor_free(out_in_gpu);
    ds4_gpu_tensor_free(b_gpu);
    ds4_gpu_tensor_free(a_gpu);
    ds4_gpu_tensor_free(z_gpu);
    ds4_gpu_tensor_free(qkv_gpu);
    ds4_gpu_tensor_free(input_ln);
    ds4_gpu_tensor_free(input);
    free(layer_input_seq);
    free(qkv_raw);
    free(qkv);
    free(z_raw);
    free(z);
    free(a_raw);
    free(a);
    free(b_raw);
    free(b);
    free(conv);
    free(q);
    free(k);
    free(v);
    free(beta);
    free(g);
    free(core);
    free(state);
    free(out_in);
    free(out_in_gg);
    free(out_proj);
    free(residual);
    free(qkv_cpu_contract);
    return ok;
}

int main(int argc, char **argv) {
    const char *gguf_path, *prefix_path = NULL, *dump_seq_path = NULL;
    qwen36_gguf_file gf;
    qwen36_35a3b_q8_model q8;
    dwf_fixture fx;
    dwf_fixture *fxs = NULL;
    prefix_fixture pfx;
    mapped_file mf;
    float *out_seq = NULL;
    float *state_seq = NULL;
    int32_t *router_selected = NULL;
    float *router_weights = NULL;
    char err[512];
    int ok = 0;
    int n_fx = 0;
    int use_prefix_input_seq = 0;

    memset(&gf, 0, sizeof(gf));
    memset(&fx, 0, sizeof(fx));
    memset(&pfx, 0, sizeof(pfx));
    memset(&mf, 0, sizeof(mf));
    mf.fd = -1;

    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL.gguf LAYER.bin [LAYER.bin ...] --prefix PREFIX.bin [--use-prefix-input-seq] [--dump-seq PATH]\n", argv[0]);
        fprintf(stderr, "note: one fixture runs one GPU-owned hybrid layer; multiple fixtures chain a GPU-owned hybrid prefix.\n");
        return 1;
    }
    gguf_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--prefix") == 0 && i + 1 < argc) { prefix_path = argv[++i]; continue; }
        if (strcmp(argv[i], "--dump-seq") == 0 && i + 1 < argc) { dump_seq_path = argv[++i]; continue; }
        if (strcmp(argv[i], "--use-prefix-input-seq") == 0) { use_prefix_input_seq = 1; continue; }
        ++n_fx;
    }

    if (!prefix_path || n_fx <= 0) { fprintf(stderr, "--prefix is required\n"); return 1; }
    if (!prefix_load(prefix_path, &pfx)) {
        fprintf(stderr, "failed to load prefix fixture: %s\n", prefix_path);
        goto cleanup;
    }
    if (!qwen36_gguf_open(&gf, gguf_path, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        goto cleanup;
    }
    if (!qwen36_35a3b_q8_bind(&gf, &q8, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        goto cleanup;
    }
    fxs = (dwf_fixture *)calloc((size_t)n_fx, sizeof(dwf_fixture));
    if (!fxs) goto cleanup;
    {
        int fxi = 0;
        for (int i = 2; i < argc; ++i) {
            if (strcmp(argv[i], "--prefix") == 0 || strcmp(argv[i], "--dump-seq") == 0) { ++i; continue; }
            if (strcmp(argv[i], "--use-prefix-input-seq") == 0) continue;
            if (!fixture_load(argv[i], &fxs[fxi])) {
                fprintf(stderr, "failed to load layer fixture: %s\n", argv[i]);
                goto cleanup;
            }
            if (fxs[fxi].hidden != pfx.hidden) {
                fprintf(stderr, "fixture hidden mismatch: %s\n", argv[i]);
                goto cleanup;
            }
            if (q8.layers[fxs[fxi].layer].kind != QWEN36_LAYER_KIND_HYBRID_SSM) {
                fprintf(stderr, "only hybrid SSM layers are supported: blk.%u\n", fxs[fxi].layer);
                goto cleanup;
            }
            ++fxi;
        }
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

    out_seq = (float *)malloc((size_t)pfx.seq_len * pfx.hidden * sizeof(float));
    router_selected = (int32_t *)malloc((size_t)pfx.seq_len * fxs[0].topk * sizeof(int32_t));
    router_weights = (float *)malloc((size_t)pfx.seq_len * fxs[0].topk * sizeof(float));
    if (!out_seq || !router_selected || !router_weights) goto cleanup;

    state_seq = (float *)malloc((size_t)pfx.seq_len * pfx.hidden * sizeof(float));
    if (!state_seq) goto cleanup;
    if (use_prefix_input_seq) {
        memcpy(state_seq, pfx.input_seq_ref, (size_t)pfx.seq_len * pfx.hidden * sizeof(float));
        printf("prefix_source: input_seq_ref\n");
    } else {
        memset(state_seq, 0, (size_t)pfx.seq_len * pfx.hidden * sizeof(float));
        printf("prefix_source: token_embd\n");
    }

    for (int fxi = 0; fxi < n_fx; ++fxi) {
        dwf_fixture *cur = &fxs[fxi];
        const qwen36_35a3b_q8_layer *layer = &q8.layers[cur->layer];
        const float *layer_input_seq_in = ((fxi == 0) && !use_prefix_input_seq) ? NULL : state_seq;
        memset(router_selected, 0, (size_t)pfx.seq_len * cur->topk * sizeof(int32_t));
        memset(router_weights, 0, (size_t)pfx.seq_len * cur->topk * sizeof(float));
        if (!run_gpu_hybrid_dynamic(&mf, &q8, layer, &pfx, cur, layer_input_seq_in, out_seq, router_selected, router_weights)) {
            fprintf(stderr, "gpu hybrid dynamic failed: blk.%u\n", cur->layer);
            goto cleanup;
        }
        printf("gpu_layer_done: blk.%u seq_len=%u\n", cur->layer, pfx.seq_len);
        if (n_fx == 1 && pfx.seq_len == cur->seq_len) {
            printf("blk%u_dynamic_seq_rmse: %.8f\n", cur->layer, rmse(out_seq, cur->layer_output_seq_ref, (size_t)cur->seq_len * cur->hidden));
            printf("blk%u_dynamic_seq_cosine: %.8f\n", cur->layer, cosine(out_seq, cur->layer_output_seq_ref, (size_t)cur->seq_len * cur->hidden));
            printf("blk%u_dynamic_last_rmse: %.8f\n", cur->layer, rmse(out_seq + (size_t)(cur->seq_len - 1) * cur->hidden, cur->layer_output_ref, cur->hidden));
            printf("blk%u_dynamic_last_cosine: %.8f\n", cur->layer, cosine(out_seq + (size_t)(cur->seq_len - 1) * cur->hidden, cur->layer_output_ref, cur->hidden));
            {
                uint32_t topk_match = 1;
                for (uint32_t t = 0; t < cur->seq_len && topk_match; ++t) {
                    for (uint32_t j = 0; j < cur->topk; ++j) {
                        if (router_selected[(size_t)t * cur->topk + j] != (int32_t)cur->router_indices_seq_ref_f32[(size_t)t * cur->topk + j]) {
                            topk_match = 0;
                            break;
                        }
                    }
                }
                printf("router_topk_exact: %s\n", topk_match ? "true" : "false");
                printf("router_scores_rmse: %.8f\n", rmse(router_weights, cur->router_scores_seq_ref, (size_t)cur->seq_len * cur->topk));
            }
        }
        if (state_seq) {
            memcpy(state_seq, out_seq, (size_t)pfx.seq_len * pfx.hidden * sizeof(float));
        }
    }

    if (dump_seq_path) {
        FILE *fp = fopen(dump_seq_path, "wb");
        if (!fp) {
            fprintf(stderr, "failed to open dump path: %s\n", dump_seq_path);
            goto cleanup;
        }
        fwrite(out_seq, sizeof(float), (size_t)pfx.seq_len * pfx.hidden, fp);
        fclose(fp);
        printf("wrote_seq: %s\n", dump_seq_path);
    }
    ok = 1;

cleanup:
    if (mf.map || mf.fd >= 0) mapped_file_close(&mf);
    ds4_gpu_cleanup();
    qwen36_gguf_close(&gf);
    prefix_free(&pfx);
    if (fxs) {
        for (int i = 0; i < n_fx; ++i) fixture_free(&fxs[i]);
    }
    free(fxs);
    free(out_seq);
    free(state_seq);
    free(router_selected);
    free(router_weights);
    return ok ? 0 : 1;
}
