#include "qwen36_35a3b_q8.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "Q36DWF02"
#define MAGIC_LEN 8
#define ROUTER_COUNT 256

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

static int read_exact(FILE *fp, void *buf, size_t n) {
    return fread(buf, 1, n, fp) == n;
}

static int alloc_read_f32(FILE *fp, float **out, size_t count) {
    float *p = (float *)malloc(count * sizeof(float));
    if (!p) return 0;
    if (!read_exact(fp, p, count * sizeof(float))) {
        free(p);
        return 0;
    }
    *out = p;
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
    if (!read_exact(fp, magic, MAGIC_LEN) || memcmp(magic, MAGIC, MAGIC_LEN) != 0) {
        fclose(fp);
        return 0;
    }
    if (!read_exact(fp, &fx->layer, sizeof(uint32_t)) || !read_exact(fp, &fx->seq_len, sizeof(uint32_t)) || !read_exact(fp, &fx->hidden, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->num_v_heads, sizeof(uint32_t)) || !read_exact(fp, &fx->num_k_heads, sizeof(uint32_t)) || !read_exact(fp, &fx->head_k_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->head_v_dim, sizeof(uint32_t)) || !read_exact(fp, &fx->key_dim, sizeof(uint32_t)) || !read_exact(fp, &fx->value_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->topk, sizeof(uint32_t)) || !read_exact(fp, &fx->union_experts, sizeof(uint32_t))) {
        fclose(fp);
        return 0;
    }
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
    R(gate_sel, (size_t)fx->union_experts * 512 * fx->hidden); R(up_sel, (size_t)fx->union_experts * 512 * fx->hidden); R(down_sel, (size_t)fx->union_experts * fx->hidden * 512);
    R(gate_shexp, (size_t)512 * fx->hidden); R(up_shexp, (size_t)512 * fx->hidden); R(down_shexp, (size_t)fx->hidden * 512); R(gate_inp_shexp, fx->hidden);
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

static void build_perms(uint32_t n_heads, uint32_t *gg_to_hf, uint32_t *hf_to_gg) {
    uint32_t pos = 0;
    for (uint32_t i = 0; i < n_heads; i += 2) gg_to_hf[pos++] = i;
    for (uint32_t i = 1; i < n_heads; i += 2) gg_to_hf[pos++] = i;
    for (uint32_t i = 0; i < n_heads; ++i) hf_to_gg[gg_to_hf[i]] = i;
}

static void reorder_head_rows_matrix(float *dst, const float *src, uint32_t n_heads, uint32_t rows_per_head, uint32_t cols) {
    uint32_t *gg_to_hf = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    if (!gg_to_hf || !hf_to_gg) {
        free(gg_to_hf);
        free(hf_to_gg);
        return;
    }
    build_perms(n_heads, gg_to_hf, hf_to_gg);
    for (uint32_t h = 0; h < n_heads; ++h) {
        memcpy(dst + (size_t)h * rows_per_head * cols,
               src + (size_t)hf_to_gg[h] * rows_per_head * cols,
               (size_t)rows_per_head * cols * sizeof(float));
    }
    free(gg_to_hf);
    free(hf_to_gg);
}

static void reorder_head_scalars(float *dst, const float *src, uint32_t n_heads) {
    uint32_t *gg_to_hf = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    if (!gg_to_hf || !hf_to_gg) {
        free(gg_to_hf);
        free(hf_to_gg);
        return;
    }
    build_perms(n_heads, gg_to_hf, hf_to_gg);
    for (uint32_t h = 0; h < n_heads; ++h) {
        dst[h] = src[hf_to_gg[h]];
    }
    free(gg_to_hf);
    free(hf_to_gg);
}

static void reorder_qkv_v_rows_matrix_gg_to_hf(
        float *dst,
        const float *src,
        uint32_t key_dim,
        uint32_t n_heads,
        uint32_t head_dim,
        uint32_t cols) {
    uint32_t *gg_to_hf = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    const uint32_t v_off = key_dim * 2u;
    if (!gg_to_hf || !hf_to_gg) {
        free(gg_to_hf);
        free(hf_to_gg);
        return;
    }
    build_perms(n_heads, gg_to_hf, hf_to_gg);
    memcpy(dst, src, (size_t)(key_dim * 2u) * cols * sizeof(float));
    for (uint32_t h = 0; h < n_heads; ++h) {
        memcpy(dst + (size_t)(v_off + h * head_dim) * cols,
               src + (size_t)(v_off + hf_to_gg[h] * head_dim) * cols,
               (size_t)head_dim * cols * sizeof(float));
    }
    free(gg_to_hf);
    free(hf_to_gg);
}

static void reorder_v_head_cols_matrix_gg_to_hf(float *dst, const float *src, uint32_t rows, uint32_t n_heads, uint32_t head_dim) {
    uint32_t *gg_to_hf = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    if (!gg_to_hf || !hf_to_gg) {
        free(gg_to_hf);
        free(hf_to_gg);
        return;
    }
    build_perms(n_heads, gg_to_hf, hf_to_gg);
    for (uint32_t row = 0; row < rows; ++row) {
        const float *src_row = src + (size_t)row * n_heads * head_dim;
        float *dst_row = dst + (size_t)row * n_heads * head_dim;
        for (uint32_t h_hf = 0; h_hf < n_heads; ++h_hf) {
            memcpy(dst_row + (size_t)h_hf * head_dim,
                   src_row + (size_t)hf_to_gg[h_hf] * head_dim,
                   (size_t)head_dim * sizeof(float));
        }
    }
    free(gg_to_hf);
    free(hf_to_gg);
}

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

static void report_cmp(const char *label, const float *got, const float *ref, size_t n) {
    printf("%s_rmse: %.8f\n", label, rmse(got, ref, n));
    printf("%s_cosine: %.8f\n", label, cosine(got, ref, n));
}

int main(int argc, char **argv) {
    const char *gguf_path, *fixture_path;
    qwen36_gguf_file gf;
    qwen36_35a3b_q8_model q8;
    dwf_fixture fx;
    char err[512];
    const qwen36_35a3b_q8_layer *layer;
    float *tmp = NULL, *tmp2 = NULL, *tmp3 = NULL;

    setvbuf(stdout, NULL, _IONBF, 0);

    memset(&gf, 0, sizeof(gf));
    memset(&q8, 0, sizeof(q8));
    memset(&fx, 0, sizeof(fx));

    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL.gguf LAYER_FIXTURE.bin\n", argv[0]);
        return 1;
    }
    gguf_path = argv[1];
    fixture_path = argv[2];

    if (!fixture_load(fixture_path, &fx)) {
        fprintf(stderr, "failed to load fixture: %s\n", fixture_path);
        return 1;
    }
    if (!qwen36_gguf_open(&gf, gguf_path, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        fixture_free(&fx);
        return 1;
    }
    if (!qwen36_35a3b_q8_bind(&gf, &q8, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        qwen36_gguf_close(&gf);
        fixture_free(&fx);
        return 1;
    }
    if (fx.layer >= QWEN36_35A3B_Q8_BLOCK_COUNT) {
        fprintf(stderr, "fixture layer out of range: %u\n", fx.layer);
        qwen36_gguf_close(&gf);
        fixture_free(&fx);
        return 1;
    }
    layer = &q8.layers[fx.layer];
    if (layer->kind != QWEN36_LAYER_KIND_HYBRID_SSM) {
        fprintf(stderr, "fixture is not a hybrid layer: blk.%u\n", fx.layer);
        qwen36_gguf_close(&gf);
        fixture_free(&fx);
        return 1;
    }

    printf("layer: blk.%u\n", fx.layer);
    printf("hidden: %u\n", fx.hidden);
    printf("num_v_heads: %u\n", fx.num_v_heads);
    printf("head_k_dim: %u\n", fx.head_k_dim);
    printf("head_v_dim: %u\n", fx.head_v_dim);
    printf("key_dim: %u\n", fx.key_dim);
    printf("value_dim: %u\n", fx.value_dim);

    if (!read_f32_tensor(&gf, layer->attn_norm, &tmp, fx.hidden, err, sizeof(err))) goto fail;
    report_cmp("attn_norm", tmp, fx.attn_norm_w, fx.hidden);
    free(tmp); tmp = NULL;

    if (!read_f32_tensor(&gf, layer->post_attn_norm, &tmp, fx.hidden, err, sizeof(err))) goto fail;
    report_cmp("post_attn_norm", tmp, fx.post_attn_norm_w, fx.hidden);
    free(tmp); tmp = NULL;

    printf("phase: attn_qkv\n");
    tmp = (float *)malloc((size_t)(fx.key_dim * 2u + fx.value_dim) * fx.hidden * sizeof(float));
    tmp2 = (float *)malloc((size_t)(fx.key_dim * 2u + fx.value_dim) * fx.hidden * sizeof(float));
    if (!tmp || !tmp2) goto fail;
    if (!decode_q8_rows(&gf, layer->attn_qkv, 0, fx.key_dim, fx.hidden, tmp, err, sizeof(err))) goto fail;
    if (!decode_q8_rows(&gf, layer->attn_qkv, fx.key_dim, fx.key_dim, fx.hidden, tmp + fx.key_dim * fx.hidden, err, sizeof(err))) goto fail;
    if (!decode_q8_rows(&gf, layer->attn_qkv, fx.key_dim * 2u, fx.value_dim, fx.hidden, tmp + (size_t)fx.key_dim * 2u * fx.hidden, err, sizeof(err))) goto fail;
    report_cmp("attn_qkv_raw", tmp, fx.w_qkv, (size_t)(fx.key_dim * 2u + fx.value_dim) * fx.hidden);
    reorder_qkv_v_rows_matrix_gg_to_hf(tmp2, tmp, fx.key_dim, fx.num_v_heads, fx.head_v_dim, fx.hidden);
    report_cmp("attn_qkv_v_reordered", tmp2, fx.w_qkv, (size_t)(fx.key_dim * 2u + fx.value_dim) * fx.hidden);
    free(tmp); tmp = NULL;
    free(tmp2); tmp2 = NULL;

    printf("phase: attn_gate\n");
    tmp = (float *)malloc((size_t)fx.value_dim * fx.hidden * sizeof(float));
    tmp2 = (float *)malloc((size_t)fx.value_dim * fx.hidden * sizeof(float));
    if (!tmp || !tmp2) goto fail;
    if (!decode_q8_rows(&gf, layer->attn_gate, 0, fx.value_dim, fx.hidden, tmp, err, sizeof(err))) goto fail;
    report_cmp("attn_gate_raw", tmp, fx.w_z, (size_t)fx.value_dim * fx.hidden);
    reorder_head_rows_matrix(tmp2, tmp, fx.num_v_heads, fx.head_v_dim, fx.hidden);
    report_cmp("attn_gate_reordered", tmp2, fx.w_z, (size_t)fx.value_dim * fx.hidden);
    free(tmp); tmp = NULL;
    free(tmp2); tmp2 = NULL;

    printf("phase: ssm_alpha\n");
    tmp = (float *)malloc((size_t)fx.num_v_heads * fx.hidden * sizeof(float));
    tmp2 = (float *)malloc((size_t)fx.num_v_heads * fx.hidden * sizeof(float));
    if (!tmp || !tmp2) goto fail;
    if (!decode_q8_rows(&gf, layer->ssm_alpha, 0, fx.num_v_heads, fx.hidden, tmp, err, sizeof(err))) goto fail;
    report_cmp("ssm_alpha_raw", tmp, fx.w_a, (size_t)fx.num_v_heads * fx.hidden);
    reorder_head_rows_matrix(tmp2, tmp, fx.num_v_heads, 1, fx.hidden);
    report_cmp("ssm_alpha_reordered", tmp2, fx.w_a, (size_t)fx.num_v_heads * fx.hidden);
    free(tmp); tmp = NULL;
    free(tmp2); tmp2 = NULL;

    printf("phase: ssm_beta\n");
    tmp = (float *)malloc((size_t)fx.num_v_heads * fx.hidden * sizeof(float));
    tmp2 = (float *)malloc((size_t)fx.num_v_heads * fx.hidden * sizeof(float));
    if (!tmp || !tmp2) goto fail;
    if (!decode_q8_rows(&gf, layer->ssm_beta, 0, fx.num_v_heads, fx.hidden, tmp, err, sizeof(err))) goto fail;
    report_cmp("ssm_beta_raw", tmp, fx.w_b, (size_t)fx.num_v_heads * fx.hidden);
    reorder_head_rows_matrix(tmp2, tmp, fx.num_v_heads, 1, fx.hidden);
    report_cmp("ssm_beta_reordered", tmp2, fx.w_b, (size_t)fx.num_v_heads * fx.hidden);
    free(tmp); tmp = NULL;
    free(tmp2); tmp2 = NULL;

    printf("phase: ssm_conv1d\n");
    if (!read_f32_tensor(&gf, layer->ssm_conv1d, &tmp, (size_t)4u * (fx.key_dim * 2u + fx.value_dim), err, sizeof(err))) goto fail;
    report_cmp("ssm_conv1d_raw", tmp, fx.conv_w, (size_t)4u * (fx.key_dim * 2u + fx.value_dim));
    tmp2 = (float *)malloc((size_t)4u * (fx.key_dim * 2u + fx.value_dim) * sizeof(float));
    if (!tmp2) goto fail;
    for (uint32_t d = 0; d < fx.key_dim * 2u + fx.value_dim; ++d) {
        for (uint32_t k = 0; k < 4u; ++k) {
            tmp2[(size_t)d * 4u + k] = tmp[(size_t)k * (fx.key_dim * 2u + fx.value_dim) + d];
        }
    }
    report_cmp("ssm_conv1d_transposed", tmp2, fx.conv_w, (size_t)4u * (fx.key_dim * 2u + fx.value_dim));
    tmp3 = (float *)malloc((size_t)4u * (fx.key_dim * 2u + fx.value_dim) * sizeof(float));
    if (!tmp3) goto fail;
    reorder_qkv_v_rows_matrix_gg_to_hf(tmp3, tmp, fx.key_dim, fx.num_v_heads, fx.head_v_dim, 4u);
    report_cmp("ssm_conv1d_v_reordered", tmp3, fx.conv_w, (size_t)4u * (fx.key_dim * 2u + fx.value_dim));
    free(tmp); tmp = NULL;
    free(tmp2); tmp2 = NULL;
    free(tmp3); tmp3 = NULL;

    if (!read_f32_tensor(&gf, layer->ssm_a, &tmp, fx.num_v_heads, err, sizeof(err))) goto fail;
    report_cmp("ssm_a", tmp, fx.A_log, fx.num_v_heads);
    tmp2 = (float *)malloc((size_t)fx.num_v_heads * sizeof(float));
    if (!tmp2) goto fail;
    reorder_head_scalars(tmp2, tmp, fx.num_v_heads);
    report_cmp("ssm_a_reordered", tmp2, fx.A_log, fx.num_v_heads);
    for (uint32_t i = 0; i < fx.num_v_heads; ++i) tmp2[i] = -expf(tmp[i]);
    report_cmp("ssm_a_neg_exp", tmp2, fx.A_log, fx.num_v_heads);
    reorder_head_scalars(tmp2, tmp, fx.num_v_heads);
    for (uint32_t i = 0; i < fx.num_v_heads; ++i) tmp2[i] = -expf(tmp2[i]);
    report_cmp("ssm_a_reordered_neg_exp", tmp2, fx.A_log, fx.num_v_heads);
    free(tmp); tmp = NULL;
    free(tmp2); tmp2 = NULL;

    if (!read_f32_tensor(&gf, layer->ssm_dt_bias, &tmp, fx.num_v_heads, err, sizeof(err))) goto fail;
    report_cmp("ssm_dt_bias", tmp, fx.dt_bias, fx.num_v_heads);
    tmp2 = (float *)malloc((size_t)fx.num_v_heads * sizeof(float));
    if (!tmp2) goto fail;
    reorder_head_scalars(tmp2, tmp, fx.num_v_heads);
    report_cmp("ssm_dt_bias_reordered", tmp2, fx.dt_bias, fx.num_v_heads);
    free(tmp); tmp = NULL;
    free(tmp2); tmp2 = NULL;

    if (!read_f32_tensor(&gf, layer->ssm_norm, &tmp, fx.head_v_dim, err, sizeof(err))) goto fail;
    report_cmp("ssm_norm", tmp, fx.ssm_norm_w, fx.head_v_dim);
    free(tmp); tmp = NULL;

    printf("phase: ssm_out\n");
    tmp = (float *)malloc((size_t)fx.hidden * fx.value_dim * sizeof(float));
    tmp2 = (float *)malloc((size_t)fx.hidden * fx.value_dim * sizeof(float));
    if (!tmp || !tmp2) goto fail;
    if (!decode_q8_rows(&gf, layer->ssm_out, 0, fx.hidden, fx.value_dim, tmp, err, sizeof(err))) goto fail;
    report_cmp("ssm_out_raw", tmp, fx.w_out, (size_t)fx.hidden * fx.value_dim);
    reorder_v_head_cols_matrix_gg_to_hf(tmp2, tmp, fx.hidden, fx.num_v_heads, fx.head_v_dim);
    report_cmp("ssm_out_input_reordered", tmp2, fx.w_out, (size_t)fx.hidden * fx.value_dim);
    free(tmp); tmp = NULL;
    free(tmp2); tmp2 = NULL;

    if (!read_f32_tensor(&gf, layer->ffn_gate_inp, &tmp, (size_t)ROUTER_COUNT * fx.hidden, err, sizeof(err))) goto fail;
    report_cmp("router_w", tmp, fx.router_w, (size_t)ROUTER_COUNT * fx.hidden);
    free(tmp); tmp = NULL;

    tmp = (float *)malloc((size_t)512u * fx.hidden * sizeof(float));
    if (!tmp) goto fail;
    if (!decode_q8_rows(&gf, layer->ffn_gate_shexp, 0, 512u, fx.hidden, tmp, err, sizeof(err))) goto fail;
    report_cmp("gate_shexp", tmp, fx.gate_shexp, (size_t)512u * fx.hidden);
    free(tmp); tmp = NULL;

    tmp = (float *)malloc((size_t)512u * fx.hidden * sizeof(float));
    if (!tmp) goto fail;
    if (!decode_q8_rows(&gf, layer->ffn_up_shexp, 0, 512u, fx.hidden, tmp, err, sizeof(err))) goto fail;
    report_cmp("up_shexp", tmp, fx.up_shexp, (size_t)512u * fx.hidden);
    free(tmp); tmp = NULL;

    tmp = (float *)malloc((size_t)fx.hidden * 512u * sizeof(float));
    if (!tmp) goto fail;
    if (!decode_q8_rows(&gf, layer->ffn_down_shexp, 0, fx.hidden, 512u, tmp, err, sizeof(err))) goto fail;
    report_cmp("down_shexp", tmp, fx.down_shexp, (size_t)fx.hidden * 512u);
    free(tmp); tmp = NULL;

    if (!read_f32_tensor(&gf, layer->ffn_gate_inp_shexp, &tmp, fx.hidden, err, sizeof(err))) goto fail;
    report_cmp("gate_inp_shexp", tmp, fx.gate_inp_shexp, fx.hidden);
    free(tmp); tmp = NULL;

    qwen36_gguf_close(&gf);
    fixture_free(&fx);
    return 0;

fail:
    fprintf(stderr, "contract check failed: %s\n", err[0] ? err : "oom");
    free(tmp);
    free(tmp2);
    qwen36_gguf_close(&gf);
    fixture_free(&fx);
    return 1;
}
