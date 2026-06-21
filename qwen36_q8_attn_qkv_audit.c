#include "qwen36_35a3b_q8.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "Q36DWF02"
#define MAGIC_LEN 8

typedef struct audit_fixture {
    uint32_t layer;
    uint32_t seq_len;
    uint32_t hidden;
    uint32_t num_v_heads;
    uint32_t num_k_heads;
    uint32_t head_k_dim;
    uint32_t head_v_dim;
    uint32_t key_dim;
    uint32_t value_dim;
    uint32_t topk;
    uint32_t union_experts;
    float *input_ln_seq;
    float *qkv_seq;
    float *w_qkv;
} audit_fixture;

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

static int skip_f32(FILE *fp, size_t count) {
    return fseek(fp, (long)(count * sizeof(float)), SEEK_CUR) == 0;
}

static void fixture_free(audit_fixture *fx) {
    if (!fx) return;
    free(fx->input_ln_seq);
    free(fx->qkv_seq);
    free(fx->w_qkv);
    memset(fx, 0, sizeof(*fx));
}

static int fixture_load_min(const char *path, audit_fixture *fx) {
    FILE *fp = fopen(path, "rb");
    char magic[MAGIC_LEN];
    memset(fx, 0, sizeof(*fx));
    if (!fp) return 0;
    if (!read_exact(fp, magic, MAGIC_LEN) || memcmp(magic, MAGIC, MAGIC_LEN) != 0) {
        fclose(fp);
        return 0;
    }
    if (!read_exact(fp, &fx->layer, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->seq_len, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->hidden, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->num_v_heads, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->num_k_heads, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->head_k_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->head_v_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->key_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->value_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->topk, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->union_experts, sizeof(uint32_t))) {
        fclose(fp);
        return 0;
    }

    {
        const size_t seq_hidden2 = (size_t)fx->seq_len * fx->hidden;
        const size_t seq_qkv2 = (size_t)fx->seq_len * (fx->key_dim * 2u + fx->value_dim);
        const size_t seq_z2 = (size_t)fx->seq_len * fx->value_dim;
        const size_t seq_heads2 = (size_t)fx->seq_len * fx->num_v_heads;
        const size_t seq_q2 = (size_t)fx->seq_len * fx->num_v_heads * fx->head_k_dim;
        const size_t seq_v2 = (size_t)fx->seq_len * fx->num_v_heads * fx->head_v_dim;
        const size_t conv_raw2 = (size_t)(fx->key_dim * 2u + fx->value_dim) * (fx->seq_len + 3u);
        const size_t hidden2 = fx->hidden;
        const size_t qkv_dim2 = (size_t)fx->key_dim * 2u + fx->value_dim;
        const size_t topk2 = fx->topk;
        if (!skip_f32(fp, seq_hidden2)) goto fail;                       /* layer_input_seq */
        if (!alloc_read_f32(fp, &fx->input_ln_seq, seq_hidden2)) goto fail;
        if (!alloc_read_f32(fp, &fx->qkv_seq, seq_qkv2)) goto fail;
        if (!skip_f32(fp, seq_z2)) goto fail;                            /* z_seq */
        if (!skip_f32(fp, seq_heads2)) goto fail;                        /* a_seq */
        if (!skip_f32(fp, seq_heads2)) goto fail;                        /* b_seq */
        if (!skip_f32(fp, conv_raw2)) goto fail;                         /* conv_raw */
        if (!skip_f32(fp, seq_q2)) goto fail;                            /* q_ref */
        if (!skip_f32(fp, seq_q2)) goto fail;                            /* k_ref */
        if (!skip_f32(fp, seq_v2)) goto fail;                            /* v_ref */
        if (!skip_f32(fp, seq_heads2)) goto fail;                        /* beta_ref */
        if (!skip_f32(fp, seq_heads2)) goto fail;                        /* g_ref */
        if (!skip_f32(fp, seq_v2)) goto fail;                            /* core_ref */
        if (!skip_f32(fp, seq_z2)) goto fail;                            /* out_in_seq */
        if (!skip_f32(fp, seq_hidden2)) goto fail;                       /* out_proj_out_seq */
        if (!skip_f32(fp, hidden2)) goto fail;                           /* mixer_out_ref */
        if (!skip_f32(fp, hidden2)) goto fail;                           /* layer_input_last_ref */
        if (!skip_f32(fp, hidden2)) goto fail;                           /* residual_after_mixer_ref */
        if (!skip_f32(fp, hidden2)) goto fail;                           /* post_attn_ln_ref */
        if (!skip_f32(fp, hidden2)) goto fail;                           /* mlp_out_ref */
        if (!skip_f32(fp, hidden2)) goto fail;                           /* layer_output_ref */
        if (!skip_f32(fp, 256)) goto fail;                               /* router_logits_ref */
        if (!skip_f32(fp, topk2)) goto fail;                             /* router_indices_ref_f32 */
        if (!skip_f32(fp, topk2)) goto fail;                             /* router_scores_ref */
        if (!skip_f32(fp, 1)) goto fail;                                 /* shared_gate_pre_ref */
        if (!skip_f32(fp, seq_hidden2)) goto fail;                       /* layer_input_seq_ref */
        if (!skip_f32(fp, seq_hidden2)) goto fail;                       /* residual_after_mixer_seq_ref */
        if (!skip_f32(fp, seq_hidden2)) goto fail;                       /* post_attn_ln_seq_ref */
        if (!skip_f32(fp, seq_hidden2)) goto fail;                       /* mlp_out_seq_ref */
        if (!skip_f32(fp, seq_hidden2)) goto fail;                       /* layer_output_seq_ref */
        if (!skip_f32(fp, (size_t)fx->seq_len * 256u)) goto fail;        /* router_logits_seq_ref */
        if (!skip_f32(fp, (size_t)fx->seq_len * topk2)) goto fail;       /* router_indices_seq_ref_f32 */
        if (!skip_f32(fp, (size_t)fx->seq_len * topk2)) goto fail;       /* router_scores_seq_ref */
        if (!skip_f32(fp, fx->seq_len)) goto fail;                       /* shared_gate_pre_seq_ref */
        if (!skip_f32(fp, hidden2)) goto fail;                           /* attn_norm_w */
        if (!skip_f32(fp, hidden2)) goto fail;                           /* post_attn_norm_w */
        if (!alloc_read_f32(fp, &fx->w_qkv, qkv_dim2 * hidden2)) goto fail;
    }

    fclose(fp);
    return 1;

fail:
    fclose(fp);
    fixture_free(fx);
    return 0;
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1f;
    const uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            uint32_t e = 127 - 15 + 1;
            uint32_t m = mant;
            while ((m & 0x400) == 0) { m <<= 1; e--; }
            m &= 0x3ff;
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

static double vec_rmse(const float *a, const float *b, size_t n) {
    double acc = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double)a[i] - b[i];
        acc += d * d;
    }
    return sqrt(acc / (double)n);
}

static double vec_cosine(const float *a, const float *b, size_t n) {
    double dot = 0.0, an = 0.0, bn = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += (double)a[i] * b[i];
        an += (double)a[i] * a[i];
        bn += (double)b[i] * b[i];
    }
    if (an == 0.0 || bn == 0.0) return NAN;
    return dot / sqrt(an * bn);
}

static void matmul_cpu_rows(const float *mat, const float *x, float *out, uint32_t rows, uint32_t cols) {
    for (uint32_t r = 0; r < rows; ++r) {
        const float *row = mat + (size_t)r * cols;
        double sum = 0.0;
        for (uint32_t c = 0; c < cols; ++c) sum += (double)row[c] * x[c];
        out[r] = (float)sum;
    }
}

static void build_perms(uint32_t n_heads, uint32_t *gg_to_hf, uint32_t *hf_to_gg) {
    uint32_t pos = 0;
    for (uint32_t i = 0; i < n_heads; i += 2) gg_to_hf[pos++] = i;
    for (uint32_t i = 1; i < n_heads; i += 2) gg_to_hf[pos++] = i;
    for (uint32_t i = 0; i < n_heads; ++i) hf_to_gg[gg_to_hf[i]] = i;
}

static void reorder_v_rows_gg_to_hf(float *dst, const float *src, uint32_t key_dim, uint32_t value_dim, uint32_t n_heads, uint32_t head_dim, uint32_t hidden) {
    uint32_t *gg_to_hf = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    const uint32_t qk_rows = key_dim * 2u;
    if (!gg_to_hf || !hf_to_gg) {
        free(gg_to_hf);
        free(hf_to_gg);
        return;
    }
    memcpy(dst, src, (size_t)qk_rows * hidden * sizeof(float));
    build_perms(n_heads, gg_to_hf, hf_to_gg);
    for (uint32_t h = 0; h < n_heads; ++h) {
        memcpy(dst + ((size_t)qk_rows + (size_t)h * head_dim) * hidden,
               src + ((size_t)qk_rows + (size_t)hf_to_gg[h] * head_dim) * hidden,
               (size_t)head_dim * hidden * sizeof(float));
    }
    free(gg_to_hf);
    free(hf_to_gg);
    (void)value_dim;
}

static int decode_q8_rows(const qwen36_gguf_file *gf, const qwen36_gguf_tensor *t, uint32_t row_elems, float *out) {
    const size_t row_size = 34u * (row_elems / 32u);
    const size_t total_rows = (size_t)t->dims[1];
    uint8_t *buf = (uint8_t *)malloc(total_rows * row_size);
    char err[256];
    if (!buf) return 0;
    if (!qwen36_gguf_read_tensor_bytes(gf, t, 0, buf, total_rows * row_size, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        free(buf);
        return 0;
    }
    for (size_t r = 0; r < total_rows; ++r) {
        const uint8_t *row = buf + r * row_size;
        float *dst = out + r * row_elems;
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

int main(int argc, char **argv) {
    audit_fixture fx;
    qwen36_gguf_file gf;
    qwen36_35a3b_q8_model q8;
    const qwen36_35a3b_q8_layer *layer = NULL;
    char err[512];
    float *raw_w = NULL, *reordered_w = NULL;
    float *raw_qkv = NULL, *reordered_qkv = NULL;
    uint32_t slice_sizes[3];

    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL.gguf FIXTURE.bin\n", argv[0]);
        return 1;
    }
    if (!fixture_load_min(argv[2], &fx)) {
        fprintf(stderr, "failed to load audit fixture\n");
        return 1;
    }
    if (!qwen36_gguf_open(&gf, argv[1], err, sizeof(err))) {
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
        fprintf(stderr, "fixture layer %u out of range\n", fx.layer);
        qwen36_gguf_close(&gf);
        fixture_free(&fx);
        return 1;
    }
    layer = &q8.layers[fx.layer];
    if (layer->kind != QWEN36_LAYER_KIND_HYBRID_SSM || !layer->attn_qkv) {
        fprintf(stderr, "fixture layer %u is not a hybrid attn_qkv layer\n", fx.layer);
        qwen36_gguf_close(&gf);
        fixture_free(&fx);
        return 1;
    }

    {
        const uint32_t qkv_dim = fx.key_dim * 2u + fx.value_dim;
        const size_t mat_elems = (size_t)qkv_dim * fx.hidden;
        const size_t seq_elems = (size_t)fx.seq_len * qkv_dim;
        slice_sizes[0] = fx.key_dim;
        slice_sizes[1] = fx.key_dim;
        slice_sizes[2] = fx.value_dim;

        raw_w = (float *)malloc(mat_elems * sizeof(float));
        reordered_w = (float *)malloc(mat_elems * sizeof(float));
        raw_qkv = (float *)malloc(seq_elems * sizeof(float));
        reordered_qkv = (float *)malloc(seq_elems * sizeof(float));
        if (!raw_w || !reordered_w || !raw_qkv || !reordered_qkv) {
            fprintf(stderr, "out of memory\n");
            qwen36_gguf_close(&gf);
            fixture_free(&fx);
            free(raw_w); free(reordered_w); free(raw_qkv); free(reordered_qkv);
            return 1;
        }

        if (!decode_q8_rows(&gf, layer->attn_qkv, fx.hidden, raw_w)) {
            qwen36_gguf_close(&gf);
            fixture_free(&fx);
            free(raw_w); free(reordered_w); free(raw_qkv); free(reordered_qkv);
            return 1;
        }
        reorder_v_rows_gg_to_hf(reordered_w, raw_w, fx.key_dim, fx.value_dim, fx.num_v_heads, fx.head_v_dim, fx.hidden);

        for (uint32_t t = 0; t < fx.seq_len; ++t) {
            const float *x = fx.input_ln_seq + (size_t)t * fx.hidden;
            matmul_cpu_rows(raw_w, x, raw_qkv + (size_t)t * qkv_dim, qkv_dim, fx.hidden);
            matmul_cpu_rows(reordered_w, x, reordered_qkv + (size_t)t * qkv_dim, qkv_dim, fx.hidden);
        }

        printf("layer: %u\n", fx.layer);
        printf("attn_qkv_rows: %u\n", qkv_dim);
        printf("attn_qkv_cols: %u\n", fx.hidden);
        printf("matrix_raw_vs_fixture_rmse: %.8f\n", vec_rmse(raw_w, fx.w_qkv, mat_elems));
        printf("matrix_raw_vs_fixture_cosine: %.8f\n", vec_cosine(raw_w, fx.w_qkv, mat_elems));
        printf("matrix_reordered_vs_fixture_rmse: %.8f\n", vec_rmse(reordered_w, fx.w_qkv, mat_elems));
        printf("matrix_reordered_vs_fixture_cosine: %.8f\n", vec_cosine(reordered_w, fx.w_qkv, mat_elems));
        printf("qkv_raw_vs_trace_rmse: %.8f\n", vec_rmse(raw_qkv, fx.qkv_seq, seq_elems));
        printf("qkv_raw_vs_trace_cosine: %.8f\n", vec_cosine(raw_qkv, fx.qkv_seq, seq_elems));
        printf("qkv_reordered_vs_trace_rmse: %.8f\n", vec_rmse(reordered_qkv, fx.qkv_seq, seq_elems));
        printf("qkv_reordered_vs_trace_cosine: %.8f\n", vec_cosine(reordered_qkv, fx.qkv_seq, seq_elems));

        {
            size_t off = 0;
            const char *names[3] = { "q", "k", "v" };
            for (int i = 0; i < 3; ++i) {
                const size_t n = (size_t)fx.seq_len * slice_sizes[i];
                printf("%s_raw_vs_trace_rmse: %.8f\n", names[i], vec_rmse(raw_qkv + off, fx.qkv_seq + off, n));
                printf("%s_raw_vs_trace_cosine: %.8f\n", names[i], vec_cosine(raw_qkv + off, fx.qkv_seq + off, n));
                printf("%s_reordered_vs_trace_rmse: %.8f\n", names[i], vec_rmse(reordered_qkv + off, fx.qkv_seq + off, n));
                printf("%s_reordered_vs_trace_cosine: %.8f\n", names[i], vec_cosine(reordered_qkv + off, fx.qkv_seq + off, n));
                off += n;
            }
        }
    }

    qwen36_gguf_close(&gf);
    fixture_free(&fx);
    free(raw_w);
    free(reordered_w);
    free(raw_qkv);
    free(reordered_qkv);
    return 0;
}
