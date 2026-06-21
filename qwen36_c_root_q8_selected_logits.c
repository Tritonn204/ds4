#include "qwen36_35a3b_q8.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "Q36RHF01"
#define MAGIC_LEN 8

typedef struct probe_fixture {
    uint32_t hidden;
    uint32_t topk;
    uint32_t prompt_tokens;
    float *hidden_pre;
    float *hidden_post_ref;
    uint32_t *top_ids;
    float *top_logits_ref;
} probe_fixture;

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
static void fixture_free(probe_fixture *fx) {
    if (!fx) return;
    free(fx->hidden_pre); free(fx->hidden_post_ref); free(fx->top_ids); free(fx->top_logits_ref);
    memset(fx, 0, sizeof(*fx));
}
static int fixture_load(const char *path, probe_fixture *fx) {
    FILE *fp = fopen(path, "rb");
    char magic[MAGIC_LEN];
    memset(fx, 0, sizeof(*fx));
    if (!fp) return 0;
    if (!read_exact(fp, magic, MAGIC_LEN) || memcmp(magic, MAGIC, MAGIC_LEN) != 0) { fclose(fp); return 0; }
    if (!read_exact(fp, &fx->hidden, sizeof(uint32_t)) || !read_exact(fp, &fx->topk, sizeof(uint32_t)) || !read_exact(fp, &fx->prompt_tokens, sizeof(uint32_t))) { fclose(fp); return 0; }
    if (!alloc_read_f32(fp, &fx->hidden_pre, fx->hidden) ||
        !alloc_read_f32(fp, &fx->hidden_post_ref, fx->hidden) ||
        !alloc_read_u32(fp, &fx->top_ids, fx->topk) ||
        !alloc_read_f32(fp, &fx->top_logits_ref, fx->topk)) { fclose(fp); fixture_free(fx); return 0; }
    fclose(fp);
    return 1;
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
    size_t i; double acc = 0.0;
    for (i = 0; i < n; ++i) { double d = (double)a[i] - b[i]; acc += d * d; }
    return sqrt(acc / (double)n);
}
static double vec_cosine(const float *a, const float *b, size_t n) {
    size_t i; double dot = 0.0, an = 0.0, bn = 0.0;
    for (i = 0; i < n; ++i) { dot += (double)a[i] * b[i]; an += (double)a[i] * a[i]; bn += (double)b[i] * b[i]; }
    if (an == 0.0 || bn == 0.0) return NAN;
    return dot / sqrt(an * bn);
}

int main(int argc, char **argv) {
    probe_fixture fx;
    qwen36_gguf_file gf;
    qwen36_35a3b_q8_model q8;
    char err[512];
    float *norm_w = NULL, *hidden_post = NULL, *logits = NULL, *row = NULL;
    size_t row_size;
    uint32_t i, j;
    if (argc != 3) { fprintf(stderr, "usage: %s MODEL.gguf PROBE.bin\n", argv[0]); return 1; }
    if (!fixture_load(argv[2], &fx)) { fprintf(stderr, "failed to load probe fixture\n"); return 1; }
    if (!qwen36_gguf_open(&gf, argv[1], err, sizeof(err))) { fprintf(stderr, "%s\n", err); fixture_free(&fx); return 1; }
    if (!qwen36_35a3b_q8_bind(&gf, &q8, err, sizeof(err))) { fprintf(stderr, "%s\n", err); qwen36_gguf_close(&gf); fixture_free(&fx); return 1; }
    norm_w = (float *)malloc((size_t)fx.hidden * sizeof(float));
    hidden_post = (float *)calloc(fx.hidden, sizeof(float));
    logits = (float *)calloc(fx.topk, sizeof(float));
    row_size = 34u * (fx.hidden / 32u);
    row = (float *)malloc((size_t)fx.hidden * sizeof(float));
    if (!norm_w || !hidden_post || !logits || !row) { qwen36_gguf_close(&gf); fixture_free(&fx); free(norm_w); free(hidden_post); free(logits); free(row); return 1; }
    if (!qwen36_gguf_read_tensor_bytes(&gf, q8.output_norm, 0, norm_w, (size_t)fx.hidden * sizeof(float), err, sizeof(err))) {
        fprintf(stderr, "%s\n", err); qwen36_gguf_close(&gf); fixture_free(&fx); free(norm_w); free(hidden_post); free(logits); free(row); return 1;
    }
    {
        double var = 0.0;
        for (i = 0; i < fx.hidden; ++i) var += (double)fx.hidden_pre[i] * fx.hidden_pre[i];
        var /= (double)fx.hidden;
        for (i = 0; i < fx.hidden; ++i) hidden_post[i] = (float)(fx.hidden_pre[i] / sqrt(var + 1e-6)) * norm_w[i];
    }
    for (i = 0; i < fx.topk; ++i) {
        const uint64_t byte_off = (uint64_t)fx.top_ids[i] * row_size;
        uint8_t *buf = (uint8_t *)malloc(row_size);
        if (!buf) return 1;
        if (!qwen36_gguf_read_tensor_bytes(&gf, q8.output, byte_off, buf, row_size, err, sizeof(err))) {
            fprintf(stderr, "%s\n", err); free(buf); qwen36_gguf_close(&gf); fixture_free(&fx); free(norm_w); free(hidden_post); free(logits); free(row); return 1;
        }
        for (j = 0; j < fx.hidden / 32u; ++j) {
            const uint8_t *block = buf + j * 34u;
            float dscale = f16_to_f32((uint16_t)(block[0] | (block[1] << 8)));
            const int8_t *qs = (const int8_t *)(block + 2);
            uint32_t k;
            for (k = 0; k < 32u; ++k) row[j * 32u + k] = dscale * (float)qs[k];
        }
        free(buf);
        {
            double sum = 0.0;
            for (j = 0; j < fx.hidden; ++j) sum += (double)row[j] * hidden_post[j];
            logits[i] = (float)sum;
        }
    }
    printf("prompt_tokens: %u\n", fx.prompt_tokens);
    printf("topk: %u\n", fx.topk);
    printf("final_norm_rmse: %.8f\n", vec_rmse(hidden_post, fx.hidden_post_ref, fx.hidden));
    printf("final_norm_cosine: %.8f\n", vec_cosine(hidden_post, fx.hidden_post_ref, fx.hidden));
    printf("selected_logits_rmse: %.8f\n", vec_rmse(logits, fx.top_logits_ref, fx.topk));
    printf("selected_logits_cosine: %.8f\n", vec_cosine(logits, fx.top_logits_ref, fx.topk));
    for (i = 0; i < fx.topk && i < 8; ++i) {
        printf("logit[%u]: token_id=%u ref=%.6f got=%.6f\n", i, fx.top_ids[i], fx.top_logits_ref[i], logits[i]);
    }
    qwen36_gguf_close(&gf);
    fixture_free(&fx);
    free(norm_w); free(hidden_post); free(logits); free(row);
    return 0;
}
