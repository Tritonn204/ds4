#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "Q36ROOT1"
#define MAGIC_LEN 8

typedef struct root_fixture {
    uint32_t hidden;
    uint32_t topk;
    uint32_t prompt_tokens;
    float *hidden_pre;
    float *hidden_post_ref;
    float *norm_w;
    float *top_ids_f32;
    float *top_logits_ref;
    float *output_rows;
} root_fixture;

static int read_exact(FILE *fp, void *buf, size_t n) { return fread(buf, 1, n, fp) == n; }
static int alloc_read_f32(FILE *fp, float **out, size_t count) {
    float *p = (float *)malloc(count * sizeof(float));
    if (!p) return 0;
    if (!read_exact(fp, p, count * sizeof(float))) { free(p); return 0; }
    *out = p;
    return 1;
}
static void fixture_free(root_fixture *fx) {
    if (!fx) return;
    free(fx->hidden_pre); free(fx->hidden_post_ref); free(fx->norm_w); free(fx->top_ids_f32); free(fx->top_logits_ref); free(fx->output_rows);
    memset(fx, 0, sizeof(*fx));
}
static int fixture_load(const char *path, root_fixture *fx) {
    FILE *fp = fopen(path, "rb");
    char magic[MAGIC_LEN];
    memset(fx, 0, sizeof(*fx));
    if (!fp) return 0;
    if (!read_exact(fp, magic, MAGIC_LEN) || memcmp(magic, MAGIC, MAGIC_LEN) != 0) { fclose(fp); return 0; }
    if (!read_exact(fp, &fx->hidden, sizeof(uint32_t)) || !read_exact(fp, &fx->topk, sizeof(uint32_t)) || !read_exact(fp, &fx->prompt_tokens, sizeof(uint32_t))) { fclose(fp); return 0; }
    if (!alloc_read_f32(fp, &fx->hidden_pre, fx->hidden) ||
        !alloc_read_f32(fp, &fx->hidden_post_ref, fx->hidden) ||
        !alloc_read_f32(fp, &fx->norm_w, fx->hidden) ||
        !alloc_read_f32(fp, &fx->top_ids_f32, fx->topk) ||
        !alloc_read_f32(fp, &fx->top_logits_ref, fx->topk) ||
        !alloc_read_f32(fp, &fx->output_rows, (size_t)fx->topk * fx->hidden)) { fclose(fp); fixture_free(fx); return 0; }
    fclose(fp);
    return 1;
}
static void matvec(const float *mat, const float *vec, float *out, uint32_t rows, uint32_t cols) {
    uint32_t r, c;
    for (r = 0; r < rows; ++r) {
        const float *row = mat + (size_t)r * cols;
        double sum = 0.0;
        for (c = 0; c < cols; ++c) sum += (double)row[c] * vec[c];
        out[r] = (float)sum;
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
    root_fixture fx;
    float *hidden_post = NULL, *logits = NULL;
    uint32_t i;
    if (argc != 2) { fprintf(stderr, "usage: %s FIXTURE.bin\n", argv[0]); return 1; }
    if (!fixture_load(argv[1], &fx)) { fprintf(stderr, "failed to load fixture\n"); return 1; }
    hidden_post = calloc(fx.hidden, sizeof(float));
    logits = calloc(fx.topk, sizeof(float));
    if (!hidden_post || !logits) { fixture_free(&fx); free(hidden_post); free(logits); return 1; }
    {
        double var = 0.0;
        for (i = 0; i < fx.hidden; ++i) var += (double)fx.hidden_pre[i] * fx.hidden_pre[i];
        var /= (double)fx.hidden;
        for (i = 0; i < fx.hidden; ++i) hidden_post[i] = (float)(fx.hidden_pre[i] / sqrt(var + 1e-6)) * fx.norm_w[i];
    }
    matvec(fx.output_rows, hidden_post, logits, fx.topk, fx.hidden);
    printf("prompt_tokens: %u\n", fx.prompt_tokens);
    printf("topk: %u\n", fx.topk);
    printf("final_norm_rmse: %.8f\n", vec_rmse(hidden_post, fx.hidden_post_ref, fx.hidden));
    printf("final_norm_cosine: %.8f\n", vec_cosine(hidden_post, fx.hidden_post_ref, fx.hidden));
    printf("selected_logits_rmse: %.8f\n", vec_rmse(logits, fx.top_logits_ref, fx.topk));
    printf("selected_logits_cosine: %.8f\n", vec_cosine(logits, fx.top_logits_ref, fx.topk));
    for (i = 0; i < fx.topk && i < 8; ++i) {
        printf("logit[%u]: token_id=%u ref=%.6f got=%.6f\n", i, (uint32_t)fx.top_ids_f32[i], fx.top_logits_ref[i], logits[i]);
    }
    fixture_free(&fx);
    free(hidden_post); free(logits);
    return 0;
}
