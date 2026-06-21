#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "Q36LNRF1"
#define MAGIC_LEN 8

typedef struct norm_fixture {
    uint32_t layer, seq_len, heads, head_dim;
    float *core;
    float *z_seq;
    float *out_in_ref;
    float *norm_w;
} norm_fixture;

static int read_exact(FILE *fp, void *buf, size_t n) { return fread(buf, 1, n, fp) == n; }

static int alloc_read_f32(FILE *fp, float **out, size_t count) {
    float *p = (float *)malloc(count * sizeof(float));
    if (!p) return 0;
    if (!read_exact(fp, p, count * sizeof(float))) { free(p); return 0; }
    *out = p;
    return 1;
}

static void fixture_free(norm_fixture *fx) {
    if (!fx) return;
    free(fx->core); free(fx->z_seq); free(fx->out_in_ref); free(fx->norm_w);
    memset(fx, 0, sizeof(*fx));
}

static int fixture_load(const char *path, norm_fixture *fx) {
    FILE *fp = fopen(path, "rb");
    char magic[MAGIC_LEN];
    memset(fx, 0, sizeof(*fx));
    if (!fp) return 0;
    if (!read_exact(fp, magic, MAGIC_LEN) || memcmp(magic, MAGIC, MAGIC_LEN) != 0) { fclose(fp); return 0; }
    if (!read_exact(fp, &fx->layer, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->seq_len, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->heads, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->head_dim, sizeof(uint32_t))) { fclose(fp); return 0; }
    if (!alloc_read_f32(fp, &fx->core, (size_t)fx->seq_len * fx->heads * fx->head_dim) ||
        !alloc_read_f32(fp, &fx->z_seq, (size_t)fx->seq_len * fx->heads * fx->head_dim) ||
        !alloc_read_f32(fp, &fx->out_in_ref, (size_t)fx->seq_len * fx->heads * fx->head_dim) ||
        !alloc_read_f32(fp, &fx->norm_w, fx->head_dim)) { fclose(fp); fixture_free(fx); return 0; }
    fclose(fp);
    return 1;
}

static inline float siluf_local(float x) { return x / (1.0f + expf(-x)); }

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
    norm_fixture fx;
    float *out = NULL;
    size_t i, t, h, base;
    if (argc != 2) {
        fprintf(stderr, "usage: %s FIXTURE.bin\n", argv[0]);
        return 1;
    }
    if (!fixture_load(argv[1], &fx)) {
        fprintf(stderr, "failed to load fixture\n");
        return 1;
    }
    out = (float *)calloc((size_t)fx.seq_len * fx.heads * fx.head_dim, sizeof(float));
    if (!out) { fixture_free(&fx); return 1; }

    for (t = 0; t < fx.seq_len; ++t) {
        for (h = 0; h < fx.heads; ++h) {
            double var = 0.0;
            float gate_scale;
            base = ((size_t)t * fx.heads + h) * fx.head_dim;
            for (i = 0; i < fx.head_dim; ++i) {
                double v = fx.core[base + i];
                var += v * v;
            }
            var /= (double)fx.head_dim;
            gate_scale = siluf_local(fx.z_seq[base]);
            for (i = 0; i < fx.head_dim; ++i) {
                float normed = (float)(fx.core[base + i] / sqrt(var + 1e-6));
                out[base + i] = normed * fx.norm_w[i] * siluf_local(fx.z_seq[base + i]);
            }
            (void)gate_scale;
        }
    }

    printf("layer: %u\n", fx.layer);
    printf("seq_len: %u\n", fx.seq_len);
    printf("gated_norm_rmse: %.8f\n", vec_rmse(out, fx.out_in_ref, (size_t)fx.seq_len * fx.heads * fx.head_dim));
    printf("gated_norm_cosine: %.8f\n", vec_cosine(out, fx.out_in_ref, (size_t)fx.seq_len * fx.heads * fx.head_dim));

    fixture_free(&fx);
    free(out);
    return 0;
}
