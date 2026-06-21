#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "Q36LLSF1"
#define MAGIC_LEN 8

typedef struct layer_fixture {
    uint32_t layer, heads, head_dim, hidden;
    float *layer_input;
    float *core_last;
    float *z_last;
    float *mixer_out_ref;
    float *residual_after_mixer_ref;
    float *post_attn_ln_ref;
    float *norm_w;
    float *out_w;
} layer_fixture;

static int read_exact(FILE *fp, void *buf, size_t n) { return fread(buf, 1, n, fp) == n; }

static int alloc_read_f32(FILE *fp, float **out, size_t count) {
    float *p = (float *)malloc(count * sizeof(float));
    if (!p) return 0;
    if (!read_exact(fp, p, count * sizeof(float))) { free(p); return 0; }
    *out = p;
    return 1;
}

static void fixture_free(layer_fixture *fx) {
    if (!fx) return;
    free(fx->layer_input); free(fx->core_last); free(fx->z_last); free(fx->mixer_out_ref);
    free(fx->residual_after_mixer_ref); free(fx->post_attn_ln_ref); free(fx->norm_w); free(fx->out_w);
    memset(fx, 0, sizeof(*fx));
}

static int fixture_load(const char *path, layer_fixture *fx) {
    FILE *fp = fopen(path, "rb");
    char magic[MAGIC_LEN];
    memset(fx, 0, sizeof(*fx));
    if (!fp) return 0;
    if (!read_exact(fp, magic, MAGIC_LEN) || memcmp(magic, MAGIC, MAGIC_LEN) != 0) { fclose(fp); return 0; }
    if (!read_exact(fp, &fx->layer, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->heads, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->head_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->hidden, sizeof(uint32_t))) { fclose(fp); return 0; }
    if (!alloc_read_f32(fp, &fx->layer_input, fx->hidden) ||
        !alloc_read_f32(fp, &fx->core_last, (size_t)fx->heads * fx->head_dim) ||
        !alloc_read_f32(fp, &fx->z_last, (size_t)fx->heads * fx->head_dim) ||
        !alloc_read_f32(fp, &fx->mixer_out_ref, fx->hidden) ||
        !alloc_read_f32(fp, &fx->residual_after_mixer_ref, fx->hidden) ||
        !alloc_read_f32(fp, &fx->post_attn_ln_ref, fx->hidden) ||
        !alloc_read_f32(fp, &fx->norm_w, fx->head_dim) ||
        !alloc_read_f32(fp, &fx->out_w, (size_t)fx->hidden * fx->heads * fx->head_dim)) { fclose(fp); fixture_free(fx); return 0; }
    fclose(fp);
    return 1;
}

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
    layer_fixture fx;
    float *out_in = NULL, *mixer = NULL, *layer = NULL;
    size_t h, i, base;
    if (argc != 2) {
        fprintf(stderr, "usage: %s FIXTURE.bin\n", argv[0]);
        return 1;
    }
    if (!fixture_load(argv[1], &fx)) {
        fprintf(stderr, "failed to load fixture\n");
        return 1;
    }
    out_in = (float *)calloc((size_t)fx.heads * fx.head_dim, sizeof(float));
    mixer = (float *)calloc(fx.hidden, sizeof(float));
    layer = (float *)calloc(fx.hidden, sizeof(float));
    if (!out_in || !mixer || !layer) { fixture_free(&fx); free(out_in); free(mixer); free(layer); return 1; }

    for (h = 0; h < fx.heads; ++h) {
        double var = 0.0;
        base = h * fx.head_dim;
        for (i = 0; i < fx.head_dim; ++i) {
            double v = fx.core_last[base + i];
            var += v * v;
        }
        var /= (double)fx.head_dim;
        for (i = 0; i < fx.head_dim; ++i) {
            float normed = (float)(fx.core_last[base + i] / sqrt(var + 1e-6));
            out_in[base + i] = normed * fx.norm_w[i] * siluf_local(fx.z_last[base + i]);
        }
    }
    matvec(fx.out_w, out_in, mixer, fx.hidden, fx.heads * fx.head_dim);
    for (i = 0; i < fx.hidden; ++i) layer[i] = fx.layer_input[i] + mixer[i];

    printf("layer: %u\n", fx.layer);
    printf("mixer_rmse: %.8f\n", vec_rmse(mixer, fx.mixer_out_ref, fx.hidden));
    printf("mixer_cosine: %.8f\n", vec_cosine(mixer, fx.mixer_out_ref, fx.hidden));
    printf("residual_after_mixer_rmse: %.8f\n", vec_rmse(layer, fx.residual_after_mixer_ref, fx.hidden));
    printf("residual_after_mixer_cosine: %.8f\n", vec_cosine(layer, fx.residual_after_mixer_ref, fx.hidden));

    fixture_free(&fx);
    free(out_in); free(mixer); free(layer);
    return 0;
}
