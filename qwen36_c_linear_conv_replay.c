#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "Q36LCVF1"
#define MAGIC_LEN 8

typedef struct conv_fixture {
    uint32_t layer;
    uint32_t seq_len;
    uint32_t hidden;
    uint32_t qkv_dim;
    float *input_ln_seq;
    float *qkv_ref;
    float *conv_post_ref;
    float *conv_w;
    float *w_qkv;
} conv_fixture;

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

static void fixture_free(conv_fixture *fx) {
    if (!fx) return;
    free(fx->input_ln_seq);
    free(fx->qkv_ref);
    free(fx->conv_post_ref);
    free(fx->conv_w);
    free(fx->w_qkv);
    memset(fx, 0, sizeof(*fx));
}

static int fixture_load(const char *path, conv_fixture *fx) {
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
        !read_exact(fp, &fx->qkv_dim, sizeof(uint32_t))) {
        fclose(fp);
        return 0;
    }
    if (!alloc_read_f32(fp, &fx->input_ln_seq, (size_t)fx->seq_len * fx->hidden) ||
        !alloc_read_f32(fp, &fx->qkv_ref, (size_t)fx->seq_len * fx->qkv_dim) ||
        !alloc_read_f32(fp, &fx->conv_post_ref, (size_t)fx->seq_len * fx->qkv_dim) ||
        !alloc_read_f32(fp, &fx->conv_w, (size_t)fx->qkv_dim * 4) ||
        !alloc_read_f32(fp, &fx->w_qkv, (size_t)fx->qkv_dim * fx->hidden)) {
        fclose(fp);
        fixture_free(fx);
        return 0;
    }
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

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

static double vec_rmse(const float *a, const float *b, uint32_t n) {
    uint32_t i;
    double acc = 0.0;
    for (i = 0; i < n; ++i) {
        double d = (double)a[i] - b[i];
        acc += d * d;
    }
    return sqrt(acc / n);
}

static double vec_cosine(const float *a, const float *b, uint32_t n) {
    uint32_t i;
    double dot = 0.0, an = 0.0, bn = 0.0;
    for (i = 0; i < n; ++i) {
        dot += (double)a[i] * b[i];
        an += (double)a[i] * a[i];
        bn += (double)b[i] * b[i];
    }
    if (an == 0.0 || bn == 0.0) return NAN;
    return dot / sqrt(an * bn);
}

int main(int argc, char **argv) {
    conv_fixture fx;
    float *qkv = NULL, *conv = NULL;
    uint32_t t, c, k;
    if (argc != 2) {
        fprintf(stderr, "usage: %s FIXTURE.bin\n", argv[0]);
        return 1;
    }
    if (!fixture_load(argv[1], &fx)) {
        fprintf(stderr, "failed to load fixture\n");
        return 1;
    }
    qkv = (float *)calloc((size_t)fx.seq_len * fx.qkv_dim, sizeof(float));
    conv = (float *)calloc((size_t)fx.seq_len * fx.qkv_dim, sizeof(float));
    if (!qkv || !conv) {
        fixture_free(&fx);
        free(qkv); free(conv);
        return 1;
    }

    for (t = 0; t < fx.seq_len; ++t) {
        matvec(fx.w_qkv, fx.input_ln_seq + (size_t)t * fx.hidden, qkv + (size_t)t * fx.qkv_dim, fx.qkv_dim, fx.hidden);
    }

    for (t = 0; t < fx.seq_len; ++t) {
        for (c = 0; c < fx.qkv_dim; ++c) {
            double sum = 0.0;
            for (k = 0; k < 4; ++k) {
                int src_t = (int)t - 3 + (int)k;
                if (src_t >= 0 && src_t < (int)fx.seq_len) {
                    sum += (double)fx.conv_w[(size_t)c * 4 + k] * qkv[(size_t)src_t * fx.qkv_dim + c];
                }
            }
            conv[(size_t)t * fx.qkv_dim + c] = silu((float)sum);
        }
    }

    printf("layer: %u\n", fx.layer);
    printf("seq_len: %u\n", fx.seq_len);
    printf("qkv_rmse: %.8f\n", vec_rmse(qkv, fx.qkv_ref, fx.seq_len * fx.qkv_dim));
    printf("qkv_cosine: %.8f\n", vec_cosine(qkv, fx.qkv_ref, fx.seq_len * fx.qkv_dim));
    printf("conv_post_rmse: %.8f\n", vec_rmse(conv, fx.conv_post_ref, fx.seq_len * fx.qkv_dim));
    printf("conv_post_cosine: %.8f\n", vec_cosine(conv, fx.conv_post_ref, fx.seq_len * fx.qkv_dim));

    fixture_free(&fx);
    free(qkv);
    free(conv);
    return 0;
}
