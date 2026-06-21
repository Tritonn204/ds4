#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "Q36LINF1"
#define MAGIC_LEN 8

typedef struct linear_fixture {
    uint32_t layer;
    uint32_t hidden;
    uint32_t qkv_dim;
    uint32_t z_dim;
    uint32_t a_dim;
    uint32_t b_dim;
    uint32_t out_in_dim;
    uint32_t out_dim;
    float *input_ln;
    float *qkv_ref;
    float *z_ref;
    float *a_ref;
    float *b_ref;
    float *out_in;
    float *out_ref;
    float *mixer_out_ref;
    float *layer_out_ref;
    float *w_qkv;
    float *w_z;
    float *w_a;
    float *w_b;
    float *w_out;
} linear_fixture;

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

static void fixture_free(linear_fixture *fx) {
    if (!fx) return;
    free(fx->input_ln);
    free(fx->qkv_ref);
    free(fx->z_ref);
    free(fx->a_ref);
    free(fx->b_ref);
    free(fx->out_in);
    free(fx->out_ref);
    free(fx->mixer_out_ref);
    free(fx->layer_out_ref);
    free(fx->w_qkv);
    free(fx->w_z);
    free(fx->w_a);
    free(fx->w_b);
    free(fx->w_out);
    memset(fx, 0, sizeof(*fx));
}

static int fixture_load(const char *path, linear_fixture *fx) {
    FILE *fp = fopen(path, "rb");
    char magic[MAGIC_LEN];
    memset(fx, 0, sizeof(*fx));
    if (!fp) return 0;
    if (!read_exact(fp, magic, MAGIC_LEN) || memcmp(magic, MAGIC, MAGIC_LEN) != 0) {
        fclose(fp);
        return 0;
    }
    if (!read_exact(fp, &fx->layer, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->hidden, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->qkv_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->z_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->a_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->b_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->out_in_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->out_dim, sizeof(uint32_t))) {
        fclose(fp);
        return 0;
    }
    if (!alloc_read_f32(fp, &fx->input_ln, fx->hidden) ||
        !alloc_read_f32(fp, &fx->qkv_ref, fx->qkv_dim) ||
        !alloc_read_f32(fp, &fx->z_ref, fx->z_dim) ||
        !alloc_read_f32(fp, &fx->a_ref, fx->a_dim) ||
        !alloc_read_f32(fp, &fx->b_ref, fx->b_dim) ||
        !alloc_read_f32(fp, &fx->out_in, fx->out_in_dim) ||
        !alloc_read_f32(fp, &fx->out_ref, fx->out_dim) ||
        !alloc_read_f32(fp, &fx->mixer_out_ref, fx->out_dim) ||
        !alloc_read_f32(fp, &fx->layer_out_ref, fx->out_dim) ||
        !alloc_read_f32(fp, &fx->w_qkv, (size_t)fx->qkv_dim * fx->hidden) ||
        !alloc_read_f32(fp, &fx->w_z, (size_t)fx->z_dim * fx->hidden) ||
        !alloc_read_f32(fp, &fx->w_a, (size_t)fx->a_dim * fx->hidden) ||
        !alloc_read_f32(fp, &fx->w_b, (size_t)fx->b_dim * fx->hidden) ||
        !alloc_read_f32(fp, &fx->w_out, (size_t)fx->out_dim * fx->out_in_dim)) {
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
    linear_fixture fx;
    float *qkv = NULL, *z = NULL, *a = NULL, *b = NULL, *out = NULL;
    if (argc != 2) {
        fprintf(stderr, "usage: %s FIXTURE.bin\n", argv[0]);
        return 1;
    }
    if (!fixture_load(argv[1], &fx)) {
        fprintf(stderr, "failed to load fixture\n");
        return 1;
    }
    qkv = (float *)calloc(fx.qkv_dim, sizeof(float));
    z = (float *)calloc(fx.z_dim, sizeof(float));
    a = (float *)calloc(fx.a_dim, sizeof(float));
    b = (float *)calloc(fx.b_dim, sizeof(float));
    out = (float *)calloc(fx.out_dim, sizeof(float));
    if (!qkv || !z || !a || !b || !out) {
        fixture_free(&fx);
        free(qkv); free(z); free(a); free(b); free(out);
        return 1;
    }

    matvec(fx.w_qkv, fx.input_ln, qkv, fx.qkv_dim, fx.hidden);
    matvec(fx.w_z, fx.input_ln, z, fx.z_dim, fx.hidden);
    matvec(fx.w_a, fx.input_ln, a, fx.a_dim, fx.hidden);
    matvec(fx.w_b, fx.input_ln, b, fx.b_dim, fx.hidden);
    matvec(fx.w_out, fx.out_in, out, fx.out_dim, fx.out_in_dim);

    printf("layer: %u\n", fx.layer);
    printf("qkv_rmse: %.8f\n", vec_rmse(qkv, fx.qkv_ref, fx.qkv_dim));
    printf("qkv_cosine: %.8f\n", vec_cosine(qkv, fx.qkv_ref, fx.qkv_dim));
    printf("z_rmse: %.8f\n", vec_rmse(z, fx.z_ref, fx.z_dim));
    printf("z_cosine: %.8f\n", vec_cosine(z, fx.z_ref, fx.z_dim));
    printf("a_rmse: %.8f\n", vec_rmse(a, fx.a_ref, fx.a_dim));
    printf("a_cosine: %.8f\n", vec_cosine(a, fx.a_ref, fx.a_dim));
    printf("b_rmse: %.8f\n", vec_rmse(b, fx.b_ref, fx.b_dim));
    printf("b_cosine: %.8f\n", vec_cosine(b, fx.b_ref, fx.b_dim));
    printf("out_proj_rmse: %.8f\n", vec_rmse(out, fx.out_ref, fx.out_dim));
    printf("out_proj_cosine: %.8f\n", vec_cosine(out, fx.out_ref, fx.out_dim));

    fixture_free(&fx);
    free(qkv); free(z); free(a); free(b); free(out);
    return 0;
}
