#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "Q36LLFF1"
#define MAGIC_LEN 8

typedef struct full_fixture {
    uint32_t layer, seq_len, num_v_heads, num_k_heads, head_k_dim, head_v_dim, out_in_dim, hidden;
    float *layer_input;
    float *q;
    float *k;
    float *v;
    float *beta;
    float *g;
    float *z_last;
    float *mixer_out_ref;
    float *residual_after_mixer_ref;
    float *post_attn_ln_ref;
    float *layer_out_ref;
    float *norm_w;
    float *out_w;
} full_fixture;

static int read_exact(FILE *fp, void *buf, size_t n) { return fread(buf, 1, n, fp) == n; }

static int alloc_read_f32(FILE *fp, float **out, size_t count) {
    float *p = (float *)malloc(count * sizeof(float));
    if (!p) return 0;
    if (!read_exact(fp, p, count * sizeof(float))) { free(p); return 0; }
    *out = p;
    return 1;
}

static void fixture_free(full_fixture *fx) {
    if (!fx) return;
    free(fx->layer_input); free(fx->q); free(fx->k); free(fx->v); free(fx->beta); free(fx->g);
    free(fx->z_last); free(fx->mixer_out_ref); free(fx->residual_after_mixer_ref); free(fx->post_attn_ln_ref); free(fx->layer_out_ref);
    free(fx->norm_w); free(fx->out_w);
    memset(fx, 0, sizeof(*fx));
}

static int fixture_load(const char *path, full_fixture *fx) {
    FILE *fp = fopen(path, "rb");
    char magic[MAGIC_LEN];
    memset(fx, 0, sizeof(*fx));
    if (!fp) return 0;
    if (!read_exact(fp, magic, MAGIC_LEN) || memcmp(magic, MAGIC, MAGIC_LEN) != 0) { fclose(fp); return 0; }
    if (!read_exact(fp, &fx->layer, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->seq_len, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->num_v_heads, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->num_k_heads, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->head_k_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->head_v_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->out_in_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->hidden, sizeof(uint32_t))) { fclose(fp); return 0; }
    if (!alloc_read_f32(fp, &fx->layer_input, fx->hidden) ||
        !alloc_read_f32(fp, &fx->q, (size_t)fx->seq_len * fx->num_v_heads * fx->head_k_dim) ||
        !alloc_read_f32(fp, &fx->k, (size_t)fx->seq_len * fx->num_v_heads * fx->head_k_dim) ||
        !alloc_read_f32(fp, &fx->v, (size_t)fx->seq_len * fx->num_v_heads * fx->head_v_dim) ||
        !alloc_read_f32(fp, &fx->beta, (size_t)fx->seq_len * fx->num_v_heads) ||
        !alloc_read_f32(fp, &fx->g, (size_t)fx->seq_len * fx->num_v_heads) ||
        !alloc_read_f32(fp, &fx->z_last, (size_t)fx->num_v_heads * fx->head_v_dim) ||
        !alloc_read_f32(fp, &fx->mixer_out_ref, fx->hidden) ||
        !alloc_read_f32(fp, &fx->residual_after_mixer_ref, fx->hidden) ||
        !alloc_read_f32(fp, &fx->post_attn_ln_ref, fx->hidden) ||
        !alloc_read_f32(fp, &fx->layer_out_ref, fx->hidden) ||
        !alloc_read_f32(fp, &fx->norm_w, fx->head_v_dim) ||
        !alloc_read_f32(fp, &fx->out_w, (size_t)fx->hidden * fx->out_in_dim)) { fclose(fp); fixture_free(fx); return 0; }
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
    full_fixture fx;
    float *state = NULL, *core = NULL, *out_in = NULL, *mixer = NULL, *resid = NULL;
    uint32_t t, h, hd, vd;
    if (argc != 2) {
        fprintf(stderr, "usage: %s FIXTURE.bin\n", argv[0]);
        return 1;
    }
    if (!fixture_load(argv[1], &fx)) {
        fprintf(stderr, "failed to load fixture\n");
        return 1;
    }
    state = (float *)calloc((size_t)fx.num_v_heads * fx.head_k_dim * fx.head_v_dim, sizeof(float));
    core = (float *)calloc((size_t)fx.seq_len * fx.num_v_heads * fx.head_v_dim, sizeof(float));
    out_in = (float *)calloc(fx.out_in_dim, sizeof(float));
    mixer = (float *)calloc(fx.hidden, sizeof(float));
    resid = (float *)calloc(fx.hidden, sizeof(float));
    if (!state || !core || !out_in || !mixer || !resid) { fixture_free(&fx); free(state); free(core); free(out_in); free(mixer); free(resid); return 1; }

    for (t = 0; t < fx.seq_len; ++t) {
        for (h = 0; h < fx.num_v_heads; ++h) {
            float gexp = expf(fx.g[t * fx.num_v_heads + h]);
            float beta_t = fx.beta[t * fx.num_v_heads + h];
            float qnorm = 0.0f, knorm = 0.0f;
            size_t qbase = ((size_t)t * fx.num_v_heads + h) * fx.head_k_dim;
            size_t vbase = ((size_t)t * fx.num_v_heads + h) * fx.head_v_dim;
            size_t sbase = (size_t)h * fx.head_k_dim * fx.head_v_dim;
            for (hd = 0; hd < fx.head_k_dim; ++hd) {
                qnorm += fx.q[qbase + hd] * fx.q[qbase + hd];
                knorm += fx.k[qbase + hd] * fx.k[qbase + hd];
            }
            qnorm = 1.0f / sqrtf(qnorm + 1e-6f);
            knorm = 1.0f / sqrtf(knorm + 1e-6f);
            for (hd = 0; hd < fx.head_k_dim * fx.head_v_dim; ++hd) state[sbase + hd] *= gexp;
            {
                float delta[128];
                for (vd = 0; vd < fx.head_v_dim; ++vd) {
                    float kv_mem = 0.0f;
                    for (hd = 0; hd < fx.head_k_dim; ++hd) {
                        kv_mem += state[sbase + (size_t)hd * fx.head_v_dim + vd] * (fx.k[qbase + hd] * knorm);
                    }
                    delta[vd] = (fx.v[vbase + vd] - kv_mem) * beta_t;
                }
                for (hd = 0; hd < fx.head_k_dim; ++hd) {
                    float kval = fx.k[qbase + hd] * knorm;
                    for (vd = 0; vd < fx.head_v_dim; ++vd) {
                        state[sbase + (size_t)hd * fx.head_v_dim + vd] += kval * delta[vd];
                    }
                }
                for (vd = 0; vd < fx.head_v_dim; ++vd) {
                    float outv = 0.0f;
                    for (hd = 0; hd < fx.head_k_dim; ++hd) {
                        outv += state[sbase + (size_t)hd * fx.head_v_dim + vd] * (fx.q[qbase + hd] * qnorm / sqrtf((float)fx.head_k_dim));
                    }
                    core[vbase + vd] = outv;
                }
            }
        }
    }

    {
        size_t base;
        for (h = 0; h < fx.num_v_heads; ++h) {
            double var = 0.0;
            base = (size_t)h * fx.head_v_dim;
            for (vd = 0; vd < fx.head_v_dim; ++vd) {
                double v = core[((size_t)(fx.seq_len - 1) * fx.num_v_heads + h) * fx.head_v_dim + vd];
                var += v * v;
            }
            var /= (double)fx.head_v_dim;
            for (vd = 0; vd < fx.head_v_dim; ++vd) {
                float cv = core[((size_t)(fx.seq_len - 1) * fx.num_v_heads + h) * fx.head_v_dim + vd];
                float normed = (float)(cv / sqrt(var + 1e-6));
                out_in[base + vd] = normed * fx.norm_w[vd] * siluf_local(fx.z_last[base + vd]);
            }
        }
    }
    matvec(fx.out_w, out_in, mixer, fx.hidden, fx.out_in_dim);
    for (vd = 0; vd < fx.hidden; ++vd) resid[vd] = fx.layer_input[vd] + mixer[vd];

    printf("layer: %u\n", fx.layer);
    printf("mixer_rmse: %.8f\n", vec_rmse(mixer, fx.mixer_out_ref, fx.hidden));
    printf("mixer_cosine: %.8f\n", vec_cosine(mixer, fx.mixer_out_ref, fx.hidden));
    printf("residual_after_mixer_rmse: %.8f\n", vec_rmse(resid, fx.residual_after_mixer_ref, fx.hidden));
    printf("residual_after_mixer_cosine: %.8f\n", vec_cosine(resid, fx.residual_after_mixer_ref, fx.hidden));

    fixture_free(&fx);
    free(state); free(core); free(out_in); free(mixer); free(resid);
    return 0;
}
