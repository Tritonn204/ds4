#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "Q36LCRF2"
#define MAGIC_LEN 8

typedef struct core_fixture {
    uint32_t layer, seq_len, num_v_heads, num_k_heads, head_k_dim, head_v_dim, key_dim, value_dim;
    float *conv_raw;
    float *a_seq;
    float *b_seq;
    float *A_log;
    float *dt_bias;
    float *q_ref;
    float *k_ref;
    float *v_ref;
    float *beta_ref;
    float *g_ref;
    float *core_ref;
} core_fixture;

static int read_exact(FILE *fp, void *buf, size_t n) { return fread(buf, 1, n, fp) == n; }

static int alloc_read_f32(FILE *fp, float **out, size_t count) {
    float *p = (float *)malloc(count * sizeof(float));
    if (!p) return 0;
    if (!read_exact(fp, p, count * sizeof(float))) { free(p); return 0; }
    *out = p;
    return 1;
}

static void fixture_free(core_fixture *fx) {
    if (!fx) return;
    free(fx->conv_raw); free(fx->a_seq); free(fx->b_seq); free(fx->A_log); free(fx->dt_bias);
    free(fx->q_ref); free(fx->k_ref); free(fx->v_ref); free(fx->beta_ref); free(fx->g_ref); free(fx->core_ref);
    memset(fx, 0, sizeof(*fx));
}

static int fixture_load(const char *path, core_fixture *fx) {
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
        !read_exact(fp, &fx->key_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->value_dim, sizeof(uint32_t))) { fclose(fp); return 0; }
    if (!alloc_read_f32(fp, &fx->conv_raw, (size_t)(fx->key_dim * 2 + fx->value_dim) * (fx->seq_len + 3)) ||
        !alloc_read_f32(fp, &fx->a_seq, (size_t)fx->seq_len * fx->num_v_heads) ||
        !alloc_read_f32(fp, &fx->b_seq, (size_t)fx->seq_len * fx->num_v_heads) ||
        !alloc_read_f32(fp, &fx->A_log, fx->num_v_heads) ||
        !alloc_read_f32(fp, &fx->dt_bias, fx->num_v_heads) ||
        !alloc_read_f32(fp, &fx->q_ref, (size_t)fx->seq_len * fx->num_v_heads * fx->head_k_dim) ||
        !alloc_read_f32(fp, &fx->k_ref, (size_t)fx->seq_len * fx->num_v_heads * fx->head_k_dim) ||
        !alloc_read_f32(fp, &fx->v_ref, (size_t)fx->seq_len * fx->num_v_heads * fx->head_v_dim) ||
        !alloc_read_f32(fp, &fx->beta_ref, (size_t)fx->seq_len * fx->num_v_heads) ||
        !alloc_read_f32(fp, &fx->g_ref, (size_t)fx->seq_len * fx->num_v_heads) ||
        !alloc_read_f32(fp, &fx->core_ref, (size_t)fx->seq_len * fx->num_v_heads * fx->head_v_dim)) { fclose(fp); fixture_free(fx); return 0; }
    fclose(fp);
    return 1;
}

static inline float sigmoidf_local(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float softplusf_local(float x) { return log1pf(expf(x)); }
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
    core_fixture fx;
    float *q = NULL, *k = NULL, *v = NULL, *beta = NULL, *g = NULL, *core = NULL, *state = NULL;
    uint32_t t, h, d, rep, hd, vd;
    if (argc != 2) {
        fprintf(stderr, "usage: %s FIXTURE.bin\n", argv[0]);
        return 1;
    }
    if (!fixture_load(argv[1], &fx)) {
        fprintf(stderr, "failed to load fixture\n");
        return 1;
    }
    q = (float *)calloc((size_t)fx.seq_len * fx.num_v_heads * fx.head_k_dim, sizeof(float));
    k = (float *)calloc((size_t)fx.seq_len * fx.num_v_heads * fx.head_k_dim, sizeof(float));
    v = (float *)calloc((size_t)fx.seq_len * fx.num_v_heads * fx.head_v_dim, sizeof(float));
    beta = (float *)calloc((size_t)fx.seq_len * fx.num_v_heads, sizeof(float));
    g = (float *)calloc((size_t)fx.seq_len * fx.num_v_heads, sizeof(float));
    core = (float *)calloc((size_t)fx.seq_len * fx.num_v_heads * fx.head_v_dim, sizeof(float));
    state = (float *)calloc((size_t)fx.num_v_heads * fx.head_k_dim * fx.head_v_dim, sizeof(float));
    if (!q || !k || !v || !beta || !g || !core || !state) { fixture_free(&fx); free(q); free(k); free(v); free(beta); free(g); free(core); free(state); return 1; }

    rep = fx.num_v_heads / fx.num_k_heads;

    for (t = 0; t < fx.seq_len; ++t) {
        const float *src_q = fx.conv_raw + (size_t)0 * (fx.seq_len + 3) + t;
        const float *src_k = fx.conv_raw + (size_t)fx.key_dim * (fx.seq_len + 3) + t;
        const float *src_v = fx.conv_raw + (size_t)(fx.key_dim * 2) * (fx.seq_len + 3) + t;
        for (h = 0; h < fx.num_k_heads; ++h) {
            for (d = 0; d < fx.head_k_dim; ++d) {
                uint32_t qidx = h * fx.head_k_dim + d;
                float qv = siluf_local(src_q[(size_t)qidx * (fx.seq_len + 3)]);
                float kv = siluf_local(src_k[(size_t)qidx * (fx.seq_len + 3)]);
                uint32_t vh;
                for (vh = 0; vh < rep; ++vh) {
                    uint32_t dst_h = h * rep + vh;
                    q[((size_t)t * fx.num_v_heads + dst_h) * fx.head_k_dim + d] = qv;
                    k[((size_t)t * fx.num_v_heads + dst_h) * fx.head_k_dim + d] = kv;
                }
            }
        }
        for (h = 0; h < fx.num_v_heads; ++h) {
            for (d = 0; d < fx.head_v_dim; ++d) {
                uint32_t vidx = h * fx.head_v_dim + d;
                v[((size_t)t * fx.num_v_heads + h) * fx.head_v_dim + d] =
                    siluf_local(src_v[(size_t)vidx * (fx.seq_len + 3)]);
            }
            beta[t * fx.num_v_heads + h] = sigmoidf_local(fx.b_seq[t * fx.num_v_heads + h]);
            /* GGUF stores blk.*.ssm_a as the pre-expanded decay coefficient -exp(A_log). */
            g[t * fx.num_v_heads + h] =
                fx.A_log[h] * softplusf_local(fx.a_seq[t * fx.num_v_heads + h] + fx.dt_bias[h]);
        }
    }

    printf("layer: %u\n", fx.layer);
    printf("seq_len: %u\n", fx.seq_len);
    printf("query_rmse: %.8f\n", vec_rmse(q, fx.q_ref, (size_t)fx.seq_len * fx.num_v_heads * fx.head_k_dim));
    printf("query_cosine: %.8f\n", vec_cosine(q, fx.q_ref, (size_t)fx.seq_len * fx.num_v_heads * fx.head_k_dim));
    printf("key_rmse: %.8f\n", vec_rmse(k, fx.k_ref, (size_t)fx.seq_len * fx.num_v_heads * fx.head_k_dim));
    printf("key_cosine: %.8f\n", vec_cosine(k, fx.k_ref, (size_t)fx.seq_len * fx.num_v_heads * fx.head_k_dim));
    printf("value_rmse: %.8f\n", vec_rmse(v, fx.v_ref, (size_t)fx.seq_len * fx.num_v_heads * fx.head_v_dim));
    printf("value_cosine: %.8f\n", vec_cosine(v, fx.v_ref, (size_t)fx.seq_len * fx.num_v_heads * fx.head_v_dim));
    printf("beta_rmse: %.8f\n", vec_rmse(beta, fx.beta_ref, (size_t)fx.seq_len * fx.num_v_heads));
    printf("beta_cosine: %.8f\n", vec_cosine(beta, fx.beta_ref, (size_t)fx.seq_len * fx.num_v_heads));
    printf("g_rmse: %.8f\n", vec_rmse(g, fx.g_ref, (size_t)fx.seq_len * fx.num_v_heads));
    printf("g_cosine: %.8f\n", vec_cosine(g, fx.g_ref, (size_t)fx.seq_len * fx.num_v_heads));

    for (t = 0; t < fx.seq_len; ++t) {
        for (h = 0; h < fx.num_v_heads; ++h) {
            float qnorm = 0.0f, knorm = 0.0f;
            float gexp = expf(g[t * fx.num_v_heads + h]);
            float beta_t = beta[t * fx.num_v_heads + h];
            size_t qbase = ((size_t)t * fx.num_v_heads + h) * fx.head_k_dim;
            size_t vbase = ((size_t)t * fx.num_v_heads + h) * fx.head_v_dim;
            size_t sbase = (size_t)h * fx.head_k_dim * fx.head_v_dim;
            for (hd = 0; hd < fx.head_k_dim; ++hd) {
                qnorm += q[qbase + hd] * q[qbase + hd];
                knorm += k[qbase + hd] * k[qbase + hd];
            }
            qnorm = 1.0f / sqrtf(qnorm + 1e-6f);
            knorm = 1.0f / sqrtf(knorm + 1e-6f);
            for (hd = 0; hd < fx.head_k_dim * fx.head_v_dim; ++hd) state[sbase + hd] *= gexp;
            {
                float delta[128];
                for (vd = 0; vd < fx.head_v_dim; ++vd) {
                    float kv_mem = 0.0f;
                    for (hd = 0; hd < fx.head_k_dim; ++hd) {
                        kv_mem += state[sbase + (size_t)hd * fx.head_v_dim + vd] * (k[qbase + hd] * knorm);
                    }
                    delta[vd] = (v[vbase + vd] - kv_mem) * beta_t;
                }
                for (hd = 0; hd < fx.head_k_dim; ++hd) {
                    float kval = k[qbase + hd] * knorm;
                    for (vd = 0; vd < fx.head_v_dim; ++vd) {
                        state[sbase + (size_t)hd * fx.head_v_dim + vd] += kval * delta[vd];
                    }
                }
                for (vd = 0; vd < fx.head_v_dim; ++vd) {
                    float outv = 0.0f;
                    for (hd = 0; hd < fx.head_k_dim; ++hd) {
                        outv += state[sbase + (size_t)hd * fx.head_v_dim + vd] * (q[qbase + hd] * qnorm / sqrtf((float)fx.head_k_dim));
                    }
                    core[vbase + vd] = outv;
                }
            }
        }
    }

    printf("core_rmse: %.8f\n", vec_rmse(core, fx.core_ref, (size_t)fx.seq_len * fx.num_v_heads * fx.head_v_dim));
    printf("core_cosine: %.8f\n", vec_cosine(core, fx.core_ref, (size_t)fx.seq_len * fx.num_v_heads * fx.head_v_dim));

    fixture_free(&fx);
    free(q); free(k); free(v); free(beta); free(g); free(core); free(state);
    return 0;
}
