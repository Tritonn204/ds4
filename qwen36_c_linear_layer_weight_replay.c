#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "Q36LWF01"
#define MAGIC_LEN 8

typedef struct wf_fixture {
    uint32_t layer, seq_len, hidden, num_v_heads, num_k_heads, head_k_dim, head_v_dim, key_dim, value_dim;
    float *layer_input_seq, *input_ln_seq, *qkv_seq, *z_seq, *a_seq, *b_seq, *conv_raw;
    float *q_ref, *k_ref, *v_ref, *beta_ref, *g_ref, *core_ref, *out_in_seq, *out_proj_out_seq;
    float *mixer_out_ref, *residual_after_mixer_ref;
    float *attn_norm_w, *w_qkv, *w_z, *w_a, *w_b, *conv_w, *A_log, *dt_bias, *ssm_norm_w, *w_out;
} wf_fixture;

static int read_exact(FILE *fp, void *buf, size_t n) { return fread(buf, 1, n, fp) == n; }
static int alloc_read_f32(FILE *fp, float **out, size_t count) {
    float *p = (float *)malloc(count * sizeof(float));
    if (!p) return 0;
    if (!read_exact(fp, p, count * sizeof(float))) { free(p); return 0; }
    *out = p;
    return 1;
}
static void fixture_free(wf_fixture *fx) {
    if (!fx) return;
    free(fx->layer_input_seq); free(fx->input_ln_seq); free(fx->qkv_seq); free(fx->z_seq); free(fx->a_seq); free(fx->b_seq); free(fx->conv_raw);
    free(fx->q_ref); free(fx->k_ref); free(fx->v_ref); free(fx->beta_ref); free(fx->g_ref); free(fx->core_ref); free(fx->out_in_seq); free(fx->out_proj_out_seq);
    free(fx->mixer_out_ref); free(fx->residual_after_mixer_ref);
    free(fx->attn_norm_w); free(fx->w_qkv); free(fx->w_z); free(fx->w_a); free(fx->w_b); free(fx->conv_w); free(fx->A_log); free(fx->dt_bias); free(fx->ssm_norm_w); free(fx->w_out);
    memset(fx, 0, sizeof(*fx));
}
static int fixture_load(const char *path, wf_fixture *fx) {
    FILE *fp = fopen(path, "rb");
    char magic[MAGIC_LEN];
    memset(fx, 0, sizeof(*fx));
    if (!fp) return 0;
    if (!read_exact(fp, magic, MAGIC_LEN) || memcmp(magic, MAGIC, MAGIC_LEN) != 0) { fclose(fp); return 0; }
    if (!read_exact(fp, &fx->layer, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->seq_len, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->hidden, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->num_v_heads, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->num_k_heads, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->head_k_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->head_v_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->key_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->value_dim, sizeof(uint32_t))) { fclose(fp); return 0; }
#define R(name,count) if (!alloc_read_f32(fp, &fx->name, (count))) { fclose(fp); fixture_free(fx); return 0; }
    R(layer_input_seq, (size_t)fx->seq_len * fx->hidden);
    R(input_ln_seq, (size_t)fx->seq_len * fx->hidden);
    R(qkv_seq, (size_t)fx->seq_len * (fx->key_dim * 2 + fx->value_dim));
    R(z_seq, (size_t)fx->seq_len * fx->value_dim);
    R(a_seq, (size_t)fx->seq_len * fx->num_v_heads);
    R(b_seq, (size_t)fx->seq_len * fx->num_v_heads);
    R(conv_raw, (size_t)(fx->key_dim * 2 + fx->value_dim) * (fx->seq_len + 3));
    R(q_ref, (size_t)fx->seq_len * fx->num_v_heads * fx->head_k_dim);
    R(k_ref, (size_t)fx->seq_len * fx->num_v_heads * fx->head_k_dim);
    R(v_ref, (size_t)fx->seq_len * fx->num_v_heads * fx->head_v_dim);
    R(beta_ref, (size_t)fx->seq_len * fx->num_v_heads);
    R(g_ref, (size_t)fx->seq_len * fx->num_v_heads);
    R(core_ref, (size_t)fx->seq_len * fx->num_v_heads * fx->head_v_dim);
    R(out_in_seq, (size_t)fx->seq_len * fx->value_dim);
    R(out_proj_out_seq, (size_t)fx->seq_len * fx->hidden);
    R(mixer_out_ref, fx->hidden);
    R(residual_after_mixer_ref, fx->hidden);
    R(attn_norm_w, fx->hidden);
    R(w_qkv, (size_t)(fx->key_dim * 2 + fx->value_dim) * fx->hidden);
    R(w_z, (size_t)fx->value_dim * fx->hidden);
    R(w_a, (size_t)fx->num_v_heads * fx->hidden);
    R(w_b, (size_t)fx->num_v_heads * fx->hidden);
    R(conv_w, (size_t)(fx->key_dim * 2 + fx->value_dim) * 4);
    R(A_log, fx->num_v_heads);
    R(dt_bias, fx->num_v_heads);
    R(ssm_norm_w, fx->head_v_dim);
    R(w_out, (size_t)fx->hidden * fx->value_dim);
#undef R
    fclose(fp);
    return 1;
}

static inline float sigmoidf_local(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float softplusf_local(float x) { return log1pf(expf(x)); }
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
    wf_fixture fx;
    float *input_ln = NULL, *qkv = NULL, *z = NULL, *a = NULL, *b = NULL, *conv = NULL;
    float *q = NULL, *k = NULL, *v = NULL, *beta = NULL, *g = NULL, *core = NULL, *state = NULL;
    float *out_in = NULL, *out_proj = NULL, *resid = NULL, *conv_ref = NULL;
    uint32_t t, h, d, hd, vd, rep, qkv_dim;
    if (argc != 2) {
        fprintf(stderr, "usage: %s FIXTURE.bin\n", argv[0]);
        return 1;
    }
    if (!fixture_load(argv[1], &fx)) {
        fprintf(stderr, "failed to load fixture\n");
        return 1;
    }
    qkv_dim = fx.key_dim * 2 + fx.value_dim;
    rep = fx.num_v_heads / fx.num_k_heads;
    input_ln = (float *)calloc((size_t)fx.seq_len * fx.hidden, sizeof(float));
    qkv = (float *)calloc((size_t)fx.seq_len * qkv_dim, sizeof(float));
    z = (float *)calloc((size_t)fx.seq_len * fx.value_dim, sizeof(float));
    a = (float *)calloc((size_t)fx.seq_len * fx.num_v_heads, sizeof(float));
    b = (float *)calloc((size_t)fx.seq_len * fx.num_v_heads, sizeof(float));
    conv = (float *)calloc((size_t)fx.seq_len * qkv_dim, sizeof(float));
    q = (float *)calloc((size_t)fx.seq_len * fx.num_v_heads * fx.head_k_dim, sizeof(float));
    k = (float *)calloc((size_t)fx.seq_len * fx.num_v_heads * fx.head_k_dim, sizeof(float));
    v = (float *)calloc((size_t)fx.seq_len * fx.num_v_heads * fx.head_v_dim, sizeof(float));
    beta = (float *)calloc((size_t)fx.seq_len * fx.num_v_heads, sizeof(float));
    g = (float *)calloc((size_t)fx.seq_len * fx.num_v_heads, sizeof(float));
    core = (float *)calloc((size_t)fx.seq_len * fx.num_v_heads * fx.head_v_dim, sizeof(float));
    state = (float *)calloc((size_t)fx.num_v_heads * fx.head_k_dim * fx.head_v_dim, sizeof(float));
    out_in = (float *)calloc((size_t)fx.seq_len * fx.value_dim, sizeof(float));
    out_proj = (float *)calloc((size_t)fx.seq_len * fx.hidden, sizeof(float));
    resid = (float *)calloc(fx.hidden, sizeof(float));
    conv_ref = (float *)calloc((size_t)fx.seq_len * qkv_dim, sizeof(float));
    if (!input_ln || !qkv || !z || !a || !b || !conv || !q || !k || !v || !beta || !g || !core || !state || !out_in || !out_proj || !resid || !conv_ref) {
        fixture_free(&fx);
        free(input_ln); free(qkv); free(z); free(a); free(b); free(conv); free(q); free(k); free(v); free(beta); free(g); free(core); free(state); free(out_in); free(out_proj); free(resid); free(conv_ref);
        return 1;
    }

    for (t = 0; t < fx.seq_len; ++t) {
        const float *in = fx.layer_input_seq + (size_t)t * fx.hidden;
        float *out = input_ln + (size_t)t * fx.hidden;
        double var = 0.0;
        for (d = 0; d < fx.hidden; ++d) var += (double)in[d] * in[d];
        var /= (double)fx.hidden;
        for (d = 0; d < fx.hidden; ++d) out[d] = (float)(in[d] / sqrt(var + 1e-6)) * fx.attn_norm_w[d];
        matvec(fx.w_qkv, out, qkv + (size_t)t * qkv_dim, qkv_dim, fx.hidden);
        matvec(fx.w_z, out, z + (size_t)t * fx.value_dim, fx.value_dim, fx.hidden);
        matvec(fx.w_a, out, a + (size_t)t * fx.num_v_heads, fx.num_v_heads, fx.hidden);
        matvec(fx.w_b, out, b + (size_t)t * fx.num_v_heads, fx.num_v_heads, fx.hidden);
    }

    for (t = 0; t < fx.seq_len; ++t) {
        for (d = 0; d < qkv_dim; ++d) {
            double sum = 0.0;
            for (uint32_t kk = 0; kk < 4; ++kk) {
                int src_t = (int)t - 3 + (int)kk;
                if (src_t >= 0 && src_t < (int)fx.seq_len) {
                    sum += (double)fx.conv_w[(size_t)d * 4 + kk] * qkv[(size_t)src_t * qkv_dim + d];
                }
            }
            conv[(size_t)t * qkv_dim + d] = siluf_local((float)sum);
        }
    }

    for (t = 0; t < fx.seq_len; ++t) {
        const float *cq = conv + (size_t)t * qkv_dim;
        const float *ck = cq + fx.key_dim;
        const float *cv = ck + fx.key_dim;
        for (h = 0; h < fx.num_k_heads; ++h) {
            for (d = 0; d < fx.head_k_dim; ++d) {
                float qv = cq[h * fx.head_k_dim + d];
                float kv = ck[h * fx.head_k_dim + d];
                for (uint32_t vh = 0; vh < rep; ++vh) {
                    uint32_t dst_h = h * rep + vh;
                    q[((size_t)t * fx.num_v_heads + dst_h) * fx.head_k_dim + d] = qv;
                    k[((size_t)t * fx.num_v_heads + dst_h) * fx.head_k_dim + d] = kv;
                }
            }
        }
        for (h = 0; h < fx.num_v_heads; ++h) {
            memcpy(v + ((size_t)t * fx.num_v_heads + h) * fx.head_v_dim, cv + (size_t)h * fx.head_v_dim, fx.head_v_dim * sizeof(float));
            beta[t * fx.num_v_heads + h] = sigmoidf_local(b[(size_t)t * fx.num_v_heads + h]);
            /* GGUF stores blk.*.ssm_a as the pre-expanded decay coefficient -exp(A_log). */
            g[t * fx.num_v_heads + h] = fx.A_log[h] * softplusf_local(a[(size_t)t * fx.num_v_heads + h] + fx.dt_bias[h]);
        }
    }

    for (t = 0; t < fx.seq_len; ++t) {
        for (d = 0; d < qkv_dim; ++d) {
            conv_ref[(size_t)t * qkv_dim + d] = siluf_local(fx.conv_raw[(size_t)d * (fx.seq_len + 3) + t]);
        }
    }

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
                    for (hd = 0; hd < fx.head_k_dim; ++hd) kv_mem += state[sbase + (size_t)hd * fx.head_v_dim + vd] * (k[qbase + hd] * knorm);
                    delta[vd] = (v[vbase + vd] - kv_mem) * beta_t;
                }
                for (hd = 0; hd < fx.head_k_dim; ++hd) {
                    float kval = k[qbase + hd] * knorm;
                    for (vd = 0; vd < fx.head_v_dim; ++vd) state[sbase + (size_t)hd * fx.head_v_dim + vd] += kval * delta[vd];
                }
                for (vd = 0; vd < fx.head_v_dim; ++vd) {
                    float outv = 0.0f;
                    for (hd = 0; hd < fx.head_k_dim; ++hd) outv += state[sbase + (size_t)hd * fx.head_v_dim + vd] * (q[qbase + hd] * qnorm / sqrtf((float)fx.head_k_dim));
                    core[vbase + vd] = outv;
                }
            }
        }
    }

    for (t = 0; t < fx.seq_len; ++t) {
        for (h = 0; h < fx.num_v_heads; ++h) {
            size_t base = (size_t)t * fx.value_dim + (size_t)h * fx.head_v_dim;
            double var = 0.0;
            for (vd = 0; vd < fx.head_v_dim; ++vd) {
                double cv = core[((size_t)t * fx.num_v_heads + h) * fx.head_v_dim + vd];
                var += cv * cv;
            }
            var /= (double)fx.head_v_dim;
            for (vd = 0; vd < fx.head_v_dim; ++vd) {
                float cv = core[((size_t)t * fx.num_v_heads + h) * fx.head_v_dim + vd];
                float normed = (float)(cv / sqrt(var + 1e-6));
                out_in[base + vd] = normed * fx.ssm_norm_w[vd] * siluf_local(z[(size_t)t * fx.value_dim + (size_t)h * fx.head_v_dim + vd]);
            }
        }
        matvec(fx.w_out, out_in + (size_t)t * fx.value_dim, out_proj + (size_t)t * fx.hidden, fx.hidden, fx.value_dim);
    }
    for (d = 0; d < fx.hidden; ++d) resid[d] = fx.layer_input_seq[(size_t)(fx.seq_len - 1) * fx.hidden + d] + out_proj[(size_t)(fx.seq_len - 1) * fx.hidden + d];

    printf("layer: %u\n", fx.layer);
    printf("seq_len: %u\n", fx.seq_len);
    printf("input_ln_rmse: %.8f\n", vec_rmse(input_ln, fx.input_ln_seq, (size_t)fx.seq_len * fx.hidden));
    printf("input_ln_cosine: %.8f\n", vec_cosine(input_ln, fx.input_ln_seq, (size_t)fx.seq_len * fx.hidden));
    printf("qkv_rmse: %.8f\n", vec_rmse(qkv, fx.qkv_seq, (size_t)fx.seq_len * qkv_dim));
    printf("qkv_cosine: %.8f\n", vec_cosine(qkv, fx.qkv_seq, (size_t)fx.seq_len * qkv_dim));
    printf("z_rmse: %.8f\n", vec_rmse(z, fx.z_seq, (size_t)fx.seq_len * fx.value_dim));
    printf("z_cosine: %.8f\n", vec_cosine(z, fx.z_seq, (size_t)fx.seq_len * fx.value_dim));
    printf("a_rmse: %.8f\n", vec_rmse(a, fx.a_seq, (size_t)fx.seq_len * fx.num_v_heads));
    printf("a_cosine: %.8f\n", vec_cosine(a, fx.a_seq, (size_t)fx.seq_len * fx.num_v_heads));
    printf("b_rmse: %.8f\n", vec_rmse(b, fx.b_seq, (size_t)fx.seq_len * fx.num_v_heads));
    printf("b_cosine: %.8f\n", vec_cosine(b, fx.b_seq, (size_t)fx.seq_len * fx.num_v_heads));
    printf("conv_post_rmse: %.8f\n", vec_rmse(conv, conv_ref, (size_t)fx.seq_len * qkv_dim));
    printf("conv_post_cosine: %.8f\n", vec_cosine(conv, conv_ref, (size_t)fx.seq_len * qkv_dim));
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
    printf("core_rmse: %.8f\n", vec_rmse(core, fx.core_ref, (size_t)fx.seq_len * fx.num_v_heads * fx.head_v_dim));
    printf("core_cosine: %.8f\n", vec_cosine(core, fx.core_ref, (size_t)fx.seq_len * fx.num_v_heads * fx.head_v_dim));
    printf("out_in_rmse: %.8f\n", vec_rmse(out_in, fx.out_in_seq, (size_t)fx.seq_len * fx.value_dim));
    printf("out_in_cosine: %.8f\n", vec_cosine(out_in, fx.out_in_seq, (size_t)fx.seq_len * fx.value_dim));
    printf("out_proj_rmse: %.8f\n", vec_rmse(out_proj, fx.out_proj_out_seq, (size_t)fx.seq_len * fx.hidden));
    printf("out_proj_cosine: %.8f\n", vec_cosine(out_proj, fx.out_proj_out_seq, (size_t)fx.seq_len * fx.hidden));
    printf("mixer_rmse: %.8f\n", vec_rmse(out_proj + (size_t)(fx.seq_len - 1) * fx.hidden, fx.mixer_out_ref, fx.hidden));
    printf("mixer_cosine: %.8f\n", vec_cosine(out_proj + (size_t)(fx.seq_len - 1) * fx.hidden, fx.mixer_out_ref, fx.hidden));
    printf("residual_after_mixer_rmse: %.8f\n", vec_rmse(resid, fx.residual_after_mixer_ref, fx.hidden));
    printf("residual_after_mixer_cosine: %.8f\n", vec_cosine(resid, fx.residual_after_mixer_ref, fx.hidden));

    fixture_free(&fx);
    free(input_ln); free(qkv); free(z); free(a); free(b); free(conv); free(q); free(k); free(v); free(beta); free(g); free(core); free(state); free(out_in); free(out_proj); free(resid); free(conv_ref);
    return 0;
}
