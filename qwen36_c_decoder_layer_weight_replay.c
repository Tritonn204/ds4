#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "Q36DWF01"
#define MAGIC_LEN 8
#define ROUTER_COUNT 256
#define INTER 512

typedef struct dwf_fixture {
    uint32_t layer, seq_len, hidden, num_v_heads, num_k_heads, head_k_dim, head_v_dim, key_dim, value_dim, topk;
    float *layer_input_seq, *input_ln_seq, *qkv_seq, *z_seq, *a_seq, *b_seq, *conv_raw;
    float *q_ref, *k_ref, *v_ref, *beta_ref, *g_ref, *core_ref, *out_in_seq, *out_proj_out_seq;
    float *mixer_out_ref, *layer_input_last_ref, *residual_after_mixer_ref, *post_attn_ln_ref, *mlp_out_ref, *layer_output_ref;
    float *router_logits_ref, *router_indices_ref_f32, *router_scores_ref, *shared_gate_pre_ref;
    float *attn_norm_w, *post_attn_norm_w, *w_qkv, *w_z, *w_a, *w_b, *conv_w, *A_log, *dt_bias, *ssm_norm_w, *w_out;
    float *router_w;
    float *gate_sel, *up_sel, *down_sel, *gate_shexp, *up_shexp, *down_shexp, *gate_inp_shexp;
} dwf_fixture;

static int read_exact(FILE *fp, void *buf, size_t n) { return fread(buf, 1, n, fp) == n; }
static int alloc_read_f32(FILE *fp, float **out, size_t count) {
    float *p = (float *)malloc(count * sizeof(float));
    if (!p) return 0;
    if (!read_exact(fp, p, count * sizeof(float))) { free(p); return 0; }
    *out = p;
    return 1;
}
static void fixture_free(dwf_fixture *fx) {
    if (!fx) return;
    free(fx->layer_input_seq); free(fx->input_ln_seq); free(fx->qkv_seq); free(fx->z_seq); free(fx->a_seq); free(fx->b_seq); free(fx->conv_raw);
    free(fx->q_ref); free(fx->k_ref); free(fx->v_ref); free(fx->beta_ref); free(fx->g_ref); free(fx->core_ref); free(fx->out_in_seq); free(fx->out_proj_out_seq);
    free(fx->mixer_out_ref); free(fx->layer_input_last_ref); free(fx->residual_after_mixer_ref); free(fx->post_attn_ln_ref); free(fx->mlp_out_ref); free(fx->layer_output_ref);
    free(fx->router_logits_ref); free(fx->router_indices_ref_f32); free(fx->router_scores_ref); free(fx->shared_gate_pre_ref);
    free(fx->attn_norm_w); free(fx->post_attn_norm_w); free(fx->w_qkv); free(fx->w_z); free(fx->w_a); free(fx->w_b); free(fx->conv_w); free(fx->A_log); free(fx->dt_bias); free(fx->ssm_norm_w); free(fx->w_out);
    free(fx->router_w);
    free(fx->gate_sel); free(fx->up_sel); free(fx->down_sel); free(fx->gate_shexp); free(fx->up_shexp); free(fx->down_shexp); free(fx->gate_inp_shexp);
    memset(fx, 0, sizeof(*fx));
}
static int fixture_load(const char *path, dwf_fixture *fx) {
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
        !read_exact(fp, &fx->value_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->topk, sizeof(uint32_t))) { fclose(fp); return 0; }
#define R(name,count) if (!alloc_read_f32(fp, &fx->name, (count))) { fclose(fp); fixture_free(fx); return 0; }
    R(layer_input_seq, (size_t)fx->seq_len * fx->hidden); R(input_ln_seq, (size_t)fx->seq_len * fx->hidden); R(qkv_seq, (size_t)fx->seq_len * (fx->key_dim * 2 + fx->value_dim));
    R(z_seq, (size_t)fx->seq_len * fx->value_dim); R(a_seq, (size_t)fx->seq_len * fx->num_v_heads); R(b_seq, (size_t)fx->seq_len * fx->num_v_heads); R(conv_raw, (size_t)(fx->key_dim * 2 + fx->value_dim) * (fx->seq_len + 3));
    R(q_ref, (size_t)fx->seq_len * fx->num_v_heads * fx->head_k_dim); R(k_ref, (size_t)fx->seq_len * fx->num_v_heads * fx->head_k_dim); R(v_ref, (size_t)fx->seq_len * fx->num_v_heads * fx->head_v_dim);
    R(beta_ref, (size_t)fx->seq_len * fx->num_v_heads); R(g_ref, (size_t)fx->seq_len * fx->num_v_heads); R(core_ref, (size_t)fx->seq_len * fx->num_v_heads * fx->head_v_dim);
    R(out_in_seq, (size_t)fx->seq_len * fx->value_dim); R(out_proj_out_seq, (size_t)fx->seq_len * fx->hidden);
    R(mixer_out_ref, fx->hidden); R(layer_input_last_ref, fx->hidden); R(residual_after_mixer_ref, fx->hidden); R(post_attn_ln_ref, fx->hidden); R(mlp_out_ref, fx->hidden); R(layer_output_ref, fx->hidden);
    R(router_logits_ref, ROUTER_COUNT); R(router_indices_ref_f32, fx->topk); R(router_scores_ref, fx->topk); R(shared_gate_pre_ref, 1);
    R(attn_norm_w, fx->hidden); R(post_attn_norm_w, fx->hidden); R(w_qkv, (size_t)(fx->key_dim * 2 + fx->value_dim) * fx->hidden); R(w_z, (size_t)fx->value_dim * fx->hidden);
    R(w_a, (size_t)fx->num_v_heads * fx->hidden); R(w_b, (size_t)fx->num_v_heads * fx->hidden); R(conv_w, (size_t)(fx->key_dim * 2 + fx->value_dim) * 4); R(A_log, fx->num_v_heads); R(dt_bias, fx->num_v_heads);
    R(ssm_norm_w, fx->head_v_dim); R(w_out, (size_t)fx->hidden * fx->value_dim); R(router_w, (size_t)ROUTER_COUNT * fx->hidden);
    R(gate_sel, (size_t)fx->topk * INTER * fx->hidden); R(up_sel, (size_t)fx->topk * INTER * fx->hidden); R(down_sel, (size_t)fx->topk * fx->hidden * INTER);
    R(gate_shexp, (size_t)INTER * fx->hidden); R(up_shexp, (size_t)INTER * fx->hidden); R(down_shexp, (size_t)fx->hidden * INTER); R(gate_inp_shexp, fx->hidden);
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

static void topk_softmax256(const float *logits, uint32_t k, uint32_t *idx, float *scores) {
    uint32_t i, j, m;
    for (i = 0; i < k; ++i) idx[i] = i;
    for (i = k; i < ROUTER_COUNT; ++i) {
        m = 0;
        for (j = 1; j < k; ++j) if (logits[idx[j]] < logits[idx[m]]) m = j;
        if (logits[i] > logits[idx[m]]) idx[m] = i;
    }
    for (i = 0; i < k; ++i) {
        for (j = i + 1; j < k; ++j) {
            if (logits[idx[j]] > logits[idx[i]]) {
                uint32_t tmp = idx[i];
                idx[i] = idx[j];
                idx[j] = tmp;
            }
        }
    }
    {
        float maxv = logits[idx[0]];
        double sum = 0.0;
        for (i = 0; i < k; ++i) sum += exp((double)logits[idx[i]] - maxv);
        for (i = 0; i < k; ++i) scores[i] = (float)(exp((double)logits[idx[i]] - maxv) / sum);
    }
}

int main(int argc, char **argv) {
    dwf_fixture fx;
    float *input_ln = NULL, *qkv = NULL, *z = NULL, *a = NULL, *b = NULL, *conv = NULL, *conv_ref = NULL;
    float *q = NULL, *k = NULL, *v = NULL, *beta = NULL, *g = NULL, *core = NULL, *state = NULL, *out_in = NULL, *out_proj = NULL;
    float *resid = NULL, *post_attn_ln = NULL, *router_logits = NULL, *router_scores = NULL, *mlp = NULL, *final_out = NULL;
    float *gate = NULL, *up = NULL, *act = NULL, *down = NULL, *shared_gate = NULL, *shared_up = NULL, *shared_act = NULL, *shared = NULL;
    uint32_t *router_idx = NULL;
    uint32_t t, h, d, hd, vd, rep, qkv_dim, i;

    if (argc != 2) { fprintf(stderr, "usage: %s FIXTURE.bin\n", argv[0]); return 1; }
    if (!fixture_load(argv[1], &fx)) { fprintf(stderr, "failed to load fixture\n"); return 1; }
    qkv_dim = fx.key_dim * 2 + fx.value_dim;
    rep = fx.num_v_heads / fx.num_k_heads;

    input_ln = calloc((size_t)fx.seq_len * fx.hidden, sizeof(float));
    qkv = calloc((size_t)fx.seq_len * qkv_dim, sizeof(float));
    z = calloc((size_t)fx.seq_len * fx.value_dim, sizeof(float));
    a = calloc((size_t)fx.seq_len * fx.num_v_heads, sizeof(float));
    b = calloc((size_t)fx.seq_len * fx.num_v_heads, sizeof(float));
    conv = calloc((size_t)fx.seq_len * qkv_dim, sizeof(float));
    conv_ref = calloc((size_t)fx.seq_len * qkv_dim, sizeof(float));
    q = calloc((size_t)fx.seq_len * fx.num_v_heads * fx.head_k_dim, sizeof(float));
    k = calloc((size_t)fx.seq_len * fx.num_v_heads * fx.head_k_dim, sizeof(float));
    v = calloc((size_t)fx.seq_len * fx.num_v_heads * fx.head_v_dim, sizeof(float));
    beta = calloc((size_t)fx.seq_len * fx.num_v_heads, sizeof(float));
    g = calloc((size_t)fx.seq_len * fx.num_v_heads, sizeof(float));
    core = calloc((size_t)fx.seq_len * fx.num_v_heads * fx.head_v_dim, sizeof(float));
    state = calloc((size_t)fx.num_v_heads * fx.head_k_dim * fx.head_v_dim, sizeof(float));
    out_in = calloc((size_t)fx.seq_len * fx.value_dim, sizeof(float));
    out_proj = calloc((size_t)fx.seq_len * fx.hidden, sizeof(float));
    resid = calloc(fx.hidden, sizeof(float));
    post_attn_ln = calloc(fx.hidden, sizeof(float));
    router_logits = calloc(ROUTER_COUNT, sizeof(float));
    router_idx = calloc(fx.topk, sizeof(uint32_t));
    router_scores = calloc(fx.topk, sizeof(float));
    mlp = calloc(fx.hidden, sizeof(float));
    final_out = calloc(fx.hidden, sizeof(float));
    gate = calloc(INTER, sizeof(float)); up = calloc(INTER, sizeof(float)); act = calloc(INTER, sizeof(float)); down = calloc(fx.hidden, sizeof(float));
    shared_gate = calloc(INTER, sizeof(float)); shared_up = calloc(INTER, sizeof(float)); shared_act = calloc(INTER, sizeof(float)); shared = calloc(fx.hidden, sizeof(float));
    if (!input_ln || !qkv || !z || !a || !b || !conv || !conv_ref || !q || !k || !v || !beta || !g || !core || !state || !out_in || !out_proj || !resid || !post_attn_ln || !router_logits || !router_idx || !router_scores || !mlp || !final_out || !gate || !up || !act || !down || !shared_gate || !shared_up || !shared_act || !shared) return 1;

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
                if (src_t >= 0 && src_t < (int)fx.seq_len) sum += (double)fx.conv_w[(size_t)d * 4 + kk] * qkv[(size_t)src_t * qkv_dim + d];
            }
            conv[(size_t)t * qkv_dim + d] = siluf_local((float)sum);
            conv_ref[(size_t)t * qkv_dim + d] = siluf_local(fx.conv_raw[(size_t)d * (fx.seq_len + 3) + t]);
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
        for (h = 0; h < fx.num_v_heads; ++h) {
            float qnorm = 0.0f, knorm = 0.0f;
            float gexp = expf(g[t * fx.num_v_heads + h]);
            float beta_t = beta[t * fx.num_v_heads + h];
            size_t qbase = ((size_t)t * fx.num_v_heads + h) * fx.head_k_dim;
            size_t vbase = ((size_t)t * fx.num_v_heads + h) * fx.head_v_dim;
            size_t sbase = (size_t)h * fx.head_k_dim * fx.head_v_dim;
            for (hd = 0; hd < fx.head_k_dim; ++hd) { qnorm += q[qbase + hd] * q[qbase + hd]; knorm += k[qbase + hd] * k[qbase + hd]; }
            qnorm = 1.0f / sqrtf(qnorm + 1e-6f); knorm = 1.0f / sqrtf(knorm + 1e-6f);
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
            for (vd = 0; vd < fx.head_v_dim; ++vd) { double cv = core[((size_t)t * fx.num_v_heads + h) * fx.head_v_dim + vd]; var += cv * cv; }
            var /= (double)fx.head_v_dim;
            for (vd = 0; vd < fx.head_v_dim; ++vd) {
                float cv = core[((size_t)t * fx.num_v_heads + h) * fx.head_v_dim + vd];
                out_in[base + vd] = (float)(cv / sqrt(var + 1e-6)) * fx.ssm_norm_w[vd] * siluf_local(z[(size_t)t * fx.value_dim + (size_t)h * fx.head_v_dim + vd]);
            }
        }
        matvec(fx.w_out, out_in + (size_t)t * fx.value_dim, out_proj + (size_t)t * fx.hidden, fx.hidden, fx.value_dim);
    }

    for (d = 0; d < fx.hidden; ++d) resid[d] = fx.layer_input_last_ref[d] + out_proj[(size_t)(fx.seq_len - 1) * fx.hidden + d];
    {
        double var = 0.0;
        for (d = 0; d < fx.hidden; ++d) var += (double)resid[d] * resid[d];
        var /= (double)fx.hidden;
        for (d = 0; d < fx.hidden; ++d) post_attn_ln[d] = (float)(resid[d] / sqrt(var + 1e-6)) * fx.post_attn_norm_w[d];
    }
    matvec(fx.router_w, post_attn_ln, router_logits, ROUTER_COUNT, fx.hidden);
    topk_softmax256(router_logits, fx.topk, router_idx, router_scores);

    for (i = 0; i < fx.topk; ++i) {
        const float *gate_mat = fx.gate_sel + (size_t)i * INTER * fx.hidden;
        const float *up_mat = fx.up_sel + (size_t)i * INTER * fx.hidden;
        const float *down_mat = fx.down_sel + (size_t)i * fx.hidden * INTER;
        matvec(gate_mat, post_attn_ln, gate, INTER, fx.hidden);
        matvec(up_mat, post_attn_ln, up, INTER, fx.hidden);
        for (vd = 0; vd < INTER; ++vd) act[vd] = siluf_local(gate[vd]) * up[vd];
        matvec(down_mat, act, down, fx.hidden, INTER);
        for (vd = 0; vd < fx.hidden; ++vd) mlp[vd] += down[vd] * router_scores[i];
    }
    matvec(fx.gate_shexp, post_attn_ln, shared_gate, INTER, fx.hidden);
    matvec(fx.up_shexp, post_attn_ln, shared_up, INTER, fx.hidden);
    for (vd = 0; vd < INTER; ++vd) shared_act[vd] = siluf_local(shared_gate[vd]) * shared_up[vd];
    matvec(fx.down_shexp, shared_act, shared, fx.hidden, INTER);
    {
        float scale_in = 0.0f;
        for (vd = 0; vd < fx.hidden; ++vd) scale_in += post_attn_ln[vd] * fx.gate_inp_shexp[vd];
        {
            float s = sigmoidf_local(scale_in);
            for (vd = 0; vd < fx.hidden; ++vd) {
                mlp[vd] += shared[vd] * s;
                final_out[vd] = resid[vd] + mlp[vd];
            }
        }
    }

    printf("layer: %u\n", fx.layer);
    printf("input_ln_cosine: %.8f\n", vec_cosine(input_ln, fx.input_ln_seq, (size_t)fx.seq_len * fx.hidden));
    printf("qkv_cosine: %.8f\n", vec_cosine(qkv, fx.qkv_seq, (size_t)fx.seq_len * qkv_dim));
    printf("conv_post_cosine: %.8f\n", vec_cosine(conv, conv_ref, (size_t)fx.seq_len * qkv_dim));
    printf("core_cosine: %.8f\n", vec_cosine(core, fx.core_ref, (size_t)fx.seq_len * fx.num_v_heads * fx.head_v_dim));
    printf("mixer_cosine: %.8f\n", vec_cosine(out_proj + (size_t)(fx.seq_len - 1) * fx.hidden, fx.mixer_out_ref, fx.hidden));
    printf("residual_after_mixer_cosine: %.8f\n", vec_cosine(resid, fx.residual_after_mixer_ref, fx.hidden));
    printf("post_attn_ln_cosine: %.8f\n", vec_cosine(post_attn_ln, fx.post_attn_ln_ref, fx.hidden));
    printf("router_logits_cosine: %.8f\n", vec_cosine(router_logits, fx.router_logits_ref, ROUTER_COUNT));
    printf("router_logits_rmse: %.8f\n", vec_rmse(router_logits, fx.router_logits_ref, ROUTER_COUNT));
    {
        uint32_t same = 1;
        for (i = 0; i < fx.topk; ++i) if (router_idx[i] != (uint32_t)fx.router_indices_ref_f32[i]) same = 0;
        printf("router_topk_exact: %u\n", same);
    }
    printf("router_scores_cosine: %.8f\n", vec_cosine(router_scores, fx.router_scores_ref, fx.topk));
    printf("router_scores_rmse: %.8f\n", vec_rmse(router_scores, fx.router_scores_ref, fx.topk));
    {
        float scale_in = 0.0f;
        for (vd = 0; vd < fx.hidden; ++vd) scale_in += post_attn_ln[vd] * fx.gate_inp_shexp[vd];
        printf("shared_gate_pre_rmse: %.8f\n", vec_rmse(&scale_in, fx.shared_gate_pre_ref, 1));
    }
    printf("mlp_cosine: %.8f\n", vec_cosine(mlp, fx.mlp_out_ref, fx.hidden));
    printf("mlp_rmse: %.8f\n", vec_rmse(mlp, fx.mlp_out_ref, fx.hidden));
    printf("layer_output_cosine: %.8f\n", vec_cosine(final_out, fx.layer_output_ref, fx.hidden));
    printf("layer_output_rmse: %.8f\n", vec_rmse(final_out, fx.layer_output_ref, fx.hidden));

    fixture_free(&fx);
    free(input_ln); free(qkv); free(z); free(a); free(b); free(conv); free(conv_ref); free(q); free(k); free(v); free(beta); free(g); free(core); free(state); free(out_in); free(out_proj);
    free(resid); free(post_attn_ln); free(router_logits); free(router_idx); free(router_scores); free(mlp); free(final_out); free(gate); free(up); free(act); free(down); free(shared_gate); free(shared_up); free(shared_act); free(shared);
    return 0;
}
