#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "Q36DWF02"
#define MAGIC_LEN 8
#define ROUTER_COUNT 256
#define INTER 512

typedef struct dwf_fixture {
    uint32_t layer, seq_len, hidden, num_v_heads, num_k_heads, head_k_dim, head_v_dim, key_dim, value_dim, topk, union_experts;
    float *layer_input_seq, *input_ln_seq, *qkv_seq, *z_seq, *a_seq, *b_seq, *conv_raw;
    float *q_ref, *k_ref, *v_ref, *beta_ref, *g_ref, *core_ref, *out_in_seq, *out_proj_out_seq;
    float *mixer_out_ref, *layer_input_last_ref, *residual_after_mixer_ref, *post_attn_ln_ref, *mlp_out_ref, *layer_output_ref;
    float *router_logits_ref, *router_indices_ref_f32, *router_scores_ref, *shared_gate_pre_ref;
    float *layer_input_seq_ref, *residual_after_mixer_seq_ref, *post_attn_ln_seq_ref, *mlp_out_seq_ref, *layer_output_seq_ref;
    float *router_logits_seq_ref, *router_indices_seq_ref_f32, *router_scores_seq_ref, *shared_gate_pre_seq_ref;
    float *attn_norm_w, *post_attn_norm_w, *w_qkv, *w_z, *w_a, *w_b, *conv_w, *A_log, *dt_bias, *ssm_norm_w, *w_out;
    float *router_w;
    float *union_expert_ids_f32, *router_union_pos_seq_f32;
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
    free(fx->layer_input_seq_ref); free(fx->residual_after_mixer_seq_ref); free(fx->post_attn_ln_seq_ref); free(fx->mlp_out_seq_ref); free(fx->layer_output_seq_ref);
    free(fx->router_logits_seq_ref); free(fx->router_indices_seq_ref_f32); free(fx->router_scores_seq_ref); free(fx->shared_gate_pre_seq_ref);
    free(fx->attn_norm_w); free(fx->post_attn_norm_w); free(fx->w_qkv); free(fx->w_z); free(fx->w_a); free(fx->w_b); free(fx->conv_w); free(fx->A_log); free(fx->dt_bias); free(fx->ssm_norm_w); free(fx->w_out);
    free(fx->router_w); free(fx->union_expert_ids_f32); free(fx->router_union_pos_seq_f32);
    free(fx->gate_sel); free(fx->up_sel); free(fx->down_sel); free(fx->gate_shexp); free(fx->up_shexp); free(fx->down_shexp); free(fx->gate_inp_shexp);
    memset(fx, 0, sizeof(*fx));
}
static int fixture_load(const char *path, dwf_fixture *fx) {
    FILE *fp = fopen(path, "rb");
    char magic[MAGIC_LEN];
    memset(fx, 0, sizeof(*fx));
    if (!fp) return 0;
    if (!read_exact(fp, magic, MAGIC_LEN) || memcmp(magic, MAGIC, MAGIC_LEN) != 0) { fclose(fp); return 0; }
    if (!read_exact(fp, &fx->layer, sizeof(uint32_t)) || !read_exact(fp, &fx->seq_len, sizeof(uint32_t)) || !read_exact(fp, &fx->hidden, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->num_v_heads, sizeof(uint32_t)) || !read_exact(fp, &fx->num_k_heads, sizeof(uint32_t)) || !read_exact(fp, &fx->head_k_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->head_v_dim, sizeof(uint32_t)) || !read_exact(fp, &fx->key_dim, sizeof(uint32_t)) || !read_exact(fp, &fx->value_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->topk, sizeof(uint32_t)) || !read_exact(fp, &fx->union_experts, sizeof(uint32_t))) { fclose(fp); return 0; }
#define R(name,count) if (!alloc_read_f32(fp, &fx->name, (count))) { fclose(fp); fixture_free(fx); return 0; }
    R(layer_input_seq, (size_t)fx->seq_len * fx->hidden); R(input_ln_seq, (size_t)fx->seq_len * fx->hidden); R(qkv_seq, (size_t)fx->seq_len * (fx->key_dim * 2 + fx->value_dim));
    R(z_seq, (size_t)fx->seq_len * fx->value_dim); R(a_seq, (size_t)fx->seq_len * fx->num_v_heads); R(b_seq, (size_t)fx->seq_len * fx->num_v_heads); R(conv_raw, (size_t)(fx->key_dim * 2 + fx->value_dim) * (fx->seq_len + 3));
    R(q_ref, (size_t)fx->seq_len * fx->num_v_heads * fx->head_k_dim); R(k_ref, (size_t)fx->seq_len * fx->num_v_heads * fx->head_k_dim); R(v_ref, (size_t)fx->seq_len * fx->num_v_heads * fx->head_v_dim);
    R(beta_ref, (size_t)fx->seq_len * fx->num_v_heads); R(g_ref, (size_t)fx->seq_len * fx->num_v_heads); R(core_ref, (size_t)fx->seq_len * fx->num_v_heads * fx->head_v_dim);
    R(out_in_seq, (size_t)fx->seq_len * fx->value_dim); R(out_proj_out_seq, (size_t)fx->seq_len * fx->hidden);
    R(mixer_out_ref, fx->hidden); R(layer_input_last_ref, fx->hidden); R(residual_after_mixer_ref, fx->hidden); R(post_attn_ln_ref, fx->hidden); R(mlp_out_ref, fx->hidden); R(layer_output_ref, fx->hidden);
    R(router_logits_ref, ROUTER_COUNT); R(router_indices_ref_f32, fx->topk); R(router_scores_ref, fx->topk); R(shared_gate_pre_ref, 1);
    R(layer_input_seq_ref, (size_t)fx->seq_len * fx->hidden); R(residual_after_mixer_seq_ref, (size_t)fx->seq_len * fx->hidden); R(post_attn_ln_seq_ref, (size_t)fx->seq_len * fx->hidden);
    R(mlp_out_seq_ref, (size_t)fx->seq_len * fx->hidden); R(layer_output_seq_ref, (size_t)fx->seq_len * fx->hidden);
    R(router_logits_seq_ref, (size_t)fx->seq_len * ROUTER_COUNT); R(router_indices_seq_ref_f32, (size_t)fx->seq_len * fx->topk); R(router_scores_seq_ref, (size_t)fx->seq_len * fx->topk); R(shared_gate_pre_seq_ref, fx->seq_len);
    R(attn_norm_w, fx->hidden); R(post_attn_norm_w, fx->hidden); R(w_qkv, (size_t)(fx->key_dim * 2 + fx->value_dim) * fx->hidden); R(w_z, (size_t)fx->value_dim * fx->hidden);
    R(w_a, (size_t)fx->num_v_heads * fx->hidden); R(w_b, (size_t)fx->num_v_heads * fx->hidden); R(conv_w, (size_t)(fx->key_dim * 2 + fx->value_dim) * 4); R(A_log, fx->num_v_heads); R(dt_bias, fx->num_v_heads);
    R(ssm_norm_w, fx->head_v_dim); R(w_out, (size_t)fx->hidden * fx->value_dim); R(router_w, (size_t)ROUTER_COUNT * fx->hidden);
    R(union_expert_ids_f32, fx->union_experts); R(router_union_pos_seq_f32, (size_t)fx->seq_len * fx->topk);
    R(gate_sel, (size_t)fx->union_experts * INTER * fx->hidden); R(up_sel, (size_t)fx->union_experts * INTER * fx->hidden); R(down_sel, (size_t)fx->union_experts * fx->hidden * INTER);
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
    size_t i;
    double acc = 0.0;
    for (i = 0; i < n; ++i) {
        double d = (double)a[i] - b[i];
        acc += d * d;
    }
    return sqrt(acc / (double)n);
}
static double vec_cosine(const float *a, const float *b, size_t n) {
    size_t i;
    double dot = 0.0, an = 0.0, bn = 0.0;
    for (i = 0; i < n; ++i) {
        dot += (double)a[i] * b[i];
        an += (double)a[i] * a[i];
        bn += (double)b[i] * b[i];
    }
    if (an == 0.0 || bn == 0.0) return NAN;
    return dot / sqrt(an * bn);
}

typedef struct layer_debug {
    float *input_ln_seq;
    float *qkv_seq;
    float *z_seq;
    float *a_seq;
    float *b_seq;
    float *out_proj_seq;
    float *resid_seq;
    float *post_attn_ln_seq;
    float *mlp_seq;
} layer_debug;

static int run_layer(const dwf_fixture *fx, const float *input_override_seq, float *out_final_seq, layer_debug *dbg) {
    uint32_t qkv_dim = fx->key_dim * 2 + fx->value_dim;
    uint32_t rep = fx->num_v_heads / fx->num_k_heads;
    size_t seq_hidden = (size_t)fx->seq_len * fx->hidden;
    float *input_ln = (float *)calloc(seq_hidden, sizeof(float));
    float *qkv = (float *)calloc((size_t)fx->seq_len * qkv_dim, sizeof(float));
    float *z = (float *)calloc((size_t)fx->seq_len * fx->value_dim, sizeof(float));
    float *a = (float *)calloc((size_t)fx->seq_len * fx->num_v_heads, sizeof(float));
    float *b = (float *)calloc((size_t)fx->seq_len * fx->num_v_heads, sizeof(float));
    float *conv = (float *)calloc((size_t)fx->seq_len * qkv_dim, sizeof(float));
    float *q = (float *)calloc((size_t)fx->seq_len * fx->num_v_heads * fx->head_k_dim, sizeof(float));
    float *k = (float *)calloc((size_t)fx->seq_len * fx->num_v_heads * fx->head_k_dim, sizeof(float));
    float *v = (float *)calloc((size_t)fx->seq_len * fx->num_v_heads * fx->head_v_dim, sizeof(float));
    float *beta = (float *)calloc((size_t)fx->seq_len * fx->num_v_heads, sizeof(float));
    float *g = (float *)calloc((size_t)fx->seq_len * fx->num_v_heads, sizeof(float));
    float *core = (float *)calloc((size_t)fx->seq_len * fx->num_v_heads * fx->head_v_dim, sizeof(float));
    float *state = (float *)calloc((size_t)fx->num_v_heads * fx->head_k_dim * fx->head_v_dim, sizeof(float));
    float *out_in = (float *)calloc((size_t)fx->seq_len * fx->value_dim, sizeof(float));
    float *out_proj = (float *)calloc(seq_hidden, sizeof(float));
    float *resid_seq = (float *)calloc(seq_hidden, sizeof(float));
    float *post_attn_ln_seq = (float *)calloc(seq_hidden, sizeof(float));
    float *mlp_seq = (float *)calloc(seq_hidden, sizeof(float));
    float *gate = (float *)calloc(INTER, sizeof(float));
    float *up = (float *)calloc(INTER, sizeof(float));
    float *act = (float *)calloc(INTER, sizeof(float));
    float *down = (float *)calloc(fx->hidden, sizeof(float));
    float *shared_gate = (float *)calloc(INTER, sizeof(float));
    float *shared_up = (float *)calloc(INTER, sizeof(float));
    float *shared_act = (float *)calloc(INTER, sizeof(float));
    float *shared = (float *)calloc(fx->hidden, sizeof(float));
    const float *layer_input_seq = input_override_seq ? input_override_seq : fx->layer_input_seq;
    uint32_t t, h, d, hd, vd, i;

    if (!input_ln || !qkv || !z || !a || !b || !conv || !q || !k || !v || !beta || !g || !core || !state || !out_in ||
        !out_proj || !resid_seq || !post_attn_ln_seq || !mlp_seq || !gate || !up || !act || !down || !shared_gate || !shared_up ||
        !shared_act || !shared) goto oom;

    for (t = 0; t < fx->seq_len; ++t) {
        const float *in = layer_input_seq + (size_t)t * fx->hidden;
        float *out = input_ln + (size_t)t * fx->hidden;
        double var = 0.0;
        for (d = 0; d < fx->hidden; ++d) var += (double)in[d] * in[d];
        var /= (double)fx->hidden;
        for (d = 0; d < fx->hidden; ++d) out[d] = (float)(in[d] / sqrt(var + 1e-6)) * fx->attn_norm_w[d];
        matvec(fx->w_qkv, out, qkv + (size_t)t * qkv_dim, qkv_dim, fx->hidden);
        matvec(fx->w_z, out, z + (size_t)t * fx->value_dim, fx->value_dim, fx->hidden);
        matvec(fx->w_a, out, a + (size_t)t * fx->num_v_heads, fx->num_v_heads, fx->hidden);
        matvec(fx->w_b, out, b + (size_t)t * fx->num_v_heads, fx->num_v_heads, fx->hidden);
    }
    for (t = 0; t < fx->seq_len; ++t) {
        for (d = 0; d < qkv_dim; ++d) {
            double sum = 0.0;
            uint32_t kk;
            for (kk = 0; kk < 4; ++kk) {
                int src_t = (int)t - 3 + (int)kk;
                if (src_t >= 0 && src_t < (int)fx->seq_len) sum += (double)fx->conv_w[(size_t)d * 4 + kk] * qkv[(size_t)src_t * qkv_dim + d];
            }
            conv[(size_t)t * qkv_dim + d] = siluf_local((float)sum);
        }
    }
    for (t = 0; t < fx->seq_len; ++t) {
        const float *cq = conv + (size_t)t * qkv_dim;
        const float *ck = cq + fx->key_dim;
        const float *cv = ck + fx->key_dim;
        for (h = 0; h < fx->num_k_heads; ++h) {
            for (d = 0; d < fx->head_k_dim; ++d) {
                float qv = cq[h * fx->head_k_dim + d];
                float kv = ck[h * fx->head_k_dim + d];
                uint32_t vh;
                for (vh = 0; vh < rep; ++vh) {
                    uint32_t dst_h = h * rep + vh;
                    q[((size_t)t * fx->num_v_heads + dst_h) * fx->head_k_dim + d] = qv;
                    k[((size_t)t * fx->num_v_heads + dst_h) * fx->head_k_dim + d] = kv;
                }
            }
        }
        for (h = 0; h < fx->num_v_heads; ++h) {
            memcpy(v + ((size_t)t * fx->num_v_heads + h) * fx->head_v_dim, cv + (size_t)h * fx->head_v_dim, fx->head_v_dim * sizeof(float));
            beta[t * fx->num_v_heads + h] = sigmoidf_local(b[(size_t)t * fx->num_v_heads + h]);
            /* GGUF stores blk.*.ssm_a as the pre-expanded decay coefficient -exp(A_log). */
            g[t * fx->num_v_heads + h] = fx->A_log[h] * softplusf_local(a[(size_t)t * fx->num_v_heads + h] + fx->dt_bias[h]);
        }
    }
    for (t = 0; t < fx->seq_len; ++t) {
        for (h = 0; h < fx->num_v_heads; ++h) {
            float qnorm = 0.0f, knorm = 0.0f, gexp = expf(g[t * fx->num_v_heads + h]), beta_t = beta[t * fx->num_v_heads + h];
            size_t qbase = ((size_t)t * fx->num_v_heads + h) * fx->head_k_dim;
            size_t vbase = ((size_t)t * fx->num_v_heads + h) * fx->head_v_dim;
            size_t sbase = (size_t)h * fx->head_k_dim * fx->head_v_dim;
            for (hd = 0; hd < fx->head_k_dim; ++hd) {
                qnorm += q[qbase + hd] * q[qbase + hd];
                knorm += k[qbase + hd] * k[qbase + hd];
            }
            qnorm = 1.0f / sqrtf(qnorm + 1e-6f);
            knorm = 1.0f / sqrtf(knorm + 1e-6f);
            for (hd = 0; hd < fx->head_k_dim * fx->head_v_dim; ++hd) state[sbase + hd] *= gexp;
            {
                float delta[128];
                for (vd = 0; vd < fx->head_v_dim; ++vd) {
                    float kv_mem = 0.0f;
                    for (hd = 0; hd < fx->head_k_dim; ++hd) kv_mem += state[sbase + (size_t)hd * fx->head_v_dim + vd] * (k[qbase + hd] * knorm);
                    delta[vd] = (v[vbase + vd] - kv_mem) * beta_t;
                }
                for (hd = 0; hd < fx->head_k_dim; ++hd) {
                    float kval = k[qbase + hd] * knorm;
                    for (vd = 0; vd < fx->head_v_dim; ++vd) state[sbase + (size_t)hd * fx->head_v_dim + vd] += kval * delta[vd];
                }
                for (vd = 0; vd < fx->head_v_dim; ++vd) {
                    float outv = 0.0f;
                    for (hd = 0; hd < fx->head_k_dim; ++hd) outv += state[sbase + (size_t)hd * fx->head_v_dim + vd] * (q[qbase + hd] * qnorm / sqrtf((float)fx->head_k_dim));
                    core[vbase + vd] = outv;
                }
            }
        }
    }
    for (t = 0; t < fx->seq_len; ++t) {
        for (h = 0; h < fx->num_v_heads; ++h) {
            size_t base = (size_t)t * fx->value_dim + (size_t)h * fx->head_v_dim;
            double var = 0.0;
            for (vd = 0; vd < fx->head_v_dim; ++vd) {
                double cv = core[((size_t)t * fx->num_v_heads + h) * fx->head_v_dim + vd];
                var += cv * cv;
            }
            var /= (double)fx->head_v_dim;
            for (vd = 0; vd < fx->head_v_dim; ++vd) {
                float cv = core[((size_t)t * fx->num_v_heads + h) * fx->head_v_dim + vd];
                out_in[base + vd] = (float)(cv / sqrt(var + 1e-6)) * fx->ssm_norm_w[vd] * siluf_local(z[(size_t)t * fx->value_dim + (size_t)h * fx->head_v_dim + vd]);
            }
        }
        matvec(fx->w_out, out_in + (size_t)t * fx->value_dim, out_proj + (size_t)t * fx->hidden, fx->hidden, fx->value_dim);
    }
    for (t = 0; t < fx->seq_len; ++t) {
        const float *in_row = layer_input_seq + (size_t)t * fx->hidden;
        float *resid_row = resid_seq + (size_t)t * fx->hidden;
        float *post_row = post_attn_ln_seq + (size_t)t * fx->hidden;
        float *mlp_row = mlp_seq + (size_t)t * fx->hidden;
        float *out_row = out_final_seq + (size_t)t * fx->hidden;
        double var = 0.0;
        float scale_in = 0.0f;
        for (d = 0; d < fx->hidden; ++d) {
            resid_row[d] = in_row[d] + out_proj[(size_t)t * fx->hidden + d];
            var += (double)resid_row[d] * resid_row[d];
        }
        var /= (double)fx->hidden;
        for (d = 0; d < fx->hidden; ++d) {
            post_row[d] = (float)(resid_row[d] / sqrt(var + 1e-6)) * fx->post_attn_norm_w[d];
            scale_in += post_row[d] * fx->gate_inp_shexp[d];
        }
        for (i = 0; i < fx->topk; ++i) {
            uint32_t union_pos = (uint32_t)fx->router_union_pos_seq_f32[(size_t)t * fx->topk + i];
            float router_score = fx->router_scores_seq_ref[(size_t)t * fx->topk + i];
            const float *gate_mat = fx->gate_sel + (size_t)union_pos * INTER * fx->hidden;
            const float *up_mat = fx->up_sel + (size_t)union_pos * INTER * fx->hidden;
            const float *down_mat = fx->down_sel + (size_t)union_pos * fx->hidden * INTER;
            matvec(gate_mat, post_row, gate, INTER, fx->hidden);
            matvec(up_mat, post_row, up, INTER, fx->hidden);
            for (vd = 0; vd < INTER; ++vd) act[vd] = siluf_local(gate[vd]) * up[vd];
            matvec(down_mat, act, down, fx->hidden, INTER);
            for (d = 0; d < fx->hidden; ++d) mlp_row[d] += down[d] * router_score;
        }
        matvec(fx->gate_shexp, post_row, shared_gate, INTER, fx->hidden);
        matvec(fx->up_shexp, post_row, shared_up, INTER, fx->hidden);
        for (vd = 0; vd < INTER; ++vd) shared_act[vd] = siluf_local(shared_gate[vd]) * shared_up[vd];
        matvec(fx->down_shexp, shared_act, shared, fx->hidden, INTER);
        {
            float s = sigmoidf_local(scale_in);
            for (d = 0; d < fx->hidden; ++d) out_row[d] = resid_row[d] + mlp_row[d] + shared[d] * s;
        }
    }

    if (dbg) {
        dbg->input_ln_seq = input_ln;
        dbg->qkv_seq = qkv;
        dbg->z_seq = z;
        dbg->a_seq = a;
        dbg->b_seq = b;
        dbg->out_proj_seq = out_proj;
        dbg->resid_seq = resid_seq;
        dbg->post_attn_ln_seq = post_attn_ln_seq;
        dbg->mlp_seq = mlp_seq;
    } else {
        free(input_ln); free(qkv); free(z); free(a); free(b); free(out_proj); free(resid_seq); free(post_attn_ln_seq); free(mlp_seq);
    }
    free(conv); free(q); free(k); free(v); free(beta); free(g); free(core); free(state); free(out_in); free(gate); free(up); free(act); free(down); free(shared_gate); free(shared_up); free(shared_act); free(shared);
    return 1;
oom:
    free(input_ln); free(qkv); free(z); free(a); free(b); free(conv); free(q); free(k); free(v); free(beta); free(g); free(core); free(state); free(out_in); free(out_proj);
    free(resid_seq); free(post_attn_ln_seq); free(mlp_seq); free(gate); free(up); free(act); free(down); free(shared_gate); free(shared_up); free(shared_act); free(shared);
    return 0;
}

int main(int argc, char **argv) {
    int dump_last = 0, dump_seq = 0;
    const char *dump_last_path = NULL, *dump_seq_path = NULL;
    int n_fx = 0, i, argi;
    dwf_fixture *fxs;
    float *state_seq = NULL;

    if (argc < 2) {
        fprintf(stderr, "usage: %s FIX0.bin [FIX1.bin ...] [--dump-last out.f32] [--dump-seq out.f32]\n", argv[0]);
        return 1;
    }
    for (argi = 1; argi < argc; ++argi) {
        if (strcmp(argv[argi], "--dump-last") == 0 || strcmp(argv[argi], "--dump-seq") == 0) {
            if (argi + 1 >= argc) return 1;
            if (strcmp(argv[argi], "--dump-last") == 0) {
                dump_last = 1;
                dump_last_path = argv[argi + 1];
            } else {
                dump_seq = 1;
                dump_seq_path = argv[argi + 1];
            }
            ++argi;
        } else {
            ++n_fx;
        }
    }
    if (n_fx < 1) {
        fprintf(stderr, "no fixtures provided\n");
        return 1;
    }
    fxs = (dwf_fixture *)calloc((size_t)n_fx, sizeof(dwf_fixture));
    if (!fxs) return 1;
    argi = 1;
    for (i = 0; i < n_fx; ++i) {
        while (argi < argc && (strcmp(argv[argi], "--dump-last") == 0 || strcmp(argv[argi], "--dump-seq") == 0)) argi += 2;
        if (argi >= argc || !fixture_load(argv[argi], &fxs[i])) {
            fprintf(stderr, "failed to load fixture\n");
            return 1;
        }
        ++argi;
    }
    for (i = 0; i < n_fx; ++i) {
        float *next_seq = (float *)calloc((size_t)fxs[i].seq_len * fxs[i].hidden, sizeof(float));
        const float *last_row;
        layer_debug dbg = {0};
        if (!next_seq) return 1;
        if (!run_layer(&fxs[i], i == 0 ? NULL : state_seq, next_seq, &dbg)) return 1;
        if (i == 0) {
            printf("layer0_output_seq_rmse: %.8f\n", vec_rmse(next_seq, fxs[i].layer_output_seq_ref, (size_t)fxs[i].seq_len * fxs[i].hidden));
            printf("layer0_output_seq_cosine: %.8f\n", vec_cosine(next_seq, fxs[i].layer_output_seq_ref, (size_t)fxs[i].seq_len * fxs[i].hidden));
        } else {
            printf("layer%d_input_handoff_seq_rmse: %.8f\n", i, vec_rmse(state_seq, fxs[i].layer_input_seq_ref, (size_t)fxs[i].seq_len * fxs[i].hidden));
            printf("layer%d_input_handoff_seq_cosine: %.8f\n", i, vec_cosine(state_seq, fxs[i].layer_input_seq_ref, (size_t)fxs[i].seq_len * fxs[i].hidden));
        }
        last_row = next_seq + (size_t)(fxs[i].seq_len - 1) * fxs[i].hidden;
        printf("layer%d_output_rmse: %.8f\n", i, vec_rmse(last_row, fxs[i].layer_output_ref, fxs[i].hidden));
        printf("layer%d_output_cosine: %.8f\n", i, vec_cosine(last_row, fxs[i].layer_output_ref, fxs[i].hidden));
        printf("layer%d_output_seq_rmse: %.8f\n", i, vec_rmse(next_seq, fxs[i].layer_output_seq_ref, (size_t)fxs[i].seq_len * fxs[i].hidden));
        printf("layer%d_output_seq_cosine: %.8f\n", i, vec_cosine(next_seq, fxs[i].layer_output_seq_ref, (size_t)fxs[i].seq_len * fxs[i].hidden));
        printf("layer%d_input_ln_seq_rmse: %.8f\n", i, vec_rmse(dbg.input_ln_seq, fxs[i].input_ln_seq, (size_t)fxs[i].seq_len * fxs[i].hidden));
        printf("layer%d_input_ln_seq_cosine: %.8f\n", i, vec_cosine(dbg.input_ln_seq, fxs[i].input_ln_seq, (size_t)fxs[i].seq_len * fxs[i].hidden));
        printf("layer%d_qkv_seq_rmse: %.8f\n", i, vec_rmse(dbg.qkv_seq, fxs[i].qkv_seq, (size_t)fxs[i].seq_len * (fxs[i].key_dim * 2 + fxs[i].value_dim)));
        printf("layer%d_qkv_seq_cosine: %.8f\n", i, vec_cosine(dbg.qkv_seq, fxs[i].qkv_seq, (size_t)fxs[i].seq_len * (fxs[i].key_dim * 2 + fxs[i].value_dim)));
        printf("layer%d_z_seq_rmse: %.8f\n", i, vec_rmse(dbg.z_seq, fxs[i].z_seq, (size_t)fxs[i].seq_len * fxs[i].value_dim));
        printf("layer%d_z_seq_cosine: %.8f\n", i, vec_cosine(dbg.z_seq, fxs[i].z_seq, (size_t)fxs[i].seq_len * fxs[i].value_dim));
        printf("layer%d_residual_after_mixer_seq_rmse: %.8f\n", i, vec_rmse(dbg.resid_seq, fxs[i].residual_after_mixer_seq_ref, (size_t)fxs[i].seq_len * fxs[i].hidden));
        printf("layer%d_residual_after_mixer_seq_cosine: %.8f\n", i, vec_cosine(dbg.resid_seq, fxs[i].residual_after_mixer_seq_ref, (size_t)fxs[i].seq_len * fxs[i].hidden));
        printf("layer%d_post_attn_ln_seq_rmse: %.8f\n", i, vec_rmse(dbg.post_attn_ln_seq, fxs[i].post_attn_ln_seq_ref, (size_t)fxs[i].seq_len * fxs[i].hidden));
        printf("layer%d_post_attn_ln_seq_cosine: %.8f\n", i, vec_cosine(dbg.post_attn_ln_seq, fxs[i].post_attn_ln_seq_ref, (size_t)fxs[i].seq_len * fxs[i].hidden));
        printf("layer%d_mlp_seq_rmse: %.8f\n", i, vec_rmse(dbg.mlp_seq, fxs[i].mlp_out_seq_ref, (size_t)fxs[i].seq_len * fxs[i].hidden));
        printf("layer%d_mlp_seq_cosine: %.8f\n", i, vec_cosine(dbg.mlp_seq, fxs[i].mlp_out_seq_ref, (size_t)fxs[i].seq_len * fxs[i].hidden));
        free(dbg.input_ln_seq); free(dbg.qkv_seq); free(dbg.z_seq); free(dbg.a_seq); free(dbg.b_seq); free(dbg.out_proj_seq); free(dbg.resid_seq); free(dbg.post_attn_ln_seq); free(dbg.mlp_seq);
        free(state_seq);
        state_seq = next_seq;
    }
    if (dump_last) {
        FILE *fp = fopen(dump_last_path, "wb");
        const dwf_fixture *last_fx = &fxs[n_fx - 1];
        const float *last_row = state_seq + (size_t)(last_fx->seq_len - 1) * last_fx->hidden;
        if (!fp) return 1;
        if (fwrite(last_row, sizeof(float), last_fx->hidden, fp) != last_fx->hidden) return 1;
        fclose(fp);
        printf("wrote_last_hidden: %s\n", dump_last_path);
    }
    if (dump_seq) {
        FILE *fp = fopen(dump_seq_path, "wb");
        const dwf_fixture *last_fx = &fxs[n_fx - 1];
        size_t count = (size_t)last_fx->seq_len * last_fx->hidden;
        if (!fp) return 1;
        if (fwrite(state_seq, sizeof(float), count, fp) != count) return 1;
        fclose(fp);
        printf("wrote_seq_hidden: %s\n", dump_seq_path);
    }
    for (i = 0; i < n_fx; ++i) fixture_free(&fxs[i]);
    free(fxs);
    free(state_seq);
    return 0;
}
