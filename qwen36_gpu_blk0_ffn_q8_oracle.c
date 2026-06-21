#include "ds4_gpu.h"
#include "qwen36_35a3b_q8.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAGIC "Q36DWF02"
#define MAGIC_LEN 8
#define PREFIX_MAGIC "Q36PFX01"
#define PREFIX_MAGIC_LEN 8
#define ROUTER_COUNT 256
#define INTER 512
#define QWEN_RMS_EPS 1e-6f
#define QWEN_SWIGLU_CLAMP 80.0f

typedef struct prefix_fixture {
    uint32_t seq_len;
    uint32_t hidden;
    uint32_t *token_ids;
    float *input_seq_ref;
} prefix_fixture;

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

typedef struct mapped_file {
    int fd;
    void *map;
    uint64_t size;
} mapped_file;

/* Exported by the DS4 ROCm/CUDA substrate, but not yet declared in ds4_gpu.h. */
extern int ds4_gpu_embed_tokens_hc_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens_t,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc);

static int read_exact(FILE *fp, void *buf, size_t n) { return fread(buf, 1, n, fp) == n; }

static int alloc_read_f32(FILE *fp, float **out, size_t count) {
    float *p = (float *)malloc(count * sizeof(float));
    if (!p) return 0;
    if (!read_exact(fp, p, count * sizeof(float))) { free(p); return 0; }
    *out = p;
    return 1;
}

static int alloc_read_u32(FILE *fp, uint32_t **out, size_t count) {
    uint32_t *p = (uint32_t *)malloc(count * sizeof(uint32_t));
    if (!p) return 0;
    if (!read_exact(fp, p, count * sizeof(uint32_t))) { free(p); return 0; }
    *out = p;
    return 1;
}

static void prefix_free(prefix_fixture *fx) {
    if (!fx) return;
    free(fx->token_ids);
    free(fx->input_seq_ref);
    memset(fx, 0, sizeof(*fx));
}

static int prefix_load(const char *path, prefix_fixture *fx) {
    FILE *fp = fopen(path, "rb");
    char magic[PREFIX_MAGIC_LEN];
    memset(fx, 0, sizeof(*fx));
    if (!fp) return 0;
    if (!read_exact(fp, magic, PREFIX_MAGIC_LEN) || memcmp(magic, PREFIX_MAGIC, PREFIX_MAGIC_LEN) != 0) { fclose(fp); return 0; }
    if (!read_exact(fp, &fx->seq_len, sizeof(uint32_t)) || !read_exact(fp, &fx->hidden, sizeof(uint32_t))) { fclose(fp); return 0; }
    if (!alloc_read_u32(fp, &fx->token_ids, fx->seq_len) ||
        !alloc_read_f32(fp, &fx->input_seq_ref, (size_t)fx->seq_len * fx->hidden)) {
        fclose(fp); prefix_free(fx); return 0;
    }
    fclose(fp);
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

static int mapped_file_open(mapped_file *mf, const char *path) {
    struct stat st;
    memset(mf, 0, sizeof(*mf));
    mf->fd = open(path, O_RDONLY);
    if (mf->fd == -1) return 0;
    if (fstat(mf->fd, &st) == -1) { close(mf->fd); mf->fd = -1; return 0; }
    mf->map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, mf->fd, 0);
    if (mf->map == MAP_FAILED) { close(mf->fd); mf->fd = -1; mf->map = NULL; return 0; }
    mf->size = (uint64_t)st.st_size;
    return 1;
}

static void mapped_file_close(mapped_file *mf) {
    if (!mf) return;
    if (mf->map && mf->map != MAP_FAILED && mf->size) munmap(mf->map, (size_t)mf->size);
    if (mf->fd >= 0) close(mf->fd);
    memset(mf, 0, sizeof(*mf));
    mf->fd = -1;
}

static float sigmoidf_local(float x) { return 1.0f / (1.0f + expf(-x)); }

static float cosine(const float *a, const float *b, size_t n) {
    double dot = 0.0, an = 0.0, bn = 0.0;
    for (size_t i = 0; i < n; ++i) {
        dot += (double)a[i] * b[i];
        an += (double)a[i] * a[i];
        bn += (double)b[i] * b[i];
    }
    if (an == 0.0 || bn == 0.0) return 0.0f;
    return (float)(dot / sqrt(an * bn));
}

static float rmse(const float *a, const float *b, size_t n) {
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = (double)a[i] - b[i];
        s += d * d;
    }
    return (float)sqrt(s / (double)n);
}

static uint64_t q8_row_bytes(uint32_t cols) {
    return ((uint64_t)cols / 32u) * 34u;
}

static void topk_softmax256(const float *logits, uint32_t k, int32_t *idx, float *scores) {
    uint32_t i, j, m;
    for (i = 0; i < k; ++i) idx[i] = (int32_t)i;
    for (i = k; i < ROUTER_COUNT; ++i) {
        m = 0;
        for (j = 1; j < k; ++j) {
            if (logits[idx[j]] < logits[idx[m]]) m = j;
        }
        if (logits[i] > logits[idx[m]]) idx[m] = (int32_t)i;
    }
    for (i = 0; i < k; ++i) {
        for (j = i + 1; j < k; ++j) {
            if (logits[idx[j]] > logits[idx[i]]) {
                const int32_t tmp = idx[i];
                idx[i] = idx[j];
                idx[j] = tmp;
            }
        }
    }
    {
        const float maxv = logits[idx[0]];
        double sum = 0.0;
        for (i = 0; i < k; ++i) sum += exp((double)logits[idx[i]] - maxv);
        for (i = 0; i < k; ++i) scores[i] = (float)(exp((double)logits[idx[i]] - maxv) / sum);
    }
}

static int run_gpu_blk0_ffn(
        const mapped_file *mf,
        const qwen36_35a3b_q8_layer *layer,
        const dwf_fixture *fx,
        const uint32_t *token_ids,
        float *out_seq,
        int32_t *router_selected_cpu,
        float *router_weights_cpu) {
    const uint32_t n_tokens = fx->seq_len;
    const uint32_t hidden = fx->hidden;
    const uint32_t topk = fx->topk;
    const uint32_t shared_dim = INTER;
    const uint64_t seq_hidden_bytes = (uint64_t)n_tokens * hidden * sizeof(float);
    const uint64_t router_logits_bytes = (uint64_t)n_tokens * ROUTER_COUNT * sizeof(float);
    const uint64_t shared_bytes = (uint64_t)n_tokens * shared_dim * sizeof(float);
    const uint64_t gate_row_bytes = q8_row_bytes(hidden);
    const uint64_t gate_expert_bytes = gate_row_bytes * INTER;
    const uint64_t down_row_bytes = q8_row_bytes(INTER);
    const uint64_t down_expert_bytes = down_row_bytes * hidden;
    ds4_gpu_tensor *residual = NULL, *post = NULL, *router_logits = NULL;
    ds4_gpu_tensor *shared_gate = NULL, *shared_up = NULL, *shared_mid = NULL, *shared_out = NULL;
    ds4_gpu_tensor *expert_gate = NULL, *expert_up = NULL, *expert_mid = NULL, *expert_down = NULL;
    float *router_logits_cpu = NULL, *shared_out_cpu = NULL, *expert_down_cpu = NULL, *routed_out_cpu = NULL;
    int32_t union_ids[ROUTER_COUNT];
    int32_t union_pos[ROUTER_COUNT];
    uint32_t n_union = 0;
    int ok = 0;

    residual = ds4_gpu_tensor_alloc(seq_hidden_bytes);
    post = ds4_gpu_tensor_alloc(seq_hidden_bytes);
    router_logits = ds4_gpu_tensor_alloc(router_logits_bytes);
    shared_gate = ds4_gpu_tensor_alloc(shared_bytes);
    shared_up = ds4_gpu_tensor_alloc(shared_bytes);
    shared_mid = ds4_gpu_tensor_alloc(shared_bytes);
    shared_out = ds4_gpu_tensor_alloc(seq_hidden_bytes);
    expert_gate = ds4_gpu_tensor_alloc(shared_bytes);
    expert_up = ds4_gpu_tensor_alloc(shared_bytes);
    expert_mid = ds4_gpu_tensor_alloc(shared_bytes);
    expert_down = ds4_gpu_tensor_alloc(seq_hidden_bytes);
    if (!residual || !post || !router_logits || !shared_gate || !shared_up || !shared_mid || !shared_out ||
        !expert_gate || !expert_up || !expert_mid || !expert_down) goto cleanup;
    (void)token_ids;
    fprintf(stderr, "[gpu-blk0] write residual\n");
    if (ds4_gpu_tensor_write(residual, 0, fx->residual_after_mixer_seq_ref, seq_hidden_bytes) == 0) goto cleanup;
    fprintf(stderr, "[gpu-blk0] begin commands\n");
    if (ds4_gpu_begin_commands() == 0) goto cleanup;
    fprintf(stderr, "[gpu-blk0] post-attn norm\n");
    if (ds4_gpu_rms_norm_weight_rows_tensor(post,
                                            residual,
                                            mf->map,
                                            mf->size,
                                            layer->post_attn_norm->abs_offset,
                                            hidden,
                                            n_tokens,
                                            QWEN_RMS_EPS) == 0) goto cleanup;
    fprintf(stderr, "[gpu-blk0] router matmul\n");
    if (ds4_gpu_matmul_f32_tensor(router_logits,
                                  mf->map,
                                  mf->size,
                                  layer->ffn_gate_inp->abs_offset,
                                  hidden,
                                  ROUTER_COUNT,
                                  post,
                                  n_tokens) == 0) goto cleanup;
    fprintf(stderr, "[gpu-blk0] shared gate/up/swiglu\n");
    if (ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(shared_gate,
                                                  shared_up,
                                                  shared_mid,
                                                  mf->map,
                                                  mf->size,
                                                  layer->ffn_gate_shexp->abs_offset,
                                                  layer->ffn_up_shexp->abs_offset,
                                                  hidden,
                                                  shared_dim,
                                                  post,
                                                  QWEN_SWIGLU_CLAMP) == 0) goto cleanup;
    fprintf(stderr, "[gpu-blk0] shared down\n");
    if (ds4_gpu_matmul_q8_0_tensor(shared_out,
                                   mf->map,
                                   mf->size,
                                   layer->ffn_down_shexp->abs_offset,
                                   shared_dim,
                                   hidden,
                                                  shared_mid,
                                                  n_tokens) == 0) goto cleanup;
    fprintf(stderr, "[gpu-blk0] end commands\n");
    if (ds4_gpu_end_commands() == 0) goto cleanup;

    router_logits_cpu = (float *)malloc(router_logits_bytes);
    shared_out_cpu = (float *)malloc(seq_hidden_bytes);
    expert_down_cpu = (float *)malloc(seq_hidden_bytes);
    routed_out_cpu = (float *)malloc(seq_hidden_bytes);
    if (!router_logits_cpu || !shared_out_cpu || !expert_down_cpu || !routed_out_cpu) goto cleanup;
    memset(routed_out_cpu, 0, seq_hidden_bytes);
    memset(union_pos, 0xff, sizeof(union_pos));
    fprintf(stderr, "[gpu-blk0] read router logits\n");
    if (ds4_gpu_tensor_read(router_logits, 0, router_logits_cpu, router_logits_bytes) == 0) goto cleanup;
    fprintf(stderr, "[gpu-blk0] read shared\n");
    if (ds4_gpu_tensor_read(shared_out, 0, shared_out_cpu, seq_hidden_bytes) == 0) goto cleanup;

    fprintf(stderr, "[gpu-blk0] cpu router topk\n");
    for (uint32_t t = 0; t < n_tokens; ++t) {
        topk_softmax256(router_logits_cpu + (size_t)t * ROUTER_COUNT,
                        topk,
                        router_selected_cpu + (size_t)t * topk,
                        router_weights_cpu + (size_t)t * topk);
        for (uint32_t i = 0; i < topk; ++i) {
            const int32_t expert_id = router_selected_cpu[(size_t)t * topk + i];
            if (union_pos[expert_id] < 0) {
                union_pos[expert_id] = (int32_t)n_union;
                union_ids[n_union++] = expert_id;
            }
        }
    }

    fprintf(stderr, "[gpu-blk0] routed experts union=%u\n", n_union);
    for (uint32_t u = 0; u < n_union; ++u) {
        const uint32_t expert_id = (uint32_t)union_ids[u];
        fprintf(stderr, "[gpu-blk0] expert %u/%u id=%u\n", u + 1u, n_union, expert_id);
        if (ds4_gpu_begin_commands() == 0) goto cleanup;
        if (ds4_gpu_matmul_q8_0_pair_tensor(expert_gate,
                                            expert_up,
                                            mf->map,
                                            mf->size,
                                            layer->ffn_gate_exps->abs_offset + gate_expert_bytes * expert_id,
                                            layer->ffn_up_exps->abs_offset + gate_expert_bytes * expert_id,
                                            hidden,
                                            shared_dim,
                                            shared_dim,
                                            post,
                                            n_tokens) == 0) goto cleanup;
        if (ds4_gpu_swiglu_tensor(expert_mid,
                                  expert_gate,
                                  expert_up,
                                  n_tokens * shared_dim,
                                  QWEN_SWIGLU_CLAMP,
                                  1.0f) == 0) goto cleanup;
        if (ds4_gpu_matmul_q8_0_tensor(expert_down,
                                       mf->map,
                                       mf->size,
                                       layer->ffn_down_exps->abs_offset + down_expert_bytes * expert_id,
                                       shared_dim,
                                       hidden,
                                       expert_mid,
                                       n_tokens) == 0) goto cleanup;
        if (ds4_gpu_end_commands() == 0) goto cleanup;
        if (ds4_gpu_tensor_read(expert_down, 0, expert_down_cpu, seq_hidden_bytes) == 0) goto cleanup;
        for (uint32_t t = 0; t < n_tokens; ++t) {
            float weight = 0.0f;
            for (uint32_t i = 0; i < topk; ++i) {
                const size_t base = (size_t)t * topk + i;
                if ((uint32_t)router_selected_cpu[base] == expert_id) {
                    weight = router_weights_cpu[base];
                    break;
                }
            }
            if (weight == 0.0f) continue;
            for (uint32_t d = 0; d < hidden; ++d) {
                const size_t idx = (size_t)t * hidden + d;
                routed_out_cpu[idx] += expert_down_cpu[idx] * weight;
            }
        }
    }

    for (uint32_t t = 0; t < n_tokens; ++t) {
        const float shared_scale = sigmoidf_local(fx->shared_gate_pre_seq_ref[t]);
        for (uint32_t d = 0; d < hidden; ++d) {
            const size_t idx = (size_t)t * hidden + d;
            out_seq[idx] = fx->residual_after_mixer_seq_ref[idx] + routed_out_cpu[idx] + shared_out_cpu[idx] * shared_scale;
        }
    }
    ok = 1;

cleanup:
    if (!ok) {
        fprintf(stderr, "[gpu-blk0] failure before completion\n");
        ds4_gpu_print_memory_report("qwen36_gpu_blk0_ffn_q8_oracle_failure");
    }
    free(router_logits_cpu);
    free(shared_out_cpu);
    free(expert_down_cpu);
    free(routed_out_cpu);
    ds4_gpu_tensor_free(expert_down);
    ds4_gpu_tensor_free(expert_mid);
    ds4_gpu_tensor_free(expert_up);
    ds4_gpu_tensor_free(expert_gate);
    ds4_gpu_tensor_free(shared_out);
    ds4_gpu_tensor_free(shared_mid);
    ds4_gpu_tensor_free(shared_up);
    ds4_gpu_tensor_free(shared_gate);
    ds4_gpu_tensor_free(router_logits);
    ds4_gpu_tensor_free(post);
    ds4_gpu_tensor_free(residual);
    return ok;
}

static int run_gpu_q8_embed(
        const mapped_file *mf,
        const qwen36_35a3b_q8_model *q8,
        const prefix_fixture *pfx,
        float *embed_seq_out) {
    ds4_gpu_tensor *tokens = NULL;
    ds4_gpu_tensor *embed = NULL;
    const uint64_t tokens_bytes = (uint64_t)pfx->seq_len * sizeof(int32_t);
    const uint64_t embed_bytes = (uint64_t)pfx->seq_len * pfx->hidden * sizeof(float);
    int ok = 0;

    tokens = ds4_gpu_tensor_alloc(tokens_bytes);
    embed = ds4_gpu_tensor_alloc(embed_bytes);
    if (!tokens || !embed) goto cleanup;
    if (ds4_gpu_tensor_write(tokens, 0, pfx->token_ids, tokens_bytes) == 0) goto cleanup;
    if (ds4_gpu_begin_commands() == 0) goto cleanup;
    if (ds4_gpu_embed_tokens_hc_q8_0_tensor(embed,
                                            tokens,
                                            mf->map,
                                            mf->size,
                                            q8->token_embd->abs_offset,
                                            (uint32_t)q8->token_embd->dims[1],
                                            pfx->seq_len,
                                            pfx->hidden,
                                            1) == 0) goto cleanup;
    if (ds4_gpu_end_commands() == 0) goto cleanup;
    if (ds4_gpu_tensor_read(embed, 0, embed_seq_out, embed_bytes) == 0) goto cleanup;
    ok = 1;

cleanup:
    ds4_gpu_tensor_free(embed);
    ds4_gpu_tensor_free(tokens);
    return ok;
}

int main(int argc, char **argv) {
    const char *gguf_path, *fixture_path, *prefix_path = NULL, *dump_seq_path = NULL;
    qwen36_gguf_file gf;
    qwen36_35a3b_q8_model q8;
    dwf_fixture fx;
    prefix_fixture pfx;
    mapped_file mf;
    float *out_seq = NULL;
    float *embed_seq = NULL;
    int32_t *router_selected = NULL;
    float *router_weights = NULL;
    uint32_t *fallback_tokens = NULL;
    char err[512];
    int ok = 0;

    memset(&gf, 0, sizeof(gf));
    memset(&fx, 0, sizeof(fx));
    memset(&pfx, 0, sizeof(pfx));
    memset(&mf, 0, sizeof(mf));
    mf.fd = -1;

    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf LAYER0.bin [--prefix PREFIX.bin] [--dump-seq PATH]\n", argv[0]);
        return 1;
    }
    gguf_path = argv[1];
    fixture_path = argv[2];
    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--prefix") == 0 && i + 1 < argc) {
            prefix_path = argv[++i];
        } else if (strcmp(argv[i], "--dump-seq") == 0 && i + 1 < argc) {
            dump_seq_path = argv[++i];
        }
    }

    if (!fixture_load(fixture_path, &fx)) {
        fprintf(stderr, "failed to load layer fixture: %s\n", fixture_path);
        goto cleanup;
    }
    if (fx.layer != 0) {
        fprintf(stderr, "expected blk.0 fixture, got blk.%u\n", fx.layer);
        goto cleanup;
    }
    if (prefix_path && !prefix_load(prefix_path, &pfx)) {
        fprintf(stderr, "failed to load prefix fixture: %s\n", prefix_path);
        goto cleanup;
    }
    if (!qwen36_gguf_open(&gf, gguf_path, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        goto cleanup;
    }
    if (!qwen36_35a3b_q8_bind(&gf, &q8, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        goto cleanup;
    }
    if (!mapped_file_open(&mf, gguf_path)) {
        fprintf(stderr, "failed to mmap gguf: %s\n", strerror(errno));
        goto cleanup;
    }
    if (ds4_gpu_init() == 0) {
        fprintf(stderr, "ds4_gpu_init failed\n");
        goto cleanup;
    }
    (void)ds4_gpu_set_model_map(mf.map, mf.size);
    (void)ds4_gpu_set_model_fd_for_map(mf.fd, mf.map);

    out_seq = (float *)malloc((size_t)fx.seq_len * fx.hidden * sizeof(float));
    router_selected = (int32_t *)malloc((size_t)fx.seq_len * fx.topk * sizeof(int32_t));
    router_weights = (float *)malloc((size_t)fx.seq_len * fx.topk * sizeof(float));
    fallback_tokens = (uint32_t *)calloc(fx.seq_len ? fx.seq_len : 1u, sizeof(uint32_t));
    if (!out_seq || !router_selected || !router_weights || !fallback_tokens) goto cleanup;

    if (prefix_path) {
        embed_seq = (float *)malloc((size_t)pfx.seq_len * pfx.hidden * sizeof(float));
        if (!embed_seq) goto cleanup;
        if (!run_gpu_q8_embed(&mf, &q8, &pfx, embed_seq)) {
            fprintf(stderr, "gpu embedding failed\n");
            goto cleanup;
        }
        printf("embed_seq_rmse: %.8f\n", rmse(embed_seq, pfx.input_seq_ref, (size_t)pfx.seq_len * pfx.hidden));
        printf("embed_seq_cosine: %.8f\n", cosine(embed_seq, pfx.input_seq_ref, (size_t)pfx.seq_len * pfx.hidden));
    }

    if (!run_gpu_blk0_ffn(&mf, &q8.layers[0], &fx, pfx.token_ids ? pfx.token_ids : fallback_tokens, out_seq, router_selected, router_weights)) {
        fprintf(stderr, "gpu blk0 ffn failed\n");
        goto cleanup;
    }

    printf("blk0_ffn_seq_rmse: %.8f\n", rmse(out_seq, fx.layer_output_seq_ref, (size_t)fx.seq_len * fx.hidden));
    printf("blk0_ffn_seq_cosine: %.8f\n", cosine(out_seq, fx.layer_output_seq_ref, (size_t)fx.seq_len * fx.hidden));
    printf("blk0_ffn_last_rmse: %.8f\n", rmse(out_seq + (size_t)(fx.seq_len - 1) * fx.hidden, fx.layer_output_ref, fx.hidden));
    printf("blk0_ffn_last_cosine: %.8f\n", cosine(out_seq + (size_t)(fx.seq_len - 1) * fx.hidden, fx.layer_output_ref, fx.hidden));

    {
        uint32_t topk_match = 1;
        for (uint32_t t = 0; t < fx.seq_len && topk_match; ++t) {
            for (uint32_t i = 0; i < fx.topk; ++i) {
                const int32_t got = router_selected[(size_t)t * fx.topk + i];
                const int32_t ref = (int32_t)fx.router_indices_seq_ref_f32[(size_t)t * fx.topk + i];
                if (got != ref) { topk_match = 0; break; }
            }
        }
        printf("router_topk_exact: %s\n", topk_match ? "true" : "false");
        printf("router_scores_rmse: %.8f\n", rmse(router_weights, fx.router_scores_seq_ref, (size_t)fx.seq_len * fx.topk));
    }

    if (dump_seq_path) {
        FILE *fp = fopen(dump_seq_path, "wb");
        if (!fp) {
            fprintf(stderr, "failed to open dump path: %s\n", dump_seq_path);
            goto cleanup;
        }
        fwrite(out_seq, sizeof(float), (size_t)fx.seq_len * fx.hidden, fp);
        fclose(fp);
        printf("wrote_seq: %s\n", dump_seq_path);
    }
    ok = 1;

cleanup:
    if (mf.map || mf.fd >= 0) mapped_file_close(&mf);
    ds4_gpu_cleanup();
    qwen36_gguf_close(&gf);
    prefix_free(&pfx);
    fixture_free(&fx);
    free(out_seq);
    free(embed_seq);
    free(router_selected);
    free(router_weights);
    free(fallback_tokens);
    return ok ? 0 : 1;
}
