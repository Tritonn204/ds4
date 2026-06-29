#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#define QWEN_RMS_EPS 1e-6f
#define QWEN_SWIGLU_CLAMP 80.0f

typedef struct mapped_file {
    int fd;
    void *map;
    uint64_t size;
} mapped_file;

typedef struct full_layer_small_cache {
    float *q_norm_w;
    float *k_norm_w;
    float *shared_gate_inp_w;
} full_layer_small_cache;

static int mapped_file_open(mapped_file *mf, const char *path) {
    struct stat st;
    memset(mf, 0, sizeof(*mf));
    mf->fd = open(path, O_RDONLY);
    if (mf->fd == -1) return 0;
    if (fstat(mf->fd, &st) == -1) {
        close(mf->fd);
        mf->fd = -1;
        return 0;
    }
    mf->map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, mf->fd, 0);
    if (mf->map == MAP_FAILED) {
        close(mf->fd);
        mf->fd = -1;
        mf->map = NULL;
        return 0;
    }
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

static void free_small_cache(full_layer_small_cache *cache) {
    if (!cache) return;
    free(cache->q_norm_w);
    free(cache->k_norm_w);
    free(cache->shared_gate_inp_w);
    memset(cache, 0, sizeof(*cache));
}

static int load_small_cache(const qwen36_gguf_file *gf,
                            const qwen36_35a3b_q8_layer *layer,
                            full_layer_small_cache *cache,
                            char *err,
                            size_t err_cap) {
    memset(cache, 0, sizeof(*cache));
    if (!read_f32_tensor(gf, layer->attn_q_norm, &cache->q_norm_w, FULL_HEAD_DIM, err, err_cap)) return 0;
    if (!read_f32_tensor(gf, layer->attn_k_norm, &cache->k_norm_w, FULL_HEAD_DIM, err, err_cap)) return 0;
    if (!read_f32_tensor(gf, layer->ffn_gate_inp_shexp, &cache->shared_gate_inp_w, FULL_HIDDEN, err, err_cap)) return 0;
    return 1;
}

static uint64_t q8_row_bytes(uint32_t cols) {
    return ((uint64_t)cols / 32u) * 34u;
}

static void build_perms(uint32_t n_heads, uint32_t *gg_to_hf, uint32_t *hf_to_gg) {
    uint32_t pos = 0;
    for (uint32_t i = 0; i < n_heads; i += 2) gg_to_hf[pos++] = i;
    for (uint32_t i = 1; i < n_heads; i += 2) gg_to_hf[pos++] = i;
    for (uint32_t i = 0; i < n_heads; ++i) hf_to_gg[gg_to_hf[i]] = i;
}

static void reorder_head_rows_seq_f32(float *dst, const float *src, uint32_t seq_len, uint32_t n_heads, uint32_t head_dim) {
    uint32_t *gg_to_hf = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    if (!gg_to_hf || !hf_to_gg) {
        free(gg_to_hf); free(hf_to_gg);
        return;
    }
    build_perms(n_heads, gg_to_hf, hf_to_gg);
    for (uint32_t t = 0; t < seq_len; ++t) {
        for (uint32_t h = 0; h < n_heads; ++h) {
            memcpy(dst + ((size_t)t * n_heads + h) * head_dim,
                   src + ((size_t)t * n_heads + hf_to_gg[h]) * head_dim,
                   (size_t)head_dim * sizeof(float));
        }
    }
    free(gg_to_hf);
    free(hf_to_gg);
}

static void reorder_head_scalars_seq_f32(float *dst, const float *src, uint32_t seq_len, uint32_t n_heads) {
    uint32_t *gg_to_hf = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    if (!gg_to_hf || !hf_to_gg) {
        free(gg_to_hf); free(hf_to_gg);
        return;
    }
    build_perms(n_heads, gg_to_hf, hf_to_gg);
    for (uint32_t t = 0; t < seq_len; ++t) {
        for (uint32_t h = 0; h < n_heads; ++h) {
            dst[(size_t)t * n_heads + h] = src[(size_t)t * n_heads + hf_to_gg[h]];
        }
    }
    free(gg_to_hf);
    free(hf_to_gg);
}

static void reorder_qkv_v_tokenmajor_gg_to_hf(
        float *dst,
        const float *src,
        uint32_t seq_len,
        uint32_t key_dim,
        uint32_t n_heads,
        uint32_t head_dim) {
    uint32_t *gg_to_hf = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    const uint32_t value_dim = n_heads * head_dim;
    const uint32_t qkv_dim = key_dim * 2u + value_dim;
    const uint32_t v_off = key_dim * 2u;
    if (!gg_to_hf || !hf_to_gg) {
        free(gg_to_hf);
        free(hf_to_gg);
        return;
    }
    build_perms(n_heads, gg_to_hf, hf_to_gg);
    for (uint32_t t = 0; t < seq_len; ++t) {
        const float *src_row = src + (size_t)t * qkv_dim;
        float *dst_row = dst + (size_t)t * qkv_dim;
        memcpy(dst_row, src_row, (size_t)(key_dim * 2u) * sizeof(float));
        for (uint32_t h = 0; h < n_heads; ++h) {
            memcpy(dst_row + v_off + (size_t)h * head_dim,
                   src_row + v_off + (size_t)hf_to_gg[h] * head_dim,
                   (size_t)head_dim * sizeof(float));
        }
    }
    free(gg_to_hf);
    free(hf_to_gg);
}

static void reorder_out_in_hf_to_gg(float *dst, const float *src, uint32_t seq_len, uint32_t n_heads, uint32_t head_dim) {
    uint32_t *gg_to_hf = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg = (uint32_t *)malloc((size_t)n_heads * sizeof(uint32_t));
    if (!gg_to_hf || !hf_to_gg) {
        free(gg_to_hf); free(hf_to_gg);
        return;
    }
    build_perms(n_heads, gg_to_hf, hf_to_gg);
    for (uint32_t t = 0; t < seq_len; ++t) {
        for (uint32_t h_gg = 0; h_gg < n_heads; ++h_gg) {
            memcpy(dst + ((size_t)t * n_heads + h_gg) * head_dim,
                   src + ((size_t)t * n_heads + gg_to_hf[h_gg]) * head_dim,
                   (size_t)head_dim * sizeof(float));
        }
    }
    free(gg_to_hf);
    free(hf_to_gg);
}

static int run_gpu_q8_qkv_split_rowwise(const mapped_file *mf,
                                        const qwen36_35a3b_q8_layer *layer,
                                        uint32_t in_dim,
                                        uint32_t key_dim,
                                        uint32_t value_dim,
                                        ds4_gpu_tensor *input_gpu,
                                        uint32_t n_tokens,
                                        float *out_qkv) {
    const uint32_t qkv_dim = key_dim * 2u + value_dim;
    const uint64_t bytes = (uint64_t)n_tokens * qkv_dim * sizeof(float);
    ds4_gpu_tensor *qkv_gpu = ds4_gpu_tensor_alloc(bytes);
    int ok = 0;
    if (!qkv_gpu) return 0;
    if (ds4_gpu_begin_commands() == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(qkv_gpu, mf->map, mf->size,
                                   layer->attn_qkv->abs_offset,
                                   in_dim, qkv_dim, input_gpu, n_tokens) == 0) goto cleanup;
    if (ds4_gpu_end_commands() == 0) goto cleanup;
    if (ds4_gpu_tensor_read(qkv_gpu, 0, out_qkv, bytes) == 0) goto cleanup;
    ok = 1;
cleanup:
    ds4_gpu_tensor_free(qkv_gpu);
    return ok;
}

static int run_gpu_ffn_from_residual_unified(
        const mapped_file *mf,
        const qwen36_35a3b_q8_layer *layer,
        const float *shared_gate_inp_w,
        uint32_t n_tokens,
        const float *residual_in,
        float *out_seq,
        void *unused_timing) {
    const uint32_t hidden = FULL_HIDDEN;
    const uint32_t topk = 8u;
    const uint32_t shared_dim = FULL_INTER;
    const uint64_t seq_hidden_bytes = (uint64_t)n_tokens * hidden * sizeof(float);
    const uint64_t router_logits_bytes = (uint64_t)n_tokens * ROUTER_COUNT * sizeof(float);
    const uint64_t shared_bytes = (uint64_t)n_tokens * shared_dim * sizeof(float);
    const uint64_t gate_row_bytes = q8_row_bytes(hidden);
    const uint64_t gate_expert_bytes = gate_row_bytes * FULL_INTER;
    const uint64_t down_row_bytes = q8_row_bytes(FULL_INTER);
    const uint64_t down_expert_bytes = down_row_bytes * hidden;
    ds4_gpu_tensor *residual = NULL, *post = NULL, *router_logits = NULL;
    ds4_gpu_tensor *shared_gate = NULL, *shared_up = NULL, *shared_mid = NULL, *shared_out = NULL;
    ds4_gpu_tensor *expert_gate = NULL, *expert_up = NULL, *expert_mid = NULL, *expert_down = NULL;
    float *router_logits_cpu = NULL, *shared_out_cpu = NULL, *expert_down_cpu = NULL, *routed_out_cpu = NULL, *post_cpu = NULL;
    uint32_t *router_selected_cpu = NULL;
    float *router_weights_cpu = NULL;
    uint32_t union_ids[ROUTER_COUNT];
    int32_t union_pos[ROUTER_COUNT];
    uint32_t n_union = 0;
    int ok = 0;
    (void)unused_timing;

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

    if (ds4_gpu_tensor_write(residual, 0, residual_in, seq_hidden_bytes) == 0) goto cleanup;
    if (ds4_gpu_begin_commands() == 0) goto cleanup;
    if (ds4_gpu_rms_norm_weight_rows_tensor(post, residual, mf->map, mf->size,
                                            layer->post_attn_norm->abs_offset,
                                            hidden, n_tokens, QWEN_RMS_EPS) == 0) goto cleanup;
    if (ds4_gpu_matmul_f32_tensor(router_logits, mf->map, mf->size,
                                  layer->ffn_gate_inp->abs_offset,
                                  hidden, ROUTER_COUNT, post, n_tokens) == 0) goto cleanup;
    if (ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(shared_gate, shared_up, shared_mid,
                                                  mf->map, mf->size,
                                                  layer->ffn_gate_shexp->abs_offset,
                                                  layer->ffn_up_shexp->abs_offset,
                                                  hidden, shared_dim, post,
                                                  QWEN_SWIGLU_CLAMP) == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(shared_out, mf->map, mf->size,
                                   layer->ffn_down_shexp->abs_offset,
                                   shared_dim, hidden, shared_mid, n_tokens) == 0) goto cleanup;
    if (ds4_gpu_end_commands() == 0) goto cleanup;

    router_logits_cpu = (float *)malloc(router_logits_bytes);
    shared_out_cpu = (float *)malloc(seq_hidden_bytes);
    expert_down_cpu = (float *)malloc(seq_hidden_bytes);
    routed_out_cpu = (float *)calloc((size_t)n_tokens * hidden, sizeof(float));
    post_cpu = (float *)malloc(seq_hidden_bytes);
    router_selected_cpu = (uint32_t *)malloc((size_t)n_tokens * topk * sizeof(uint32_t));
    router_weights_cpu = (float *)malloc((size_t)n_tokens * topk * sizeof(float));
    if (!router_logits_cpu || !shared_out_cpu || !expert_down_cpu || !routed_out_cpu || !post_cpu ||
        !router_selected_cpu || !router_weights_cpu) goto cleanup;
    memset(union_pos, 0xff, sizeof(union_pos));
    if (ds4_gpu_tensor_read(router_logits, 0, router_logits_cpu, router_logits_bytes) == 0) goto cleanup;
    if (ds4_gpu_tensor_read(shared_out, 0, shared_out_cpu, seq_hidden_bytes) == 0) goto cleanup;
    if (ds4_gpu_tensor_read(post, 0, post_cpu, seq_hidden_bytes) == 0) goto cleanup;

    for (uint32_t t = 0; t < n_tokens; ++t) {
        topk_softmax256(router_logits_cpu + (size_t)t * ROUTER_COUNT, topk,
                        router_selected_cpu + (size_t)t * topk,
                        router_weights_cpu + (size_t)t * topk);
        for (uint32_t i = 0; i < topk; ++i) {
            const uint32_t expert_id = router_selected_cpu[(size_t)t * topk + i];
            if (union_pos[expert_id] < 0) {
                union_pos[expert_id] = (int32_t)n_union;
                union_ids[n_union++] = expert_id;
            }
        }
    }

    for (uint32_t u = 0; u < n_union; ++u) {
        const uint32_t expert_id = union_ids[u];
        if (ds4_gpu_begin_commands() == 0) goto cleanup;
        if (ds4_gpu_matmul_q8_0_pair_tensor(expert_gate, expert_up,
                                            mf->map, mf->size,
                                            layer->ffn_gate_exps->abs_offset + gate_expert_bytes * expert_id,
                                            layer->ffn_up_exps->abs_offset + gate_expert_bytes * expert_id,
                                            hidden, shared_dim, shared_dim, post, n_tokens) == 0) goto cleanup;
        if (ds4_gpu_swiglu_tensor(expert_mid, expert_gate, expert_up,
                                  n_tokens * shared_dim, QWEN_SWIGLU_CLAMP, 1.0f) == 0) goto cleanup;
        if (ds4_gpu_matmul_q8_0_tensor(expert_down, mf->map, mf->size,
                                       layer->ffn_down_exps->abs_offset + down_expert_bytes * expert_id,
                                       shared_dim, hidden, expert_mid, n_tokens) == 0) goto cleanup;
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
        float scale_in = 0.0f;
        for (uint32_t d = 0; d < hidden; ++d) scale_in += post_cpu[(size_t)t * hidden + d] * shared_gate_inp_w[d];
        {
            const float s = sigmoidf_local(scale_in);
            for (uint32_t d = 0; d < hidden; ++d) {
                const size_t idx = (size_t)t * hidden + d;
                out_seq[idx] = residual_in[idx] + routed_out_cpu[idx] + shared_out_cpu[idx] * s;
            }
        }
    }
    ok = 1;

cleanup:
    free(router_logits_cpu);
    free(shared_out_cpu);
    free(expert_down_cpu);
    free(routed_out_cpu);
    free(post_cpu);
    free(router_selected_cpu);
    free(router_weights_cpu);
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

static int run_full_layer_step_gpu(const qwen36_35a3b_q8_layer *layer,
                                   const mapped_file *mf,
                                   const full_layer_small_cache *cache,
                                   full_layer_state *state,
                                   const float *input_row,
                                   float *out_row) {
    ds4_gpu_tensor *input_gpu = NULL, *attn_in_gpu = NULL, *qg_gpu = NULL, *k_gpu = NULL, *v_gpu = NULL;
    ds4_gpu_tensor *attn_out_gpu = NULL, *proj_out_gpu = NULL;
    float *attn_in = NULL, *qg = NULL, *kk = NULL, *vv = NULL;
    float q_cur[FULL_HEAD_DIM * FULL_NUM_HEADS];
    float k_cur[FULL_HEAD_DIM * FULL_NUM_KV_HEADS];
    float v_cur[FULL_HEAD_DIM * FULL_NUM_KV_HEADS];
    float gate_cur[FULL_ATTN_GATE_DIM];
    float attn_out_flat[FULL_ATTN_GATE_DIM];
    float proj_out[FULL_HIDDEN];
    float residual[FULL_HIDDEN];
    uint32_t pos = state->seq_len;
    int ok = 0;

    attn_in = (float *)malloc((size_t)FULL_HIDDEN * sizeof(float));
    qg = (float *)malloc((size_t)FULL_ATTN_GATE_DIM * 2u * sizeof(float));
    kk = (float *)malloc((size_t)FULL_NUM_KV_HEADS * FULL_HEAD_DIM * sizeof(float));
    vv = (float *)malloc((size_t)FULL_NUM_KV_HEADS * FULL_HEAD_DIM * sizeof(float));
    if (!attn_in || !qg || !kk || !vv) goto cleanup;

    input_gpu = ds4_gpu_tensor_alloc((uint64_t)FULL_HIDDEN * sizeof(float));
    attn_in_gpu = ds4_gpu_tensor_alloc((uint64_t)FULL_HIDDEN * sizeof(float));
    qg_gpu = ds4_gpu_tensor_alloc((uint64_t)FULL_ATTN_GATE_DIM * 2u * sizeof(float));
    k_gpu = ds4_gpu_tensor_alloc((uint64_t)FULL_NUM_KV_HEADS * FULL_HEAD_DIM * sizeof(float));
    v_gpu = ds4_gpu_tensor_alloc((uint64_t)FULL_NUM_KV_HEADS * FULL_HEAD_DIM * sizeof(float));
    attn_out_gpu = ds4_gpu_tensor_alloc((uint64_t)FULL_ATTN_GATE_DIM * sizeof(float));
    proj_out_gpu = ds4_gpu_tensor_alloc((uint64_t)FULL_HIDDEN * sizeof(float));
    if (!input_gpu || !attn_in_gpu || !qg_gpu || !k_gpu || !v_gpu || !attn_out_gpu || !proj_out_gpu) goto cleanup;

    if (ds4_gpu_tensor_write(input_gpu, 0, input_row, (uint64_t)FULL_HIDDEN * sizeof(float)) == 0) goto cleanup;
    if (ds4_gpu_begin_commands() == 0) goto cleanup;
    if (ds4_gpu_rms_norm_weight_rows_tensor(attn_in_gpu, input_gpu, mf->map, mf->size,
                                            layer->attn_norm->abs_offset, FULL_HIDDEN, 1u, QWEN_RMS_EPS) == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(qg_gpu, mf->map, mf->size,
                                   layer->attn_q->abs_offset, FULL_HIDDEN, FULL_ATTN_GATE_DIM * 2u, attn_in_gpu, 1u) == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(k_gpu, mf->map, mf->size,
                                   layer->attn_k->abs_offset, FULL_HIDDEN, FULL_NUM_KV_HEADS * FULL_HEAD_DIM, attn_in_gpu, 1u) == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(v_gpu, mf->map, mf->size,
                                   layer->attn_v->abs_offset, FULL_HIDDEN, FULL_NUM_KV_HEADS * FULL_HEAD_DIM, attn_in_gpu, 1u) == 0) goto cleanup;
    if (ds4_gpu_end_commands() == 0) goto cleanup;

    if (ds4_gpu_tensor_read(attn_in_gpu, 0, attn_in, (uint64_t)FULL_HIDDEN * sizeof(float)) == 0) goto cleanup;
    if (ds4_gpu_tensor_read(qg_gpu, 0, qg, (uint64_t)FULL_ATTN_GATE_DIM * 2u * sizeof(float)) == 0) goto cleanup;
    if (ds4_gpu_tensor_read(k_gpu, 0, kk, (uint64_t)FULL_NUM_KV_HEADS * FULL_HEAD_DIM * sizeof(float)) == 0) goto cleanup;
    if (ds4_gpu_tensor_read(v_gpu, 0, vv, (uint64_t)FULL_NUM_KV_HEADS * FULL_HEAD_DIM * sizeof(float)) == 0) goto cleanup;

    for (uint32_t h = 0; h < FULL_NUM_HEADS; ++h) {
        const float *src = qg + (size_t)h * (FULL_HEAD_DIM * 2u);
        float *qdst = q_cur + (size_t)h * FULL_HEAD_DIM;
        rmsnorm_weight_plus1(src, qdst, cache->q_norm_w, FULL_HEAD_DIM);
        memcpy(gate_cur + (size_t)h * FULL_HEAD_DIM, src + FULL_HEAD_DIM, FULL_HEAD_DIM * sizeof(float));
    }
    for (uint32_t h = 0; h < FULL_NUM_KV_HEADS; ++h) {
        rmsnorm_weight_plus1(kk + (size_t)h * FULL_HEAD_DIM, k_cur + (size_t)h * FULL_HEAD_DIM, cache->k_norm_w, FULL_HEAD_DIM);
        memcpy(v_cur + (size_t)h * FULL_HEAD_DIM, vv + (size_t)h * FULL_HEAD_DIM, FULL_HEAD_DIM * sizeof(float));
    }
    for (uint32_t h = 0; h < FULL_NUM_KV_HEADS; ++h) apply_rope_one_inplace(k_cur + (size_t)h * FULL_HEAD_DIM, pos);
    for (uint32_t h = 0; h < FULL_NUM_HEADS; ++h) apply_rope_one_inplace(q_cur + (size_t)h * FULL_HEAD_DIM, pos);

    memset(attn_out_flat, 0, sizeof(attn_out_flat));
    for (uint32_t h = 0; h < FULL_NUM_HEADS; ++h) {
        const uint32_t kvh = h / (FULL_NUM_HEADS / FULL_NUM_KV_HEADS);
        float *out = attn_out_flat + (size_t)h * FULL_HEAD_DIM;
        const float *qv = q_cur + (size_t)h * FULL_HEAD_DIM;
        double scores[8192];
        double maxv = -1e300, sum = 0.0;
        for (uint32_t s = 0; s <= pos; ++s) {
            const float *kv = (s == pos) ? (k_cur + (size_t)kvh * FULL_HEAD_DIM) : (state->k_all + ((size_t)s * FULL_NUM_KV_HEADS + kvh) * FULL_HEAD_DIM);
            double dot = 0.0;
            for (uint32_t d = 0; d < FULL_HEAD_DIM; ++d) dot += (double)qv[d] * kv[d];
            scores[s] = dot / sqrt((double)FULL_HEAD_DIM);
            if (scores[s] > maxv) maxv = scores[s];
        }
        for (uint32_t s = 0; s <= pos; ++s) {
            scores[s] = exp(scores[s] - maxv);
            sum += scores[s];
        }
        for (uint32_t s = 0; s <= pos; ++s) {
            const float w = (float)(scores[s] / sum);
            const float *vvh = (s == pos) ? (v_cur + (size_t)kvh * FULL_HEAD_DIM) : (state->v_all + ((size_t)s * FULL_NUM_KV_HEADS + kvh) * FULL_HEAD_DIM);
            for (uint32_t d = 0; d < FULL_HEAD_DIM; ++d) out[d] += w * vvh[d];
        }
        for (uint32_t d = 0; d < FULL_HEAD_DIM; ++d) out[d] *= sigmoidf_local(gate_cur[(size_t)h * FULL_HEAD_DIM + d]);
    }

    if (ds4_gpu_tensor_write(attn_out_gpu, 0, attn_out_flat, (uint64_t)FULL_ATTN_GATE_DIM * sizeof(float)) == 0) goto cleanup;
    if (ds4_gpu_begin_commands() == 0) goto cleanup;
    if (ds4_gpu_matmul_q8_0_tensor(proj_out_gpu, mf->map, mf->size,
                                   layer->attn_output->abs_offset, FULL_ATTN_GATE_DIM, FULL_HIDDEN, attn_out_gpu, 1u) == 0) goto cleanup;
    if (ds4_gpu_end_commands() == 0) goto cleanup;
    if (ds4_gpu_tensor_read(proj_out_gpu, 0, proj_out, (uint64_t)FULL_HIDDEN * sizeof(float)) == 0) goto cleanup;

    for (uint32_t d = 0; d < FULL_HIDDEN; ++d) residual[d] = input_row[d] + proj_out[d];
    if (!run_gpu_ffn_from_residual_unified(mf, layer, cache->shared_gate_inp_w, 1u, residual, out_row, NULL)) goto cleanup;

    if (!ensure_full_layer_state_cap(state, pos + 1u, NULL, 0)) goto cleanup;
    memcpy(state->k_all + (size_t)pos * FULL_NUM_KV_HEADS * FULL_HEAD_DIM, k_cur, sizeof(k_cur));
    memcpy(state->v_all + (size_t)pos * FULL_NUM_KV_HEADS * FULL_HEAD_DIM, v_cur, sizeof(v_cur));
    memcpy(state->output_seq + (size_t)pos * FULL_HIDDEN, out_row, (size_t)FULL_HIDDEN * sizeof(float));
    state->seq_len = pos + 1u;
    ok = 1;

cleanup:
    ds4_gpu_tensor_free(proj_out_gpu);
    ds4_gpu_tensor_free(attn_out_gpu);
    ds4_gpu_tensor_free(v_gpu);
    ds4_gpu_tensor_free(k_gpu);
    ds4_gpu_tensor_free(qg_gpu);
    ds4_gpu_tensor_free(attn_in_gpu);
    ds4_gpu_tensor_free(input_gpu);
    free(attn_in);
    free(qg);
    free(kk);
    free(vv);
    return ok;
}

static int run_full_layer_prefill_gpu(uint32_t full_layer,
                                      const qwen36_35a3b_q8_layer *layer,
                                      const mapped_file *mf,
                                      const full_layer_small_cache *cache,
                                      full_layer_state *state,
                                      const float *input_seq,
                                      uint32_t seq_len) {
    float row_out[FULL_HIDDEN];
    (void)full_layer;
    free_full_layer_state(state);
    memset(state, 0, sizeof(*state));
    for (uint32_t t = 0; t < seq_len; ++t) {
        if (!run_full_layer_step_gpu(layer, mf, cache, state, input_seq + (size_t)t * FULL_HIDDEN, row_out)) return 0;
    }
    return 1;
}

static int run_hybrid_layer_prefill_gpuproj(const live_fixture *fx,
                                            const qwen36_gguf_file *gf,
                                            const qwen36_35a3b_q8_layer *layer,
                                            const mapped_file *mf,
                                            layer_runtime *ls,
                                            uint32_t seq_len,
                                            const float *layer_input_seq,
                                            float *out_final_seq,
                                            char *err,
                                            size_t err_cap) {
    uint32_t i;
    (void)gf;
    if (ls->deltanet_state) {
        memset(ls->deltanet_state, 0, (size_t)fx->num_v_heads * fx->head_k_dim * fx->head_v_dim * sizeof(float));
    }
    for (i = 0; i < 3; ++i) {
        if (ls->conv_ring[i]) memset(ls->conv_ring[i], 0, (size_t)(fx->key_dim * 2u + fx->value_dim) * sizeof(float));
    }
    ls->conv_ring_count = 0;
    for (i = 0; i < seq_len; ++i) {
        if (!run_hybrid_layer_step_gpuproj(fx, gf, layer, mf, ls,
                                           layer_input_seq + (size_t)i * fx->hidden,
                                           out_final_seq + (size_t)i * fx->hidden,
                                           err, err_cap)) {
            return 0;
        }
    }
    return 1;
}
