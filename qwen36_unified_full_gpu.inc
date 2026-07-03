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

static int full_gpu_env_enabled_default_off(const char *name) {
    const char *env = getenv(name);
    if (!env || !env[0]) return 0;
    return !(strcmp(env, "0") == 0 ||
             strcmp(env, "false") == 0 ||
             strcmp(env, "FALSE") == 0 ||
             strcmp(env, "no") == 0 ||
             strcmp(env, "NO") == 0);
}

static int g_qwen36_full_gpu_prefill_mode = 0;

static int qwen36_routed_moe_supported_types(uint32_t gate_type, uint32_t down_type) {
    return (gate_type == QWEN36_GGUF_TYPE_Q4_K && down_type == QWEN36_GGUF_TYPE_Q4_K) ||
           (gate_type == QWEN36_GGUF_TYPE_IQ2_XXS && down_type == QWEN36_GGUF_TYPE_Q2_K) ||
           (gate_type == QWEN36_GGUF_TYPE_Q2_K && down_type == QWEN36_GGUF_TYPE_Q2_K);
}

static int load_small_cache(const qwen36_gguf_file *gf,
                            const qwen36_contract_layer *layer,
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
        uint32_t num_v_heads,
        uint32_t head_v_dim) {
    uint32_t *gg_to_hf_v = (uint32_t *)malloc((size_t)num_v_heads * sizeof(uint32_t));
    uint32_t *hf_to_gg_v = (uint32_t *)malloc((size_t)num_v_heads * sizeof(uint32_t));
    const uint32_t value_dim = num_v_heads * head_v_dim;
    const uint32_t qkv_dim = key_dim * 2u + value_dim;
    const uint32_t v_off = key_dim * 2u;
    if (!gg_to_hf_v || !hf_to_gg_v) {
        free(gg_to_hf_v);
        free(hf_to_gg_v);
        return;
    }
    build_perms(num_v_heads, gg_to_hf_v, hf_to_gg_v);
    for (uint32_t t = 0; t < seq_len; ++t) {
        const float *src_row = src + (size_t)t * qkv_dim;
        float *dst_row = dst + (size_t)t * qkv_dim;
        memcpy(dst_row, src_row, (size_t)(qkv_dim - value_dim) * sizeof(float));
        for (uint32_t h = 0; h < num_v_heads; ++h) {
            memcpy(dst_row + v_off + (size_t)h * head_v_dim,
                   src_row + v_off + (size_t)hf_to_gg_v[h] * head_v_dim,
                   (size_t)head_v_dim * sizeof(float));
        }
    }
    free(gg_to_hf_v);
    free(hf_to_gg_v);
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
                                        const qwen36_contract_layer *layer,
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

static int run_hybrid_layer_prefill_gpuproj_batched(const live_fixture *fx,
                                                    const qwen36_gguf_file *gf,
                                                    const qwen36_contract_layer *layer,
                                                    const mapped_file *mf,
                                                    layer_runtime *ls,
                                                    uint32_t seq_len,
                                                    const float *layer_input_seq,
                                                    float *out_final_seq,
                                                    char *err,
                                                    size_t err_cap) {
    layer_step_scratch *sc = &ls->step;
    const uint32_t qkv_dim = fx->key_dim * 2u + fx->value_dim;
    const uint32_t rep = fx->num_v_heads / fx->num_k_heads;
    const uint64_t hidden_bytes = (uint64_t)seq_len * fx->hidden * sizeof(float);
    const uint64_t qkv_bytes = (uint64_t)seq_len * qkv_dim * sizeof(float);
    const uint64_t z_bytes = (uint64_t)seq_len * fx->value_dim * sizeof(float);
    const uint64_t ab_bytes = (uint64_t)seq_len * fx->num_v_heads * sizeof(float);
    const uint64_t out_in_bytes = (uint64_t)seq_len * fx->value_dim * sizeof(float);
    const size_t qk_seq_elems = (size_t)seq_len * fx->num_v_heads * fx->head_k_dim;
    const size_t vv_seq_elems = (size_t)seq_len * fx->value_dim;
    const size_t hv_seq_elems = (size_t)seq_len * fx->num_v_heads;
    ds4_gpu_tensor *input_gpu = NULL, *input_ln_gpu = NULL, *z_gpu = NULL, *a_gpu = NULL, *b_gpu = NULL;
    ds4_gpu_tensor *out_in_gpu = NULL, *out_proj_gpu = NULL;
    float *qkv_raw = NULL, *qkv = NULL, *z_raw = NULL, *z = NULL, *a_raw = NULL, *a = NULL, *b_raw = NULL, *b = NULL;
    float *conv = NULL, *q = NULL, *k = NULL, *v = NULL, *beta = NULL, *g = NULL, *core = NULL;
    float *out_in = NULL, *out_in_gg = NULL, *out_proj = NULL;
    uint32_t t, h, d, vd, i;

    if (seq_len == 0) return 1;

    qkv_raw = (float *)malloc(qkv_bytes);
    qkv = (float *)malloc(qkv_bytes);
    z_raw = (float *)malloc(z_bytes);
    z = (float *)malloc(z_bytes);
    a_raw = (float *)malloc(ab_bytes);
    a = (float *)malloc(ab_bytes);
    b_raw = (float *)malloc(ab_bytes);
    b = (float *)malloc(ab_bytes);
    conv = (float *)malloc(qkv_bytes);
    q = (float *)malloc(qk_seq_elems * sizeof(float));
    k = (float *)malloc(qk_seq_elems * sizeof(float));
    v = (float *)malloc(vv_seq_elems * sizeof(float));
    beta = (float *)malloc(hv_seq_elems * sizeof(float));
    g = (float *)malloc(hv_seq_elems * sizeof(float));
    core = (float *)malloc(vv_seq_elems * sizeof(float));
    out_in = (float *)malloc(out_in_bytes);
    out_in_gg = (float *)malloc(out_in_bytes);
    out_proj = (float *)malloc(hidden_bytes);
    if (!qkv_raw || !qkv || !z_raw || !z || !a_raw || !a || !b_raw || !b ||
        !conv || !q || !k || !v || !beta || !g || !core || !out_in || !out_in_gg || !out_proj) {
        snprintf(err, err_cap, "oom hybrid gpu prefill batch blk.%u", fx->layer);
        goto fail;
    }

    input_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    input_ln_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    z_gpu = ds4_gpu_tensor_alloc(z_bytes);
    a_gpu = ds4_gpu_tensor_alloc(ab_bytes);
    b_gpu = ds4_gpu_tensor_alloc(ab_bytes);
    out_in_gpu = ds4_gpu_tensor_alloc(out_in_bytes);
    out_proj_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    if (!input_gpu || !input_ln_gpu || !z_gpu || !a_gpu || !b_gpu || !out_in_gpu || !out_proj_gpu) {
        snprintf(err, err_cap, "gpu prefill temp tensor alloc failed blk.%u", fx->layer);
        goto fail;
    }

    if (ds4_gpu_tensor_write(input_gpu, 0, layer_input_seq, hidden_bytes) == 0) {
        snprintf(err, err_cap, "gpu prefill input write failed blk.%u", fx->layer);
        goto fail;
    }
    if (ds4_gpu_begin_commands() == 0) {
        snprintf(err, err_cap, "gpu prefill begin failed blk.%u", fx->layer);
        goto fail;
    }
    if (ds4_gpu_rms_norm_weight_rows_tensor(input_ln_gpu, input_gpu, mf->map, mf->size,
                                            layer->attn_norm->abs_offset, fx->hidden, seq_len, 1e-6f) == 0) {
        snprintf(err, err_cap, "gpu prefill rmsnorm failed blk.%u", fx->layer);
        goto fail;
    }
    if (hybrid_gpu_dense_matmul_tensor(z_gpu, mf, layer->attn_gate,
                                       fx->hidden, fx->value_dim, input_ln_gpu, seq_len) == 0) {
        snprintf(err, err_cap, "gpu prefill z matmul failed blk.%u", fx->layer);
        goto fail;
    }
    if (hybrid_gpu_dense_matmul_tensor(a_gpu, mf, layer->ssm_alpha,
                                       fx->hidden, fx->num_v_heads, input_ln_gpu, seq_len) == 0) {
        snprintf(err, err_cap, "gpu prefill a matmul failed blk.%u", fx->layer);
        goto fail;
    }
    if (hybrid_gpu_dense_matmul_tensor(b_gpu, mf, layer->ssm_beta,
                                       fx->hidden, fx->num_v_heads, input_ln_gpu, seq_len) == 0) {
        snprintf(err, err_cap, "gpu prefill b matmul failed blk.%u", fx->layer);
        goto fail;
    }
    if (ds4_gpu_end_commands() == 0) {
        snprintf(err, err_cap, "gpu prefill end failed blk.%u", fx->layer);
        goto fail;
    }
    if (!run_gpu_q8_qkv_split_rowwise(mf, layer, fx->hidden, fx->key_dim, fx->value_dim, input_ln_gpu, seq_len, qkv_raw)) {
        snprintf(err, err_cap, "gpu prefill qkv split failed blk.%u", fx->layer);
        goto fail;
    }
    if (ds4_gpu_tensor_read(z_gpu, 0, z_raw, z_bytes) == 0 ||
        ds4_gpu_tensor_read(a_gpu, 0, a_raw, ab_bytes) == 0 ||
        ds4_gpu_tensor_read(b_gpu, 0, b_raw, ab_bytes) == 0) {
        snprintf(err, err_cap, "gpu prefill zab readback failed blk.%u", fx->layer);
        goto fail;
    }

    reorder_head_rows_seq_f32(z, z_raw, seq_len, fx->num_v_heads, fx->head_v_dim);
    reorder_head_scalars_seq_f32(a, a_raw, seq_len, fx->num_v_heads);
    reorder_head_scalars_seq_f32(b, b_raw, seq_len, fx->num_v_heads);
    reorder_qkv_v_tokenmajor_gg_to_hf(qkv, qkv_raw, seq_len, fx->key_dim, fx->num_v_heads, fx->head_v_dim);

    memset(ls->deltanet_state, 0, (size_t)fx->num_v_heads * fx->head_k_dim * fx->head_v_dim * sizeof(float));
    for (i = 0; i < 3; ++i) memset(ls->conv_ring[i], 0, (size_t)qkv_dim * sizeof(float));
    ls->conv_ring_count = 0;

    for (t = 0; t < seq_len; ++t) {
        for (d = 0; d < qkv_dim; ++d) {
            double sum = 0.0;
            for (i = 0; i < 4; ++i) {
                int src_t = (int)t - 3 + (int)i;
                if (src_t >= 0) {
                    sum += (double)fx->conv_w[(size_t)d * 4u + i] * qkv[(size_t)src_t * qkv_dim + d];
                }
            }
            conv[(size_t)t * qkv_dim + d] = siluf_local((float)sum);
        }
    }
    for (t = 0; t < seq_len; ++t) {
        const float *cq = conv + (size_t)t * qkv_dim;
        const float *ck = cq + fx->key_dim;
        const float *cv = ck + fx->key_dim;
        for (h = 0; h < fx->num_k_heads; ++h) {
            for (d = 0; d < fx->head_k_dim; ++d) {
                const float qv = cq[h * fx->head_k_dim + d];
                const float kv = ck[h * fx->head_k_dim + d];
                uint32_t vh;
                for (vh = 0; vh < rep; ++vh) {
                    const uint32_t dst_h = h * rep + vh;
                    q[((size_t)t * fx->num_v_heads + dst_h) * fx->head_k_dim + d] = qv;
                    k[((size_t)t * fx->num_v_heads + dst_h) * fx->head_k_dim + d] = kv;
                }
            }
        }
        for (h = 0; h < fx->num_v_heads; ++h) {
            memcpy(v + ((size_t)t * fx->num_v_heads + h) * fx->head_v_dim,
                   cv + (size_t)h * fx->head_v_dim,
                   (size_t)fx->head_v_dim * sizeof(float));
            beta[(size_t)t * fx->num_v_heads + h] = sigmoidf_local(b[(size_t)t * fx->num_v_heads + h]);
            g[(size_t)t * fx->num_v_heads + h] = fx->A_log[h] * softplusf_local(a[(size_t)t * fx->num_v_heads + h] + fx->dt_bias[h]);
        }
    }
    memset(core, 0, vv_seq_elems * sizeof(float));
    for (t = 0; t < seq_len; ++t) {
        for (h = 0; h < fx->num_v_heads; ++h) {
            const size_t qbase = ((size_t)t * fx->num_v_heads + h) * fx->head_k_dim;
            const size_t vbase = ((size_t)t * fx->num_v_heads + h) * fx->head_v_dim;
            const size_t sbase = (size_t)h * fx->head_k_dim * fx->head_v_dim;
            run_deltanet_head_step(ls->deltanet_state + sbase,
                                   fx->head_k_dim,
                                   fx->head_v_dim,
                                   k + qbase,
                                   q + qbase,
                                   v + vbase,
                                   expf(g[(size_t)t * fx->num_v_heads + h]),
                                   beta[(size_t)t * fx->num_v_heads + h],
                                   sc->k_scaled,
                                   sc->q_scaled,
                                   sc->delta_tmp,
                                   core + vbase);
        }
    }
    for (t = 0; t < seq_len; ++t) {
        for (h = 0; h < fx->num_v_heads; ++h) {
            const size_t base = (size_t)t * fx->value_dim + (size_t)h * fx->head_v_dim;
            double var = 0.0;
            for (vd = 0; vd < fx->head_v_dim; ++vd) {
                const double cv = core[((size_t)t * fx->num_v_heads + h) * fx->head_v_dim + vd];
                var += cv * cv;
            }
            var /= (double)fx->head_v_dim;
            for (vd = 0; vd < fx->head_v_dim; ++vd) {
                const float cv = core[((size_t)t * fx->num_v_heads + h) * fx->head_v_dim + vd];
                out_in[base + vd] = (float)(cv / sqrt(var + 1e-6)) *
                                    fx->ssm_norm_w[vd] *
                                    siluf_local(z[(size_t)t * fx->value_dim + (size_t)h * fx->head_v_dim + vd]);
            }
        }
    }

    reorder_out_in_hf_to_gg(out_in_gg, out_in, seq_len, fx->num_v_heads, fx->head_v_dim);
    if (ds4_gpu_tensor_write(out_in_gpu, 0, out_in_gg, out_in_bytes) == 0) {
        snprintf(err, err_cap, "gpu prefill out_in write failed blk.%u", fx->layer);
        goto fail;
    }
    if (ds4_gpu_begin_commands() == 0) {
        snprintf(err, err_cap, "gpu prefill out_proj begin failed blk.%u", fx->layer);
        goto fail;
    }
    if (hybrid_gpu_dense_matmul_tensor(out_proj_gpu, mf, layer->ssm_out,
                                       fx->value_dim, fx->hidden, out_in_gpu, seq_len) == 0) {
        snprintf(err, err_cap, "gpu prefill out_proj failed blk.%u", fx->layer);
        goto fail;
    }
    if (ds4_gpu_end_commands() == 0) {
        snprintf(err, err_cap, "gpu prefill out_proj end failed blk.%u", fx->layer);
        goto fail;
    }
    if (ds4_gpu_tensor_read(out_proj_gpu, 0, out_proj, hidden_bytes) == 0) {
        snprintf(err, err_cap, "gpu prefill out_proj readback failed blk.%u", fx->layer);
        goto fail;
    }

    for (t = 0; t < seq_len; ++t) {
        float scale_in = 0.0f;
        double var = 0.0;
        const float *row_in = layer_input_seq + (size_t)t * fx->hidden;
        const float *row_proj = out_proj + (size_t)t * fx->hidden;
        float *resid = sc->resid;
        float *post_ln = sc->post_ln;
        float *mlp = sc->mlp;
        for (d = 0; d < fx->hidden; ++d) {
            resid[d] = row_in[d] + row_proj[d];
            var += (double)resid[d] * resid[d];
        }
        var /= (double)fx->hidden;
        for (d = 0; d < fx->hidden; ++d) {
            post_ln[d] = (float)(resid[d] / sqrt(var + 1e-6)) * fx->post_attn_norm_w[d];
            scale_in += post_ln[d] * fx->gate_inp_shexp[d];
            mlp[d] = 0.0f;
        }
        matvec(fx->router_w, post_ln, sc->shared, ROUTER_COUNT, fx->hidden);
        topk_softmax256(sc->shared, fx->topk, (uint32_t *)sc->beta, (float *)sc->g);
        for (i = 0; i < fx->topk; ++i) {
            const uint32_t expert_id = ((uint32_t *)sc->beta)[i];
            const float expert_w = ((float *)sc->g)[i];
            if (!ensure_expert_loaded(gf, layer, ls->experts, expert_id, fx->hidden, fx->inter, err, err_cap)) {
                goto fail;
            }
            matvec(ls->experts[expert_id].gate, post_ln, sc->gate, fx->inter, fx->hidden);
            matvec(ls->experts[expert_id].up, post_ln, sc->up, fx->inter, fx->hidden);
            for (vd = 0; vd < fx->inter; ++vd) sc->act[vd] = siluf_local(sc->gate[vd]) * sc->up[vd];
            matvec(ls->experts[expert_id].down, sc->act, sc->down, fx->hidden, fx->inter);
            for (d = 0; d < fx->hidden; ++d) mlp[d] += sc->down[d] * expert_w;
        }
        matvec(fx->gate_shexp, post_ln, sc->shared_gate, fx->inter, fx->hidden);
        matvec(fx->up_shexp, post_ln, sc->shared_up, fx->inter, fx->hidden);
        for (vd = 0; vd < fx->inter; ++vd) sc->shared_act[vd] = siluf_local(sc->shared_gate[vd]) * sc->shared_up[vd];
        matvec(fx->down_shexp, sc->shared_act, sc->shared, fx->hidden, fx->inter);
        {
            const float s = sigmoidf_local(scale_in);
            float *row_out = out_final_seq + (size_t)t * fx->hidden;
            for (d = 0; d < fx->hidden; ++d) row_out[d] = resid[d] + mlp[d] + sc->shared[d] * s;
        }
    }

    ls->conv_ring_count = (seq_len >= 3u) ? 3u : seq_len;
    for (i = 0; i < ls->conv_ring_count; ++i) {
        memcpy(ls->conv_ring[i],
               qkv + (size_t)(seq_len - 1u - i) * qkv_dim,
               (size_t)qkv_dim * sizeof(float));
    }
    for (; i < 3u; ++i) memset(ls->conv_ring[i], 0, (size_t)qkv_dim * sizeof(float));

    ds4_gpu_tensor_free(out_proj_gpu);
    ds4_gpu_tensor_free(out_in_gpu);
    ds4_gpu_tensor_free(b_gpu);
    ds4_gpu_tensor_free(a_gpu);
    ds4_gpu_tensor_free(z_gpu);
    ds4_gpu_tensor_free(input_ln_gpu);
    ds4_gpu_tensor_free(input_gpu);
    free(qkv_raw); free(qkv); free(z_raw); free(z); free(a_raw); free(a); free(b_raw); free(b);
    free(conv); free(q); free(k); free(v); free(beta); free(g); free(core); free(out_in); free(out_in_gg); free(out_proj);
    return 1;

fail:
    ds4_gpu_tensor_free(out_proj_gpu);
    ds4_gpu_tensor_free(out_in_gpu);
    ds4_gpu_tensor_free(b_gpu);
    ds4_gpu_tensor_free(a_gpu);
    ds4_gpu_tensor_free(z_gpu);
    ds4_gpu_tensor_free(input_ln_gpu);
    ds4_gpu_tensor_free(input_gpu);
    free(qkv_raw); free(qkv); free(z_raw); free(z); free(a_raw); free(a); free(b_raw); free(b);
    free(conv); free(q); free(k); free(v); free(beta); free(g); free(core); free(out_in); free(out_in_gg); free(out_proj);
    return 0;
}

static int full_gpu_env_enabled_default_on(const char *name) {
    const char *v = getenv(name);
    return !(v && strcmp(v, "0") == 0);
}

static uint64_t full_gpu_tensor_row_bytes(const qwen36_gguf_tensor *t, uint32_t cols) {
    if (!t) return 0;
    if (t->type == QWEN36_GGUF_TYPE_Q8_0) return q8_row_bytes(cols);
    if (t->type == QWEN36_GGUF_TYPE_Q2_K) return ((uint64_t)cols / 256u) * 84u;
    if (t->type == QWEN36_GGUF_TYPE_Q4_K) return ((uint64_t)cols / 256u) * 144u;
    if (t->type == QWEN36_GGUF_TYPE_Q5_K) return ((uint64_t)cols / 256u) * 176u;
    if (t->type == QWEN36_GGUF_TYPE_Q6_K) return ((uint64_t)cols / 256u) * 210u;
    if (t->type == QWEN36_GGUF_TYPE_IQ2_XXS) return ((uint64_t)cols / 256u) * 66u;
    if (t->type == QWEN36_GGUF_TYPE_IQ2_S) return ((uint64_t)cols / 256u) * 82u;
    return 0;
}

static int full_gpu_ffn_shared_q8_supported(const qwen36_contract_layer *layer) {
    return layer->ffn_gate_shexp &&
           layer->ffn_up_shexp &&
           layer->ffn_down_shexp &&
           layer->ffn_gate_shexp->type == QWEN36_GGUF_TYPE_Q8_0 &&
           layer->ffn_up_shexp->type == QWEN36_GGUF_TYPE_Q8_0 &&
           layer->ffn_down_shexp->type == QWEN36_GGUF_TYPE_Q8_0;
}

static int full_gpu_ffn_routed_q8_supported(const qwen36_contract_layer *layer) {
    return layer->ffn_gate_exps &&
           layer->ffn_up_exps &&
           layer->ffn_down_exps &&
           layer->ffn_gate_exps->type == QWEN36_GGUF_TYPE_Q8_0 &&
           layer->ffn_up_exps->type == QWEN36_GGUF_TYPE_Q8_0 &&
           layer->ffn_down_exps->type == QWEN36_GGUF_TYPE_Q8_0;
}

static int full_gpu_ffn_routed_decoded_supported(const qwen36_contract_layer *layer) {
    return layer->ffn_gate_exps &&
           layer->ffn_up_exps &&
           layer->ffn_down_exps &&
           gpu_decoded_expert_type_supported(layer->ffn_gate_exps->type) &&
           gpu_decoded_expert_type_supported(layer->ffn_up_exps->type) &&
           gpu_decoded_expert_type_supported(layer->ffn_down_exps->type);
}

static int run_gpu_ffn_from_residual_unified(
        const qwen36_gguf_file *gf,
        const mapped_file *mf,
        const qwen36_contract_layer *layer,
        uint32_t layer_index,
        full_layer_cache *cpu_cache,
        const float *shared_gate_inp_w,
        full_layer_state *state,
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
    const uint64_t expert_bytes = (uint64_t)n_tokens * topk * shared_dim * sizeof(float);
    const uint64_t gate_row_bytes = full_gpu_tensor_row_bytes(layer->ffn_gate_exps, hidden);
    const uint64_t gate_expert_bytes = gate_row_bytes * FULL_INTER;
    const uint64_t down_row_bytes = full_gpu_tensor_row_bytes(layer->ffn_down_exps, FULL_INTER);
    const uint64_t down_expert_bytes = down_row_bytes * hidden;
    ds4_gpu_tensor *residual = NULL, *post = NULL, *router_logits = NULL;
    ds4_gpu_tensor *shared_gate = NULL, *shared_up = NULL, *shared_mid = NULL, *shared_out = NULL;
    ds4_gpu_tensor *expert_gate = NULL, *expert_up = NULL, *expert_mid = NULL, *expert_down = NULL;
    ds4_gpu_tensor *routed_out = NULL, *router_selected = NULL, *router_weights = NULL;
    float *router_logits_cpu = NULL, *shared_out_cpu = NULL, *expert_down_cpu = NULL, *routed_out_cpu = NULL, *post_cpu = NULL;
    float *expert_gate_cpu = NULL, *expert_up_cpu = NULL, *expert_mid_cpu = NULL;
    uint32_t *router_selected_cpu = NULL;
    float *router_weights_cpu = NULL;
    uint32_t union_ids[ROUTER_COUNT];
    int32_t union_pos[ROUTER_COUNT];
    uint32_t n_union = 0;
    char ffn_err[256];
    int ok = 0;
    const int use_batched = full_gpu_env_enabled_default_on("QWEN36_FULL_GPU_FFN_BATCHED_EXPERTS");
    const int use_persistent =
        full_gpu_env_enabled_default_on("QWEN36_FULL_GPU_PERSISTENT_SCRATCH") &&
        state != NULL &&
        n_tokens == 1u;
    const int use_routed_moe =
        full_gpu_env_enabled_default_on("QWEN36_FULL_GPU_ROUTED_MOE") &&
        n_tokens == 1u &&
        qwen36_routed_moe_supported_types(layer->ffn_gate_exps->type,
                                          layer->ffn_down_exps->type);
    const int use_shared_gpu = full_gpu_ffn_shared_q8_supported(layer);
    const int use_q8_expert_gpu = full_gpu_ffn_routed_q8_supported(layer);
    const int use_decoded_expert_gpu =
        full_gpu_env_enabled_default_off("QWEN36_GPU_DECODED_EXPERTS") &&
        !use_routed_moe && !use_q8_expert_gpu &&
        full_gpu_ffn_routed_decoded_supported(layer);
    const int use_cpu_experts = !use_routed_moe && !use_q8_expert_gpu && !use_decoded_expert_gpu;
    const int path_audit = getenv("QWEN36_PATH_AUDIT") != NULL;
    (void)unused_timing;

    if (path_audit) {
        fprintf(stderr,
                "qwen36_path full layer=%u shared_gpu=%d routed_moe=%d q8_expert_gpu=%d decoded_expert_gpu=%d cpu_experts=%d gate=%s up=%s down=%s shared_gate=%s shared_up=%s shared_down=%s n_tokens=%u\n",
                layer_index,
                use_shared_gpu,
                use_routed_moe,
                use_q8_expert_gpu,
                use_decoded_expert_gpu,
                use_cpu_experts,
                qwen36_gguf_type_name(layer->ffn_gate_exps->type),
                qwen36_gguf_type_name(layer->ffn_up_exps->type),
                qwen36_gguf_type_name(layer->ffn_down_exps->type),
                qwen36_gguf_type_name(layer->ffn_gate_shexp->type),
                qwen36_gguf_type_name(layer->ffn_up_shexp->type),
                qwen36_gguf_type_name(layer->ffn_down_shexp->type),
                n_tokens);
        fflush(stderr);
    }

    if (use_persistent) {
        residual = state->ffn_residual_gpu;
        post = state->ffn_post_gpu;
        router_logits = state->ffn_router_logits_gpu;
        shared_gate = state->ffn_shared_gate_gpu;
        shared_up = state->ffn_shared_up_gpu;
        shared_mid = state->ffn_shared_mid_gpu;
        shared_out = state->ffn_shared_out_gpu;
        expert_gate = state->ffn_expert_gate_gpu;
        expert_up = state->ffn_expert_up_gpu;
        expert_mid = state->ffn_expert_mid_gpu;
        expert_down = state->ffn_expert_down_gpu;
        routed_out = state->ffn_routed_out_gpu;
        router_selected = state->ffn_router_selected_gpu;
        router_weights = state->ffn_router_weights_gpu;
        router_logits_cpu = state->ffn_router_logits_cpu;
        shared_out_cpu = state->ffn_shared_out_cpu;
        expert_down_cpu = state->ffn_expert_down_cpu;
        routed_out_cpu = state->ffn_routed_out_cpu;
        post_cpu = state->ffn_post_cpu;
        router_selected_cpu = state->ffn_router_selected_cpu;
        router_weights_cpu = state->ffn_router_weights_cpu;
    } else {
        residual = ds4_gpu_tensor_alloc(seq_hidden_bytes);
        post = ds4_gpu_tensor_alloc(seq_hidden_bytes);
        router_logits = ds4_gpu_tensor_alloc(router_logits_bytes);
        shared_gate = ds4_gpu_tensor_alloc(shared_bytes);
        shared_up = ds4_gpu_tensor_alloc(shared_bytes);
        shared_mid = ds4_gpu_tensor_alloc(shared_bytes);
        shared_out = ds4_gpu_tensor_alloc(seq_hidden_bytes);
        expert_gate = ds4_gpu_tensor_alloc(expert_bytes);
        expert_up = ds4_gpu_tensor_alloc(expert_bytes);
        expert_mid = ds4_gpu_tensor_alloc(expert_bytes);
        expert_down = ds4_gpu_tensor_alloc(topk * seq_hidden_bytes);
        routed_out = ds4_gpu_tensor_alloc(seq_hidden_bytes);
        router_selected = ds4_gpu_tensor_alloc((uint64_t)n_tokens * topk * sizeof(int32_t));
        router_weights = ds4_gpu_tensor_alloc((uint64_t)n_tokens * topk * sizeof(float));
        router_logits_cpu = (float *)malloc((size_t)router_logits_bytes);
        shared_out_cpu = (float *)malloc((size_t)seq_hidden_bytes);
        expert_down_cpu = (float *)malloc((size_t)seq_hidden_bytes);
        routed_out_cpu = (float *)calloc((size_t)n_tokens * hidden, sizeof(float));
        post_cpu = (float *)malloc((size_t)seq_hidden_bytes);
        router_selected_cpu = (uint32_t *)malloc((size_t)n_tokens * topk * sizeof(uint32_t));
        router_weights_cpu = (float *)malloc((size_t)n_tokens * topk * sizeof(float));
    }

    if (!residual || !post || !router_logits || !shared_gate || !shared_up || !shared_mid || !shared_out ||
        !expert_gate || !expert_up || !expert_mid || !expert_down || !routed_out || !router_selected || !router_weights ||
        !router_logits_cpu || !shared_out_cpu ||
        !expert_down_cpu || !routed_out_cpu || !post_cpu || !router_selected_cpu || !router_weights_cpu) goto cleanup;
    if (!use_shared_gpu || use_cpu_experts) {
        char ffn_err[256];
        if (!cpu_cache || !gf) goto cleanup;
        if (!ensure_full_layer_loaded(gf, layer, cpu_cache, ffn_err, sizeof(ffn_err))) goto cleanup;
    }
    if (use_cpu_experts) {
        expert_gate_cpu = (float *)malloc((size_t)shared_dim * sizeof(float));
        expert_up_cpu = (float *)malloc((size_t)shared_dim * sizeof(float));
        expert_mid_cpu = (float *)malloc((size_t)shared_dim * sizeof(float));
        if (!expert_gate_cpu || !expert_up_cpu || !expert_mid_cpu) goto cleanup;
    }

    memset(routed_out_cpu, 0, (size_t)n_tokens * hidden * sizeof(float));

    if (ds4_gpu_tensor_write(residual, 0, residual_in, seq_hidden_bytes) == 0) goto cleanup;
    if (ds4_gpu_begin_commands() == 0) goto cleanup;
    if (ds4_gpu_rms_norm_weight_rows_tensor(post, residual, mf->map, mf->size,
                                            layer->post_attn_norm->abs_offset,
                                            hidden, n_tokens, QWEN_RMS_EPS) == 0) goto cleanup;
    if (ds4_gpu_matmul_f32_tensor(router_logits, mf->map, mf->size,
                                  layer->ffn_gate_inp->abs_offset,
                                  hidden, ROUTER_COUNT, post, n_tokens) == 0) goto cleanup;
    if (use_shared_gpu) {
        if (ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(shared_gate, shared_up, shared_mid,
                                                      mf->map, mf->size,
                                                      layer->ffn_gate_shexp->abs_offset,
                                                      layer->ffn_up_shexp->abs_offset,
                                                      hidden, shared_dim, post,
                                                      QWEN_SWIGLU_CLAMP) == 0) goto cleanup;
        if (ds4_gpu_matmul_q8_0_tensor(shared_out, mf->map, mf->size,
                                       layer->ffn_down_shexp->abs_offset,
                                       shared_dim, hidden, shared_mid, n_tokens) == 0) goto cleanup;
    }
    if (ds4_gpu_end_commands() == 0) goto cleanup;

    memset(union_pos, 0xff, sizeof(union_pos));
    if (ds4_gpu_tensor_read(router_logits, 0, router_logits_cpu, router_logits_bytes) == 0) goto cleanup;
    if (ds4_gpu_tensor_read(post, 0, post_cpu, seq_hidden_bytes) == 0) goto cleanup;
    if (use_shared_gpu) {
        if (ds4_gpu_tensor_read(shared_out, 0, shared_out_cpu, seq_hidden_bytes) == 0) goto cleanup;
    } else {
        for (uint32_t t = 0; t < n_tokens; ++t) {
            const float *post_row = post_cpu + (size_t)t * hidden;
            float *shared_row = shared_out_cpu + (size_t)t * hidden;
            matvec(cpu_cache->gate_shexp_w, post_row, expert_gate_cpu, shared_dim, hidden);
            matvec(cpu_cache->up_shexp_w, post_row, expert_up_cpu, shared_dim, hidden);
            for (uint32_t i = 0; i < shared_dim; ++i) {
                expert_mid_cpu[i] = siluf_local(expert_gate_cpu[i]) * expert_up_cpu[i];
                if (expert_mid_cpu[i] > QWEN_SWIGLU_CLAMP) expert_mid_cpu[i] = QWEN_SWIGLU_CLAMP;
                if (expert_mid_cpu[i] < -QWEN_SWIGLU_CLAMP) expert_mid_cpu[i] = -QWEN_SWIGLU_CLAMP;
            }
            matvec(cpu_cache->down_shexp_w, expert_mid_cpu, shared_row, hidden, shared_dim);
        }
    }
    if (g_full_dbg_active && n_tokens == 1u) {
        memcpy(g_full_dbg_gpu.post_ln, post_cpu, sizeof(g_full_dbg_gpu.post_ln));
        memcpy(g_full_dbg_gpu.router_logits, router_logits_cpu, sizeof(g_full_dbg_gpu.router_logits));
        memcpy(g_full_dbg_gpu.shared_out, shared_out_cpu, sizeof(g_full_dbg_gpu.shared_out));
    }

    for (uint32_t t = 0; t < n_tokens; ++t) {
        topk_softmax256(router_logits_cpu + (size_t)t * ROUTER_COUNT, topk,
                        router_selected_cpu + (size_t)t * topk,
                        router_weights_cpu + (size_t)t * topk);
        if (g_full_dbg_active && n_tokens == 1u && t == 0u) {
            memcpy(g_full_dbg_gpu.router_idx, router_selected_cpu, sizeof(g_full_dbg_gpu.router_idx));
            memcpy(g_full_dbg_gpu.router_scores, router_weights_cpu, sizeof(g_full_dbg_gpu.router_scores));
        }
        for (uint32_t i = 0; i < topk; ++i) {
            const uint32_t expert_id = router_selected_cpu[(size_t)t * topk + i];
            if (union_pos[expert_id] < 0) {
                union_pos[expert_id] = (int32_t)n_union;
                union_ids[n_union++] = expert_id;
            }
        }
    }

    if (use_routed_moe) {
        if (ds4_gpu_tensor_write(router_selected, 0, router_selected_cpu, (uint64_t)n_tokens * topk * sizeof(int32_t)) == 0) goto cleanup;
        if (ds4_gpu_tensor_write(router_weights, 0, router_weights_cpu, (uint64_t)n_tokens * topk * sizeof(float)) == 0) goto cleanup;
        if (ds4_gpu_begin_commands() == 0) goto cleanup;
        if (ds4_gpu_routed_moe_batch_tensor(routed_out,
                                            expert_gate,
                                            expert_up,
                                            expert_mid,
                                            expert_down,
                                            mf->map,
                                            mf->size,
                                            layer->ffn_gate_exps->abs_offset,
                                            layer->ffn_up_exps->abs_offset,
                                            layer->ffn_down_exps->abs_offset,
                                            layer->ffn_gate_exps->type,
                                            layer->ffn_down_exps->type,
                                            gate_expert_bytes,
                                            gate_row_bytes,
                                            down_expert_bytes,
                                            down_row_bytes,
                                            hidden,
                                            shared_dim,
                                            hidden,
                                            router_selected,
                                            router_weights,
                                            ROUTER_COUNT,
                                            topk,
                                            QWEN_SWIGLU_CLAMP,
                                            post,
                                            layer_index,
                                            n_tokens,
                                            NULL) == 0) goto cleanup;
        if (ds4_gpu_end_commands() == 0) goto cleanup;
        if (ds4_gpu_tensor_read(routed_out, 0, routed_out_cpu, seq_hidden_bytes) == 0) goto cleanup;
    } else if (use_q8_expert_gpu && use_batched) {
        if (n_union > topk) goto cleanup;
        if (n_union > 0u) {
            if (ds4_gpu_begin_commands() == 0) goto cleanup;
        }
        for (uint32_t u = 0; u < n_union; ++u) {
            const uint32_t expert_id = union_ids[u];
            ds4_gpu_tensor *expert_down_slot = ds4_gpu_tensor_view(
                expert_down,
                (uint64_t)u * seq_hidden_bytes,
                seq_hidden_bytes);
            if (!expert_down_slot) goto cleanup;
            if (ds4_gpu_matmul_q8_0_pair_tensor(expert_gate, expert_up,
                                                mf->map, mf->size,
                                                layer->ffn_gate_exps->abs_offset + gate_expert_bytes * expert_id,
                                                layer->ffn_up_exps->abs_offset + gate_expert_bytes * expert_id,
                                                hidden, shared_dim, shared_dim, post, n_tokens) == 0) goto cleanup;
            if (ds4_gpu_swiglu_tensor(expert_mid, expert_gate, expert_up,
                                      n_tokens * shared_dim, QWEN_SWIGLU_CLAMP, 1.0f) == 0) goto cleanup;
            if (ds4_gpu_matmul_q8_0_tensor(expert_down_slot, mf->map, mf->size,
                                           layer->ffn_down_exps->abs_offset + down_expert_bytes * expert_id,
                                           shared_dim, hidden, expert_mid, n_tokens) == 0) goto cleanup;
            ds4_gpu_tensor_free(expert_down_slot);
        }
        if (n_union > 0u) {
            if (ds4_gpu_end_commands() == 0) goto cleanup;
        }
        for (uint32_t u = 0; u < n_union; ++u) {
            const uint32_t expert_id = union_ids[u];
            if (ds4_gpu_tensor_read(expert_down, (uint64_t)u * seq_hidden_bytes, expert_down_cpu, seq_hidden_bytes) == 0) goto cleanup;
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
    } else if (use_q8_expert_gpu) {
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
    } else if (use_decoded_expert_gpu) {
        if (n_union > topk) goto cleanup;
        for (uint32_t u = 0; u < n_union; ++u) {
            const uint32_t expert_id = union_ids[u];
            if (!ensure_full_expert_loaded(gf, layer, cpu_cache, expert_id, ffn_err, sizeof(ffn_err))) goto cleanup;
        }
        if (n_union > 0u) {
            if (ds4_gpu_begin_commands() == 0) goto cleanup;
        }
        for (uint32_t u = 0; u < n_union; ++u) {
            const uint32_t expert_id = union_ids[u];
            const expert_cache_entry *e = &cpu_cache->experts[expert_id];
            ds4_gpu_tensor *expert_down_slot = ds4_gpu_tensor_view(
                expert_down,
                (uint64_t)u * seq_hidden_bytes,
                seq_hidden_bytes);
            if (!expert_down_slot) goto cleanup;
            if (gpu_host_f32_matmul_tensor(expert_gate, e->gate,
                                           hidden, shared_dim, post, n_tokens) == 0) goto cleanup;
            if (gpu_host_f32_matmul_tensor(expert_up, e->up,
                                           hidden, shared_dim, post, n_tokens) == 0) goto cleanup;
            if (ds4_gpu_swiglu_tensor(expert_mid, expert_gate, expert_up,
                                      n_tokens * shared_dim, QWEN_SWIGLU_CLAMP, 1.0f) == 0) goto cleanup;
            if (gpu_host_f32_matmul_tensor(expert_down_slot, e->down,
                                           shared_dim, hidden, expert_mid, n_tokens) == 0) goto cleanup;
            ds4_gpu_tensor_free(expert_down_slot);
        }
        if (n_union > 0u) {
            if (ds4_gpu_end_commands() == 0) goto cleanup;
        }
        for (uint32_t u = 0; u < n_union; ++u) {
            const uint32_t expert_id = union_ids[u];
            if (ds4_gpu_tensor_read(expert_down, (uint64_t)u * seq_hidden_bytes, expert_down_cpu, seq_hidden_bytes) == 0) goto cleanup;
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
    } else {
        for (uint32_t u = 0; u < n_union; ++u) {
            const uint32_t expert_id = union_ids[u];
            if (!ensure_full_expert_loaded(gf, layer, cpu_cache, expert_id, ffn_err, sizeof(ffn_err))) goto cleanup;
        }
        for (uint32_t t = 0; t < n_tokens; ++t) {
            const float *post_row = post_cpu + (size_t)t * hidden;
            for (uint32_t i = 0; i < topk; ++i) {
                const size_t route_base = (size_t)t * topk + i;
                const uint32_t expert_id = router_selected_cpu[route_base];
                const float weight = router_weights_cpu[route_base];
                const expert_cache_entry *e = &cpu_cache->experts[expert_id];
                matvec(e->gate, post_row, expert_gate_cpu, shared_dim, hidden);
                matvec(e->up, post_row, expert_up_cpu, shared_dim, hidden);
                for (uint32_t j = 0; j < shared_dim; ++j) {
                    expert_mid_cpu[j] = siluf_local(expert_gate_cpu[j]) * expert_up_cpu[j];
                    if (expert_mid_cpu[j] > QWEN_SWIGLU_CLAMP) expert_mid_cpu[j] = QWEN_SWIGLU_CLAMP;
                    if (expert_mid_cpu[j] < -QWEN_SWIGLU_CLAMP) expert_mid_cpu[j] = -QWEN_SWIGLU_CLAMP;
                }
                matvec(e->down, expert_mid_cpu, expert_down_cpu, hidden, shared_dim);
                for (uint32_t d = 0; d < hidden; ++d) {
                    const size_t idx = (size_t)t * hidden + d;
                    routed_out_cpu[idx] += expert_down_cpu[d] * weight;
                }
            }
        }
    }
    if (g_full_dbg_active && n_tokens == 1u) {
        memcpy(g_full_dbg_gpu.routed_out, routed_out_cpu, sizeof(g_full_dbg_gpu.routed_out));
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
    if (g_full_dbg_active && n_tokens == 1u) {
        memcpy(g_full_dbg_gpu.final_out, out_seq, sizeof(g_full_dbg_gpu.final_out));
    }
    ok = 1;

cleanup:
    if (!use_persistent) {
        ds4_gpu_tensor_free(router_weights);
        ds4_gpu_tensor_free(router_selected);
        ds4_gpu_tensor_free(routed_out);
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
        free(router_weights_cpu);
        free(router_selected_cpu);
        free(post_cpu);
        free(routed_out_cpu);
        free(expert_down_cpu);
        free(shared_out_cpu);
        free(router_logits_cpu);
    }
    free(expert_mid_cpu);
    free(expert_up_cpu);
    free(expert_gate_cpu);
    return ok;
}

static int run_full_layer_step_gpu(const qwen36_gguf_file *gf,
                                   const qwen36_contract_layer *layer,
                                   uint32_t full_layer,
                                   const mapped_file *mf,
                                   full_layer_cache *cpu_cache,
                                   const full_layer_small_cache *cache,
                                   full_layer_state *state,
                                   const float *input_row,
                                   float *out_row) {
    ds4_gpu_tensor *input_gpu = NULL, *attn_in_gpu = NULL, *qg_gpu = NULL, *k_gpu = NULL, *v_gpu = NULL;
    ds4_gpu_tensor *attn_out_gpu = NULL, *proj_out_gpu = NULL, *q_cur_gpu = NULL, *gate_gpu = NULL, *heads_gpu = NULL;
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
    const int gpu_decode_attn = full_gpu_env_enabled_default_off("QWEN36_FULL_GPU_DECODE_ATTN");

    if (full_gpu_env_enabled_default_on("QWEN36_FULL_GPU_PERSISTENT_SCRATCH")) {
        if (!ensure_full_layer_decode_scratch(state, NULL, 0)) goto cleanup;
        attn_in = state->step_attn_in;
        qg = state->step_qg;
        kk = state->step_kk;
        vv = state->step_vv;
        input_gpu = state->step_input_gpu;
        attn_in_gpu = state->step_attn_in_gpu;
        qg_gpu = state->step_qg_gpu;
        k_gpu = state->step_k_gpu;
        v_gpu = state->step_v_gpu;
        attn_out_gpu = state->step_attn_out_gpu;
        proj_out_gpu = state->step_proj_out_gpu;
        q_cur_gpu = state->step_q_cur_gpu;
        gate_gpu = state->step_gate_gpu;
        heads_gpu = state->step_heads_gpu;
    } else {
        attn_in = (float *)malloc((uint64_t)FULL_HIDDEN * sizeof(float));
        qg = (float *)malloc((uint64_t)FULL_ATTN_GATE_DIM * 2u * sizeof(float));
        kk = (float *)malloc((uint64_t)FULL_NUM_KV_HEADS * FULL_HEAD_DIM * sizeof(float));
        vv = (float *)malloc((uint64_t)FULL_NUM_KV_HEADS * FULL_HEAD_DIM * sizeof(float));
        input_gpu = ds4_gpu_tensor_alloc((uint64_t)FULL_HIDDEN * sizeof(float));
        attn_in_gpu = ds4_gpu_tensor_alloc((uint64_t)FULL_HIDDEN * sizeof(float));
        qg_gpu = ds4_gpu_tensor_alloc((uint64_t)FULL_ATTN_GATE_DIM * 2u * sizeof(float));
        k_gpu = ds4_gpu_tensor_alloc((uint64_t)FULL_NUM_KV_HEADS * FULL_HEAD_DIM * sizeof(float));
        v_gpu = ds4_gpu_tensor_alloc((uint64_t)FULL_NUM_KV_HEADS * FULL_HEAD_DIM * sizeof(float));
        attn_out_gpu = ds4_gpu_tensor_alloc((uint64_t)FULL_ATTN_GATE_DIM * sizeof(float));
        proj_out_gpu = ds4_gpu_tensor_alloc((uint64_t)FULL_HIDDEN * sizeof(float));
        q_cur_gpu = ds4_gpu_tensor_alloc((uint64_t)FULL_ATTN_GATE_DIM * sizeof(float));
        gate_gpu = ds4_gpu_tensor_alloc((uint64_t)FULL_ATTN_GATE_DIM * sizeof(float));
        heads_gpu = ds4_gpu_tensor_alloc((uint64_t)FULL_ATTN_GATE_DIM * sizeof(float));
    }
    if (!attn_in || !qg || !kk || !vv || !input_gpu || !attn_in_gpu || !qg_gpu || !k_gpu || !v_gpu || !attn_out_gpu || !proj_out_gpu || !q_cur_gpu || !gate_gpu || !heads_gpu) goto cleanup;
    if (g_full_dbg_active) memset(&g_full_dbg_gpu, 0, sizeof(g_full_dbg_gpu));
    if (g_full_dbg_active) {
        memcpy(g_full_dbg_gpu.input_row, input_row, sizeof(g_full_dbg_gpu.input_row));
        if (pos > 0u) {
            memcpy(g_full_dbg_gpu.prev_out,
                   state->output_seq + (size_t)(pos - 1u) * FULL_HIDDEN,
                   sizeof(g_full_dbg_gpu.prev_out));
            memcpy(g_full_dbg_gpu.prev_k,
                   state->k_all + (size_t)(pos - 1u) * FULL_NUM_KV_HEADS * FULL_HEAD_DIM,
                   sizeof(g_full_dbg_gpu.prev_k));
            memcpy(g_full_dbg_gpu.prev_v,
                   state->v_all + (size_t)(pos - 1u) * FULL_NUM_KV_HEADS * FULL_HEAD_DIM,
                   sizeof(g_full_dbg_gpu.prev_v));
        }
    }

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
    if (ds4_gpu_tensor_read(qg_gpu, 0, qg, (uint64_t)FULL_ATTN_GATE_DIM * 2u * sizeof(float)) == 0) goto cleanup;
    for (uint32_t h = 0; h < FULL_NUM_HEADS; ++h) {
        const float *qg_head = qg + (size_t)h * (FULL_HEAD_DIM * 2u);
        rmsnorm_weight_raw(qg_head,
                           q_cur + (size_t)h * FULL_HEAD_DIM,
                           cache->q_norm_w,
                           FULL_HEAD_DIM);
        memcpy(gate_cur + (size_t)h * FULL_HEAD_DIM,
               qg_head + FULL_HEAD_DIM,
               (size_t)FULL_HEAD_DIM * sizeof(float));
    }
    for (uint32_t h = 0; h < FULL_NUM_HEADS; ++h) {
        apply_rope_one_inplace(q_cur + (size_t)h * FULL_HEAD_DIM, pos);
    }
    if (ds4_gpu_tensor_write(q_cur_gpu, 0, q_cur, (uint64_t)FULL_ATTN_GATE_DIM * sizeof(float)) == 0) goto cleanup;
    if (ds4_gpu_tensor_write(gate_gpu, 0, gate_cur, (uint64_t)FULL_ATTN_GATE_DIM * sizeof(float)) == 0) goto cleanup;
    if (g_full_dbg_active) {
        if (ds4_gpu_tensor_read(attn_in_gpu, 0, attn_in, (uint64_t)FULL_HIDDEN * sizeof(float)) == 0) goto cleanup;
        if (ds4_gpu_tensor_read(k_gpu, 0, kk, (uint64_t)FULL_NUM_KV_HEADS * FULL_HEAD_DIM * sizeof(float)) == 0) goto cleanup;
        if (ds4_gpu_tensor_read(v_gpu, 0, vv, (uint64_t)FULL_NUM_KV_HEADS * FULL_HEAD_DIM * sizeof(float)) == 0) goto cleanup;
        memcpy(g_full_dbg_gpu.attn_in, attn_in, sizeof(g_full_dbg_gpu.attn_in));
        memcpy(g_full_dbg_gpu.qg, qg, sizeof(g_full_dbg_gpu.qg));
        memcpy(g_full_dbg_gpu.kk, kk, sizeof(g_full_dbg_gpu.kk));
        memcpy(g_full_dbg_gpu.vv, vv, sizeof(g_full_dbg_gpu.vv));
    }

    if (!ensure_full_layer_state_cap(state, pos + 1u, NULL, 0)) goto cleanup;

    memset(attn_out_flat, 0, sizeof(attn_out_flat));
    {
        const uint64_t kv_row_bytes =
            (uint64_t)FULL_NUM_KV_HEADS * FULL_HEAD_DIM * sizeof(float);
        const uint64_t kv_row_offset = (uint64_t)pos * kv_row_bytes;
        if (ds4_gpu_begin_commands() == 0) goto cleanup;
        if (ds4_gpu_rms_norm_weight_rows_tensor(k_gpu, k_gpu, mf->map, mf->size,
                                                layer->attn_k_norm->abs_offset,
                                                FULL_HEAD_DIM, FULL_NUM_KV_HEADS, QWEN_RMS_EPS) == 0) goto cleanup;
        if (ds4_gpu_rope_tail_split_half_tensor(k_gpu,
                                                1u,
                                                FULL_NUM_KV_HEADS,
                                                FULL_HEAD_DIM,
                                                FULL_ROTARY_DIM,
                                                pos,
                                                0u,
                                                false,
                                                10000000.0f,
                                                1.0f,
                                                0.0f,
                                                1.0f,
                                                0.0f,
                                                0.0f) == 0) goto cleanup;
        if (ds4_gpu_tensor_copy(state->attn_k_cache_gpu, kv_row_offset, k_gpu, 0, kv_row_bytes) == 0) goto cleanup;
        if (ds4_gpu_tensor_copy(state->attn_v_cache_gpu, kv_row_offset, v_gpu, 0, kv_row_bytes) == 0) goto cleanup;
        if (gpu_decode_attn) {
            if (ds4_gpu_attention_decode_qwen_gqa_tensor(heads_gpu,
                                                         q_cur_gpu,
                                                         state->attn_k_cache_gpu,
                                                         state->attn_v_cache_gpu,
                                                         pos + 1u,
                                                         FULL_NUM_HEADS,
                                                         FULL_NUM_KV_HEADS,
                                                         FULL_HEAD_DIM) == 0) goto cleanup;
            if (ds4_gpu_sigmoid_mul_tensor(heads_gpu, heads_gpu, gate_gpu, FULL_ATTN_GATE_DIM) == 0) goto cleanup;
            if (ds4_gpu_matmul_q8_0_tensor(proj_out_gpu, mf->map, mf->size,
                                           layer->attn_output->abs_offset, FULL_ATTN_GATE_DIM, FULL_HIDDEN, heads_gpu, 1u) == 0) goto cleanup;
        }
        if (ds4_gpu_end_commands() == 0) goto cleanup;
    }

    if (gpu_decode_attn) {
        if (ds4_gpu_tensor_read(k_gpu, 0, k_cur, (uint64_t)FULL_NUM_KV_HEADS * FULL_HEAD_DIM * sizeof(float)) == 0) goto cleanup;
        if (ds4_gpu_tensor_read(v_gpu, 0, v_cur, (uint64_t)FULL_NUM_KV_HEADS * FULL_HEAD_DIM * sizeof(float)) == 0) goto cleanup;
        if (g_full_dbg_active) {
            if (ds4_gpu_tensor_read(q_cur_gpu, 0, q_cur, (uint64_t)FULL_ATTN_GATE_DIM * sizeof(float)) == 0) goto cleanup;
            if (ds4_gpu_tensor_read(heads_gpu, 0, attn_out_flat, (uint64_t)FULL_ATTN_GATE_DIM * sizeof(float)) == 0) goto cleanup;
        }
        memcpy(state->k_all + (size_t)pos * FULL_NUM_KV_HEADS * FULL_HEAD_DIM, k_cur, sizeof(k_cur));
        memcpy(state->v_all + (size_t)pos * FULL_NUM_KV_HEADS * FULL_HEAD_DIM, v_cur, sizeof(v_cur));
        if (g_full_dbg_active) {
            memcpy(g_full_dbg_gpu.q_cur, q_cur, sizeof(g_full_dbg_gpu.q_cur));
            memcpy(g_full_dbg_gpu.k_cur, k_cur, sizeof(g_full_dbg_gpu.k_cur));
            memcpy(g_full_dbg_gpu.v_cur, v_cur, sizeof(g_full_dbg_gpu.v_cur));
        }
    } else {
        if (ds4_gpu_tensor_read(q_cur_gpu, 0, q_cur, (uint64_t)FULL_ATTN_GATE_DIM * sizeof(float)) == 0) goto cleanup;
        if (ds4_gpu_tensor_read(gate_gpu, 0, gate_cur, (uint64_t)FULL_ATTN_GATE_DIM * sizeof(float)) == 0) goto cleanup;
        if (ds4_gpu_tensor_read(k_gpu, 0, k_cur, (uint64_t)FULL_NUM_KV_HEADS * FULL_HEAD_DIM * sizeof(float)) == 0) goto cleanup;
        if (ds4_gpu_tensor_read(v_gpu, 0, v_cur, (uint64_t)FULL_NUM_KV_HEADS * FULL_HEAD_DIM * sizeof(float)) == 0) goto cleanup;
        memcpy(state->k_all + (size_t)pos * FULL_NUM_KV_HEADS * FULL_HEAD_DIM, k_cur, sizeof(k_cur));
        memcpy(state->v_all + (size_t)pos * FULL_NUM_KV_HEADS * FULL_HEAD_DIM, v_cur, sizeof(v_cur));
        if (g_full_dbg_active) {
            memcpy(g_full_dbg_gpu.q_cur, q_cur, sizeof(g_full_dbg_gpu.q_cur));
            memcpy(g_full_dbg_gpu.k_cur, k_cur, sizeof(g_full_dbg_gpu.k_cur));
            memcpy(g_full_dbg_gpu.v_cur, v_cur, sizeof(g_full_dbg_gpu.v_cur));
        }
        for (uint32_t h = 0; h < FULL_NUM_HEADS; ++h) {
            const uint32_t kvh = h / (FULL_NUM_HEADS / FULL_NUM_KV_HEADS);
            float *out = attn_out_flat + (size_t)h * FULL_HEAD_DIM;
            const float *qv = q_cur + (size_t)h * FULL_HEAD_DIM;
            double scores[8192];
            double maxv = -1e300, sum = 0.0;
            for (uint32_t s = 0; s <= pos; ++s) {
                const float *kv = state->k_all + ((size_t)s * FULL_NUM_KV_HEADS + kvh) * FULL_HEAD_DIM;
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
                const float *vvh = state->v_all + ((size_t)s * FULL_NUM_KV_HEADS + kvh) * FULL_HEAD_DIM;
                for (uint32_t d = 0; d < FULL_HEAD_DIM; ++d) out[d] += w * vvh[d];
            }
            for (uint32_t d = 0; d < FULL_HEAD_DIM; ++d) out[d] *= sigmoidf_local(gate_cur[(size_t)h * FULL_HEAD_DIM + d]);
        }
    }
    if (g_full_dbg_active) memcpy(g_full_dbg_gpu.attn_out_flat, attn_out_flat, sizeof(g_full_dbg_gpu.attn_out_flat));
    if (!gpu_decode_attn) {
        if (ds4_gpu_tensor_write(attn_out_gpu, 0, attn_out_flat, (uint64_t)FULL_ATTN_GATE_DIM * sizeof(float)) == 0) goto cleanup;
        if (ds4_gpu_begin_commands() == 0) goto cleanup;
        if (ds4_gpu_matmul_q8_0_tensor(proj_out_gpu, mf->map, mf->size,
                                       layer->attn_output->abs_offset, FULL_ATTN_GATE_DIM, FULL_HIDDEN, attn_out_gpu, 1u) == 0) goto cleanup;
        if (ds4_gpu_end_commands() == 0) goto cleanup;
    }
    if (ds4_gpu_tensor_read(proj_out_gpu, 0, proj_out, (uint64_t)FULL_HIDDEN * sizeof(float)) == 0) goto cleanup;
    if (g_full_dbg_active) memcpy(g_full_dbg_gpu.proj_out, proj_out, sizeof(g_full_dbg_gpu.proj_out));

    for (uint32_t d = 0; d < FULL_HIDDEN; ++d) residual[d] = input_row[d] + proj_out[d];
    if (g_full_dbg_active) memcpy(g_full_dbg_gpu.residual, residual, sizeof(g_full_dbg_gpu.residual));
    if (!run_gpu_ffn_from_residual_unified(gf, mf, layer, full_layer, cpu_cache, cache->shared_gate_inp_w, state, 1u, residual, out_row, NULL)) goto cleanup;

    memcpy(state->output_seq + (size_t)pos * FULL_HIDDEN, out_row, (size_t)FULL_HIDDEN * sizeof(float));
    state->seq_len = pos + 1u;
    ok = 1;

cleanup:
    if (!full_gpu_env_enabled_default_on("QWEN36_FULL_GPU_PERSISTENT_SCRATCH")) {
        ds4_gpu_tensor_free(proj_out_gpu);
        ds4_gpu_tensor_free(attn_out_gpu);
        ds4_gpu_tensor_free(v_gpu);
        ds4_gpu_tensor_free(k_gpu);
        ds4_gpu_tensor_free(qg_gpu);
        ds4_gpu_tensor_free(attn_in_gpu);
        ds4_gpu_tensor_free(input_gpu);
        ds4_gpu_tensor_free(q_cur_gpu);
        ds4_gpu_tensor_free(gate_gpu);
        ds4_gpu_tensor_free(heads_gpu);
        free(attn_in);
        free(qg);
        free(kk);
        free(vv);
    }
    return ok;
}

static int run_full_layer_prefill_gpu(uint32_t full_layer,
                                      const qwen36_gguf_file *gf,
                                      const qwen36_contract_layer *layer,
                                      const mapped_file *mf,
                                      full_layer_cache *cpu_cache,
                                      const full_layer_small_cache *cache,
                                      full_layer_state *state,
                                      const float *input_seq,
                                      uint32_t seq_len) {
    float row_out[FULL_HIDDEN];
    free_full_layer_state(state);
    memset(state, 0, sizeof(*state));
    g_qwen36_full_gpu_prefill_mode = 1;
    for (uint32_t t = 0; t < seq_len; ++t) {
        if (!run_full_layer_step_gpu(gf, layer, full_layer, mf, cpu_cache, cache, state, input_seq + (size_t)t * FULL_HIDDEN, row_out)) {
            g_qwen36_full_gpu_prefill_mode = 0;
            return 0;
        }
    }
    g_qwen36_full_gpu_prefill_mode = 0;
    return 1;
}

static int run_hybrid_layer_prefill_gpuproj(const live_fixture *fx,
                                            const qwen36_gguf_file *gf,
                                            const qwen36_contract_layer *layer,
                                            const mapped_file *mf,
                                            layer_runtime *ls,
                                            uint32_t seq_len,
                                            const float *layer_input_seq,
                                            float *out_final_seq,
                                            char *err,
                                            size_t err_cap) {
    uint32_t i;
    layer_runtime cpu_ref_ls;
    float *cpu_out_row = NULL;
    int do_prefill_dbg = 0;
    static int dbg_init = 0;
    static int dbg_enabled = 0;
    static int dbg_all_layers = 0;
    static uint32_t dbg_layer = 0;
    static int dbg_step_start = -1;
    static int dbg_step_end = -1;
    memset(&cpu_ref_ls, 0, sizeof(cpu_ref_ls));
    if (!dbg_init) {
        const char *env = getenv("QWEN36_DBG_HYBRID_LAYER");
        const char *env_step_start = getenv("QWEN36_DBG_HYBRID_STEP_START");
        const char *env_step_end = getenv("QWEN36_DBG_HYBRID_STEP_END");
        if (env && env[0]) {
            if (strcmp(env, "all") == 0) {
                dbg_enabled = 1;
                dbg_all_layers = 1;
            } else {
                char *endp = NULL;
                unsigned long v = strtoul(env, &endp, 10);
                if (endp && *endp == '\0') {
                    dbg_enabled = 1;
                    dbg_layer = (uint32_t)v;
                }
            }
        }
        if (env_step_start && env_step_start[0]) dbg_step_start = atoi(env_step_start);
        if (env_step_end && env_step_end[0]) dbg_step_end = atoi(env_step_end);
        dbg_init = 1;
    }
    do_prefill_dbg = dbg_enabled && (dbg_all_layers || fx->layer == dbg_layer);
    if (ls->deltanet_state) {
        memset(ls->deltanet_state, 0, (size_t)fx->num_v_heads * fx->head_k_dim * fx->head_v_dim * sizeof(float));
    }
    for (i = 0; i < 3; ++i) {
        if (ls->conv_ring[i]) memset(ls->conv_ring[i], 0, (size_t)(fx->key_dim * 2u + fx->value_dim) * sizeof(float));
    }
    ls->conv_ring_count = 0;
    if (do_prefill_dbg) {
        cpu_out_row = (float *)malloc((size_t)fx->hidden * sizeof(float));
        if (!cpu_out_row) {
            snprintf(err, err_cap, "oom hybrid prefill cpu row blk.%u", fx->layer);
            return 0;
        }
        if (!clone_layer_runtime_for_debug(fx, ls, &cpu_ref_ls, err, err_cap)) {
            free(cpu_out_row);
            return 0;
        }
    }
    for (i = 0; i < seq_len; ++i) {
        int ok_step = run_hybrid_layer_step_gpuproj(fx, gf, layer, mf, ls,
                                                    layer_input_seq + (size_t)i * fx->hidden,
                                                    out_final_seq + (size_t)i * fx->hidden,
                                                    err, err_cap);
        if (!ok_step) {
            free(cpu_out_row);
            free_layer_runtime_debug(&cpu_ref_ls);
            return 0;
        }
        if (do_prefill_dbg
            && (dbg_step_start < 0 || (int)i >= dbg_step_start)
            && (dbg_step_end < 0 || (int)i <= dbg_step_end)) {
            if (!run_step_layer_dynamic(fx, gf, layer, &cpu_ref_ls,
                                        layer_input_seq + (size_t)i * fx->hidden,
                                        cpu_out_row, err, err_cap)) {
                free(cpu_out_row);
                free_layer_runtime_debug(&cpu_ref_ls);
                return 0;
            }
            fprintf(stderr, "=== DBG_HYBRID_PREFILL[%u] row=%u GPU_STATE ===\n", fx->layer, i);
            fflush(stderr);
            hybrid_dbg_report_ring(fx->layer, i, "GPU_PREFILL", fx, ls, NULL);
            fprintf(stderr, "=== DBG_HYBRID_PREFILL[%u] row=%u CPU_STATE ===\n", fx->layer, i);
            fflush(stderr);
            hybrid_dbg_report_ring(fx->layer, i, "CPU_PREFILL", fx, &cpu_ref_ls, NULL);
        }
    }
    free(cpu_out_row);
    free_layer_runtime_debug(&cpu_ref_ls);
    return 1;
}
