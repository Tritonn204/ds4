#include "ds4_gpu.h"
extern "C" {
#include "qwen36_35a3b_q8.h"
}

#include <hip/hip_runtime.h>

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

#define ROUTER_COUNT 256

typedef struct mapped_file {
    int fd;
    void *map;
    uint64_t size;
} mapped_file;

typedef struct native_hybrid_state {
    float *deltanet_state;
    ds4_gpu_tensor *conv_ring_gpu[3];
    uint32_t conv_ring_count;
} native_hybrid_state;

#include "rocm/qwen36_native_hybrid_kernels.cuh"

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

static int native_hybrid_state_init(native_hybrid_state *st,
                                    uint32_t num_v_heads, uint32_t head_k_dim, uint32_t head_v_dim,
                                    uint32_t qkv_dim) {
    memset(st, 0, sizeof(*st));
    st->deltanet_state = (float *)calloc((size_t)num_v_heads * head_k_dim * head_v_dim, sizeof(float));
    st->conv_ring_gpu[0] = ds4_gpu_tensor_alloc((uint64_t)qkv_dim * sizeof(float));
    st->conv_ring_gpu[1] = ds4_gpu_tensor_alloc((uint64_t)qkv_dim * sizeof(float));
    st->conv_ring_gpu[2] = ds4_gpu_tensor_alloc((uint64_t)qkv_dim * sizeof(float));
    st->conv_ring_count = 0;
    if (!st->deltanet_state || !st->conv_ring_gpu[0] || !st->conv_ring_gpu[1] || !st->conv_ring_gpu[2]) {
        free(st->deltanet_state);
        ds4_gpu_tensor_free(st->conv_ring_gpu[0]);
        ds4_gpu_tensor_free(st->conv_ring_gpu[1]);
        ds4_gpu_tensor_free(st->conv_ring_gpu[2]);
        memset(st, 0, sizeof(*st));
        return 0;
    }
    /* zero-initialize ring tensors */
    ds4_gpu_tensor_fill_f32(st->conv_ring_gpu[0], 0.0f, qkv_dim);
    ds4_gpu_tensor_fill_f32(st->conv_ring_gpu[1], 0.0f, qkv_dim);
    ds4_gpu_tensor_fill_f32(st->conv_ring_gpu[2], 0.0f, qkv_dim);
    return 1;
}

static void native_hybrid_state_free(native_hybrid_state *st) {
    if (!st) return;
    free(st->deltanet_state);
    ds4_gpu_tensor_free(st->conv_ring_gpu[0]);
    ds4_gpu_tensor_free(st->conv_ring_gpu[1]);
    ds4_gpu_tensor_free(st->conv_ring_gpu[2]);
    memset(st, 0, sizeof(*st));
}

/*
 * run_native_hybrid_layer_gpu
 *
 * Implements a single decode-step for a Qwen3.6-35B-A3B hybrid (SSM) layer
 * using GPU projections wherever possible.  ALL GPU kernel launches are
 * funneled through ONE ds4_gpu_begin_commands … ds4_gpu_end_commands pair.
 *
 * CPU-only logic (reorder, conv-3, deltanet-step, rms-norm, silu-gate,
 * topk-softmax) is bridged via stub functions in qwen36_native_hybrid_kernels.cuh
 * that read/write GPU tensors inside the batch.  Those stubs will be replaced
 * by native GPU kernels later.
 *
 * Parameters:
 *   mf          – mmap'd GGUF model file (map + size)
 *   layer       – tensor bindings for this layer (abs_offsets)
 *   st          – persistent state (deltanet SSM state + conv ring)
 *   hidden      – 2048
 *   num_v_heads – 4
 *   num_k_heads – 1
 *   head_k_dim  – 128
 *   head_v_dim  – 64
 *   key_dim     – num_k_heads * head_k_dim = 128
 *   value_dim   – num_v_heads * head_v_dim = 256
 *   topk        – number of routed experts (e.g. 8)
 *   inter       – FFN intermediate dim (e.g. 512)
 *   layer_input – [hidden] input row (CPU)
 *   out_row     – [hidden] output row (CPU, written by this function)
 *   err         – error buffer
 *   err_cap     – error buffer capacity
 *
 * Returns: 1 on success, 0 on failure (err filled).
 */
static int run_native_hybrid_layer_gpu(const mapped_file *mf,
                                       const qwen36_35a3b_q8_layer *layer,
                                       native_hybrid_state *st,
                                       uint32_t hidden,
                                       uint32_t num_v_heads,
                                       uint32_t num_k_heads,
                                       uint32_t head_k_dim,
                                       uint32_t head_v_dim,
                                       uint32_t key_dim,
                                       uint32_t value_dim,
                                       uint32_t topk,
                                       uint32_t inter,
                                       const float *layer_input,
                                       float *out_row,
                                       char *err,
                                       size_t err_cap) {
    const uint32_t qkv_dim = key_dim * 2u + value_dim;
    const uint64_t hidden_bytes = (uint64_t)hidden * sizeof(float);
    const uint64_t value_dim_bytes = (uint64_t)value_dim * sizeof(float);
    const uint64_t key_dim_bytes = (uint64_t)key_dim * sizeof(float);
    const uint64_t qkv_dim_bytes = (uint64_t)qkv_dim * sizeof(float);
    const uint64_t ab_bytes = (uint64_t)num_v_heads * sizeof(float);
    const uint64_t inter_bytes = (uint64_t)inter * sizeof(float);
    const uint64_t router_logits_bytes = (uint64_t)ROUTER_COUNT * sizeof(float);
    const uint64_t gate_expert_bytes = (uint64_t)((uint64_t)inter * hidden / 32u) * 34u;
    const uint64_t down_expert_bytes = (uint64_t)((uint64_t)hidden * inter / 32u) * 34u;
    const uint64_t qkv_row_bytes = (uint64_t)(hidden / 32u) * 34u;
    const uint64_t q_off = layer->attn_qkv->abs_offset;
    const uint64_t k_off = q_off + (uint64_t)key_dim * qkv_row_bytes;
    const uint64_t v_off = k_off + (uint64_t)key_dim * qkv_row_bytes;

    ds4_gpu_tensor *input_gpu = NULL, *input_ln_gpu = NULL;
    ds4_gpu_tensor *z_gpu = NULL, *a_gpu = NULL, *b_gpu = NULL;
    ds4_gpu_tensor *qkv_gpu = NULL, *q_gpu = NULL, *k_gpu = NULL, *v_gpu = NULL;
    ds4_gpu_tensor *conv_gpu = NULL, *core_gpu = NULL;
    ds4_gpu_tensor *beta_gpu = NULL, *g_gpu = NULL;
    ds4_gpu_tensor *out_in_gpu = NULL, *out_proj_gpu = NULL;
    ds4_gpu_tensor *resid_gpu = NULL, *post_ln_gpu = NULL;
    ds4_gpu_tensor *router_logits_gpu = NULL;
    ds4_gpu_tensor *shared_gate_gpu = NULL, *shared_up_gpu = NULL, *shared_mid_gpu = NULL, *shared_out_gpu = NULL;
    ds4_gpu_tensor *state_gpu = NULL;
    ds4_gpu_tensor *expert_gate_gpu[8] = {NULL};
    ds4_gpu_tensor *expert_up_gpu[8] = {NULL};
    ds4_gpu_tensor *expert_mid_gpu[8] = {NULL};
    ds4_gpu_tensor *expert_down_gpu[8] = {NULL};
    ds4_gpu_tensor *conv_w_gpu = NULL, *A_log_gpu = NULL, *dt_bias_gpu = NULL;

    float *resid = NULL, *post_ln = NULL, *shared = NULL;
    float *conv_w = NULL, *A_log = NULL, *dt_bias = NULL;
    float *ssm_norm_w = NULL, *post_attn_norm_w = NULL, *gate_inp_shexp_w = NULL;
    uint32_t router_idx[256];
    float router_scores[256];
    uint32_t i, d;
    int ok = 0;
    int ex_flags[8] = {0};
    const uint8_t *base_map = (const uint8_t *)mf->map;

    if (topk > 8) { snprintf(err, err_cap, "topk %u > max 8", topk); goto cleanup; }

    conv_w = (float *)malloc((size_t)qkv_dim * 4u * sizeof(float));
    A_log = (float *)malloc((size_t)num_v_heads * sizeof(float));
    dt_bias = (float *)malloc((size_t)num_v_heads * sizeof(float));
    ssm_norm_w = (float *)malloc((size_t)head_v_dim * sizeof(float));
    post_attn_norm_w = (float *)malloc(hidden_bytes);
    gate_inp_shexp_w = (float *)malloc(hidden_bytes);
    if (!conv_w || !A_log || !dt_bias || !ssm_norm_w || !post_attn_norm_w || !gate_inp_shexp_w) {
        snprintf(err, err_cap, "weight malloc failed"); goto cleanup;
    }
    memcpy(conv_w, base_map + layer->ssm_conv1d->abs_offset, (size_t)qkv_dim * 4u * sizeof(float));
    memcpy(A_log, base_map + layer->ssm_a->abs_offset, (size_t)num_v_heads * sizeof(float));
    memcpy(dt_bias, base_map + layer->ssm_dt_bias->abs_offset, (size_t)num_v_heads * sizeof(float));
    memcpy(ssm_norm_w, base_map + layer->ssm_norm->abs_offset, (size_t)head_v_dim * sizeof(float));
    memcpy(post_attn_norm_w, base_map + layer->post_attn_norm->abs_offset, hidden_bytes);
    memcpy(gate_inp_shexp_w, base_map + layer->ffn_gate_inp_shexp->abs_offset, hidden_bytes);

    /* upload conv_w, A_log, dt_bias to GPU tensors before begin_commands */
    conv_w_gpu = ds4_gpu_tensor_alloc((uint64_t)qkv_dim * 4u * sizeof(float));
    A_log_gpu = ds4_gpu_tensor_alloc((uint64_t)num_v_heads * sizeof(float));
    dt_bias_gpu = ds4_gpu_tensor_alloc((uint64_t)num_v_heads * sizeof(float));
    if (!conv_w_gpu || !A_log_gpu || !dt_bias_gpu) {
        snprintf(err, err_cap, "gpu weight tensor alloc failed"); goto cleanup;
    }
    if (ds4_gpu_tensor_write(conv_w_gpu, 0, conv_w, (uint64_t)qkv_dim * 4u * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(A_log_gpu, 0, A_log, (uint64_t)num_v_heads * sizeof(float)) == 0 ||
        ds4_gpu_tensor_write(dt_bias_gpu, 0, dt_bias, (uint64_t)num_v_heads * sizeof(float)) == 0) {
        snprintf(err, err_cap, "gpu weight write failed"); goto cleanup;
    }

    resid = (float *)malloc(hidden_bytes);
    post_ln = (float *)malloc(hidden_bytes);
    shared = (float *)malloc(hidden_bytes);
    if (!resid || !post_ln || !shared) { snprintf(err, err_cap, "cpu malloc failed"); goto cleanup; }

    input_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    input_ln_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    z_gpu = ds4_gpu_tensor_alloc(value_dim_bytes);
    a_gpu = ds4_gpu_tensor_alloc(ab_bytes);
    b_gpu = ds4_gpu_tensor_alloc(ab_bytes);
    qkv_gpu = ds4_gpu_tensor_alloc(qkv_dim_bytes);
    q_gpu = ds4_gpu_tensor_alloc((uint64_t)num_v_heads * head_k_dim * sizeof(float));
    k_gpu = ds4_gpu_tensor_alloc((uint64_t)num_v_heads * head_k_dim * sizeof(float));
    v_gpu = ds4_gpu_tensor_alloc(value_dim_bytes);
    conv_gpu = ds4_gpu_tensor_alloc(qkv_dim_bytes);
    core_gpu = ds4_gpu_tensor_alloc((uint64_t)num_v_heads * head_v_dim * sizeof(float));
    beta_gpu = ds4_gpu_tensor_alloc(ab_bytes);
    g_gpu = ds4_gpu_tensor_alloc(ab_bytes);
    out_in_gpu = ds4_gpu_tensor_alloc(value_dim_bytes);
    out_proj_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    resid_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    post_ln_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    router_logits_gpu = ds4_gpu_tensor_alloc(router_logits_bytes);
    shared_gate_gpu = ds4_gpu_tensor_alloc(inter_bytes);
    shared_up_gpu = ds4_gpu_tensor_alloc(inter_bytes);
    shared_mid_gpu = ds4_gpu_tensor_alloc(inter_bytes);
    shared_out_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    state_gpu = ds4_gpu_tensor_alloc((uint64_t)num_v_heads * head_k_dim * head_v_dim * sizeof(float));

    if (!input_gpu || !input_ln_gpu || !z_gpu || !a_gpu || !b_gpu ||
        !qkv_gpu || !q_gpu || !k_gpu || !v_gpu || !conv_gpu || !core_gpu ||
        !beta_gpu || !g_gpu || !out_in_gpu || !out_proj_gpu ||
        !resid_gpu || !post_ln_gpu || !router_logits_gpu ||
        !shared_gate_gpu || !shared_up_gpu || !shared_mid_gpu || !shared_out_gpu ||
        !state_gpu) {
        snprintf(err, err_cap, "gpu tensor alloc failed");
        goto cleanup;
    }

    if (ds4_gpu_tensor_write(state_gpu, 0, st->deltanet_state,
                             (uint64_t)num_v_heads * head_k_dim * head_v_dim * sizeof(float)) == 0) {
        snprintf(err, err_cap, "gpu state write failed");
        goto cleanup;
    }

    /* ---- SINGLE COMMAND BATCH ---- */
    if (ds4_gpu_begin_commands() == 0) {
        snprintf(err, err_cap, "gpu begin failed");
        goto cleanup;
    }

    if (ds4_gpu_tensor_write(input_gpu, 0, layer_input, hidden_bytes) == 0) {
        snprintf(err, err_cap, "gpu input write failed");
        goto cleanup;
    }

    /* 1. RMS norm */
    if (ds4_gpu_rms_norm_weight_rows_tensor(input_ln_gpu, input_gpu,
                                            mf->map, mf->size,
                                            layer->attn_norm->abs_offset,
                                            hidden, 1u, 1e-6f) == 0) {
        snprintf(err, err_cap, "gpu rms norm failed");
        goto cleanup;
    }

    /* 2. z, a, b projections */
    if (ds4_gpu_matmul_q8_0_tensor(z_gpu, mf->map, mf->size,
                                    layer->attn_gate->abs_offset,
                                    hidden, value_dim, input_ln_gpu, 1u) == 0) {
        snprintf(err, err_cap, "gpu z matmul failed");
        goto cleanup;
    }
    if (ds4_gpu_matmul_q8_0_tensor(a_gpu, mf->map, mf->size,
                                    layer->ssm_alpha->abs_offset,
                                    hidden, num_v_heads, input_ln_gpu, 1u) == 0) {
        snprintf(err, err_cap, "gpu a matmul failed");
        goto cleanup;
    }
    if (ds4_gpu_matmul_q8_0_tensor(b_gpu, mf->map, mf->size,
                                    layer->ssm_beta->abs_offset,
                                    hidden, num_v_heads, input_ln_gpu, 1u) == 0) {
        snprintf(err, err_cap, "gpu b matmul failed");
        goto cleanup;
    }

    /* 3. q, k, v from merged QKV weight */
    if (ds4_gpu_matmul_q8_0_tensor(qkv_gpu, mf->map, mf->size, q_off,
                                    hidden, qkv_dim, input_ln_gpu, 1u) == 0) {
        snprintf(err, err_cap, "gpu qkv matmul failed");
        goto cleanup;
    }

    /* 4. GPU: reorder → conv3 → head-split → beta/g   (reads z,a,b,qkv; writes conv,q,k,v,beta,g) */
    if (ds4_gpu_fused_reorder_conv_tensor(conv_gpu, q_gpu, k_gpu, v_gpu, beta_gpu, g_gpu,
                                          z_gpu, a_gpu, b_gpu, qkv_gpu,
                                          st->conv_ring_gpu, &st->conv_ring_count,
                                          num_v_heads, num_k_heads, head_k_dim, head_v_dim, key_dim,
                                          conv_w_gpu, A_log_gpu, dt_bias_gpu) == 0) {
        snprintf(err, err_cap, "gpu reorder_conv failed");
        goto cleanup;
    }

    /* 5. stub: deltanet SSM step   (reads k,q,v,beta,g,state; writes state,core) */
    if (ds4_gpu_deltanet_step_tensor(state_gpu, core_gpu,
                                     k_gpu, q_gpu, v_gpu, g_gpu, beta_gpu,
                                     num_v_heads, head_k_dim, head_v_dim) == 0) {
        snprintf(err, err_cap, "stub deltanet step failed");
        goto cleanup;
    }

    /* 6. stub: per-head rms-norm → silu-gate → HF→GG reorder */
    if (ds4_gpu_state_norm_silu_reorder_tensor(out_in_gpu, core_gpu, z_gpu,
                                               num_v_heads, head_v_dim,
                                               ssm_norm_w) == 0) {
        snprintf(err, err_cap, "stub norm_silu_reorder failed");
        goto cleanup;
    }

    /* 7. out_proj matmul */
    if (ds4_gpu_matmul_q8_0_tensor(out_proj_gpu, mf->map, mf->size,
                                    layer->ssm_out->abs_offset,
                                    value_dim, hidden, out_in_gpu, 1u) == 0) {
        snprintf(err, err_cap, "gpu out_proj failed");
        goto cleanup;
    }

    /* 8. stub: residual + post_ln norm */
    if (ds4_gpu_resid_post_ln_tensor(resid_gpu, post_ln_gpu,
                                     input_gpu, out_proj_gpu,
                                     hidden, post_attn_norm_w) == 0) {
        snprintf(err, err_cap, "stub resid_post_ln failed");
        goto cleanup;
    }

    /* 9. router matmul (f32 weight) */
    if (ds4_gpu_matmul_f32_tensor(router_logits_gpu, mf->map, mf->size,
                                  layer->ffn_gate_inp->abs_offset,
                                  hidden, ROUTER_COUNT, post_ln_gpu, 1u) == 0) {
        snprintf(err, err_cap, "gpu router failed");
        goto cleanup;
    }

    /* 10. shared gate/up + swiglu (fused) */
    if (ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(shared_gate_gpu, shared_up_gpu, shared_mid_gpu,
                                                   mf->map, mf->size,
                                                   layer->ffn_gate_shexp->abs_offset,
                                                   layer->ffn_up_shexp->abs_offset,
                                                   hidden, inter, post_ln_gpu,
                                                   80.0f) == 0) {
        snprintf(err, err_cap, "gpu shared gate/up/swiglu failed");
        goto cleanup;
    }

    /* 11. shared down */
    if (ds4_gpu_matmul_q8_0_tensor(shared_out_gpu, mf->map, mf->size,
                                    layer->ffn_down_shexp->abs_offset,
                                    inter, hidden, shared_mid_gpu, 1u) == 0) {
        snprintf(err, err_cap, "gpu shared down failed");
        goto cleanup;
    }

    /* 12. stub: topk-softmax on router logits   (reads router_logits → CPU → sel idx + scores) */
    if (ds4_gpu_topk_softmax_tensor(router_idx, router_scores,
                                    router_logits_gpu, ROUTER_COUNT, topk) == 0) {
        snprintf(err, err_cap, "stub topk_softmax failed");
        goto cleanup;
    }

    /* 13. expert dispatch: all selected experts in the same batch */
    for (i = 0; i < topk; ++i) {
        const uint32_t eid = router_idx[i];
        expert_gate_gpu[i] = ds4_gpu_tensor_alloc(inter_bytes);
        expert_up_gpu[i] = ds4_gpu_tensor_alloc(inter_bytes);
        expert_mid_gpu[i] = ds4_gpu_tensor_alloc(inter_bytes);
        expert_down_gpu[i] = ds4_gpu_tensor_alloc(hidden_bytes);
        if (!expert_gate_gpu[i] || !expert_up_gpu[i] || !expert_mid_gpu[i] || !expert_down_gpu[i]) {
            snprintf(err, err_cap, "gpu expert tensor alloc failed for eid=%u", eid);
            goto cleanup;
        }
        ex_flags[i] = 1;
        if (ds4_gpu_matmul_q8_0_pair_tensor(expert_gate_gpu[i], expert_up_gpu[i],
                                             mf->map, mf->size,
                                             layer->ffn_gate_exps->abs_offset + gate_expert_bytes * eid,
                                             layer->ffn_up_exps->abs_offset + gate_expert_bytes * eid,
                                             hidden, inter, inter, post_ln_gpu, 1u) == 0) {
            snprintf(err, err_cap, "gpu expert gate/up failed eid=%u", eid);
            goto cleanup;
        }
        if (ds4_gpu_swiglu_tensor(expert_mid_gpu[i], expert_gate_gpu[i], expert_up_gpu[i],
                                   inter, 80.0f, 1.0f) == 0) {
            snprintf(err, err_cap, "gpu expert swiglu failed eid=%u", eid);
            goto cleanup;
        }
        if (ds4_gpu_matmul_q8_0_tensor(expert_down_gpu[i], mf->map, mf->size,
                                        layer->ffn_down_exps->abs_offset + down_expert_bytes * eid,
                                        inter, hidden, expert_mid_gpu[i], 1u) == 0) {
            snprintf(err, err_cap, "gpu expert down failed eid=%u", eid);
            goto cleanup;
        }
    }

    /* ---- END OF SINGLE COMMAND BATCH ---- */
    if (ds4_gpu_end_commands() == 0) {
        snprintf(err, err_cap, "gpu end failed");
        goto cleanup;
    }

    /* Readbacks */
    if (ds4_gpu_tensor_read(resid_gpu, 0, resid, hidden_bytes) == 0 ||
        ds4_gpu_tensor_read(post_ln_gpu, 0, post_ln, hidden_bytes) == 0 ||
        ds4_gpu_tensor_read(shared_out_gpu, 0, shared, hidden_bytes) == 0) {
        snprintf(err, err_cap, "gpu readback failed");
        goto cleanup;
    }

    {
        const uint64_t state_bytes = (uint64_t)num_v_heads * head_k_dim * head_v_dim * sizeof(float);
        float *state_cpu = (float *)malloc(state_bytes);
        if (state_cpu && ds4_gpu_tensor_read(state_gpu, 0, state_cpu, state_bytes) != 0) {
            memcpy(st->deltanet_state, state_cpu, state_bytes);
        }
        free(state_cpu);
    }

    /* CPU: final combine */
    {
        float scale_in = 0.0f;
        for (d = 0; d < hidden; ++d) scale_in += post_ln[d] * gate_inp_shexp_w[d];

        for (d = 0; d < hidden; ++d) out_row[d] = resid[d];
        for (i = 0; i < topk; ++i) {
            float *expert_down = (float *)malloc(hidden_bytes);
            if (expert_down && ds4_gpu_tensor_read(expert_down_gpu[i], 0, expert_down, hidden_bytes) != 0) {
                for (d = 0; d < hidden; ++d) out_row[d] += expert_down[d] * router_scores[i];
            }
            free(expert_down);
        }
        {
            float s = 1.0f / (1.0f + expf(-scale_in));
            for (d = 0; d < hidden; ++d) out_row[d] += shared[d] * s;
        }
    }

    ok = 1;

cleanup:
    ds4_gpu_tensor_free(input_gpu);
    ds4_gpu_tensor_free(input_ln_gpu);
    ds4_gpu_tensor_free(z_gpu);
    ds4_gpu_tensor_free(a_gpu);
    ds4_gpu_tensor_free(b_gpu);
    ds4_gpu_tensor_free(qkv_gpu);
    ds4_gpu_tensor_free(q_gpu);
    ds4_gpu_tensor_free(k_gpu);
    ds4_gpu_tensor_free(v_gpu);
    ds4_gpu_tensor_free(conv_gpu);
    ds4_gpu_tensor_free(core_gpu);
    ds4_gpu_tensor_free(beta_gpu);
    ds4_gpu_tensor_free(g_gpu);
    ds4_gpu_tensor_free(out_in_gpu);
    ds4_gpu_tensor_free(out_proj_gpu);
    ds4_gpu_tensor_free(resid_gpu);
    ds4_gpu_tensor_free(post_ln_gpu);
    ds4_gpu_tensor_free(router_logits_gpu);
    ds4_gpu_tensor_free(shared_gate_gpu);
    ds4_gpu_tensor_free(shared_up_gpu);
    ds4_gpu_tensor_free(shared_mid_gpu);
    ds4_gpu_tensor_free(shared_out_gpu);
    ds4_gpu_tensor_free(state_gpu);
    ds4_gpu_tensor_free(conv_w_gpu);
    ds4_gpu_tensor_free(A_log_gpu);
    ds4_gpu_tensor_free(dt_bias_gpu);
    for (i = 0; i < 8; ++i) {
        if (ex_flags[i]) {
            ds4_gpu_tensor_free(expert_gate_gpu[i]);
            ds4_gpu_tensor_free(expert_up_gpu[i]);
            ds4_gpu_tensor_free(expert_mid_gpu[i]);
            ds4_gpu_tensor_free(expert_down_gpu[i]);
        }
    }
    free(resid);
    free(post_ln);
    free(shared);
    free(conv_w);
    free(A_log);
    free(dt_bias);
    free(ssm_norm_w);
    free(post_attn_norm_w);
    free(gate_inp_shexp_w);
    return ok;
}

/* ---- standalone test ---- */
int main(int argc, char **argv) {
    const char *gguf_path;
    int layer_idx = 0;
    int repeat = 1;
    qwen36_gguf_file gf;
    qwen36_35a3b_q8_model q8;
    mapped_file mf;
    native_hybrid_state st;
    char err[512];
    float *layer_input;
    float *out_row;
    uint32_t hidden, topk, inter;
    uint32_t num_k_heads, num_v_heads, head_k_dim, head_v_dim, key_dim, value_dim;
    int i;

    memset(&gf, 0, sizeof(gf));
    memset(&q8, 0, sizeof(q8));
    memset(&mf, 0, sizeof(mf));
    mf.fd = -1;
    memset(&st, 0, sizeof(st));

    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [layer_idx=0] [repeat=1]\n", argv[0]);
        return 1;
    }
    gguf_path = argv[1];
    if (argc > 2) layer_idx = atoi(argv[2]);
    if (argc > 3) repeat = atoi(argv[3]);
    if (repeat < 1) repeat = 1;

    if (!qwen36_gguf_open(&gf, gguf_path, err, sizeof(err))) {
        fprintf(stderr, "gguf_open: %s\n", err);
        return 1;
    }
    if (!qwen36_35a3b_q8_bind(&gf, &q8, err, sizeof(err))) {
        fprintf(stderr, "bind: %s\n", err);
        qwen36_gguf_close(&gf);
        return 1;
    }
    if (layer_idx < 0 || (uint32_t)layer_idx >= QWEN36_35A3B_Q8_BLOCK_COUNT) {
        fprintf(stderr, "layer_idx %d out of range [0,%u)\n", layer_idx, QWEN36_35A3B_Q8_BLOCK_COUNT);
        qwen36_gguf_close(&gf);
        return 1;
    }
    if (q8.layers[layer_idx].kind != QWEN36_LAYER_KIND_HYBRID_SSM) {
        fprintf(stderr, "layer %d is not a hybrid SSM layer\n", layer_idx);
        qwen36_gguf_close(&gf);
        return 1;
    }

    if (!mapped_file_open(&mf, gguf_path)) {
        fprintf(stderr, "mmap: %s\n", strerror(errno));
        qwen36_gguf_close(&gf);
        return 1;
    }
    if (ds4_gpu_init() == 0) {
        fprintf(stderr, "ds4_gpu_init failed\n");
        mapped_file_close(&mf);
        qwen36_gguf_close(&gf);
        return 1;
    }
    (void)ds4_gpu_set_model_map(mf.map, mf.size);
    (void)ds4_gpu_set_model_fd_for_map(mf.fd, mf.map);

    hidden = 2048;
    topk = 8;
    inter = 512;
    num_k_heads = 1;
    num_v_heads = 4;
    head_k_dim = 128;
    head_v_dim = 64;
    key_dim = num_k_heads * head_k_dim;
    value_dim = num_v_heads * head_v_dim;

    layer_input = (float *)calloc(hidden, sizeof(float));
    out_row = (float *)calloc(hidden, sizeof(float));
    if (!layer_input || !out_row) {
        fprintf(stderr, "malloc failed\n");
        goto cleanup;
    }

    if (!native_hybrid_state_init(&st, num_v_heads, head_k_dim, head_v_dim, key_dim * 2u + value_dim)) {
        fprintf(stderr, "state init failed\n");
        goto cleanup;
    }

    printf("qwen36_native_hybrid_test: layer=%d hidden=%u repeat=%d\n", layer_idx, hidden, repeat);

    for (i = 0; i < repeat; ++i) {
        if (!run_native_hybrid_layer_gpu(&mf, &q8.layers[layer_idx], &st,
                                         hidden, num_v_heads, num_k_heads,
                                         head_k_dim, head_v_dim,
                                         key_dim, value_dim,
                                         topk, inter,
                                         layer_input, out_row,
                                         err, sizeof(err))) {
            fprintf(stderr, "Step %d: %s\n", i, err);
            goto cleanup;
        }
    }

    {
        double sum = 0.0;
        for (uint32_t d = 0; d < hidden; ++d) sum += (double)out_row[d];
        printf("output_sum: %.8f\n", (float)sum);
        printf("ok: true\n");
    }

cleanup:
    native_hybrid_state_free(&st);
    free(layer_input);
    free(out_row);
    ds4_gpu_cleanup();
    mapped_file_close(&mf);
    qwen36_gguf_close(&gf);
    return 0;
}
