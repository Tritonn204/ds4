#include "qwen36_runtime.h"

#include <stdio.h>
#include <string.h>

static void set_layer_common(qwen36_runtime_layer_plan *l, uint32_t i) {
    memset(l, 0, sizeof(*l));
    l->layer_index = i;
    l->routed_expert_count = 256;
    l->routed_expert_topk = 8;
    l->expert_ffn_length = 512;
    l->attention_kernel = "pending";
    l->router_kernel = "pending";
    l->routed_gate_up_kernel = "pending";
    l->routed_down_kernel = "pending";
    l->shared_kernel = "pending";
}

static void set_q8_plan_layer(qwen36_runtime_layer_plan *dst, const qwen36_35a3b_q8_layer *src, uint32_t i) {
    set_layer_common(dst, i);
    dst->kind = src->kind == QWEN36_LAYER_KIND_HYBRID_SSM
        ? QWEN36_RUNTIME_LAYER_HYBRID_SSM
        : QWEN36_RUNTIME_LAYER_FULL_ATTENTION;
    dst->attn_norm = src->attn_norm;
    dst->post_attn_norm = src->post_attn_norm;
    dst->attn_gate = src->attn_gate;
    dst->attn_qkv = src->attn_qkv;
    dst->attn_q = src->attn_q;
    dst->attn_q_norm = src->attn_q_norm;
    dst->attn_k = src->attn_k;
    dst->attn_k_norm = src->attn_k_norm;
    dst->attn_v = src->attn_v;
    dst->attn_output = src->attn_output;
    dst->ssm_alpha = src->ssm_alpha;
    dst->ssm_beta = src->ssm_beta;
    dst->ssm_out = src->ssm_out;
    dst->router_gate = src->ffn_gate_inp;
    dst->router_gate_shexp = src->ffn_gate_inp_shexp;
    dst->routed_gate = src->ffn_gate_exps;
    dst->routed_up = src->ffn_up_exps;
    dst->routed_down = src->ffn_down_exps;
    dst->shared_gate = src->ffn_gate_shexp;
    dst->shared_up = src->ffn_up_shexp;
    dst->shared_down = src->ffn_down_shexp;
    dst->attention_kernel = dst->kind == QWEN36_RUNTIME_LAYER_HYBRID_SSM ? "hybrid_ssm_attention" : "full_attention";
    dst->router_kernel = "router_topk_f32";
    dst->routed_gate_up_kernel = "matvec_q8_0";
    dst->routed_down_kernel = "matvec_q8_0";
    dst->shared_kernel = "matvec_q8_0";
}

static void set_q4xl_plan_layer(qwen36_runtime_layer_plan *dst, const qwen36_35a3b_q4xl_layer *src, uint32_t i) {
    set_layer_common(dst, i);
    dst->kind = src->kind == QWEN36_Q4XL_LAYER_KIND_HYBRID_SSM
        ? QWEN36_RUNTIME_LAYER_HYBRID_SSM
        : QWEN36_RUNTIME_LAYER_FULL_ATTENTION;
    dst->attn_norm = src->attn_norm;
    dst->post_attn_norm = src->post_attn_norm;
    dst->attn_gate = src->attn_gate;
    dst->attn_qkv = src->attn_qkv;
    dst->attn_q = src->attn_q;
    dst->attn_q_norm = src->attn_q_norm;
    dst->attn_k = src->attn_k;
    dst->attn_k_norm = src->attn_k_norm;
    dst->attn_v = src->attn_v;
    dst->attn_output = src->attn_output;
    dst->ssm_alpha = src->ssm_alpha;
    dst->ssm_beta = src->ssm_beta;
    dst->ssm_out = src->ssm_out;
    dst->router_gate = src->ffn_gate_inp;
    dst->router_gate_shexp = src->ffn_gate_inp_shexp;
    dst->routed_gate = src->ffn_gate_exps;
    dst->routed_up = src->ffn_up_exps;
    dst->routed_down = src->ffn_down_exps;
    dst->shared_gate = src->ffn_gate_shexp;
    dst->shared_up = src->ffn_up_shexp;
    dst->shared_down = src->ffn_down_shexp;
    dst->attention_kernel = dst->kind == QWEN36_RUNTIME_LAYER_HYBRID_SSM ? "hybrid_ssm_attention" : "full_attention";
    dst->router_kernel = "router_topk_f32";
    dst->routed_gate_up_kernel = "mixed_q4_k_q5_k";
    dst->routed_down_kernel = "mixed_q5_k_q6_k";
    dst->shared_kernel = "matvec_q8_0";
}

static void fill_common(qwen36_runtime_plan *out, const qwen36_gguf_file *gf) {
    memset(out, 0, sizeof(*out));
    out->gf = gf;
    out->hidden_size = 2048;
    out->context_length = 262144;
    out->expert_count = 256;
    out->expert_topk = 8;
    out->expert_ffn_length = 512;
    out->shared_expert_ffn_length = 512;
    out->full_attention_interval = 4;
    out->scratch_bytes = 0;
    out->kv_bytes = 0;
}

bool qwen36_runtime_build(const qwen36_gguf_file *gf, qwen36_runtime_plan *out, char *err, size_t err_cap) {
    qwen36_35a3b_q8_model q8;
    qwen36_35a3b_q4xl_model q4;
    uint32_t i;

    fill_common(out, gf);

    if (qwen36_35a3b_q8_bind(gf, &q8, err, err_cap)) {
        out->contract_kind = QWEN36_RUNTIME_CONTRACT_Q8_0;
        out->contract_name = "Qwen3.6-35B-A3B Q8_0";
        out->token_embd = q8.token_embd;
        out->output_norm = q8.output_norm;
        out->output = q8.output;
        out->block_count = QWEN36_35A3B_Q8_BLOCK_COUNT;
        out->hybrid_layer_count = q8.hybrid_layer_count;
        out->full_attention_layer_count = q8.full_attn_layer_count;
        for (i = 0; i < QWEN36_35A3B_Q8_BLOCK_COUNT; i++) set_q8_plan_layer(&out->layers[i], &q8.layers[i], i);
        return true;
    }
    if (qwen36_35a3b_q4xl_bind(gf, &q4, err, err_cap)) {
        out->contract_kind = QWEN36_RUNTIME_CONTRACT_Q4XL;
        out->contract_name = "Qwen3.6-35B-A3B UD_Q4_K_XL";
        out->token_embd = q4.token_embd;
        out->output_norm = q4.output_norm;
        out->output = q4.output;
        out->block_count = QWEN36_35A3B_Q4XL_BLOCK_COUNT;
        out->hybrid_layer_count = q4.hybrid_layer_count;
        out->full_attention_layer_count = q4.full_attn_layer_count;
        for (i = 0; i < QWEN36_35A3B_Q4XL_BLOCK_COUNT; i++) set_q4xl_plan_layer(&out->layers[i], &q4.layers[i], i);
        return true;
    }
    return false;
}

void qwen36_runtime_dump(const qwen36_runtime_plan *plan, FILE *fp) {
    uint32_t i;
    fprintf(fp, "contract: %s\n", plan->contract_name);
    fprintf(fp, "blocks: %u\n", plan->block_count);
    fprintf(fp, "hybrid_layers: %u\n", plan->hybrid_layer_count);
    fprintf(fp, "full_attention_layers: %u\n", plan->full_attention_layer_count);
    fprintf(fp, "expert_count: %u\n", plan->expert_count);
    fprintf(fp, "expert_topk: %u\n", plan->expert_topk);
    fprintf(fp, "hidden_size: %u\n", plan->hidden_size);
    fprintf(fp, "context_length: %u\n", plan->context_length);
    fprintf(fp, "scratch_bytes: %llu\n", (unsigned long long)plan->scratch_bytes);
    fprintf(fp, "kv_bytes: %llu\n", (unsigned long long)plan->kv_bytes);
    fputs("layer_plans:\n", fp);
    for (i = 0; i < plan->block_count; i++) {
        const qwen36_runtime_layer_plan *l = &plan->layers[i];
        fprintf(fp, "  blk.%u kind=%s attn=%s router=%s routed_gate_up=%s routed_down=%s shared=%s\n",
                i,
                l->kind == QWEN36_RUNTIME_LAYER_HYBRID_SSM ? "hybrid_ssm" : "full_attention",
                l->attention_kernel,
                l->router_kernel,
                l->routed_gate_up_kernel,
                l->routed_down_kernel,
                l->shared_kernel);
    }
}
