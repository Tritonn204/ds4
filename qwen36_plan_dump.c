#include "qwen36_35a3b_q4xl.h"
#include "qwen36_35a3b_q8.h"

#include <stdio.h>
#include <string.h>

static void usage(FILE *fp) {
    fputs("usage: qwen36-plan-dump <model.gguf>\n", fp);
}

static const char *matvec_family(uint32_t type) {
    switch (type) {
    case QWEN36_GGUF_TYPE_F32: return "matvec_f32";
    case QWEN36_GGUF_TYPE_Q4_K: return "matvec_q4_k";
    case QWEN36_GGUF_TYPE_Q5_K: return "matvec_q5_k";
    case QWEN36_GGUF_TYPE_Q6_K: return "matvec_q6_k";
    case QWEN36_GGUF_TYPE_Q8_0: return "matvec_q8_0";
    default: return "matvec_unknown";
    }
}

static const char *moe_gate_family(uint32_t gate_type, uint32_t up_type) {
    if (gate_type == up_type) return matvec_family(gate_type);
    return "matvec_mixed";
}

static void print_root_plan(const qwen36_gguf_tensor *token_embd,
                            const qwen36_gguf_tensor *output_norm,
                            const qwen36_gguf_tensor *output) {
    fputs("root:\n", stdout);
    printf("  token_embd: type=%s kernel=embedding_lookup\n", qwen36_gguf_type_name(token_embd->type));
    printf("  output_norm: type=%s kernel=rmsnorm_f32\n", qwen36_gguf_type_name(output_norm->type));
    printf("  output: type=%s kernel=%s\n", qwen36_gguf_type_name(output->type), matvec_family(output->type));
}

static void print_header(const qwen36_gguf_file *gf, const char *contract_name) {
    const qwen36_gguf_kv *arch = qwen36_gguf_find_kv(gf, "general.architecture");
    const qwen36_gguf_kv *name = qwen36_gguf_find_kv(gf, "general.name");
    const qwen36_gguf_kv *ctx = qwen36_gguf_find_kv(gf, "qwen35moe.context_length");
    const qwen36_gguf_kv *blocks = qwen36_gguf_find_kv(gf, "qwen35moe.block_count");
    const qwen36_gguf_kv *experts = qwen36_gguf_find_kv(gf, "qwen35moe.expert_count");
    const qwen36_gguf_kv *topk = qwen36_gguf_find_kv(gf, "qwen35moe.expert_used_count");

    printf("contract: %s\n", contract_name);
    printf("file: %s\n", gf->path);
    printf("model: %s\n", name && name->type == QWEN36_GGUF_VALUE_STRING ? name->v.str : "unknown");
    printf("architecture: %s\n", arch && arch->type == QWEN36_GGUF_VALUE_STRING ? arch->v.str : "unknown");
    printf("block_count: %llu\n", (unsigned long long)(blocks ? blocks->v.u32 ? blocks->v.u32 : blocks->v.u64 : 0));
    printf("context_length: %llu\n", (unsigned long long)(ctx ? ctx->v.u32 ? ctx->v.u32 : ctx->v.u64 : 0));
    printf("experts: %llu\n", (unsigned long long)(experts ? experts->v.u32 ? experts->v.u32 : experts->v.u64 : 0));
    printf("topk_experts: %llu\n", (unsigned long long)(topk ? topk->v.u32 ? topk->v.u32 : topk->v.u64 : 0));
    fputs("decode_contract:\n", stdout);
    fputs("  batch: 1\n", stdout);
    fputs("  mode: greedy\n", stdout);
    fputs("  kv_goal: short_context_first\n", stdout);
}

static void print_q8_layer(uint32_t i, const qwen36_35a3b_q8_layer *l) {
    printf("blk.%u:\n", i);
    printf("  kind: %s\n", l->kind == QWEN36_LAYER_KIND_HYBRID_SSM ? "hybrid_ssm" : "full_attention");
    fputs("  steps:\n", stdout);
    fputs("    - attn_norm -> rmsnorm_f32\n", stdout);
    if (l->kind == QWEN36_LAYER_KIND_HYBRID_SSM) {
        printf("    - attn_gate -> %s\n", matvec_family(l->attn_gate->type));
        printf("    - attn_qkv -> %s\n", matvec_family(l->attn_qkv->type));
        fputs("    - ssm_conv1d -> conv1d_f32\n", stdout);
        printf("    - ssm_alpha -> %s\n", matvec_family(l->ssm_alpha->type));
        printf("    - ssm_beta -> %s\n", matvec_family(l->ssm_beta->type));
        printf("    - ssm_out -> %s\n", matvec_family(l->ssm_out->type));
    } else {
        printf("    - attn_q -> %s\n", matvec_family(l->attn_q->type));
        printf("    - attn_k -> %s\n", matvec_family(l->attn_k->type));
        printf("    - attn_v -> %s\n", matvec_family(l->attn_v->type));
        fputs("    - rope/qk -> full_attention_cpu\n", stdout);
        printf("    - attn_output -> %s\n", matvec_family(l->attn_output->type));
    }
    fputs("    - post_attention_norm -> rmsnorm_f32\n", stdout);
    printf("    - router logits -> %s\n", matvec_family(l->ffn_gate_inp->type));
    fputs("    - router topk -> topk_f32(k=8 of 256)\n", stdout);
    printf("    - routed gate/up -> %s\n", moe_gate_family(l->ffn_gate_exps->type, l->ffn_up_exps->type));
    printf("    - routed down -> %s\n", matvec_family(l->ffn_down_exps->type));
    printf("    - shared expert gate/up/down -> %s/%s/%s\n",
           matvec_family(l->ffn_gate_shexp->type),
           matvec_family(l->ffn_up_shexp->type),
           matvec_family(l->ffn_down_shexp->type));
}

static void print_q4xl_layer(uint32_t i, const qwen36_35a3b_q4xl_layer *l) {
    printf("blk.%u:\n", i);
    printf("  kind: %s\n", l->kind == QWEN36_Q4XL_LAYER_KIND_HYBRID_SSM ? "hybrid_ssm" : "full_attention");
    fputs("  steps:\n", stdout);
    fputs("    - attn_norm -> rmsnorm_f32\n", stdout);
    if (l->kind == QWEN36_Q4XL_LAYER_KIND_HYBRID_SSM) {
        printf("    - attn_gate -> %s\n", matvec_family(l->attn_gate->type));
        printf("    - attn_qkv -> %s\n", matvec_family(l->attn_qkv->type));
        fputs("    - ssm_conv1d -> conv1d_f32\n", stdout);
        printf("    - ssm_alpha -> %s\n", matvec_family(l->ssm_alpha->type));
        printf("    - ssm_beta -> %s\n", matvec_family(l->ssm_beta->type));
        printf("    - ssm_out -> %s\n", matvec_family(l->ssm_out->type));
    } else {
        printf("    - attn_q -> %s\n", matvec_family(l->attn_q->type));
        printf("    - attn_k -> %s\n", matvec_family(l->attn_k->type));
        printf("    - attn_v -> %s\n", matvec_family(l->attn_v->type));
        fputs("    - rope/qk -> full_attention_cpu\n", stdout);
        printf("    - attn_output -> %s\n", matvec_family(l->attn_output->type));
    }
    fputs("    - post_attention_norm -> rmsnorm_f32\n", stdout);
    printf("    - router logits -> %s\n", matvec_family(l->ffn_gate_inp->type));
    fputs("    - router topk -> topk_f32(k=8 of 256)\n", stdout);
    printf("    - routed gate/up -> %s\n", moe_gate_family(l->ffn_gate_exps->type, l->ffn_up_exps->type));
    printf("    - routed down -> %s\n", matvec_family(l->ffn_down_exps->type));
    printf("    - shared expert gate/up/down -> %s/%s/%s\n",
           matvec_family(l->ffn_gate_shexp->type),
           matvec_family(l->ffn_up_shexp->type),
           matvec_family(l->ffn_down_shexp->type));
}

int main(int argc, char **argv) {
    qwen36_gguf_file gf;
    char err[512];

    if (argc != 2) {
        usage(stderr);
        return 2;
    }
    if (!qwen36_gguf_open(&gf, argv[1], err, sizeof(err))) {
        fprintf(stderr, "qwen36-plan-dump: %s\n", err);
        return 1;
    }

    {
        qwen36_35a3b_q8_model q8;
        if (qwen36_35a3b_q8_bind(&gf, &q8, err, sizeof(err))) {
            uint32_t i;
            print_header(&gf, "Qwen3.6-35B-A3B Q8_0");
            print_root_plan(q8.token_embd, q8.output_norm, q8.output);
            fputs("layers:\n", stdout);
            for (i = 0; i < QWEN36_35A3B_Q8_BLOCK_COUNT; i++) print_q8_layer(i, &q8.layers[i]);
            qwen36_gguf_close(&gf);
            return 0;
        }
    }
    {
        qwen36_35a3b_q4xl_model q4;
        if (qwen36_35a3b_q4xl_bind(&gf, &q4, err, sizeof(err))) {
            uint32_t i;
            print_header(&gf, "Qwen3.6-35B-A3B UD_Q4_K_XL");
            print_root_plan(q4.token_embd, q4.output_norm, q4.output);
            fputs("layers:\n", stdout);
            for (i = 0; i < QWEN36_35A3B_Q4XL_BLOCK_COUNT; i++) print_q4xl_layer(i, &q4.layers[i]);
            qwen36_gguf_close(&gf);
            return 0;
        }
    }

    fprintf(stderr, "qwen36-plan-dump: unsupported contract: %s\n", err);
    qwen36_gguf_close(&gf);
    return 1;
}
