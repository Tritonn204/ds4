#include "qwen36_runtime.h"
#include "qwen36_gguf.h"
#include "ds4_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_layer_status(const qwen36_runtime_layer_plan *l) {
    printf("blk.%u kind=%s\n",
           l->layer_index,
           l->kind == QWEN36_RUNTIME_LAYER_HYBRID_SSM ? "hybrid_ssm" : "full_attention");

    printf("  reusable_host_orchestration:\n");
    printf("    - q8 projections via ds4_gpu_matmul_q8_0_tensor\n");
    printf("    - RMSNorm surfaces via ds4_gpu_rms_norm_*_tensor\n");
    printf("    - router top-k via ds4_gpu_router_select_batch_tensor or logits+topk path\n");
    printf("    - routed experts via ds4_gpu_routed_moe_batch_tensor\n");
    printf("    - shared expert via q8 matmul + swiglu path\n");
    printf("    - residual adds via ds4_gpu_add_tensor\n");

    if (l->kind == QWEN36_RUNTIME_LAYER_HYBRID_SSM) {
        printf("  qwen_specific_gpu_math:\n");
        printf("    - attn_qkv split semantics for Qwen hybrid block\n");
        printf("    - conv1d preprocessing over qkv stream\n");
        printf("    - SSM/DeltaNet state update and output path\n");
        printf("    - Qwen-specific head/layout permutations already proven on CPU\n");
        printf("  immediate_plan:\n");
        printf("    - keep q8 projections/router/moe on existing DS4 GPU ops\n");
        printf("    - add one custom HIP kernel family for hybrid SSM core\n");
    } else {
        printf("  qwen_specific_gpu_math:\n");
        printf("    - q_proj split into query and gate halves\n");
        printf("    - partial/interleaved text mRoPE application\n");
        printf("    - grouped-query attention schedule for 16Q / 2KV heads\n");
        printf("    - post-attention sigmoid gate before o_proj\n");
        printf("  immediate_plan:\n");
        printf("    - q/k/v/o projections reuse existing q8 matmul ops\n");
        printf("    - RoPE likely reuses ds4_gpu_rope_tail_tensor with Qwen-specific host shaping\n");
        printf("    - attention likely reuses existing attention kernel family with Qwen tensor layout adapters\n");
    }
}

int main(int argc, char **argv) {
    const char *gguf_path;
    uint32_t prefix_layers = 4;
    qwen36_gguf_file gf;
    qwen36_runtime_plan plan;
    char err[512];
    int gpu_ok;

    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf [--prefix-layers N]\n", argv[0]);
        return 1;
    }
    gguf_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--prefix-layers") == 0 && i + 1 < argc) {
            prefix_layers = (uint32_t)strtoul(argv[++i], NULL, 10);
        }
    }

    if (!qwen36_gguf_open(&gf, gguf_path, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        return 1;
    }
    if (!qwen36_runtime_build(&gf, &plan, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        qwen36_gguf_close(&gf);
        return 1;
    }

    gpu_ok = ds4_gpu_init();

    printf("qwen36_gpu_oracle_scaffold\n");
    printf("contract: %s\n", plan.contract_name);
    printf("backend_gpu_init: %s\n", gpu_ok ? "ok" : "failed");
    printf("prefix_layers: %u\n", prefix_layers);
    printf("block_count: %u\n", plan.block_count);
    printf("expert_topk: %u\n", plan.expert_topk);
    printf("full_attention_interval: %u\n", plan.full_attention_interval);
    printf("embedding_surface:\n");
    printf("  - token_embd.weight -> ds4_gpu_embed_tokens_hc_tensor candidate reuse\n");
    printf("root_surface:\n");
    printf("  - output_norm.weight -> ds4_gpu_rms_norm_* candidate reuse\n");
    printf("  - output.weight -> ds4_gpu_matmul_q8_0_tensor candidate reuse\n");

    if (prefix_layers > plan.block_count) prefix_layers = plan.block_count;
    for (uint32_t i = 0; i < prefix_layers; ++i) {
        print_layer_status(&plan.layers[i]);
    }

    printf("gpu_oracle_first_cut:\n");
    printf("  - own token_embd + blk.0..3 on HIP\n");
    printf("  - read back last hidden or selected logits only\n");
    printf("  - splice into HF tail for token oracle\n");
    printf("  - compare greedy tokens, not internal tensors\n");

    if (gpu_ok) ds4_gpu_cleanup();
    qwen36_gguf_close(&gf);
    return 0;
}
