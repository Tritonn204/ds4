#ifndef QWEN36_RUNTIME_H
#define QWEN36_RUNTIME_H

#include "qwen36_35a3b_q4xl.h"
#include "qwen36_35a3b_q8.h"

typedef enum qwen36_runtime_contract_kind {
    QWEN36_RUNTIME_CONTRACT_Q8_0 = 1,
    QWEN36_RUNTIME_CONTRACT_Q4XL = 2,
} qwen36_runtime_contract_kind;

typedef enum qwen36_runtime_layer_kind {
    QWEN36_RUNTIME_LAYER_HYBRID_SSM = 1,
    QWEN36_RUNTIME_LAYER_FULL_ATTENTION = 2,
} qwen36_runtime_layer_kind;

typedef struct qwen36_runtime_layer_plan {
    qwen36_runtime_layer_kind kind;
    uint32_t layer_index;
    uint32_t routed_expert_count;
    uint32_t routed_expert_topk;
    uint32_t expert_ffn_length;
    const qwen36_gguf_tensor *attn_norm;
    const qwen36_gguf_tensor *post_attn_norm;
    const qwen36_gguf_tensor *attn_gate;
    const qwen36_gguf_tensor *attn_qkv;
    const qwen36_gguf_tensor *attn_q;
    const qwen36_gguf_tensor *attn_q_norm;
    const qwen36_gguf_tensor *attn_k;
    const qwen36_gguf_tensor *attn_k_norm;
    const qwen36_gguf_tensor *attn_v;
    const qwen36_gguf_tensor *attn_output;
    const qwen36_gguf_tensor *ssm_alpha;
    const qwen36_gguf_tensor *ssm_beta;
    const qwen36_gguf_tensor *ssm_out;
    const qwen36_gguf_tensor *router_gate;
    const qwen36_gguf_tensor *router_gate_shexp;
    const qwen36_gguf_tensor *routed_gate;
    const qwen36_gguf_tensor *routed_up;
    const qwen36_gguf_tensor *routed_down;
    const qwen36_gguf_tensor *shared_gate;
    const qwen36_gguf_tensor *shared_up;
    const qwen36_gguf_tensor *shared_down;
    const char *attention_kernel;
    const char *router_kernel;
    const char *routed_gate_up_kernel;
    const char *routed_down_kernel;
    const char *shared_kernel;
} qwen36_runtime_layer_plan;

typedef struct qwen36_runtime_plan {
    const qwen36_gguf_file *gf;
    qwen36_runtime_contract_kind contract_kind;
    const char *contract_name;
    const qwen36_gguf_tensor *token_embd;
    const qwen36_gguf_tensor *output_norm;
    const qwen36_gguf_tensor *output;
    uint32_t block_count;
    uint32_t hidden_size;
    uint32_t context_length;
    uint32_t expert_count;
    uint32_t expert_topk;
    uint32_t expert_ffn_length;
    uint32_t shared_expert_ffn_length;
    uint32_t full_attention_interval;
    uint32_t hybrid_layer_count;
    uint32_t full_attention_layer_count;
    uint64_t scratch_bytes;
    uint64_t kv_bytes;
    qwen36_runtime_layer_plan layers[QWEN36_35A3B_Q8_BLOCK_COUNT];
} qwen36_runtime_plan;

bool qwen36_runtime_build(const qwen36_gguf_file *gf, qwen36_runtime_plan *out, char *err, size_t err_cap);
void qwen36_runtime_dump(const qwen36_runtime_plan *plan, FILE *fp);

#endif
