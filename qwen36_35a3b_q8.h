#ifndef QWEN36_35A3B_Q8_H
#define QWEN36_35A3B_Q8_H

#include "qwen36_gguf.h"

#define QWEN36_35A3B_Q8_BLOCK_COUNT 40

typedef enum qwen36_35a3b_q8_layer_kind {
    QWEN36_LAYER_KIND_HYBRID_SSM = 1,
    QWEN36_LAYER_KIND_FULL_ATTENTION = 2,
} qwen36_35a3b_q8_layer_kind;

typedef struct qwen36_35a3b_q8_layer {
    qwen36_35a3b_q8_layer_kind kind;
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
    const qwen36_gguf_tensor *ssm_a;
    const qwen36_gguf_tensor *ssm_alpha;
    const qwen36_gguf_tensor *ssm_beta;
    const qwen36_gguf_tensor *ssm_conv1d;
    const qwen36_gguf_tensor *ssm_dt_bias;
    const qwen36_gguf_tensor *ssm_norm;
    const qwen36_gguf_tensor *ssm_out;
    const qwen36_gguf_tensor *ffn_gate_inp;
    const qwen36_gguf_tensor *ffn_gate_inp_shexp;
    const qwen36_gguf_tensor *ffn_gate_exps;
    const qwen36_gguf_tensor *ffn_up_exps;
    const qwen36_gguf_tensor *ffn_down_exps;
    const qwen36_gguf_tensor *ffn_gate_shexp;
    const qwen36_gguf_tensor *ffn_up_shexp;
    const qwen36_gguf_tensor *ffn_down_shexp;
} qwen36_35a3b_q8_layer;

typedef struct qwen36_35a3b_q8_model {
    const qwen36_gguf_file *gf;
    const qwen36_gguf_tensor *token_embd;
    const qwen36_gguf_tensor *output_norm;
    const qwen36_gguf_tensor *output;
    qwen36_35a3b_q8_layer layers[QWEN36_35A3B_Q8_BLOCK_COUNT];
    uint32_t hybrid_layer_count;
    uint32_t full_attn_layer_count;
} qwen36_35a3b_q8_model;

bool qwen36_35a3b_q8_validate(const qwen36_gguf_file *gf, char *err, size_t err_cap);
bool qwen36_35a3b_q8_bind(const qwen36_gguf_file *gf, qwen36_35a3b_q8_model *out, char *err, size_t err_cap);
void qwen36_35a3b_q8_dump_summary(const qwen36_35a3b_q8_model *m, FILE *fp);

#endif
