#include "qwen36_35a3b_q8.h"

#include <stdio.h>
#include <stdarg.h>
#include <string.h>

typedef struct qwen36_35a3b_q8_expect {
    const char *name;
    uint32_t type;
    uint32_t ndim;
    uint64_t d0;
    uint64_t d1;
    uint64_t d2;
} qwen36_35a3b_q8_expect;

static void set_err(char *err, size_t err_cap, const char *fmt, ...) {
    va_list ap;
    if (!err || err_cap == 0) return;
    va_start(ap, fmt);
    vsnprintf(err, err_cap, fmt, ap);
    va_end(ap);
}

static bool expect_kv_str(const qwen36_gguf_file *gf, const char *key, const char *want, char *err, size_t err_cap) {
    const qwen36_gguf_kv *kv = qwen36_gguf_find_kv(gf, key);
    if (!kv) {
        set_err(err, err_cap, "missing metadata key %s", key);
        return false;
    }
    if (kv->type != QWEN36_GGUF_VALUE_STRING) {
        set_err(err, err_cap, "metadata key %s has wrong type", key);
        return false;
    }
    if (strcmp(kv->v.str, want) != 0) {
        set_err(err, err_cap, "metadata key %s expected %s got %s", key, want, kv->v.str);
        return false;
    }
    return true;
}

static bool expect_kv_u64(const qwen36_gguf_file *gf, const char *key, uint64_t want, char *err, size_t err_cap) {
    const qwen36_gguf_kv *kv = qwen36_gguf_find_kv(gf, key);
    uint64_t got = 0;
    if (!kv) {
        set_err(err, err_cap, "missing metadata key %s", key);
        return false;
    }
    switch (kv->type) {
    case QWEN36_GGUF_VALUE_UINT32: got = kv->v.u32; break;
    case QWEN36_GGUF_VALUE_INT32: got = (uint64_t)kv->v.i32; break;
    case QWEN36_GGUF_VALUE_UINT64: got = kv->v.u64; break;
    case QWEN36_GGUF_VALUE_INT64: got = (uint64_t)kv->v.i64; break;
    default:
        set_err(err, err_cap, "metadata key %s has wrong integer type", key);
        return false;
    }
    if (got != want) {
        set_err(err, err_cap, "metadata key %s expected %llu got %llu",
                key, (unsigned long long)want, (unsigned long long)got);
        return false;
    }
    return true;
}

static bool expect_kv_f32(const qwen36_gguf_file *gf, const char *key, float want, char *err, size_t err_cap) {
    const qwen36_gguf_kv *kv = qwen36_gguf_find_kv(gf, key);
    if (!kv) {
        set_err(err, err_cap, "missing metadata key %s", key);
        return false;
    }
    if (kv->type != QWEN36_GGUF_VALUE_FLOAT32) {
        set_err(err, err_cap, "metadata key %s has wrong type", key);
        return false;
    }
    if (kv->v.f32 != want) {
        set_err(err, err_cap, "metadata key %s expected %.9g got %.9g", key, want, kv->v.f32);
        return false;
    }
    return true;
}

static bool tensor_matches(const qwen36_gguf_tensor *t, const qwen36_35a3b_q8_expect *e, char *err, size_t err_cap) {
    if (!t) {
        set_err(err, err_cap, "missing tensor %s", e->name);
        return false;
    }
    if (t->type != e->type || t->ndim != e->ndim ||
        t->dims[0] != e->d0 ||
        (e->ndim > 1 && t->dims[1] != e->d1) ||
        (e->ndim > 2 && t->dims[2] != e->d2)) {
        set_err(err, err_cap,
                "tensor %s mismatch: got type=%s ndim=%u dims=[%llu,%llu,%llu] expected type=%s ndim=%u dims=[%llu,%llu,%llu]",
                e->name,
                qwen36_gguf_type_name(t->type),
                t->ndim,
                (unsigned long long)t->dims[0],
                (unsigned long long)t->dims[1],
                (unsigned long long)t->dims[2],
                qwen36_gguf_type_name(e->type),
                e->ndim,
                (unsigned long long)e->d0,
                (unsigned long long)e->d1,
                (unsigned long long)e->d2);
        return false;
    }
    return true;
}

static bool expect_named_tensor(const qwen36_gguf_file *gf, const qwen36_35a3b_q8_expect *e, char *err, size_t err_cap) {
    return tensor_matches(qwen36_gguf_find_tensor(gf, e->name), e, err, err_cap);
}

static void layer_name(char *buf, size_t cap, uint32_t layer, const char *suffix) {
    snprintf(buf, cap, "blk.%u.%s", layer, suffix);
}

static bool expect_layer_tensor(const qwen36_gguf_file *gf, uint32_t layer, const char *suffix,
                                uint32_t type, uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2,
                                const qwen36_gguf_tensor **out, char *err, size_t err_cap) {
    char name[128];
    qwen36_35a3b_q8_expect e;
    layer_name(name, sizeof(name), layer, suffix);
    e.name = name;
    e.type = type;
    e.ndim = ndim;
    e.d0 = d0;
    e.d1 = d1;
    e.d2 = d2;
    *out = qwen36_gguf_find_tensor(gf, name);
    return tensor_matches(*out, &e, err, err_cap);
}

static bool tensor_exists(const qwen36_gguf_file *gf, uint32_t layer, const char *suffix) {
    char name[128];
    layer_name(name, sizeof(name), layer, suffix);
    return qwen36_gguf_find_tensor(gf, name) != NULL;
}

static bool validate_metadata(const qwen36_gguf_file *gf, char *err, size_t err_cap) {
    if (!expect_kv_str(gf, "general.architecture", "qwen35moe", err, err_cap)) return false;
    if (!expect_kv_str(gf, "general.name", "Qwen3.6-35B-A3B", err, err_cap)) return false;
    if (!expect_kv_u64(gf, "general.file_type", 7, err, err_cap)) return false;
    if (!expect_kv_u64(gf, "qwen35moe.block_count", 40, err, err_cap)) return false;
    if (!expect_kv_u64(gf, "qwen35moe.embedding_length", 2048, err, err_cap)) return false;
    if (!expect_kv_u64(gf, "qwen35moe.context_length", 262144, err, err_cap)) return false;
    if (!expect_kv_u64(gf, "qwen35moe.attention.head_count", 16, err, err_cap)) return false;
    if (!expect_kv_u64(gf, "qwen35moe.attention.head_count_kv", 2, err, err_cap)) return false;
    if (!expect_kv_u64(gf, "qwen35moe.attention.key_length", 256, err, err_cap)) return false;
    if (!expect_kv_u64(gf, "qwen35moe.attention.value_length", 256, err, err_cap)) return false;
    if (!expect_kv_u64(gf, "qwen35moe.expert_count", 256, err, err_cap)) return false;
    if (!expect_kv_u64(gf, "qwen35moe.expert_used_count", 8, err, err_cap)) return false;
    if (!expect_kv_u64(gf, "qwen35moe.expert_feed_forward_length", 512, err, err_cap)) return false;
    if (!expect_kv_u64(gf, "qwen35moe.expert_shared_feed_forward_length", 512, err, err_cap)) return false;
    if (!expect_kv_u64(gf, "qwen35moe.full_attention_interval", 4, err, err_cap)) return false;
    if (!expect_kv_u64(gf, "qwen35moe.ssm.inner_size", 4096, err, err_cap)) return false;
    if (!expect_kv_u64(gf, "qwen35moe.ssm.state_size", 128, err, err_cap)) return false;
    if (!expect_kv_u64(gf, "qwen35moe.ssm.group_count", 16, err, err_cap)) return false;
    if (!expect_kv_u64(gf, "qwen35moe.ssm.conv_kernel", 4, err, err_cap)) return false;
    if (!expect_kv_u64(gf, "qwen35moe.ssm.time_step_rank", 32, err, err_cap)) return false;
    if (!expect_kv_f32(gf, "qwen35moe.rope.freq_base", 10000000.0f, err, err_cap)) return false;
    return true;
}

static bool validate_export_surface(const qwen36_gguf_file *gf, char *err, size_t err_cap) {
    uint64_t i;
    for (i = 0; i < gf->tensor_count; i++) {
        uint32_t type = gf->tensors[i].type;
        if (type != QWEN36_GGUF_TYPE_F32 && type != QWEN36_GGUF_TYPE_Q8_0) {
            set_err(err, err_cap, "tensor %s has unsupported type %s",
                    gf->tensors[i].name, qwen36_gguf_type_name(type));
            return false;
        }
    }
    return true;
}

static bool bind_layer(const qwen36_gguf_file *gf, uint32_t i, qwen36_35a3b_q8_layer *l, char *err, size_t err_cap) {
    bool has_hybrid = tensor_exists(gf, i, "attn_qkv.weight");
    bool has_full = tensor_exists(gf, i, "attn_q.weight");

    memset(l, 0, sizeof(*l));
    if (!expect_layer_tensor(gf, i, "attn_norm.weight", QWEN36_GGUF_TYPE_F32, 1, 2048, 0, 0, &l->attn_norm, err, err_cap)) return false;
    if (!expect_layer_tensor(gf, i, "post_attention_norm.weight", QWEN36_GGUF_TYPE_F32, 1, 2048, 0, 0, &l->post_attn_norm, err, err_cap)) return false;
    if (!expect_layer_tensor(gf, i, "ffn_gate_inp.weight", QWEN36_GGUF_TYPE_F32, 2, 2048, 256, 0, &l->ffn_gate_inp, err, err_cap)) return false;
    if (!expect_layer_tensor(gf, i, "ffn_gate_inp_shexp.weight", QWEN36_GGUF_TYPE_F32, 1, 2048, 0, 0, &l->ffn_gate_inp_shexp, err, err_cap)) return false;
    if (!expect_layer_tensor(gf, i, "ffn_gate_exps.weight", QWEN36_GGUF_TYPE_Q8_0, 3, 2048, 512, 256, &l->ffn_gate_exps, err, err_cap)) return false;
    if (!expect_layer_tensor(gf, i, "ffn_up_exps.weight", QWEN36_GGUF_TYPE_Q8_0, 3, 2048, 512, 256, &l->ffn_up_exps, err, err_cap)) return false;
    if (!expect_layer_tensor(gf, i, "ffn_down_exps.weight", QWEN36_GGUF_TYPE_Q8_0, 3, 512, 2048, 256, &l->ffn_down_exps, err, err_cap)) return false;
    if (!expect_layer_tensor(gf, i, "ffn_gate_shexp.weight", QWEN36_GGUF_TYPE_Q8_0, 2, 2048, 512, 0, &l->ffn_gate_shexp, err, err_cap)) return false;
    if (!expect_layer_tensor(gf, i, "ffn_up_shexp.weight", QWEN36_GGUF_TYPE_Q8_0, 2, 2048, 512, 0, &l->ffn_up_shexp, err, err_cap)) return false;
    if (!expect_layer_tensor(gf, i, "ffn_down_shexp.weight", QWEN36_GGUF_TYPE_Q8_0, 2, 512, 2048, 0, &l->ffn_down_shexp, err, err_cap)) return false;

    if (has_hybrid == has_full) {
        set_err(err, err_cap, "layer %u must be exactly one of hybrid or full attention", i);
        return false;
    }
    if (has_hybrid) {
        l->kind = QWEN36_LAYER_KIND_HYBRID_SSM;
        if (!expect_layer_tensor(gf, i, "attn_gate.weight", QWEN36_GGUF_TYPE_Q8_0, 2, 2048, 4096, 0, &l->attn_gate, err, err_cap)) return false;
        if (!expect_layer_tensor(gf, i, "attn_qkv.weight", QWEN36_GGUF_TYPE_Q8_0, 2, 2048, 8192, 0, &l->attn_qkv, err, err_cap)) return false;
        if (!expect_layer_tensor(gf, i, "ssm_a", QWEN36_GGUF_TYPE_F32, 1, 32, 0, 0, &l->ssm_a, err, err_cap)) return false;
        if (!expect_layer_tensor(gf, i, "ssm_alpha.weight", QWEN36_GGUF_TYPE_Q8_0, 2, 2048, 32, 0, &l->ssm_alpha, err, err_cap)) return false;
        if (!expect_layer_tensor(gf, i, "ssm_beta.weight", QWEN36_GGUF_TYPE_Q8_0, 2, 2048, 32, 0, &l->ssm_beta, err, err_cap)) return false;
        if (!expect_layer_tensor(gf, i, "ssm_conv1d.weight", QWEN36_GGUF_TYPE_F32, 2, 4, 8192, 0, &l->ssm_conv1d, err, err_cap)) return false;
        if (!expect_layer_tensor(gf, i, "ssm_dt.bias", QWEN36_GGUF_TYPE_F32, 1, 32, 0, 0, &l->ssm_dt_bias, err, err_cap)) return false;
        if (!expect_layer_tensor(gf, i, "ssm_norm.weight", QWEN36_GGUF_TYPE_F32, 1, 128, 0, 0, &l->ssm_norm, err, err_cap)) return false;
        if (!expect_layer_tensor(gf, i, "ssm_out.weight", QWEN36_GGUF_TYPE_Q8_0, 2, 4096, 2048, 0, &l->ssm_out, err, err_cap)) return false;
    } else {
        l->kind = QWEN36_LAYER_KIND_FULL_ATTENTION;
        if (!expect_layer_tensor(gf, i, "attn_q.weight", QWEN36_GGUF_TYPE_Q8_0, 2, 2048, 8192, 0, &l->attn_q, err, err_cap)) return false;
        if (!expect_layer_tensor(gf, i, "attn_q_norm.weight", QWEN36_GGUF_TYPE_F32, 1, 256, 0, 0, &l->attn_q_norm, err, err_cap)) return false;
        if (!expect_layer_tensor(gf, i, "attn_k.weight", QWEN36_GGUF_TYPE_Q8_0, 2, 2048, 512, 0, &l->attn_k, err, err_cap)) return false;
        if (!expect_layer_tensor(gf, i, "attn_k_norm.weight", QWEN36_GGUF_TYPE_F32, 1, 256, 0, 0, &l->attn_k_norm, err, err_cap)) return false;
        if (!expect_layer_tensor(gf, i, "attn_v.weight", QWEN36_GGUF_TYPE_Q8_0, 2, 2048, 512, 0, &l->attn_v, err, err_cap)) return false;
        if (!expect_layer_tensor(gf, i, "attn_output.weight", QWEN36_GGUF_TYPE_Q8_0, 2, 4096, 2048, 0, &l->attn_output, err, err_cap)) return false;
    }
    return true;
}

bool qwen36_35a3b_q8_validate(const qwen36_gguf_file *gf, char *err, size_t err_cap) {
    qwen36_35a3b_q8_model tmp;
    if (!validate_metadata(gf, err, err_cap)) return false;
    if (!validate_export_surface(gf, err, err_cap)) return false;
    if (!qwen36_35a3b_q8_bind(gf, &tmp, err, err_cap)) return false;
    return true;
}

bool qwen36_35a3b_q8_bind(const qwen36_gguf_file *gf, qwen36_35a3b_q8_model *out, char *err, size_t err_cap) {
    uint32_t i;
    qwen36_35a3b_q8_expect root_expect[] = {
        {"token_embd.weight", QWEN36_GGUF_TYPE_Q8_0, 2, 2048, 248320, 0},
        {"output_norm.weight", QWEN36_GGUF_TYPE_F32, 1, 2048, 0, 0},
        {"output.weight", QWEN36_GGUF_TYPE_Q8_0, 2, 2048, 248320, 0},
    };

    memset(out, 0, sizeof(*out));
    out->gf = gf;

    if (!expect_named_tensor(gf, &root_expect[0], err, err_cap) ||
        !expect_named_tensor(gf, &root_expect[1], err, err_cap) ||
        !expect_named_tensor(gf, &root_expect[2], err, err_cap)) return false;

    out->token_embd = qwen36_gguf_find_tensor(gf, root_expect[0].name);
    out->output_norm = qwen36_gguf_find_tensor(gf, root_expect[1].name);
    out->output = qwen36_gguf_find_tensor(gf, root_expect[2].name);

    for (i = 0; i < QWEN36_35A3B_Q8_BLOCK_COUNT; i++) {
        if (!bind_layer(gf, i, &out->layers[i], err, err_cap)) return false;
        if (out->layers[i].kind == QWEN36_LAYER_KIND_HYBRID_SSM) out->hybrid_layer_count++;
        if (out->layers[i].kind == QWEN36_LAYER_KIND_FULL_ATTENTION) out->full_attn_layer_count++;
    }
    if (out->hybrid_layer_count != 30 || out->full_attn_layer_count != 10) {
        set_err(err, err_cap, "expected 30 hybrid and 10 full-attention layers, got %u and %u",
                out->hybrid_layer_count, out->full_attn_layer_count);
        return false;
    }
    return true;
}

void qwen36_35a3b_q8_dump_summary(const qwen36_35a3b_q8_model *m, FILE *fp) {
    uint32_t i;
    fprintf(fp, "target: Qwen3.6-35B-A3B Q8_0\n");
    fprintf(fp, "file: %s\n", m->gf->path);
    fprintf(fp, "gguf_version: %u\n", m->gf->version);
    fprintf(fp, "tensor_count: %llu\n", (unsigned long long)m->gf->tensor_count);
    fprintf(fp, "hybrid_layers: %u\n", m->hybrid_layer_count);
    fprintf(fp, "full_attention_layers: %u\n", m->full_attn_layer_count);
    fputs("layer_kinds:\n", fp);
    for (i = 0; i < QWEN36_35A3B_Q8_BLOCK_COUNT; i++) {
        fprintf(fp, "  blk.%u: %s\n", i,
                m->layers[i].kind == QWEN36_LAYER_KIND_HYBRID_SSM ? "hybrid_ssm" : "full_attention");
    }
}
