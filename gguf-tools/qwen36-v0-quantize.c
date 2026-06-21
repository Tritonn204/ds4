#include "../qwen36_gguf.h"
#include "../qwen36_35a3b_q8.h"
#include "quants.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QWEN36_DEFAULT_ALIGNMENT 32u

typedef struct qv0_params {
    const char *template_gguf;
    const char *out_gguf;
    const char *hf_dir;
    bool dry_run;
} qv0_params;

typedef struct qv0_stats {
    uint64_t tensor_count;
    uint64_t changed_count;
    uint64_t template_bytes;
    uint64_t planned_bytes;
    uint64_t resident_bytes;
    uint64_t streamable_bytes;
    uint64_t streamable_gate_up_bytes;
    uint64_t streamable_down_main_bytes;
    uint64_t streamable_down_tail_bytes;
} qv0_stats;

static void usage(const char *prog) {
    fprintf(stderr, "usage: %s --template MODEL.gguf [--dry-run] [--hf DIR --out OUT.gguf]\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Narrow Qwen3.6-35B-A3B v0 quantizer scaffold.\n");
    fprintf(stderr, "Current scope:\n");
    fprintf(stderr, "  - requires an exact Q8_0 template GGUF\n");
    fprintf(stderr, "  - computes the exact DS4Style_v0 tensor rewrite plan\n");
    fprintf(stderr, "  - reports resident vs streamable byte split\n");
    fprintf(stderr, "  - does not yet regenerate tensor payloads from HF safetensors\n");
}

static void die_usage(const char *prog, const char *msg) {
    if (msg) fprintf(stderr, "error: %s\n", msg);
    usage(prog);
    exit(2);
}

static qv0_params parse_args(int argc, char **argv) {
    qv0_params p = { .dry_run = true };
    int i;
    for (i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else if (strcmp(arg, "--template") == 0) {
            if (++i >= argc) die_usage(argv[0], "missing value for --template");
            p.template_gguf = argv[i];
        } else if (strcmp(arg, "--out") == 0) {
            if (++i >= argc) die_usage(argv[0], "missing value for --out");
            p.out_gguf = argv[i];
            p.dry_run = false;
        } else if (strcmp(arg, "--hf") == 0) {
            if (++i >= argc) die_usage(argv[0], "missing value for --hf");
            p.hf_dir = argv[i];
        } else if (strcmp(arg, "--dry-run") == 0) {
            p.dry_run = true;
        } else {
            die_usage(argv[0], arg);
        }
    }
    if (!p.template_gguf) die_usage(argv[0], "--template is required");
    return p;
}

static uint32_t gguf_alignment(const qwen36_gguf_file *gf) {
    const qwen36_gguf_kv *kv = qwen36_gguf_find_kv(gf, "general.alignment");
    if (!kv) return QWEN36_DEFAULT_ALIGNMENT;
    if (kv->type == QWEN36_GGUF_VALUE_UINT32 && kv->v.u32 > 0) return kv->v.u32;
    if (kv->type == QWEN36_GGUF_VALUE_UINT64 && kv->v.u64 > 0 && kv->v.u64 <= UINT32_MAX) return (uint32_t)kv->v.u64;
    return QWEN36_DEFAULT_ALIGNMENT;
}

static uint64_t pad_u64(uint64_t x, uint32_t n) {
    return ((x + n - 1) / n) * n;
}

static uint64_t tensor_nbytes(uint32_t type, const qwen36_gguf_tensor *t) {
    uint64_t rows = 1;
    uint32_t i;
    size_t row_size = ds4q_row_size((ds4q_type)type, (int64_t)t->dims[0]);
    if (row_size == 0) return 0;
    for (i = 1; i < t->ndim; ++i) rows *= t->dims[i];
    return (uint64_t)row_size * rows;
}

static bool parse_layer_suffix(const char *name, uint32_t *layer_out, const char **suffix_out) {
    unsigned layer = 0;
    int n = 0;
    if (sscanf(name, "blk.%u.%n", &layer, &n) == 1 && n > 0) {
        if (layer_out) *layer_out = layer;
        if (suffix_out) *suffix_out = name + n;
        return true;
    }
    return false;
}

static bool is_streamable_tensor_name(const char *name) {
    return strstr(name, ".ffn_gate_exps.weight") != NULL ||
           strstr(name, ".ffn_up_exps.weight") != NULL ||
           strstr(name, ".ffn_down_exps.weight") != NULL;
}

static uint32_t v0_type_for_tensor_name(const char *name) {
    uint32_t layer = 0;
    const char *suffix = NULL;

    if (strcmp(name, "token_embd.weight") == 0) return QWEN36_GGUF_TYPE_Q4_K;
    if (strcmp(name, "output_norm.weight") == 0) return QWEN36_GGUF_TYPE_F32;
    if (strcmp(name, "output.weight") == 0) return QWEN36_GGUF_TYPE_Q4_K;

    if (!parse_layer_suffix(name, &layer, &suffix)) return UINT32_MAX;

    if (strcmp(suffix, "attn_norm.weight") == 0) return QWEN36_GGUF_TYPE_F32;
    if (strcmp(suffix, "post_attention_norm.weight") == 0) return QWEN36_GGUF_TYPE_F32;
    if (strcmp(suffix, "ffn_gate_inp.weight") == 0) return QWEN36_GGUF_TYPE_F32;
    if (strcmp(suffix, "ffn_gate_inp_shexp.weight") == 0) return QWEN36_GGUF_TYPE_F32;
    if (strcmp(suffix, "ffn_gate_exps.weight") == 0) return QWEN36_GGUF_TYPE_IQ2_XXS;
    if (strcmp(suffix, "ffn_up_exps.weight") == 0) return QWEN36_GGUF_TYPE_IQ2_XXS;
    if (strcmp(suffix, "ffn_down_exps.weight") == 0) return layer >= 34 ? QWEN36_GGUF_TYPE_IQ3_S : QWEN36_GGUF_TYPE_IQ2_S;
    if (strcmp(suffix, "ffn_gate_shexp.weight") == 0) return QWEN36_GGUF_TYPE_Q5_K;
    if (strcmp(suffix, "ffn_up_shexp.weight") == 0) return QWEN36_GGUF_TYPE_Q5_K;
    if (strcmp(suffix, "ffn_down_shexp.weight") == 0) return QWEN36_GGUF_TYPE_Q6_K;

    if (strcmp(suffix, "attn_gate.weight") == 0) return QWEN36_GGUF_TYPE_Q5_K;
    if (strcmp(suffix, "attn_qkv.weight") == 0) return QWEN36_GGUF_TYPE_Q5_K;
    if (strcmp(suffix, "ssm_a") == 0) return QWEN36_GGUF_TYPE_F32;
    if (strcmp(suffix, "ssm_alpha.weight") == 0) return QWEN36_GGUF_TYPE_F32;
    if (strcmp(suffix, "ssm_beta.weight") == 0) return QWEN36_GGUF_TYPE_F32;
    if (strcmp(suffix, "ssm_conv1d.weight") == 0) return QWEN36_GGUF_TYPE_F32;
    if (strcmp(suffix, "ssm_dt.bias") == 0) return QWEN36_GGUF_TYPE_F32;
    if (strcmp(suffix, "ssm_norm.weight") == 0) return QWEN36_GGUF_TYPE_F32;
    if (strcmp(suffix, "ssm_out.weight") == 0) return QWEN36_GGUF_TYPE_Q6_K;

    if (strcmp(suffix, "attn_q.weight") == 0) return QWEN36_GGUF_TYPE_Q5_K;
    if (strcmp(suffix, "attn_q_norm.weight") == 0) return QWEN36_GGUF_TYPE_F32;
    if (strcmp(suffix, "attn_k.weight") == 0) return QWEN36_GGUF_TYPE_Q5_K;
    if (strcmp(suffix, "attn_k_norm.weight") == 0) return QWEN36_GGUF_TYPE_F32;
    if (strcmp(suffix, "attn_v.weight") == 0) return QWEN36_GGUF_TYPE_Q5_K;
    if (strcmp(suffix, "attn_output.weight") == 0) return QWEN36_GGUF_TYPE_Q5_K;

    return UINT32_MAX;
}

static void accumulate_streamable_detail(const char *name, uint64_t bytes, qv0_stats *stats) {
    uint32_t layer = 0;
    const char *suffix = NULL;
    if (!parse_layer_suffix(name, &layer, &suffix)) return;
    if (strcmp(suffix, "ffn_gate_exps.weight") == 0 || strcmp(suffix, "ffn_up_exps.weight") == 0) {
        stats->streamable_gate_up_bytes += bytes;
    } else if (strcmp(suffix, "ffn_down_exps.weight") == 0) {
        if (layer >= 34) stats->streamable_down_tail_bytes += bytes;
        else stats->streamable_down_main_bytes += bytes;
    }
}

static void print_plan(const qwen36_gguf_file *gf, uint32_t alignment, qv0_stats *stats) {
    uint64_t i;
    uint64_t template_padded = 0;
    uint64_t planned_padded = 0;

    for (i = 0; i < gf->tensor_count; ++i) {
        const qwen36_gguf_tensor *t = &gf->tensors[i];
        uint32_t target = v0_type_for_tensor_name(t->name);
        uint64_t cur = tensor_nbytes(t->type, t);
        uint64_t next = tensor_nbytes(target, t);
        bool changed = t->type != target;
        bool streamable = is_streamable_tensor_name(t->name);

        if (target == UINT32_MAX) {
            fprintf(stderr, "error: no v0 mapping for tensor %s\n", t->name);
            exit(1);
        }
        if (cur == 0 || next == 0) {
            fprintf(stderr, "error: byte size computation failed for %s\n", t->name);
            exit(1);
        }

        stats->tensor_count++;
        stats->template_bytes += cur;
        stats->planned_bytes += next;
        template_padded += pad_u64(cur, alignment);
        planned_padded += pad_u64(next, alignment);
        if (changed) stats->changed_count++;
        if (streamable) {
            stats->streamable_bytes += next;
            accumulate_streamable_detail(t->name, next, stats);
        } else {
            stats->resident_bytes += next;
        }

        if (changed) {
            printf("type_change: %s %s -> %s\n",
                   t->name,
                   qwen36_gguf_type_name(t->type),
                   qwen36_gguf_type_name(target));
        }
    }

    printf("template_tensor_bytes_unpadded: %" PRIu64 "\n", stats->template_bytes);
    printf("planned_tensor_bytes_unpadded: %" PRIu64 "\n", stats->planned_bytes);
    printf("template_tensor_bytes_padded: %" PRIu64 "\n", template_padded);
    printf("planned_tensor_bytes_padded: %" PRIu64 "\n", planned_padded);
    printf("type_changes: %" PRIu64 "\n", stats->changed_count);
    printf("resident_bytes_unpadded: %" PRIu64 "\n", stats->resident_bytes);
    printf("streamable_bytes_unpadded: %" PRIu64 "\n", stats->streamable_bytes);
    printf("streamable_gate_up_bytes: %" PRIu64 "\n", stats->streamable_gate_up_bytes);
    printf("streamable_down_main_bytes: %" PRIu64 "\n", stats->streamable_down_main_bytes);
    printf("streamable_down_tail_bytes: %" PRIu64 "\n", stats->streamable_down_tail_bytes);
    if (stats->planned_bytes) {
        printf("streamable_fraction: %.6f\n", (double)stats->streamable_bytes / (double)stats->planned_bytes);
        printf("resident_fraction: %.6f\n", (double)stats->resident_bytes / (double)stats->planned_bytes);
    }
}

int main(int argc, char **argv) {
    qv0_params p = parse_args(argc, argv);
    qwen36_gguf_file gf;
    qwen36_35a3b_q8_model q8;
    qv0_stats stats = {0};
    char err[512];
    uint32_t alignment = 0;

    if (!qwen36_gguf_open(&gf, p.template_gguf, err, sizeof(err))) {
        fprintf(stderr, "error: %s\n", err);
        return 1;
    }
    if (!qwen36_35a3b_q8_bind(&gf, &q8, err, sizeof(err))) {
        fprintf(stderr, "template mismatch: %s\n", err);
        fprintf(stderr, "this scaffold currently requires the exact Q8_0 template as its starting surface\n");
        qwen36_gguf_close(&gf);
        return 1;
    }

    alignment = gguf_alignment(&gf);
    printf("template_contract: Qwen3.6-35B-A3B-Q8_0\n");
    printf("target_contract: Qwen3.6-35B-A3B-DS4Style-v0\n");
    printf("alignment: %u\n", alignment);
    printf("tensor_count: %" PRIu64 "\n", gf.tensor_count);
    qwen36_35a3b_q8_dump_summary(&q8, stdout);
    print_plan(&gf, alignment, &stats);

    if (!p.dry_run) {
        fprintf(stderr, "write path not implemented yet\n");
        fprintf(stderr, "next missing piece: load Qwen3.6 safetensors from --hf and regenerate tensor payloads\n");
        if (!p.hf_dir) {
            fprintf(stderr, "hint: no local HF safetensors path was provided with --hf\n");
        }
        if (p.out_gguf) {
            fprintf(stderr, "requested output path would be: %s\n", p.out_gguf);
        }
        qwen36_gguf_close(&gf);
        return 2;
    }

    qwen36_gguf_close(&gf);
    return 0;
}
