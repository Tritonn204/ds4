#include "qwen36_35a3b_q8.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef QWEN36_UNIFIED_HAVE_GPU
#include "ds4_gpu.h"
#endif

#define MAGIC "Q36LCF01"
#define MAGIC_LEN 8
#define ROUTER_COUNT 256
#define FULL_INTER 512
#define FULL_ROTARY_DIM 64
#define FULL_NUM_HEADS 16
#define FULL_NUM_KV_HEADS 2
#define FULL_HEAD_DIM 256
#define FULL_HIDDEN 2048
#define FULL_ATTN_GATE_DIM (FULL_NUM_HEADS * FULL_HEAD_DIM)

typedef struct cycle_config {
    char **fixtures;
    uint32_t n_fixtures;
    uint32_t full_layer;
} cycle_config;

typedef struct worker_config {
    char *hybrid_worker_bin;
    char *full_worker_bin;
    cycle_config *cycles;
    uint32_t n_cycles;
} worker_config;

typedef struct live_fixture {
    uint32_t layer, hidden, num_v_heads, num_k_heads, head_k_dim, head_v_dim, key_dim, value_dim, topk, inter;
    float *attn_norm_w, *post_attn_norm_w, *w_qkv, *w_z, *w_a, *w_b, *conv_w, *A_log, *dt_bias, *ssm_norm_w, *w_out;
    float *router_w;
    float *gate_shexp, *up_shexp, *down_shexp, *gate_inp_shexp;
} live_fixture;

typedef struct expert_cache_entry {
    int loaded;
    float *gate;
    float *up;
    float *down;
} expert_cache_entry;

typedef struct full_layer_cache {
    int loaded;
    float *attn_norm_w;
    float *post_attn_norm_w;
    float *q_proj_w;
    float *k_proj_w;
    float *v_proj_w;
    float *o_proj_w;
    float *q_norm_w;
    float *k_norm_w;
    float *router_w;
    float *gate_shexp_w;
    float *up_shexp_w;
    float *down_shexp_w;
    float *gate_inp_shexp_w;
    expert_cache_entry experts[ROUTER_COUNT];
} full_layer_cache;

typedef struct full_layer_state {
    uint32_t seq_len;
    uint32_t seq_cap;
    float *k_all;
    float *v_all;
    float *output_seq;
} full_layer_state;

typedef struct layer_step_scratch {
    float *input_ln;
    float *qkv_raw;
    float *qkv;
    float *z_raw;
    float *z;
    float *a_raw;
    float *a;
    float *b_raw;
    float *b;
    float *conv;
    float *q;
    float *k;
    float *v;
    float *beta;
    float *g;
    float *core;
    float *out_in;
    float *out_in_gg;
    float *out_proj;
    float *resid;
    float *post_ln;
    float *mlp;
    float *gate;
    float *up;
    float *act;
    float *down;
    float *shared_gate;
    float *shared_up;
    float *shared_act;
    float *shared;
    float *k_scaled;
    float *q_scaled;
    float *delta_tmp;
#ifdef QWEN36_UNIFIED_HAVE_GPU
    ds4_gpu_tensor *input_gpu;
    ds4_gpu_tensor *input_ln_gpu;
    ds4_gpu_tensor *z_gpu;
    ds4_gpu_tensor *a_gpu;
    ds4_gpu_tensor *b_gpu;
    ds4_gpu_tensor *out_in_gpu;
    ds4_gpu_tensor *out_proj_gpu;
    ds4_gpu_tensor *post_gpu;
    ds4_gpu_tensor *router_logits_gpu;
    ds4_gpu_tensor *shared_gate_gpu;
    ds4_gpu_tensor *shared_up_gpu;
    ds4_gpu_tensor *shared_mid_gpu;
    ds4_gpu_tensor *shared_out_gpu;
    ds4_gpu_tensor *expert_gate_gpu;
    ds4_gpu_tensor *expert_up_gpu;
    ds4_gpu_tensor *expert_mid_gpu;
    ds4_gpu_tensor *expert_down_gpu;
    ds4_gpu_tensor *q_gpu;
    ds4_gpu_tensor *k_gpu;
    ds4_gpu_tensor *v_gpu;
#endif
} layer_step_scratch;

typedef struct unified_session_state unified_session_state;

static int read_f32_tensor(const qwen36_gguf_file *gf, const qwen36_gguf_tensor *t, float **out, size_t elems, char *err, size_t err_cap);
static double now_ms(void);
static void free_full_layer_state(full_layer_state *st);
static int ensure_full_layer_state_cap(full_layer_state *st, uint32_t need, char *err, size_t err_cap);
static int ensure_owned_decode_cap(unified_session_state *st, uint32_t need, char *err, size_t err_cap);
static inline float sigmoidf_local(float x);
static inline float softplusf_local(float x);
static inline float siluf_local(float x);
static void rmsnorm_weight_plus1(const float *in, float *out, const float *w, uint32_t dim);
static void apply_rope_one_inplace(float *x, uint32_t pos);
static void matvec(const float *mat, const float *vec, float *out, uint32_t rows, uint32_t cols);
static void topk_softmax256(const float *logits, uint32_t k, uint32_t *idx, float *scores);
static int ensure_expert_loaded(const qwen36_gguf_file *gf, const qwen36_35a3b_q8_layer *layer, expert_cache_entry *cache, uint32_t expert_id, uint32_t hidden, uint32_t inter, char *err, size_t err_cap);
static void run_deltanet_head_step(float *state,
                                   uint32_t head_k_dim,
                                   uint32_t head_v_dim,
                                   const float *k_row,
                                   const float *q_row,
                                   const float *v_row,
                                   float gexp,
                                   float beta_t,
                                   float *k_scaled,
                                   float *q_scaled,
                                   float *delta_tmp,
                                   float *out_row);
static int alloc_layer_step_scratch(const live_fixture *fx, layer_step_scratch *sc, char *err, size_t err_cap);
static void free_layer_step_scratch(layer_step_scratch *sc);
#ifdef QWEN36_UNIFIED_HAVE_GPU
static int ensure_layer_gpu_step_scratch(const live_fixture *fx, layer_step_scratch *sc, char *err, size_t err_cap);
#endif

typedef struct layer_runtime {
    float *deltanet_state;
    float *conv_ring[3];
    uint32_t conv_ring_count;
    expert_cache_entry experts[ROUTER_COUNT];
    layer_step_scratch step;
} layer_runtime;

#include "qwen36_unified_full_gpu.inc"

struct unified_session_state {
    worker_config cfg;
    qwen36_gguf_file gf;
    qwen36_35a3b_q8_model model;
    live_fixture *fxs;
    layer_runtime *layers;
    uint32_t n_fx;
    uint32_t *cycle_fx_offsets;
    full_layer_cache *full_caches;
    full_layer_state *full_states;
    int enable_full_prefill_cpu;
#ifdef QWEN36_UNIFIED_HAVE_GPU
    int enable_full_prefill_gpu;
    uint32_t hybrid_gpu_cycles;
    mapped_file mf;
    full_layer_small_cache *full_small_caches;
#endif
    int verbose_logs;
    uint32_t seq_len;
    uint32_t owned_cap;
    uint32_t hidden;
    uint32_t *token_ids;
    float *owned_seq;
    int prefilled;
};

static int write_all(int fd, const void *buf, size_t n) {
    const uint8_t *p = (const uint8_t *)buf;
    while (n > 0) {
        ssize_t w = write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (w == 0) return 0;
        p += (size_t)w;
        n -= (size_t)w;
    }
    return 1;
}

static int read_all_fd(int fd, void *buf, size_t n) {
    uint8_t *p = (uint8_t *)buf;
    while (n > 0) {
        ssize_t r = read(fd, p, n);
        if (r < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (r == 0) return 0;
        p += (size_t)r;
        n -= (size_t)r;
    }
    return 1;
}

static int read_exact_file(FILE *fp, void *buf, size_t n) {
    return fread(buf, 1, n, fp) == n;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static char *xstrdup_local(const char *s) {
    size_t n = strlen(s) + 1;
    char *out = (char *)malloc(n);
    if (!out) return NULL;
    memcpy(out, s, n);
    return out;
}

static int split_csv(const char *s, char ***out_items, uint32_t *out_count) {
    char *copy = xstrdup_local(s);
    char *tok = NULL;
    char *save = NULL;
    char **items = NULL;
    uint32_t count = 0;
    if (!copy) return 0;
    for (tok = strtok_r(copy, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        char **next = (char **)realloc(items, (size_t)(count + 1) * sizeof(char *));
        if (!next) {
            free(copy);
            while (count > 0) free(items[--count]);
            free(items);
            return 0;
        }
        items = next;
        items[count] = xstrdup_local(tok);
        if (!items[count]) {
            free(copy);
            while (count > 0) free(items[--count]);
            free(items);
            return 0;
        }
        count++;
    }
    free(copy);
    *out_items = items;
    *out_count = count;
    return 1;
}

static int alloc_read_f32(FILE *fp, float **out, size_t count) {
    float *p = (float *)malloc(count * sizeof(float));
    if (!p) return 0;
    if (!read_exact_file(fp, p, count * sizeof(float))) {
        free(p);
        return 0;
    }
    *out = p;
    return 1;
}

static int read_f32_tensor(const qwen36_gguf_file *gf, const qwen36_gguf_tensor *t, float **out, size_t elems, char *err, size_t err_cap) {
    float *buf = (float *)malloc(elems * sizeof(float));
    if (!buf) return 0;
    if (!qwen36_gguf_read_tensor_bytes(gf, t, 0, buf, elems * sizeof(float), err, err_cap)) {
        free(buf);
        return 0;
    }
    *out = buf;
    return 1;
}

static void free_cycle(cycle_config *cy) {
    uint32_t i;
    if (!cy) return;
    for (i = 0; i < cy->n_fixtures; ++i) free(cy->fixtures[i]);
    free(cy->fixtures);
    memset(cy, 0, sizeof(*cy));
}

static void free_worker_config(worker_config *cfg) {
    uint32_t i;
    if (!cfg) return;
    free(cfg->hybrid_worker_bin);
    free(cfg->full_worker_bin);
    for (i = 0; i < cfg->n_cycles; ++i) free_cycle(&cfg->cycles[i]);
    free(cfg->cycles);
    memset(cfg, 0, sizeof(*cfg));
}

static void fixture_free(live_fixture *fx) {
    if (!fx) return;
    free(fx->attn_norm_w); free(fx->post_attn_norm_w); free(fx->w_qkv); free(fx->w_z); free(fx->w_a); free(fx->w_b); free(fx->conv_w); free(fx->A_log); free(fx->dt_bias); free(fx->ssm_norm_w); free(fx->w_out);
    free(fx->router_w); free(fx->gate_shexp); free(fx->up_shexp); free(fx->down_shexp); free(fx->gate_inp_shexp);
    memset(fx, 0, sizeof(*fx));
}

static void free_expert_cache(expert_cache_entry *cache) {
    uint32_t i;
    for (i = 0; i < ROUTER_COUNT; ++i) {
        free(cache[i].gate);
        free(cache[i].up);
        free(cache[i].down);
        memset(&cache[i], 0, sizeof(cache[i]));
    }
}

static void free_full_layer_cache(full_layer_cache *cache) {
    uint32_t i;
    if (!cache) return;
    free(cache->attn_norm_w);
    free(cache->post_attn_norm_w);
    free(cache->q_proj_w);
    free(cache->k_proj_w);
    free(cache->v_proj_w);
    free(cache->o_proj_w);
    free(cache->q_norm_w);
    free(cache->k_norm_w);
    free(cache->router_w);
    free(cache->gate_shexp_w);
    free(cache->up_shexp_w);
    free(cache->down_shexp_w);
    free(cache->gate_inp_shexp_w);
    for (i = 0; i < ROUTER_COUNT; ++i) {
        free(cache->experts[i].gate);
        free(cache->experts[i].up);
        free(cache->experts[i].down);
    }
    memset(cache, 0, sizeof(*cache));
}

static void free_full_layer_state(full_layer_state *st) {
    if (!st) return;
    free(st->k_all);
    free(st->v_all);
    free(st->output_seq);
    memset(st, 0, sizeof(*st));
}

static int ensure_full_layer_state_cap(full_layer_state *st, uint32_t need, char *err, size_t err_cap) {
    float *new_k = NULL, *new_v = NULL, *new_out = NULL;
    uint32_t new_cap = st->seq_cap ? st->seq_cap : 1u;
    if (need <= st->seq_cap) return 1;
    while (new_cap < need) {
        if (new_cap > UINT32_MAX / 2u) {
            new_cap = need;
            break;
        }
        new_cap *= 2u;
    }
    new_k = (float *)realloc(st->k_all, (size_t)new_cap * FULL_NUM_KV_HEADS * FULL_HEAD_DIM * sizeof(float));
    new_v = (float *)realloc(st->v_all, (size_t)new_cap * FULL_NUM_KV_HEADS * FULL_HEAD_DIM * sizeof(float));
    new_out = (float *)realloc(st->output_seq, (size_t)new_cap * FULL_HIDDEN * sizeof(float));
    if (!new_k || !new_v || !new_out) {
        if (err && err_cap) snprintf(err, err_cap, "oom full layer state grow");
        return 0;
    }
    st->k_all = new_k;
    st->v_all = new_v;
    st->output_seq = new_out;
    st->seq_cap = new_cap;
    return 1;
}

static int ensure_owned_decode_cap(unified_session_state *st, uint32_t need, char *err, size_t err_cap) {
    float *new_owned = NULL;
    uint32_t *new_ids = NULL;
    uint32_t new_cap = st->owned_cap ? st->owned_cap : 1u;
    if (need <= st->owned_cap) return 1;
    while (new_cap < need) {
        if (new_cap > UINT32_MAX / 2u) {
            new_cap = need;
            break;
        }
        new_cap *= 2u;
    }
    new_owned = (float *)realloc(st->owned_seq, (size_t)new_cap * st->hidden * sizeof(float));
    new_ids = (uint32_t *)realloc(st->token_ids, (size_t)new_cap * sizeof(uint32_t));
    if (!new_owned || !new_ids) {
        if (err && err_cap) snprintf(err, err_cap, "oom owned decode grow");
        return 0;
    }
    st->owned_seq = new_owned;
    st->token_ids = new_ids;
    st->owned_cap = new_cap;
    return 1;
}

static int parse_config(const char *path, worker_config *cfg, char *err, size_t err_cap) {
    FILE *fp = fopen(path, "r");
    char line[8192];
    memset(cfg, 0, sizeof(*cfg));
    if (!fp) {
        snprintf(err, err_cap, "open config failed: %s", strerror(errno));
        return 0;
    }
    while (fgets(line, sizeof(line), fp)) {
        char *cmd = NULL;
        char *rest = NULL;
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        if (len == 0 || line[0] == '#') continue;
        cmd = strtok(line, " \t");
        rest = strtok(NULL, "");
        if (!cmd || !rest) continue;
        while (*rest == ' ' || *rest == '\t') rest++;
        if (strcmp(cmd, "hybrid_worker_bin") == 0) {
            cfg->hybrid_worker_bin = xstrdup_local(rest);
        } else if (strcmp(cmd, "full_worker_bin") == 0) {
            cfg->full_worker_bin = xstrdup_local(rest);
        } else if (strcmp(cmd, "cycle") == 0) {
            char *layer_s = strtok(rest, " \t");
            char *fixtures_s = strtok(NULL, "");
            cycle_config *next_cycles = NULL;
            cycle_config *cy = NULL;
            if (!layer_s || !fixtures_s) {
                snprintf(err, err_cap, "bad cycle line");
                fclose(fp);
                return 0;
            }
            while (*fixtures_s == ' ' || *fixtures_s == '\t') fixtures_s++;
            next_cycles = (cycle_config *)realloc(cfg->cycles, (size_t)(cfg->n_cycles + 1) * sizeof(cycle_config));
            if (!next_cycles) {
                snprintf(err, err_cap, "oom cycles");
                fclose(fp);
                return 0;
            }
            cfg->cycles = next_cycles;
            cy = &cfg->cycles[cfg->n_cycles];
            memset(cy, 0, sizeof(*cy));
            cy->full_layer = (uint32_t)strtoul(layer_s, NULL, 10);
            if (!split_csv(fixtures_s, &cy->fixtures, &cy->n_fixtures) || cy->n_fixtures == 0 || cy->n_fixtures > 3) {
                snprintf(err, err_cap, "bad cycle fixtures");
                fclose(fp);
                return 0;
            }
            cfg->n_cycles++;
        }
    }
    fclose(fp);
    if (!cfg->hybrid_worker_bin) {
        snprintf(err, err_cap, "missing hybrid_worker_bin");
        return 0;
    }
    if (!cfg->full_worker_bin) {
        snprintf(err, err_cap, "missing full_worker_bin");
        return 0;
    }
    if (cfg->n_cycles == 0) {
        snprintf(err, err_cap, "no cycles configured");
        return 0;
    }
    return 1;
}

static int fixture_load(const char *path, live_fixture *fx) {
    FILE *fp = fopen(path, "rb");
    char magic[MAGIC_LEN];
    memset(fx, 0, sizeof(*fx));
    if (!fp) return 0;
    if (!read_exact_file(fp, magic, MAGIC_LEN) || memcmp(magic, MAGIC, MAGIC_LEN) != 0) { fclose(fp); return 0; }
    if (!read_exact_file(fp, &fx->layer, sizeof(uint32_t)) || !read_exact_file(fp, &fx->hidden, sizeof(uint32_t)) ||
        !read_exact_file(fp, &fx->num_v_heads, sizeof(uint32_t)) || !read_exact_file(fp, &fx->num_k_heads, sizeof(uint32_t)) || !read_exact_file(fp, &fx->head_k_dim, sizeof(uint32_t)) ||
        !read_exact_file(fp, &fx->head_v_dim, sizeof(uint32_t)) || !read_exact_file(fp, &fx->key_dim, sizeof(uint32_t)) || !read_exact_file(fp, &fx->value_dim, sizeof(uint32_t)) ||
        !read_exact_file(fp, &fx->topk, sizeof(uint32_t)) || !read_exact_file(fp, &fx->inter, sizeof(uint32_t))) { fclose(fp); return 0; }
#define R(name,count) if (!alloc_read_f32(fp, &fx->name, (count))) { fclose(fp); fixture_free(fx); return 0; }
    R(attn_norm_w, fx->hidden); R(post_attn_norm_w, fx->hidden); R(w_qkv, (size_t)(fx->key_dim * 2 + fx->value_dim) * fx->hidden); R(w_z, (size_t)fx->value_dim * fx->hidden);
    R(w_a, (size_t)fx->num_v_heads * fx->hidden); R(w_b, (size_t)fx->num_v_heads * fx->hidden); R(conv_w, (size_t)(fx->key_dim * 2 + fx->value_dim) * 4); R(A_log, fx->num_v_heads); R(dt_bias, fx->num_v_heads);
    R(ssm_norm_w, fx->head_v_dim); R(w_out, (size_t)fx->hidden * fx->value_dim); R(router_w, (size_t)ROUTER_COUNT * fx->hidden);
    R(gate_shexp, (size_t)fx->inter * fx->hidden); R(up_shexp, (size_t)fx->inter * fx->hidden); R(down_shexp, (size_t)fx->hidden * fx->inter); R(gate_inp_shexp, fx->hidden);
#undef R
    fclose(fp);
    return 1;
}

static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    const uint32_t exp = (h >> 10) & 0x1f;
    const uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            uint32_t e = 127 - 15 + 1u, m = mant;
            while ((m & 0x400u) == 0u) { m <<= 1; e--; }
            m &= 0x3ffu;
            bits = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    {
        union { uint32_t u; float f; } v = { bits };
        return v.f;
    }
}

static inline float sigmoidf_local(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float softplusf_local(float x) { return log1pf(expf(x)); }
static inline float siluf_local(float x) { return x / (1.0f + expf(-x)); }

static void matvec(const float *mat, const float *vec, float *out, uint32_t rows, uint32_t cols) {
    uint32_t r, c;
    for (r = 0; r < rows; ++r) {
        const float *row = mat + (size_t)r * cols;
        double sum = 0.0;
        for (c = 0; c < cols; ++c) sum += (double)row[c] * vec[c];
        out[r] = (float)sum;
    }
}

static void run_deltanet_head_step(float *state,
                                   uint32_t head_k_dim,
                                   uint32_t head_v_dim,
                                   const float *k_row,
                                   const float *q_row,
                                   const float *v_row,
                                   float gexp,
                                   float beta_t,
                                   float *k_scaled,
                                   float *q_scaled,
                                   float *delta_tmp,
                                   float *out_row) {
    uint32_t hd, vd;
    float qnorm = 0.0f, knorm = 0.0f;
    const float qscale_den = sqrtf((float)head_k_dim);

    for (hd = 0; hd < head_k_dim; ++hd) {
        qnorm += q_row[hd] * q_row[hd];
        knorm += k_row[hd] * k_row[hd];
    }
    qnorm = 1.0f / sqrtf(qnorm + 1e-6f);
    knorm = 1.0f / sqrtf(knorm + 1e-6f);
    for (hd = 0; hd < head_k_dim; ++hd) {
        k_scaled[hd] = k_row[hd] * knorm;
        q_scaled[hd] = q_row[hd] * qnorm / qscale_den;
    }

    for (hd = 0; hd < head_k_dim; ++hd) {
        float *row = state + (size_t)hd * head_v_dim;
        for (vd = 0; vd < head_v_dim; ++vd) row[vd] *= gexp;
    }

    for (vd = 0; vd < head_v_dim; ++vd) delta_tmp[vd] = v_row[vd];
    for (hd = 0; hd < head_k_dim; ++hd) {
        const float *row = state + (size_t)hd * head_v_dim;
        const float ks = k_scaled[hd];
        for (vd = 0; vd < head_v_dim; ++vd) delta_tmp[vd] -= row[vd] * ks;
    }
    for (vd = 0; vd < head_v_dim; ++vd) {
        delta_tmp[vd] *= beta_t;
        out_row[vd] = 0.0f;
    }

    for (hd = 0; hd < head_k_dim; ++hd) {
        float *row = state + (size_t)hd * head_v_dim;
        const float ks = k_scaled[hd];
        const float qs = q_scaled[hd];
        for (vd = 0; vd < head_v_dim; ++vd) {
            row[vd] += ks * delta_tmp[vd];
            out_row[vd] += row[vd] * qs;
        }
    }
}

static int decode_q8_rows(const qwen36_gguf_file *gf, const qwen36_gguf_tensor *t, uint32_t row_idx, uint32_t nrows, uint32_t row_elems, float *out, char *err, size_t err_cap) {
    const size_t row_size = 34u * (row_elems / 32u);
    uint8_t *buf = (uint8_t *)malloc((size_t)nrows * row_size);
    uint32_t r, j, k;
    if (!buf) return 0;
    if (!qwen36_gguf_read_tensor_bytes(gf, t, (uint64_t)row_idx * row_size, buf, (size_t)nrows * row_size, err, err_cap)) {
        free(buf);
        return 0;
    }
    for (r = 0; r < nrows; ++r) {
        const uint8_t *row = buf + (size_t)r * row_size;
        float *dst = out + (size_t)r * row_elems;
        for (j = 0; j < row_elems / 32u; ++j) {
            const uint8_t *block = row + j * 34u;
            const float dscale = f16_to_f32((uint16_t)(block[0] | (block[1] << 8)));
            const int8_t *qs = (const int8_t *)(block + 2);
            for (k = 0; k < 32u; ++k) dst[j * 32u + k] = dscale * (float)qs[k];
        }
    }
    free(buf);
    return 1;
}

static int decode_q8_row(const qwen36_gguf_file *gf, const qwen36_gguf_tensor *t, uint32_t row_idx, uint32_t hidden, float *out, char *err, size_t err_cap) {
    return decode_q8_rows(gf, t, row_idx, 1u, hidden, out, err, err_cap);
}

static void rotate_half(const float *x, float *out, uint32_t dim) {
    uint32_t half = dim / 2u;
    uint32_t i;
    for (i = 0; i < half; ++i) {
        out[i] = -x[half + i];
        out[half + i] = x[i];
    }
}

static void apply_rope_inplace(float *q, float *k, uint32_t pos) {
    static int init = 0;
    static float inv_freq[FULL_ROTARY_DIM / 2];
    float freqs[FULL_ROTARY_DIM / 2];
    float cosv[FULL_ROTARY_DIM];
    float sinv[FULL_ROTARY_DIM];
    float tmp_q[FULL_ROTARY_DIM];
    float tmp_k[FULL_ROTARY_DIM];
    uint32_t i;
    if (!init) {
        for (i = 0; i < FULL_ROTARY_DIM / 2; ++i) {
            inv_freq[i] = powf(10000000.0f, -(float)(2 * i) / (float)FULL_ROTARY_DIM);
        }
        init = 1;
    }
    for (i = 0; i < FULL_ROTARY_DIM / 2; ++i) freqs[i] = (float)pos * inv_freq[i];
    for (i = 0; i < FULL_ROTARY_DIM / 2; ++i) {
        cosv[i] = cosf(freqs[i]);
        cosv[i + FULL_ROTARY_DIM / 2] = cosv[i];
        sinv[i] = sinf(freqs[i]);
        sinv[i + FULL_ROTARY_DIM / 2] = sinv[i];
    }
    rotate_half(q, tmp_q, FULL_ROTARY_DIM);
    rotate_half(k, tmp_k, FULL_ROTARY_DIM);
    for (i = 0; i < FULL_ROTARY_DIM; ++i) {
        q[i] = q[i] * cosv[i] + tmp_q[i] * sinv[i];
        k[i] = k[i] * cosv[i] + tmp_k[i] * sinv[i];
    }
}

static void rmsnorm_weight_plus1(const float *in, float *out, const float *w, uint32_t dim) {
    double var = 0.0;
    uint32_t i;
    for (i = 0; i < dim; ++i) var += (double)in[i] * in[i];
    var /= (double)dim;
    for (i = 0; i < dim; ++i) out[i] = (float)(in[i] / sqrt(var + 1e-6)) * (1.0f + w[i]);
}

static void apply_rope_one_inplace(float *x, uint32_t pos) {
    float tmp[FULL_ROTARY_DIM];
    apply_rope_inplace(x, tmp, pos);
}

static void topk_softmax256(const float *logits, uint32_t k, uint32_t *idx, float *scores) {
    uint32_t i, j, m;
    for (i = 0; i < k; ++i) idx[i] = i;
    for (i = k; i < ROUTER_COUNT; ++i) {
        m = 0;
        for (j = 1; j < k; ++j) if (logits[idx[j]] < logits[idx[m]]) m = j;
        if (logits[i] > logits[idx[m]]) idx[m] = i;
    }
    for (i = 0; i < k; ++i) {
        for (j = i + 1; j < k; ++j) {
            if (logits[idx[j]] > logits[idx[i]]) {
                uint32_t tmp = idx[i];
                idx[i] = idx[j];
                idx[j] = tmp;
            }
        }
    }
    {
        float maxv = logits[idx[0]];
        double sum = 0.0;
        for (i = 0; i < k; ++i) sum += exp((double)logits[idx[i]] - maxv);
        for (i = 0; i < k; ++i) scores[i] = (float)(exp((double)logits[idx[i]] - maxv) / sum);
    }
}

static int ensure_full_expert_loaded(const qwen36_gguf_file *gf, const qwen36_35a3b_q8_layer *layer, full_layer_cache *cache, uint32_t expert_id, char *err, size_t err_cap) {
    expert_cache_entry *e = &cache->experts[expert_id];
    if (e->loaded) return 1;
    e->gate = (float *)malloc((size_t)FULL_INTER * FULL_HIDDEN * sizeof(float));
    e->up = (float *)malloc((size_t)FULL_INTER * FULL_HIDDEN * sizeof(float));
    e->down = (float *)malloc((size_t)FULL_HIDDEN * FULL_INTER * sizeof(float));
    if (!e->gate || !e->up || !e->down) return 0;
    if (!decode_q8_rows(gf, layer->ffn_gate_exps, expert_id * FULL_INTER, FULL_INTER, FULL_HIDDEN, e->gate, err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->ffn_up_exps, expert_id * FULL_INTER, FULL_INTER, FULL_HIDDEN, e->up, err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->ffn_down_exps, expert_id * FULL_HIDDEN, FULL_HIDDEN, FULL_INTER, e->down, err, err_cap)) return 0;
    e->loaded = 1;
    return 1;
}

static int ensure_full_layer_loaded(const qwen36_gguf_file *gf, const qwen36_35a3b_q8_layer *layer, full_layer_cache *cache, char *err, size_t err_cap) {
    if (cache->loaded) return 1;
    if (!read_f32_tensor(gf, layer->attn_norm, &cache->attn_norm_w, FULL_HIDDEN, err, err_cap)) return 0;
    if (!read_f32_tensor(gf, layer->post_attn_norm, &cache->post_attn_norm_w, FULL_HIDDEN, err, err_cap)) return 0;
    if (!read_f32_tensor(gf, layer->attn_q_norm, &cache->q_norm_w, FULL_HEAD_DIM, err, err_cap)) return 0;
    if (!read_f32_tensor(gf, layer->attn_k_norm, &cache->k_norm_w, FULL_HEAD_DIM, err, err_cap)) return 0;
    if (!read_f32_tensor(gf, layer->ffn_gate_inp, &cache->router_w, (size_t)ROUTER_COUNT * FULL_HIDDEN, err, err_cap)) return 0;
    cache->gate_shexp_w = (float *)malloc((size_t)FULL_INTER * FULL_HIDDEN * sizeof(float));
    cache->up_shexp_w = (float *)malloc((size_t)FULL_INTER * FULL_HIDDEN * sizeof(float));
    cache->down_shexp_w = (float *)malloc((size_t)FULL_HIDDEN * FULL_INTER * sizeof(float));
    if (!cache->gate_shexp_w || !cache->up_shexp_w || !cache->down_shexp_w) return 0;
    if (!decode_q8_rows(gf, layer->ffn_gate_shexp, 0, FULL_INTER, FULL_HIDDEN, cache->gate_shexp_w, err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->ffn_up_shexp, 0, FULL_INTER, FULL_HIDDEN, cache->up_shexp_w, err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->ffn_down_shexp, 0, FULL_HIDDEN, FULL_INTER, cache->down_shexp_w, err, err_cap)) return 0;
    if (!read_f32_tensor(gf, layer->ffn_gate_inp_shexp, &cache->gate_inp_shexp_w, FULL_HIDDEN, err, err_cap)) return 0;
    cache->q_proj_w = (float *)malloc((size_t)FULL_ATTN_GATE_DIM * 2u * FULL_HIDDEN * sizeof(float));
    cache->k_proj_w = (float *)malloc((size_t)FULL_NUM_KV_HEADS * FULL_HEAD_DIM * FULL_HIDDEN * sizeof(float));
    cache->v_proj_w = (float *)malloc((size_t)FULL_NUM_KV_HEADS * FULL_HEAD_DIM * FULL_HIDDEN * sizeof(float));
    cache->o_proj_w = (float *)malloc((size_t)FULL_HIDDEN * FULL_ATTN_GATE_DIM * sizeof(float));
    if (!cache->q_proj_w || !cache->k_proj_w || !cache->v_proj_w || !cache->o_proj_w) return 0;
    if (!decode_q8_rows(gf, layer->attn_q, 0, FULL_ATTN_GATE_DIM * 2u, FULL_HIDDEN, cache->q_proj_w, err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->attn_k, 0, FULL_NUM_KV_HEADS * FULL_HEAD_DIM, FULL_HIDDEN, cache->k_proj_w, err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->attn_v, 0, FULL_NUM_KV_HEADS * FULL_HEAD_DIM, FULL_HIDDEN, cache->v_proj_w, err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->attn_output, 0, FULL_HIDDEN, FULL_ATTN_GATE_DIM, cache->o_proj_w, err, err_cap)) return 0;
    cache->loaded = 1;
    return 1;
}

static int run_full_layer_prefill_cpu(const qwen36_35a3b_q8_layer *layer,
                                      full_layer_cache *cache,
                                      full_layer_state *state,
                                      const qwen36_gguf_file *gf,
                                      const float *input_seq,
                                      uint32_t seq_len,
                                      char *err,
                                      size_t err_cap) {
    float *attn_in = NULL, *q_all = NULL, *k_all = NULL, *v_all = NULL;
    float *gate_all = NULL, *attn_out_flat = NULL, *proj_out = NULL, *post_ln = NULL;
    float *gate = NULL, *up = NULL, *act = NULL, *down = NULL;
    float *shared_gate = NULL, *shared_up = NULL, *shared_act = NULL, *shared = NULL;
    float router_logits[ROUTER_COUNT];
    uint32_t router_idx[8];
    float router_scores[8];
    float qg[FULL_ATTN_GATE_DIM * 2u];
    float kk[FULL_NUM_KV_HEADS * FULL_HEAD_DIM];
    float vv[FULL_NUM_KV_HEADS * FULL_HEAD_DIM];
    uint32_t t, h, d, vd, i;
    if (!ensure_full_layer_loaded(gf, layer, cache, err, err_cap)) return 0;
    free_full_layer_state(state);
    attn_in = (float *)calloc((size_t)seq_len * FULL_HIDDEN, sizeof(float));
    q_all = (float *)calloc((size_t)seq_len * FULL_NUM_HEADS * FULL_HEAD_DIM, sizeof(float));
    k_all = (float *)calloc((size_t)seq_len * FULL_NUM_KV_HEADS * FULL_HEAD_DIM, sizeof(float));
    v_all = (float *)calloc((size_t)seq_len * FULL_NUM_KV_HEADS * FULL_HEAD_DIM, sizeof(float));
    gate_all = (float *)calloc((size_t)seq_len * FULL_ATTN_GATE_DIM, sizeof(float));
    attn_out_flat = (float *)calloc((size_t)seq_len * FULL_ATTN_GATE_DIM, sizeof(float));
    proj_out = (float *)calloc((size_t)seq_len * FULL_HIDDEN, sizeof(float));
    post_ln = (float *)calloc((size_t)seq_len * FULL_HIDDEN, sizeof(float));
    gate = (float *)calloc(FULL_INTER, sizeof(float));
    up = (float *)calloc(FULL_INTER, sizeof(float));
    act = (float *)calloc(FULL_INTER, sizeof(float));
    down = (float *)calloc(FULL_HIDDEN, sizeof(float));
    shared_gate = (float *)calloc(FULL_INTER, sizeof(float));
    shared_up = (float *)calloc(FULL_INTER, sizeof(float));
    shared_act = (float *)calloc(FULL_INTER, sizeof(float));
    shared = (float *)calloc(FULL_HIDDEN, sizeof(float));
    if (!attn_in || !q_all || !k_all || !v_all || !gate_all || !attn_out_flat || !proj_out || !post_ln ||
        !gate || !up || !act || !down || !shared_gate || !shared_up || !shared_act || !shared) {
        snprintf(err, err_cap, "oom full prefill cpu");
        goto fail;
    }
    state->output_seq = (float *)calloc((size_t)seq_len * FULL_HIDDEN, sizeof(float));
    if (!state->output_seq) {
        snprintf(err, err_cap, "oom full output state");
        goto fail;
    }
    for (t = 0; t < seq_len; ++t) {
        rmsnorm_weight_plus1(input_seq + (size_t)t * FULL_HIDDEN, attn_in + (size_t)t * FULL_HIDDEN, cache->attn_norm_w, FULL_HIDDEN);
        matvec(cache->q_proj_w, attn_in + (size_t)t * FULL_HIDDEN, qg, FULL_ATTN_GATE_DIM * 2u, FULL_HIDDEN);
        matvec(cache->k_proj_w, attn_in + (size_t)t * FULL_HIDDEN, kk, FULL_NUM_KV_HEADS * FULL_HEAD_DIM, FULL_HIDDEN);
        matvec(cache->v_proj_w, attn_in + (size_t)t * FULL_HIDDEN, vv, FULL_NUM_KV_HEADS * FULL_HEAD_DIM, FULL_HIDDEN);
        for (h = 0; h < FULL_NUM_HEADS; ++h) {
            rmsnorm_weight_plus1(qg + (size_t)h * (FULL_HEAD_DIM * 2u), q_all + ((size_t)t * FULL_NUM_HEADS + h) * FULL_HEAD_DIM, cache->q_norm_w, FULL_HEAD_DIM);
            memcpy(gate_all + (size_t)t * FULL_ATTN_GATE_DIM + (size_t)h * FULL_HEAD_DIM, qg + (size_t)h * (FULL_HEAD_DIM * 2u) + FULL_HEAD_DIM, FULL_HEAD_DIM * sizeof(float));
        }
        for (h = 0; h < FULL_NUM_KV_HEADS; ++h) {
            rmsnorm_weight_plus1(kk + (size_t)h * FULL_HEAD_DIM, k_all + ((size_t)t * FULL_NUM_KV_HEADS + h) * FULL_HEAD_DIM, cache->k_norm_w, FULL_HEAD_DIM);
            memcpy(v_all + ((size_t)t * FULL_NUM_KV_HEADS + h) * FULL_HEAD_DIM, vv + (size_t)h * FULL_HEAD_DIM, FULL_HEAD_DIM * sizeof(float));
        }
        for (h = 0; h < FULL_NUM_HEADS; ++h) {
            apply_rope_inplace(q_all + ((size_t)t * FULL_NUM_HEADS + h) * FULL_HEAD_DIM,
                               k_all + ((size_t)t * FULL_NUM_KV_HEADS + (h / (FULL_NUM_HEADS / FULL_NUM_KV_HEADS))) * FULL_HEAD_DIM,
                               t);
        }
    }
    for (t = 0; t < seq_len; ++t) {
        for (h = 0; h < FULL_NUM_HEADS; ++h) {
            const uint32_t kvh = h / (FULL_NUM_HEADS / FULL_NUM_KV_HEADS);
            float scores[4096];
            float maxv = -1e30f;
            float sum = 0.0f;
            float *out = attn_out_flat + (size_t)t * FULL_ATTN_GATE_DIM + (size_t)h * FULL_HEAD_DIM;
            memset(out, 0, FULL_HEAD_DIM * sizeof(float));
            for (uint32_t s = 0; s <= t; ++s) {
                double dot = 0.0;
                const float *qv = q_all + ((size_t)t * FULL_NUM_HEADS + h) * FULL_HEAD_DIM;
                const float *kv = k_all + ((size_t)s * FULL_NUM_KV_HEADS + kvh) * FULL_HEAD_DIM;
                for (d = 0; d < FULL_HEAD_DIM; ++d) dot += (double)qv[d] * kv[d];
                scores[s] = (float)(dot / sqrt((double)FULL_HEAD_DIM));
                if (scores[s] > maxv) maxv = scores[s];
            }
            for (uint32_t s = 0; s <= t; ++s) {
                scores[s] = expf(scores[s] - maxv);
                sum += scores[s];
            }
            for (uint32_t s = 0; s <= t; ++s) {
                const float w = scores[s] / sum;
                const float *vvh = v_all + ((size_t)s * FULL_NUM_KV_HEADS + kvh) * FULL_HEAD_DIM;
                for (d = 0; d < FULL_HEAD_DIM; ++d) out[d] += w * vvh[d];
            }
            for (d = 0; d < FULL_HEAD_DIM; ++d) out[d] *= sigmoidf_local(gate_all[(size_t)t * FULL_ATTN_GATE_DIM + (size_t)h * FULL_HEAD_DIM + d]);
        }
        matvec(cache->o_proj_w, attn_out_flat + (size_t)t * FULL_ATTN_GATE_DIM, proj_out + (size_t)t * FULL_HIDDEN, FULL_HIDDEN, FULL_ATTN_GATE_DIM);
        for (d = 0; d < FULL_HIDDEN; ++d) state->output_seq[(size_t)t * FULL_HIDDEN + d] = input_seq[(size_t)t * FULL_HIDDEN + d] + proj_out[(size_t)t * FULL_HIDDEN + d];
        rmsnorm_weight_plus1(state->output_seq + (size_t)t * FULL_HIDDEN, post_ln + (size_t)t * FULL_HIDDEN, cache->post_attn_norm_w, FULL_HIDDEN);
        memset(state->output_seq + (size_t)t * FULL_HIDDEN, 0, FULL_HIDDEN * sizeof(float));
        matvec(cache->router_w, post_ln + (size_t)t * FULL_HIDDEN, router_logits, ROUTER_COUNT, FULL_HIDDEN);
        topk_softmax256(router_logits, 8, router_idx, router_scores);
        for (i = 0; i < 8; ++i) {
            if (!ensure_full_expert_loaded(gf, layer, cache, router_idx[i], err, err_cap)) goto fail;
            matvec(cache->experts[router_idx[i]].gate, post_ln + (size_t)t * FULL_HIDDEN, gate, FULL_INTER, FULL_HIDDEN);
            matvec(cache->experts[router_idx[i]].up, post_ln + (size_t)t * FULL_HIDDEN, up, FULL_INTER, FULL_HIDDEN);
            for (vd = 0; vd < FULL_INTER; ++vd) act[vd] = siluf_local(gate[vd]) * up[vd];
            matvec(cache->experts[router_idx[i]].down, act, down, FULL_HIDDEN, FULL_INTER);
            for (d = 0; d < FULL_HIDDEN; ++d) state->output_seq[(size_t)t * FULL_HIDDEN + d] += down[d] * router_scores[i];
        }
        matvec(cache->gate_shexp_w, post_ln + (size_t)t * FULL_HIDDEN, shared_gate, FULL_INTER, FULL_HIDDEN);
        matvec(cache->up_shexp_w, post_ln + (size_t)t * FULL_HIDDEN, shared_up, FULL_INTER, FULL_HIDDEN);
        for (vd = 0; vd < FULL_INTER; ++vd) shared_act[vd] = siluf_local(shared_gate[vd]) * shared_up[vd];
        matvec(cache->down_shexp_w, shared_act, shared, FULL_HIDDEN, FULL_INTER);
        {
            float s = 0.0f;
            for (d = 0; d < FULL_HIDDEN; ++d) s += post_ln[(size_t)t * FULL_HIDDEN + d] * cache->gate_inp_shexp_w[d];
            s = sigmoidf_local(s);
            for (d = 0; d < FULL_HIDDEN; ++d) {
                state->output_seq[(size_t)t * FULL_HIDDEN + d] =
                    input_seq[(size_t)t * FULL_HIDDEN + d] +
                    proj_out[(size_t)t * FULL_HIDDEN + d] +
                    state->output_seq[(size_t)t * FULL_HIDDEN + d] +
                    shared[d] * s;
            }
        }
    }
    state->k_all = k_all;
    state->v_all = v_all;
    state->seq_len = seq_len;
    state->seq_cap = seq_len;
    k_all = NULL;
    v_all = NULL;
    free(attn_in); free(q_all); free(k_all); free(v_all); free(gate_all); free(attn_out_flat); free(proj_out); free(post_ln);
    free(gate); free(up); free(act); free(down); free(shared_gate); free(shared_up); free(shared_act); free(shared);
    return 1;
fail:
    free(attn_in); free(q_all); free(k_all); free(v_all); free(gate_all); free(attn_out_flat); free(proj_out); free(post_ln);
    free(gate); free(up); free(act); free(down); free(shared_gate); free(shared_up); free(shared_act); free(shared);
    return 0;
}

static int run_full_layer_step_cpu(const qwen36_35a3b_q8_layer *layer,
                                   full_layer_cache *cache,
                                   full_layer_state *state,
                                   const qwen36_gguf_file *gf,
                                   const float *input_row,
                                   float *out_row,
                                   char *err,
                                   size_t err_cap) {
    float attn_in[FULL_HIDDEN];
    float qg[FULL_ATTN_GATE_DIM * 2u];
    float kk[FULL_NUM_KV_HEADS * FULL_HEAD_DIM];
    float vv[FULL_NUM_KV_HEADS * FULL_HEAD_DIM];
    float q_cur[FULL_HEAD_DIM * FULL_NUM_HEADS];
    float k_cur[FULL_HEAD_DIM * FULL_NUM_KV_HEADS];
    float v_cur[FULL_HEAD_DIM * FULL_NUM_KV_HEADS];
    float gate_cur[FULL_ATTN_GATE_DIM];
    float attn_out_flat[FULL_ATTN_GATE_DIM];
    float proj_out[FULL_HIDDEN];
    float post_ln[FULL_HIDDEN];
    float gate[FULL_INTER];
    float up[FULL_INTER];
    float act[FULL_INTER];
    float down[FULL_HIDDEN];
    float shared_gate[FULL_INTER];
    float shared_up[FULL_INTER];
    float shared_act[FULL_INTER];
    float shared[FULL_HIDDEN];
    float router_logits[ROUTER_COUNT];
    uint32_t router_idx[8];
    float router_scores[8];
    uint32_t pos, h, d, vd, i;
    if (!ensure_full_layer_loaded(gf, layer, cache, err, err_cap)) return 0;
    pos = state->seq_len;
    rmsnorm_weight_plus1(input_row, attn_in, cache->attn_norm_w, FULL_HIDDEN);
    matvec(cache->q_proj_w, attn_in, qg, FULL_ATTN_GATE_DIM * 2u, FULL_HIDDEN);
    matvec(cache->k_proj_w, attn_in, kk, FULL_NUM_KV_HEADS * FULL_HEAD_DIM, FULL_HIDDEN);
    matvec(cache->v_proj_w, attn_in, vv, FULL_NUM_KV_HEADS * FULL_HEAD_DIM, FULL_HIDDEN);
    for (h = 0; h < FULL_NUM_HEADS; ++h) {
        rmsnorm_weight_plus1(qg + (size_t)h * (FULL_HEAD_DIM * 2u), q_cur + (size_t)h * FULL_HEAD_DIM, cache->q_norm_w, FULL_HEAD_DIM);
        memcpy(gate_cur + (size_t)h * FULL_HEAD_DIM, qg + (size_t)h * (FULL_HEAD_DIM * 2u) + FULL_HEAD_DIM, FULL_HEAD_DIM * sizeof(float));
    }
    for (h = 0; h < FULL_NUM_KV_HEADS; ++h) {
        rmsnorm_weight_plus1(kk + (size_t)h * FULL_HEAD_DIM, k_cur + (size_t)h * FULL_HEAD_DIM, cache->k_norm_w, FULL_HEAD_DIM);
        memcpy(v_cur + (size_t)h * FULL_HEAD_DIM, vv + (size_t)h * FULL_HEAD_DIM, FULL_HEAD_DIM * sizeof(float));
    }
    for (h = 0; h < FULL_NUM_HEADS; ++h) {
        apply_rope_inplace(q_cur + (size_t)h * FULL_HEAD_DIM,
                           k_cur + (size_t)(h / (FULL_NUM_HEADS / FULL_NUM_KV_HEADS)) * FULL_HEAD_DIM,
                           pos);
    }
    memset(attn_out_flat, 0, sizeof(attn_out_flat));
    for (h = 0; h < FULL_NUM_HEADS; ++h) {
        const uint32_t kvh = h / (FULL_NUM_HEADS / FULL_NUM_KV_HEADS);
        double scores[8192];
        double maxv = -1e300, sum = 0.0;
        float *out = attn_out_flat + (size_t)h * FULL_HEAD_DIM;
        const float *qv = q_cur + (size_t)h * FULL_HEAD_DIM;
        for (uint32_t s = 0; s <= pos; ++s) {
            const float *kv = (s == pos) ? (k_cur + (size_t)kvh * FULL_HEAD_DIM) : (state->k_all + ((size_t)s * FULL_NUM_KV_HEADS + kvh) * FULL_HEAD_DIM);
            double dot = 0.0;
            for (d = 0; d < FULL_HEAD_DIM; ++d) dot += (double)qv[d] * kv[d];
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
            for (d = 0; d < FULL_HEAD_DIM; ++d) out[d] += w * vvh[d];
        }
        for (d = 0; d < FULL_HEAD_DIM; ++d) out[d] *= sigmoidf_local(gate_cur[(size_t)h * FULL_HEAD_DIM + d]);
    }
    matvec(cache->o_proj_w, attn_out_flat, proj_out, FULL_HIDDEN, FULL_ATTN_GATE_DIM);
    for (d = 0; d < FULL_HIDDEN; ++d) out_row[d] = input_row[d] + proj_out[d];
    rmsnorm_weight_plus1(out_row, post_ln, cache->post_attn_norm_w, FULL_HIDDEN);
    memset(out_row, 0, (size_t)FULL_HIDDEN * sizeof(float));
    matvec(cache->router_w, post_ln, router_logits, ROUTER_COUNT, FULL_HIDDEN);
    topk_softmax256(router_logits, 8, router_idx, router_scores);
    for (i = 0; i < 8; ++i) {
        if (!ensure_full_expert_loaded(gf, layer, cache, router_idx[i], err, err_cap)) return 0;
        matvec(cache->experts[router_idx[i]].gate, post_ln, gate, FULL_INTER, FULL_HIDDEN);
        matvec(cache->experts[router_idx[i]].up, post_ln, up, FULL_INTER, FULL_HIDDEN);
        for (vd = 0; vd < FULL_INTER; ++vd) act[vd] = siluf_local(gate[vd]) * up[vd];
        matvec(cache->experts[router_idx[i]].down, act, down, FULL_HIDDEN, FULL_INTER);
        for (d = 0; d < FULL_HIDDEN; ++d) out_row[d] += down[d] * router_scores[i];
    }
    matvec(cache->gate_shexp_w, post_ln, shared_gate, FULL_INTER, FULL_HIDDEN);
    matvec(cache->up_shexp_w, post_ln, shared_up, FULL_INTER, FULL_HIDDEN);
    for (vd = 0; vd < FULL_INTER; ++vd) shared_act[vd] = siluf_local(shared_gate[vd]) * shared_up[vd];
    matvec(cache->down_shexp_w, shared_act, shared, FULL_HIDDEN, FULL_INTER);
    {
        float s = 0.0f;
        for (d = 0; d < FULL_HIDDEN; ++d) s += post_ln[d] * cache->gate_inp_shexp_w[d];
        s = sigmoidf_local(s);
        for (d = 0; d < FULL_HIDDEN; ++d) out_row[d] = input_row[d] + proj_out[d] + out_row[d] + shared[d] * s;
    }
    if (!ensure_full_layer_state_cap(state, pos + 1u, err, err_cap)) return 0;
    memcpy(state->k_all + (size_t)pos * FULL_NUM_KV_HEADS * FULL_HEAD_DIM, k_cur, sizeof(k_cur));
    memcpy(state->v_all + (size_t)pos * FULL_NUM_KV_HEADS * FULL_HEAD_DIM, v_cur, sizeof(v_cur));
    memcpy(state->output_seq + (size_t)pos * FULL_HIDDEN, out_row, (size_t)FULL_HIDDEN * sizeof(float));
    state->seq_len = pos + 1u;
    return 1;
}

static int ensure_expert_loaded(const qwen36_gguf_file *gf, const qwen36_35a3b_q8_layer *layer, expert_cache_entry *cache, uint32_t expert_id, uint32_t hidden, uint32_t inter, char *err, size_t err_cap) {
    expert_cache_entry *e = &cache[expert_id];
    if (e->loaded) return 1;
    e->gate = (float *)malloc((size_t)inter * hidden * sizeof(float));
    e->up = (float *)malloc((size_t)inter * hidden * sizeof(float));
    e->down = (float *)malloc((size_t)hidden * inter * sizeof(float));
    if (!e->gate || !e->up || !e->down) return 0;
    if (!decode_q8_rows(gf, layer->ffn_gate_exps, expert_id * inter, inter, hidden, e->gate, err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->ffn_up_exps, expert_id * inter, inter, hidden, e->up, err, err_cap)) return 0;
    if (!decode_q8_rows(gf, layer->ffn_down_exps, expert_id * hidden, hidden, inter, e->down, err, err_cap)) return 0;
    e->loaded = 1;
    return 1;
}

static int alloc_layer_step_scratch(const live_fixture *fx, layer_step_scratch *sc, char *err, size_t err_cap) {
    const size_t hidden = (size_t)fx->hidden;
    const size_t qkv_dim = (size_t)fx->key_dim * 2u + fx->value_dim;
    const size_t value_dim = (size_t)fx->value_dim;
    const size_t num_v_heads = (size_t)fx->num_v_heads;
    const size_t head_k = (size_t)fx->head_k_dim;
    const size_t head_v = (size_t)fx->head_v_dim;
    const size_t inter = (size_t)fx->inter;

    memset(sc, 0, sizeof(*sc));
    sc->input_ln = (float *)malloc(hidden * sizeof(float));
    sc->qkv_raw = (float *)malloc(qkv_dim * sizeof(float));
    sc->qkv = (float *)malloc(qkv_dim * sizeof(float));
    sc->z_raw = (float *)malloc(value_dim * sizeof(float));
    sc->z = (float *)malloc(value_dim * sizeof(float));
    sc->a_raw = (float *)malloc(num_v_heads * sizeof(float));
    sc->a = (float *)malloc(num_v_heads * sizeof(float));
    sc->b_raw = (float *)malloc(num_v_heads * sizeof(float));
    sc->b = (float *)malloc(num_v_heads * sizeof(float));
    sc->conv = (float *)malloc(qkv_dim * sizeof(float));
    sc->q = (float *)malloc(num_v_heads * head_k * sizeof(float));
    sc->k = (float *)malloc(num_v_heads * head_k * sizeof(float));
    sc->v = (float *)malloc(num_v_heads * head_v * sizeof(float));
    sc->beta = (float *)malloc(num_v_heads * sizeof(float));
    sc->g = (float *)malloc(num_v_heads * sizeof(float));
    sc->core = (float *)malloc(num_v_heads * head_v * sizeof(float));
    sc->out_in = (float *)malloc(value_dim * sizeof(float));
    sc->out_in_gg = (float *)malloc(value_dim * sizeof(float));
    sc->out_proj = (float *)malloc(hidden * sizeof(float));
    sc->resid = (float *)malloc(hidden * sizeof(float));
    sc->post_ln = (float *)malloc(hidden * sizeof(float));
    sc->mlp = (float *)malloc(hidden * sizeof(float));
    sc->gate = (float *)malloc(inter * sizeof(float));
    sc->up = (float *)malloc(inter * sizeof(float));
    sc->act = (float *)malloc(inter * sizeof(float));
    sc->down = (float *)malloc(hidden * sizeof(float));
    sc->shared_gate = (float *)malloc(inter * sizeof(float));
    sc->shared_up = (float *)malloc(inter * sizeof(float));
    sc->shared_act = (float *)malloc(inter * sizeof(float));
    sc->shared = (float *)malloc(hidden * sizeof(float));
    sc->k_scaled = (float *)malloc(head_k * sizeof(float));
    sc->q_scaled = (float *)malloc(head_k * sizeof(float));
    sc->delta_tmp = (float *)malloc(head_v * sizeof(float));
    if (!sc->input_ln || !sc->qkv_raw || !sc->qkv || !sc->z_raw || !sc->z || !sc->a_raw || !sc->a || !sc->b_raw || !sc->b ||
        !sc->conv || !sc->q || !sc->k || !sc->v || !sc->beta || !sc->g || !sc->core || !sc->out_in || !sc->out_in_gg ||
        !sc->out_proj || !sc->resid || !sc->post_ln || !sc->mlp || !sc->gate || !sc->up || !sc->act || !sc->down ||
        !sc->shared_gate || !sc->shared_up || !sc->shared_act || !sc->shared || !sc->k_scaled || !sc->q_scaled || !sc->delta_tmp) {
        snprintf(err, err_cap, "oom step scratch blk.%u", fx->layer);
        free_layer_step_scratch(sc);
        return 0;
    }
    return 1;
}

static void free_layer_step_scratch(layer_step_scratch *sc) {
    if (!sc) return;
    free(sc->input_ln);
    free(sc->qkv_raw);
    free(sc->qkv);
    free(sc->z_raw);
    free(sc->z);
    free(sc->a_raw);
    free(sc->a);
    free(sc->b_raw);
    free(sc->b);
    free(sc->conv);
    free(sc->q);
    free(sc->k);
    free(sc->v);
    free(sc->beta);
    free(sc->g);
    free(sc->core);
    free(sc->out_in);
    free(sc->out_in_gg);
    free(sc->out_proj);
    free(sc->resid);
    free(sc->post_ln);
    free(sc->mlp);
    free(sc->gate);
    free(sc->up);
    free(sc->act);
    free(sc->down);
    free(sc->shared_gate);
    free(sc->shared_up);
    free(sc->shared_act);
    free(sc->shared);
    free(sc->k_scaled);
    free(sc->q_scaled);
    free(sc->delta_tmp);
#ifdef QWEN36_UNIFIED_HAVE_GPU
    ds4_gpu_tensor_free(sc->out_proj_gpu);
    ds4_gpu_tensor_free(sc->out_in_gpu);
    ds4_gpu_tensor_free(sc->b_gpu);
    ds4_gpu_tensor_free(sc->a_gpu);
    ds4_gpu_tensor_free(sc->z_gpu);
    ds4_gpu_tensor_free(sc->input_ln_gpu);
    ds4_gpu_tensor_free(sc->input_gpu);
    ds4_gpu_tensor_free(sc->expert_down_gpu);
    ds4_gpu_tensor_free(sc->expert_mid_gpu);
    ds4_gpu_tensor_free(sc->expert_up_gpu);
    ds4_gpu_tensor_free(sc->expert_gate_gpu);
    ds4_gpu_tensor_free(sc->shared_out_gpu);
    ds4_gpu_tensor_free(sc->shared_mid_gpu);
    ds4_gpu_tensor_free(sc->shared_up_gpu);
    ds4_gpu_tensor_free(sc->shared_gate_gpu);
    ds4_gpu_tensor_free(sc->router_logits_gpu);
    ds4_gpu_tensor_free(sc->post_gpu);
    ds4_gpu_tensor_free(sc->q_gpu);
    ds4_gpu_tensor_free(sc->k_gpu);
    ds4_gpu_tensor_free(sc->v_gpu);
#endif
    memset(sc, 0, sizeof(*sc));
}

#ifdef QWEN36_UNIFIED_HAVE_GPU
static int ensure_layer_gpu_step_scratch(const live_fixture *fx, layer_step_scratch *sc, char *err, size_t err_cap) {
    const uint64_t hidden_bytes = (uint64_t)fx->hidden * sizeof(float);
    const uint64_t z_bytes = (uint64_t)fx->value_dim * sizeof(float);
    const uint64_t ab_bytes = (uint64_t)fx->num_v_heads * sizeof(float);
    const uint64_t out_in_bytes = (uint64_t)fx->value_dim * sizeof(float);
    const uint64_t router_logits_bytes = (uint64_t)ROUTER_COUNT * sizeof(float);
    const uint64_t inter_bytes = (uint64_t)fx->inter * sizeof(float);
    if (sc->input_gpu) return 1;
    sc->input_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    sc->input_ln_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    sc->z_gpu = ds4_gpu_tensor_alloc(z_bytes);
    sc->a_gpu = ds4_gpu_tensor_alloc(ab_bytes);
    sc->b_gpu = ds4_gpu_tensor_alloc(ab_bytes);
    sc->out_in_gpu = ds4_gpu_tensor_alloc(out_in_bytes);
    sc->out_proj_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    sc->post_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    sc->router_logits_gpu = ds4_gpu_tensor_alloc(router_logits_bytes);
    sc->shared_gate_gpu = ds4_gpu_tensor_alloc(inter_bytes);
    sc->shared_up_gpu = ds4_gpu_tensor_alloc(inter_bytes);
    sc->shared_mid_gpu = ds4_gpu_tensor_alloc(inter_bytes);
    sc->shared_out_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    sc->expert_gate_gpu = ds4_gpu_tensor_alloc(inter_bytes);
    sc->expert_up_gpu = ds4_gpu_tensor_alloc(inter_bytes);
    sc->expert_mid_gpu = ds4_gpu_tensor_alloc(inter_bytes);
    sc->expert_down_gpu = ds4_gpu_tensor_alloc(hidden_bytes);
    sc->q_gpu = ds4_gpu_tensor_alloc((uint64_t)fx->key_dim * sizeof(float));
    sc->k_gpu = ds4_gpu_tensor_alloc((uint64_t)fx->key_dim * sizeof(float));
    sc->v_gpu = ds4_gpu_tensor_alloc((uint64_t)fx->value_dim * sizeof(float));
    if (!sc->input_gpu || !sc->input_ln_gpu || !sc->z_gpu || !sc->a_gpu || !sc->b_gpu || !sc->out_in_gpu || !sc->out_proj_gpu ||
        !sc->post_gpu || !sc->router_logits_gpu || !sc->shared_gate_gpu || !sc->shared_up_gpu || !sc->shared_mid_gpu ||
        !sc->shared_out_gpu || !sc->expert_gate_gpu || !sc->expert_up_gpu || !sc->expert_mid_gpu || !sc->expert_down_gpu ||
        !sc->q_gpu || !sc->k_gpu || !sc->v_gpu) {
        snprintf(err, err_cap, "gpu tensor alloc failed blk.%u", fx->layer);
        return 0;
    }
    return 1;
}
#endif

static int run_step_layer_dynamic(const live_fixture *fx,
                                  const qwen36_gguf_file *gf,
                                  const qwen36_35a3b_q8_layer *layer,
                                  layer_runtime *ls,
                                  const float *layer_input,
                                  float *out_row,
                                  char *err,
                                  size_t err_cap) {
    layer_step_scratch *sc = &ls->step;
    const uint32_t qkv_dim = fx->key_dim * 2u + fx->value_dim;
    const uint32_t rep = fx->num_v_heads / fx->num_k_heads;
    float *input_ln = sc->input_ln, *qkv = sc->qkv, *z = sc->z, *a = sc->a, *b = sc->b;
    float *conv = sc->conv, *q = sc->q, *k = sc->k, *v = sc->v, *beta = sc->beta, *g = sc->g, *core = sc->core, *out_in = sc->out_in, *out_proj = sc->out_proj;
    float *resid = sc->resid, *post_ln = sc->post_ln, *mlp = sc->mlp;
    float *gate = sc->gate, *up = sc->up, *act = sc->act, *down = sc->down;
    float *shared_gate = sc->shared_gate, *shared_up = sc->shared_up, *shared_act = sc->shared_act, *shared = sc->shared;
    float *k_scaled = sc->k_scaled, *q_scaled = sc->q_scaled, *delta_tmp = sc->delta_tmp;
    float router_logits[ROUTER_COUNT];
    uint32_t router_idx[ROUTER_COUNT];
    float router_scores[ROUTER_COUNT];
    uint32_t h, d, vd, i;

    {
        double var = 0.0;
        for (d = 0; d < fx->hidden; ++d) var += (double)layer_input[d] * layer_input[d];
        var /= (double)fx->hidden;
        for (d = 0; d < fx->hidden; ++d) input_ln[d] = (float)(layer_input[d] / sqrt(var + 1e-6)) * fx->attn_norm_w[d];
    }
    matvec(fx->w_qkv, input_ln, qkv, qkv_dim, fx->hidden);
    matvec(fx->w_z, input_ln, z, fx->value_dim, fx->hidden);
    matvec(fx->w_a, input_ln, a, fx->num_v_heads, fx->hidden);
    matvec(fx->w_b, input_ln, b, fx->num_v_heads, fx->hidden);

    for (d = 0; d < qkv_dim; ++d) {
        double sum = 0.0;
        if (ls->conv_ring_count >= 3) sum += (double)fx->conv_w[(size_t)d * 4u + 0u] * ls->conv_ring[2][d];
        if (ls->conv_ring_count >= 2) sum += (double)fx->conv_w[(size_t)d * 4u + 1u] * ls->conv_ring[1][d];
        if (ls->conv_ring_count >= 1) sum += (double)fx->conv_w[(size_t)d * 4u + 2u] * ls->conv_ring[0][d];
        sum += (double)fx->conv_w[(size_t)d * 4u + 3u] * qkv[d];
        conv[d] = siluf_local((float)sum);
    }
    {
        const float *cq = conv;
        const float *ck = cq + fx->key_dim;
        const float *cv = ck + fx->key_dim;
        for (h = 0; h < fx->num_k_heads; ++h) {
            for (d = 0; d < fx->head_k_dim; ++d) {
                float qv = cq[h * fx->head_k_dim + d], kv = ck[h * fx->head_k_dim + d];
                uint32_t vh;
                for (vh = 0; vh < rep; ++vh) {
                    uint32_t dst_h = h * rep + vh;
                    q[(size_t)dst_h * fx->head_k_dim + d] = qv;
                    k[(size_t)dst_h * fx->head_k_dim + d] = kv;
                }
            }
        }
        for (h = 0; h < fx->num_v_heads; ++h) {
            memcpy(v + (size_t)h * fx->head_v_dim, cv + (size_t)h * fx->head_v_dim, fx->head_v_dim * sizeof(float));
            beta[h] = sigmoidf_local(b[h]);
            g[h] = fx->A_log[h] * softplusf_local(a[h] + fx->dt_bias[h]);
        }
    }
    for (h = 0; h < fx->num_v_heads; ++h) {
        size_t qbase = (size_t)h * fx->head_k_dim;
        size_t vbase = (size_t)h * fx->head_v_dim;
        size_t sbase = (size_t)h * fx->head_k_dim * fx->head_v_dim;
        run_deltanet_head_step(ls->deltanet_state + sbase,
                               fx->head_k_dim,
                               fx->head_v_dim,
                               k + qbase,
                               q + qbase,
                               v + vbase,
                               expf(g[h]),
                               beta[h],
                               k_scaled,
                               q_scaled,
                               delta_tmp,
                               core + vbase);
    }
    for (h = 0; h < fx->num_v_heads; ++h) {
        size_t base = (size_t)h * fx->head_v_dim;
        double var = 0.0;
        for (vd = 0; vd < fx->head_v_dim; ++vd) {
            double cv = core[(size_t)h * fx->head_v_dim + vd];
            var += cv * cv;
        }
        var /= (double)fx->head_v_dim;
        for (vd = 0; vd < fx->head_v_dim; ++vd) {
            float cv = core[(size_t)h * fx->head_v_dim + vd];
            out_in[base + vd] = (float)(cv / sqrt(var + 1e-6)) * fx->ssm_norm_w[vd] * siluf_local(z[(size_t)h * fx->head_v_dim + vd]);
        }
    }
    matvec(fx->w_out, out_in, out_proj, fx->hidden, fx->value_dim);

    {
        double var = 0.0;
        float scale_in = 0.0f;
        for (d = 0; d < fx->hidden; ++d) { resid[d] = layer_input[d] + out_proj[d]; var += (double)resid[d] * resid[d]; }
        var /= (double)fx->hidden;
        for (d = 0; d < fx->hidden; ++d) {
            post_ln[d] = (float)(resid[d] / sqrt(var + 1e-6)) * fx->post_attn_norm_w[d];
            scale_in += post_ln[d] * fx->gate_inp_shexp[d];
            mlp[d] = 0.0f;
        }
        matvec(fx->router_w, post_ln, router_logits, ROUTER_COUNT, fx->hidden);
        topk_softmax256(router_logits, fx->topk, router_idx, router_scores);
        for (i = 0; i < fx->topk; ++i) {
            if (!ensure_expert_loaded(gf, layer, ls->experts, router_idx[i], fx->hidden, fx->inter, err, err_cap)) return 0;
            matvec(ls->experts[router_idx[i]].gate, post_ln, gate, fx->inter, fx->hidden);
            matvec(ls->experts[router_idx[i]].up, post_ln, up, fx->inter, fx->hidden);
            for (vd = 0; vd < fx->inter; ++vd) act[vd] = siluf_local(gate[vd]) * up[vd];
            matvec(ls->experts[router_idx[i]].down, act, down, fx->hidden, fx->inter);
            for (d = 0; d < fx->hidden; ++d) mlp[d] += down[d] * router_scores[i];
        }
        matvec(fx->gate_shexp, post_ln, shared_gate, fx->inter, fx->hidden);
        matvec(fx->up_shexp, post_ln, shared_up, fx->inter, fx->hidden);
        for (vd = 0; vd < fx->inter; ++vd) shared_act[vd] = siluf_local(shared_gate[vd]) * shared_up[vd];
        matvec(fx->down_shexp, shared_act, shared, fx->hidden, fx->inter);
        {
            float s = sigmoidf_local(scale_in);
            for (d = 0; d < fx->hidden; ++d) out_row[d] = resid[d] + mlp[d] + shared[d] * s;
        }
    }

    if (ls->conv_ring_count >= 2) memcpy(ls->conv_ring[2], ls->conv_ring[1], (size_t)qkv_dim * sizeof(float));
    if (ls->conv_ring_count >= 1) memcpy(ls->conv_ring[1], ls->conv_ring[0], (size_t)qkv_dim * sizeof(float));
    memcpy(ls->conv_ring[0], qkv, (size_t)qkv_dim * sizeof(float));
    if (ls->conv_ring_count < 3) ls->conv_ring_count++;
    return err[0] == '\0';
}

#ifdef QWEN36_UNIFIED_HAVE_GPU
static int run_hybrid_layer_step_gpuproj(const live_fixture *fx,
                                         const qwen36_gguf_file *gf,
                                         const qwen36_35a3b_q8_layer *layer,
                                         const mapped_file *mf,
                                         layer_runtime *ls,
                                         const float *layer_input,
                                         float *out_row,
                                         char *err,
                                         size_t err_cap) {
    layer_step_scratch *sc = &ls->step;
    const uint32_t qkv_dim = fx->key_dim * 2u + fx->value_dim;
    const uint32_t rep = fx->num_v_heads / fx->num_k_heads;
    const uint64_t hidden_bytes = (uint64_t)fx->hidden * sizeof(float);
    const uint64_t z_bytes = (uint64_t)fx->value_dim * sizeof(float);
    const uint64_t ab_bytes = (uint64_t)fx->num_v_heads * sizeof(float);
    const uint64_t out_in_bytes = (uint64_t)fx->value_dim * sizeof(float);
    ds4_gpu_tensor *input_gpu = NULL, *input_ln_gpu = NULL, *z_gpu = NULL, *a_gpu = NULL, *b_gpu = NULL, *out_in_gpu = NULL, *out_proj_gpu = NULL, *q_gpu = NULL, *k_gpu = NULL, *v_gpu = NULL;
    float *qkv_raw = sc->qkv_raw, *qkv = sc->qkv, *z_raw = sc->z_raw, *z = sc->z, *a_raw = sc->a_raw, *a = sc->a, *b_raw = sc->b_raw, *b = sc->b;
    float *conv = sc->conv, *q = sc->q, *k = sc->k, *v = sc->v, *beta = sc->beta, *g = sc->g, *core = sc->core;
    float *out_in = sc->out_in, *out_in_gg = sc->out_in_gg, *out_proj = sc->out_proj, *resid = sc->resid, *post_ln = sc->post_ln, *mlp = sc->mlp;
    float *gate = sc->gate, *up = sc->up, *act = sc->act, *down = sc->down;
    float *shared_gate = sc->shared_gate, *shared_up = sc->shared_up, *shared_act = sc->shared_act, *shared = sc->shared;
    float *k_scaled = sc->k_scaled, *q_scaled = sc->q_scaled, *delta_tmp = sc->delta_tmp;
    float router_logits[ROUTER_COUNT];
    uint32_t router_idx[ROUTER_COUNT];
    float router_scores[ROUTER_COUNT];
    uint32_t h, d, vd, i;
    int ok = 0;

    memset(core, 0, (size_t)fx->num_v_heads * fx->head_v_dim * sizeof(float));
    if (!ensure_layer_gpu_step_scratch(fx, sc, err, err_cap)) {
        goto cleanup;
    }
    input_gpu = sc->input_gpu;
    input_ln_gpu = sc->input_ln_gpu;
    z_gpu = sc->z_gpu;
    a_gpu = sc->a_gpu;
    b_gpu = sc->b_gpu;
    out_in_gpu = sc->out_in_gpu;
    out_proj_gpu = sc->out_proj_gpu;
    q_gpu = sc->q_gpu;
    k_gpu = sc->k_gpu;
    v_gpu = sc->v_gpu;

    if (ds4_gpu_tensor_write(input_gpu, 0, layer_input, hidden_bytes) == 0) {
        snprintf(err, err_cap, "gpu input write failed blk.%u", fx->layer);
        goto cleanup;
    }
    if (ds4_gpu_begin_commands() == 0) {
        snprintf(err, err_cap, "gpu begin failed blk.%u", fx->layer);
        goto cleanup;
    }
    if (ds4_gpu_rms_norm_weight_rows_tensor(input_ln_gpu, input_gpu, mf->map, mf->size,
                                            layer->attn_norm->abs_offset, fx->hidden, 1u, 1e-6f) == 0) {
        snprintf(err, err_cap, "gpu rmsnorm failed blk.%u", fx->layer);
        goto cleanup;
    }
    if (ds4_gpu_matmul_q8_0_tensor(z_gpu, mf->map, mf->size,
                                   layer->attn_gate->abs_offset, fx->hidden, fx->value_dim, input_ln_gpu, 1u) == 0) {
        snprintf(err, err_cap, "gpu z matmul failed blk.%u", fx->layer);
        goto cleanup;
    }
    if (ds4_gpu_matmul_q8_0_tensor(a_gpu, mf->map, mf->size,
                                   layer->ssm_alpha->abs_offset, fx->hidden, fx->num_v_heads, input_ln_gpu, 1u) == 0) {
        snprintf(err, err_cap, "gpu a matmul failed blk.%u", fx->layer);
        goto cleanup;
    }
    if (ds4_gpu_matmul_q8_0_tensor(b_gpu, mf->map, mf->size,
                                    layer->ssm_beta->abs_offset, fx->hidden, fx->num_v_heads, input_ln_gpu, 1u) == 0) {
        snprintf(err, err_cap, "gpu b matmul failed blk.%u", fx->layer);
        goto cleanup;
    }
    {
        const uint64_t row_bytes = (uint64_t)(fx->hidden / 32u) * 34u;
        const uint64_t q_off = layer->attn_qkv->abs_offset;
        const uint64_t k_off = q_off + (uint64_t)fx->key_dim * row_bytes;
        const uint64_t v_off = k_off + (uint64_t)fx->key_dim * row_bytes;
        if (ds4_gpu_matmul_q8_0_tensor(q_gpu, mf->map, mf->size, q_off, fx->hidden, fx->key_dim, input_ln_gpu, 1u) == 0) {
            snprintf(err, err_cap, "gpu q matmul failed blk.%u", fx->layer);
            goto cleanup;
        }
        if (ds4_gpu_matmul_q8_0_tensor(k_gpu, mf->map, mf->size, k_off, fx->hidden, fx->key_dim, input_ln_gpu, 1u) == 0) {
            snprintf(err, err_cap, "gpu k matmul failed blk.%u", fx->layer);
            goto cleanup;
        }
        if (ds4_gpu_matmul_q8_0_tensor(v_gpu, mf->map, mf->size, v_off, fx->hidden, fx->value_dim, input_ln_gpu, 1u) == 0) {
            snprintf(err, err_cap, "gpu v matmul failed blk.%u", fx->layer);
            goto cleanup;
        }
    }
    if (ds4_gpu_end_commands() == 0) {
        snprintf(err, err_cap, "gpu end failed blk.%u", fx->layer);
        goto cleanup;
    }
    if (ds4_gpu_tensor_read(z_gpu, 0, z_raw, z_bytes) == 0 ||
        ds4_gpu_tensor_read(a_gpu, 0, a_raw, ab_bytes) == 0 ||
        ds4_gpu_tensor_read(b_gpu, 0, b_raw, ab_bytes) == 0 ||
        ds4_gpu_tensor_read(q_gpu, 0, qkv_raw, (uint64_t)fx->key_dim * sizeof(float)) == 0 ||
        ds4_gpu_tensor_read(k_gpu, 0, qkv_raw + fx->key_dim, (uint64_t)fx->key_dim * sizeof(float)) == 0 ||
        ds4_gpu_tensor_read(v_gpu, 0, qkv_raw + fx->key_dim * 2u, (uint64_t)fx->value_dim * sizeof(float)) == 0) {
        snprintf(err, err_cap, "gpu qkv+zab readback failed blk.%u", fx->layer);
        goto cleanup;
    }

    reorder_head_rows_seq_f32(z, z_raw, 1u, fx->num_v_heads, fx->head_v_dim);
    reorder_head_scalars_seq_f32(a, a_raw, 1u, fx->num_v_heads);
    reorder_head_scalars_seq_f32(b, b_raw, 1u, fx->num_v_heads);
    reorder_qkv_v_tokenmajor_gg_to_hf(qkv, qkv_raw, 1u, fx->key_dim, fx->num_v_heads, fx->head_v_dim);

    for (d = 0; d < qkv_dim; ++d) {
        double sum = 0.0;
        if (ls->conv_ring_count >= 3) sum += (double)fx->conv_w[(size_t)d * 4u + 0u] * ls->conv_ring[2][d];
        if (ls->conv_ring_count >= 2) sum += (double)fx->conv_w[(size_t)d * 4u + 1u] * ls->conv_ring[1][d];
        if (ls->conv_ring_count >= 1) sum += (double)fx->conv_w[(size_t)d * 4u + 2u] * ls->conv_ring[0][d];
        sum += (double)fx->conv_w[(size_t)d * 4u + 3u] * qkv[d];
        conv[d] = siluf_local((float)sum);
    }
    {
        const float *cq = conv;
        const float *ck = cq + fx->key_dim;
        const float *cv = ck + fx->key_dim;
        for (h = 0; h < fx->num_k_heads; ++h) {
            for (d = 0; d < fx->head_k_dim; ++d) {
                float qv = cq[h * fx->head_k_dim + d];
                float kv = ck[h * fx->head_k_dim + d];
                uint32_t vh;
                for (vh = 0; vh < rep; ++vh) {
                    uint32_t dst_h = h * rep + vh;
                    q[(size_t)dst_h * fx->head_k_dim + d] = qv;
                    k[(size_t)dst_h * fx->head_k_dim + d] = kv;
                }
            }
        }
        for (h = 0; h < fx->num_v_heads; ++h) {
            memcpy(v + (size_t)h * fx->head_v_dim, cv + (size_t)h * fx->head_v_dim, (size_t)fx->head_v_dim * sizeof(float));
            beta[h] = sigmoidf_local(b[h]);
            g[h] = fx->A_log[h] * softplusf_local(a[h] + fx->dt_bias[h]);
        }
    }
    for (h = 0; h < fx->num_v_heads; ++h) {
        size_t qbase = (size_t)h * fx->head_k_dim;
        size_t vbase = (size_t)h * fx->head_v_dim;
        size_t sbase = (size_t)h * fx->head_k_dim * fx->head_v_dim;
        run_deltanet_head_step(ls->deltanet_state + sbase,
                               fx->head_k_dim,
                               fx->head_v_dim,
                               k + qbase,
                               q + qbase,
                               v + vbase,
                               expf(g[h]),
                               beta[h],
                               k_scaled,
                               q_scaled,
                               delta_tmp,
                               core + vbase);
    }
    for (h = 0; h < fx->num_v_heads; ++h) {
        size_t base = (size_t)h * fx->head_v_dim;
        double var = 0.0;
        for (vd = 0; vd < fx->head_v_dim; ++vd) {
            double cv = core[(size_t)h * fx->head_v_dim + vd];
            var += cv * cv;
        }
        var /= (double)fx->head_v_dim;
        for (vd = 0; vd < fx->head_v_dim; ++vd) {
            float cv = core[(size_t)h * fx->head_v_dim + vd];
            out_in[base + vd] = (float)(cv / sqrt(var + 1e-6)) * fx->ssm_norm_w[vd] * siluf_local(z[(size_t)h * fx->head_v_dim + vd]);
        }
    }
    reorder_out_in_hf_to_gg(out_in_gg, out_in, 1u, fx->num_v_heads, fx->head_v_dim);
    if (ds4_gpu_tensor_write(out_in_gpu, 0, out_in_gg, out_in_bytes) == 0) {
        snprintf(err, err_cap, "gpu out_in write failed blk.%u", fx->layer);
        goto cleanup;
    }
    if (ds4_gpu_begin_commands() == 0) {
        snprintf(err, err_cap, "gpu begin out_proj failed blk.%u", fx->layer);
        goto cleanup;
    }
    if (ds4_gpu_matmul_q8_0_tensor(out_proj_gpu, mf->map, mf->size,
                                   layer->ssm_out->abs_offset, fx->value_dim, fx->hidden, out_in_gpu, 1u) == 0) {
        snprintf(err, err_cap, "gpu out_proj failed blk.%u", fx->layer);
        goto cleanup;
    }
    if (ds4_gpu_end_commands() == 0) {
        snprintf(err, err_cap, "gpu end out_proj failed blk.%u", fx->layer);
        goto cleanup;
    }
    if (ds4_gpu_tensor_read(out_proj_gpu, 0, out_proj, hidden_bytes) == 0) {
        snprintf(err, err_cap, "gpu out_proj readback failed blk.%u", fx->layer);
        goto cleanup;
    }

    {
        double var = 0.0;
        float scale_in = 0.0f;
        for (d = 0; d < fx->hidden; ++d) { resid[d] = layer_input[d] + out_proj[d]; var += (double)resid[d] * resid[d]; }
        var /= (double)fx->hidden;
        for (d = 0; d < fx->hidden; ++d) {
            post_ln[d] = (float)(resid[d] / sqrt(var + 1e-6)) * fx->post_attn_norm_w[d];
            scale_in += post_ln[d] * fx->gate_inp_shexp[d];
        }
        memset(mlp, 0, (size_t)fx->hidden * sizeof(float));
        if (ds4_gpu_tensor_write(sc->post_gpu, 0, post_ln, hidden_bytes) == 0) {
            snprintf(err, err_cap, "gpu post write failed blk.%u", fx->layer);
            goto cleanup;
        }
        if (ds4_gpu_begin_commands() == 0) {
            snprintf(err, err_cap, "gpu ffn begin failed blk.%u", fx->layer);
            goto cleanup;
        }
        if (ds4_gpu_matmul_f32_tensor(sc->router_logits_gpu, mf->map, mf->size,
                                      layer->ffn_gate_inp->abs_offset,
                                      fx->hidden, ROUTER_COUNT, sc->post_gpu, 1u) == 0) {
            snprintf(err, err_cap, "gpu router failed blk.%u", fx->layer);
            goto cleanup;
        }
        if (ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(sc->shared_gate_gpu, sc->shared_up_gpu, sc->shared_mid_gpu,
                                                      mf->map, mf->size,
                                                      layer->ffn_gate_shexp->abs_offset,
                                                      layer->ffn_up_shexp->abs_offset,
                                                      fx->hidden, fx->inter, sc->post_gpu,
                                                      80.0f) == 0) {
            snprintf(err, err_cap, "gpu shared gate/up failed blk.%u", fx->layer);
            goto cleanup;
        }
        if (ds4_gpu_matmul_q8_0_tensor(sc->shared_out_gpu, mf->map, mf->size,
                                       layer->ffn_down_shexp->abs_offset,
                                       fx->inter, fx->hidden, sc->shared_mid_gpu, 1u) == 0) {
            snprintf(err, err_cap, "gpu shared down failed blk.%u", fx->layer);
            goto cleanup;
        }
        if (ds4_gpu_end_commands() == 0) {
            snprintf(err, err_cap, "gpu ffn end failed blk.%u", fx->layer);
            goto cleanup;
        }
        if (ds4_gpu_tensor_read(sc->router_logits_gpu, 0, router_logits, (uint64_t)ROUTER_COUNT * sizeof(float)) == 0) {
            snprintf(err, err_cap, "gpu router readback failed blk.%u", fx->layer);
            goto cleanup;
        }
        if (ds4_gpu_tensor_read(sc->shared_out_gpu, 0, shared, hidden_bytes) == 0) {
            snprintf(err, err_cap, "gpu shared readback failed blk.%u", fx->layer);
            goto cleanup;
        }
        topk_softmax256(router_logits, fx->topk, router_idx, router_scores);
        for (i = 0; i < fx->topk; ++i) {
            const uint64_t gate_expert_bytes = (uint64_t)(fx->inter * fx->hidden / 32u) * 34u;
            const uint64_t down_expert_bytes = (uint64_t)(fx->hidden * fx->inter / 32u) * 34u;
            const uint32_t expert_id = router_idx[i];
            if (ds4_gpu_begin_commands() == 0) {
                snprintf(err, err_cap, "gpu expert begin failed blk.%u", fx->layer);
                goto cleanup;
            }
            if (ds4_gpu_matmul_q8_0_pair_tensor(sc->expert_gate_gpu, sc->expert_up_gpu,
                                                mf->map, mf->size,
                                                layer->ffn_gate_exps->abs_offset + gate_expert_bytes * expert_id,
                                                layer->ffn_up_exps->abs_offset + gate_expert_bytes * expert_id,
                                                fx->hidden, fx->inter, fx->inter, sc->post_gpu, 1u) == 0) {
                snprintf(err, err_cap, "gpu expert gate/up failed blk.%u", fx->layer);
                goto cleanup;
            }
            if (ds4_gpu_swiglu_tensor(sc->expert_mid_gpu, sc->expert_gate_gpu, sc->expert_up_gpu,
                                      fx->inter, 80.0f, 1.0f) == 0) {
                snprintf(err, err_cap, "gpu expert swiglu failed blk.%u", fx->layer);
                goto cleanup;
            }
            if (ds4_gpu_matmul_q8_0_tensor(sc->expert_down_gpu, mf->map, mf->size,
                                           layer->ffn_down_exps->abs_offset + down_expert_bytes * expert_id,
                                           fx->inter, fx->hidden, sc->expert_mid_gpu, 1u) == 0) {
                snprintf(err, err_cap, "gpu expert down failed blk.%u", fx->layer);
                goto cleanup;
            }
            if (ds4_gpu_end_commands() == 0) {
                snprintf(err, err_cap, "gpu expert end failed blk.%u", fx->layer);
                goto cleanup;
            }
            if (ds4_gpu_tensor_read(sc->expert_down_gpu, 0, down, hidden_bytes) == 0) {
                snprintf(err, err_cap, "gpu expert readback failed blk.%u", fx->layer);
                goto cleanup;
            }
            for (d = 0; d < fx->hidden; ++d) mlp[d] += down[d] * router_scores[i];
        }
        {
            float s = sigmoidf_local(scale_in);
            for (d = 0; d < fx->hidden; ++d) out_row[d] = resid[d] + mlp[d] + shared[d] * s;
        }
    }

    if (ls->conv_ring_count >= 2) memcpy(ls->conv_ring[2], ls->conv_ring[1], (size_t)qkv_dim * sizeof(float));
    if (ls->conv_ring_count >= 1) memcpy(ls->conv_ring[1], ls->conv_ring[0], (size_t)qkv_dim * sizeof(float));
    memcpy(ls->conv_ring[0], qkv, (size_t)qkv_dim * sizeof(float));
    if (ls->conv_ring_count < 3) ls->conv_ring_count++;
    ok = 1;

cleanup:
    return ok;
}
#endif

static void session_clear_decode_state(unified_session_state *st) {
    uint32_t li, i;
    free(st->token_ids);
    free(st->owned_seq);
    st->token_ids = NULL;
    st->owned_seq = NULL;
    st->seq_len = 0;
    st->owned_cap = 0;
    st->prefilled = 0;
    for (li = 0; li < st->n_fx; ++li) {
        live_fixture *fx = &st->fxs[li];
        layer_runtime *ls = &st->layers[li];
        if (ls->deltanet_state) memset(ls->deltanet_state, 0, (size_t)fx->num_v_heads * fx->head_k_dim * fx->head_v_dim * sizeof(float));
        for (i = 0; i < 3; ++i) {
            if (ls->conv_ring[i]) memset(ls->conv_ring[i], 0, (size_t)(fx->key_dim * 2u + fx->value_dim) * sizeof(float));
        }
        ls->conv_ring_count = 0;
    }
    if (st->full_states) {
        for (li = 0; li < st->cfg.n_cycles; ++li) {
            free_full_layer_state(&st->full_states[li]);
        }
    }
}

static int session_init_runtime(unified_session_state *st, const char *gguf_path, char *err, size_t err_cap) {
    uint32_t total_fixtures = 0, ci, fi, out_idx = 0;
    const char *enable_full = getenv("QWEN36_UNIFIED_PREFILL_FULL_CPU");
    const char *verbose_logs_env = getenv("QWEN36_UNIFIED_VERBOSE");
#ifdef QWEN36_UNIFIED_HAVE_GPU
    const char *enable_full_gpu = getenv("QWEN36_UNIFIED_FULL_GPU");
    const char *hybrid_gpu_cycles_env = getenv("QWEN36_UNIFIED_HYBRID_GPU_CYCLES");
#endif
    memset(&st->gf, 0, sizeof(st->gf));
    memset(&st->model, 0, sizeof(st->model));
    st->enable_full_prefill_cpu = (enable_full && strcmp(enable_full, "0") != 0);
    st->verbose_logs = !(verbose_logs_env && strcmp(verbose_logs_env, "0") == 0);
#ifdef QWEN36_UNIFIED_HAVE_GPU
    st->mf.fd = -1;
    st->enable_full_prefill_gpu = (enable_full_gpu && strcmp(enable_full_gpu, "0") != 0);
    st->hybrid_gpu_cycles = 0;
    if (st->enable_full_prefill_cpu && st->enable_full_prefill_gpu) {
        snprintf(err, err_cap, "QWEN36_UNIFIED_PREFILL_FULL_CPU and QWEN36_UNIFIED_FULL_GPU are mutually exclusive");
        return 0;
    }
    if (hybrid_gpu_cycles_env && hybrid_gpu_cycles_env[0]) {
        char *endp = NULL;
        unsigned long v = strtoul(hybrid_gpu_cycles_env, &endp, 10);
        if (endp != hybrid_gpu_cycles_env) st->hybrid_gpu_cycles = (uint32_t)v;
    }
#endif
    for (ci = 0; ci < st->cfg.n_cycles; ++ci) total_fixtures += st->cfg.cycles[ci].n_fixtures;
    if (!qwen36_gguf_open(&st->gf, gguf_path, err, err_cap)) return 0;
    if (!qwen36_35a3b_q8_bind(&st->gf, &st->model, err, err_cap)) {
        qwen36_gguf_close(&st->gf);
        memset(&st->gf, 0, sizeof(st->gf));
        return 0;
    }
    st->fxs = (live_fixture *)calloc(total_fixtures, sizeof(live_fixture));
    st->layers = (layer_runtime *)calloc(total_fixtures, sizeof(layer_runtime));
    st->cycle_fx_offsets = (uint32_t *)calloc(st->cfg.n_cycles + 1u, sizeof(uint32_t));
    st->full_caches = st->enable_full_prefill_cpu ? (full_layer_cache *)calloc(st->cfg.n_cycles, sizeof(full_layer_cache)) : NULL;
    st->full_states =
        (st->enable_full_prefill_cpu
#ifdef QWEN36_UNIFIED_HAVE_GPU
         || st->enable_full_prefill_gpu
#endif
        )
            ? (full_layer_state *)calloc(st->cfg.n_cycles, sizeof(full_layer_state))
            : NULL;
#ifdef QWEN36_UNIFIED_HAVE_GPU
    st->full_small_caches = st->enable_full_prefill_gpu ? (full_layer_small_cache *)calloc(st->cfg.n_cycles, sizeof(full_layer_small_cache)) : NULL;
#endif
    if (!st->fxs || !st->layers || !st->cycle_fx_offsets || (st->enable_full_prefill_cpu && (!st->full_caches || !st->full_states))
#ifdef QWEN36_UNIFIED_HAVE_GPU
        || (st->enable_full_prefill_gpu && (!st->full_small_caches))
#endif
        ) {
        snprintf(err, err_cap, "oom unified runtime arrays");
        return 0;
    }
    st->n_fx = total_fixtures;
    for (ci = 0; ci < st->cfg.n_cycles; ++ci) {
        st->cycle_fx_offsets[ci] = out_idx;
        for (fi = 0; fi < st->cfg.cycles[ci].n_fixtures; ++fi) {
            live_fixture *fx = &st->fxs[out_idx];
            layer_runtime *ls = &st->layers[out_idx];
            const char *path = st->cfg.cycles[ci].fixtures[fi];
            if (!fixture_load(path, fx)) {
                snprintf(err, err_cap, "failed to load fixture: %s", path);
                return 0;
            }
            if (out_idx == 0) st->hidden = fx->hidden;
            if (fx->hidden != st->hidden) {
                snprintf(err, err_cap, "hidden mismatch across fixtures");
                return 0;
            }
            ls->deltanet_state = (float *)calloc((size_t)fx->num_v_heads * fx->head_k_dim * fx->head_v_dim, sizeof(float));
            ls->conv_ring[0] = (float *)calloc((size_t)(fx->key_dim * 2u + fx->value_dim), sizeof(float));
            ls->conv_ring[1] = (float *)calloc((size_t)(fx->key_dim * 2u + fx->value_dim), sizeof(float));
            ls->conv_ring[2] = (float *)calloc((size_t)(fx->key_dim * 2u + fx->value_dim), sizeof(float));
            if (!ls->deltanet_state || !ls->conv_ring[0] || !ls->conv_ring[1] || !ls->conv_ring[2]) {
                snprintf(err, err_cap, "oom unified layer state");
                return 0;
            }
            if (!alloc_layer_step_scratch(fx, &ls->step, err, err_cap)) return 0;
            out_idx++;
        }
        if (st->enable_full_prefill_cpu
#ifdef QWEN36_UNIFIED_HAVE_GPU
            || st->enable_full_prefill_gpu
#endif
            ) {
            const uint32_t full_layer = st->cfg.cycles[ci].full_layer;
            if (full_layer >= QWEN36_35A3B_Q8_BLOCK_COUNT || st->model.layers[full_layer].kind != QWEN36_LAYER_KIND_FULL_ATTENTION) {
                snprintf(err, err_cap, "bad full layer %u", full_layer);
                return 0;
            }
#ifdef QWEN36_UNIFIED_HAVE_GPU
            if (st->enable_full_prefill_gpu &&
                !load_small_cache(&st->gf, &st->model.layers[full_layer], &st->full_small_caches[ci], err, err_cap)) {
                snprintf(err, err_cap, "full small cache load failed for layer %u: %s", full_layer, err);
                return 0;
            }
#endif
        }
    }
    st->cycle_fx_offsets[st->cfg.n_cycles] = out_idx;
#ifdef QWEN36_UNIFIED_HAVE_GPU
    if (st->enable_full_prefill_gpu) {
        if (!mapped_file_open(&st->mf, gguf_path)) {
            snprintf(err, err_cap, "failed to mmap gguf: %s", strerror(errno));
            return 0;
        }
        if (ds4_gpu_init() == 0) {
            snprintf(err, err_cap, "ds4_gpu_init failed");
            return 0;
        }
        if (ds4_gpu_set_model_map(st->mf.map, st->mf.size) == 0) {
            snprintf(err, err_cap, "ds4_gpu_set_model_map failed");
            return 0;
        }
        if (ds4_gpu_set_model_fd_for_map(st->mf.fd, st->mf.map) == 0) {
            snprintf(err, err_cap, "ds4_gpu_set_model_fd_for_map failed");
            return 0;
        }
    }
#endif
    return 1;
}

static void session_close_runtime(unified_session_state *st) {
    uint32_t li, i;
    session_clear_decode_state(st);
    for (li = 0; li < st->n_fx; ++li) {
        free_expert_cache(st->layers[li].experts);
        free_layer_step_scratch(&st->layers[li].step);
        for (i = 0; i < 3; ++i) free(st->layers[li].conv_ring[i]);
        free(st->layers[li].deltanet_state);
        fixture_free(&st->fxs[li]);
    }
    free(st->layers);
    free(st->fxs);
    if (st->full_caches) {
        for (i = 0; i < st->cfg.n_cycles; ++i) free_full_layer_cache(&st->full_caches[i]);
    }
    if (st->full_states) {
        for (i = 0; i < st->cfg.n_cycles; ++i) free_full_layer_state(&st->full_states[i]);
    }
    free(st->full_caches);
    free(st->full_states);
#ifdef QWEN36_UNIFIED_HAVE_GPU
    if (st->full_small_caches) {
        for (i = 0; i < st->cfg.n_cycles; ++i) free_small_cache(&st->full_small_caches[i]);
    }
    free(st->full_small_caches);
    st->full_small_caches = NULL;
    if (st->enable_full_prefill_gpu) {
        ds4_gpu_cleanup();
        mapped_file_close(&st->mf);
    }
#endif
    free(st->cycle_fx_offsets);
    st->layers = NULL;
    st->fxs = NULL;
    st->full_caches = NULL;
    st->full_states = NULL;
    st->cycle_fx_offsets = NULL;
    st->n_fx = 0;
    qwen36_gguf_close(&st->gf);
    memset(&st->gf, 0, sizeof(st->gf));
}

static int session_prefill(unified_session_state *st,
                           const uint32_t *token_ids,
                           uint32_t seq_len,
                           uint32_t hidden,
                           const float *input_seq,
                           char *err,
                           size_t err_cap) {
    float *state_seq = NULL;
    float *next_seq = NULL;
    size_t seq_hidden = (size_t)seq_len * hidden;
    uint32_t ci, li, t;
    double t0;
    if (st->n_fx == 0) {
        snprintf(err, err_cap, "no hybrid fixtures loaded");
        return 0;
    }
    if (hidden != st->hidden) {
        snprintf(err, err_cap, "hidden mismatch");
        return 0;
    }
    session_clear_decode_state(st);
    st->token_ids = (uint32_t *)malloc((size_t)seq_len * sizeof(uint32_t));
    if (!st->token_ids) {
        snprintf(err, err_cap, "oom token ids");
        return 0;
    }
    memcpy(st->token_ids, token_ids, (size_t)seq_len * sizeof(uint32_t));
    state_seq = (float *)calloc(seq_hidden, sizeof(float));
    if (!state_seq) {
        snprintf(err, err_cap, "oom initial seq");
        return 0;
    }
    t0 = now_ms();
    if (st->fxs[0].layer == 0) {
        for (t = 0; t < seq_len; ++t) {
            if (!decode_q8_row(&st->gf, st->model.token_embd, token_ids[t], hidden, state_seq + (size_t)t * hidden, err, err_cap)) goto fail;
        }
    } else {
        memcpy(state_seq, input_seq, seq_hidden * sizeof(float));
    }
    for (ci = 0; ci < st->cfg.n_cycles; ++ci) {
        uint32_t start = st->cycle_fx_offsets[ci];
        uint32_t end = st->cycle_fx_offsets[ci + 1u];
        for (li = start; li < end; ++li) {
            double layer_t0 = now_ms();
            next_seq = (float *)calloc(seq_hidden, sizeof(float));
            if (!next_seq) {
                snprintf(err, err_cap, "oom prefill layer seq");
                goto fail;
            }
            fprintf(stderr, "qwen36_unified prefill layer=%u rows=%u start\n", st->fxs[li].layer, seq_len);
            fflush(stderr);
            {
#ifdef QWEN36_UNIFIED_HAVE_GPU
                if (st->enable_full_prefill_gpu && ci < st->hybrid_gpu_cycles) {
                    if (!run_hybrid_layer_prefill_gpuproj(&st->fxs[li], &st->gf, &st->model.layers[st->fxs[li].layer], &st->mf, &st->layers[li], seq_len, state_seq, next_seq, err, err_cap)) {
                        goto fail;
                    }
                } else
#endif
                {
                    uint32_t row;
                    for (row = 0; row < seq_len; ++row) {
                        if (!run_step_layer_dynamic(&st->fxs[li], &st->gf, &st->model.layers[st->fxs[li].layer], &st->layers[li], state_seq + (size_t)row * hidden, next_seq + (size_t)row * hidden, err, err_cap)) {
                            goto fail;
                        }
                    }
                }
            }
            fprintf(stderr, "qwen36_unified prefill layer=%u ms=%.2f\n", st->fxs[li].layer, now_ms() - layer_t0);
            fflush(stderr);
            free(state_seq);
            state_seq = next_seq;
            next_seq = NULL;
        }
        if (st->enable_full_prefill_cpu) {
            const uint32_t full_layer = st->cfg.cycles[ci].full_layer;
            double full_t0 = now_ms();
            fprintf(stderr, "qwen36_unified prefill full layer=%u rows=%u start\n", full_layer, seq_len);
            fflush(stderr);
            if (!run_full_layer_prefill_cpu(&st->model.layers[full_layer], &st->full_caches[ci], &st->full_states[ci], &st->gf, state_seq, seq_len, err, err_cap)) {
                goto fail;
            }
            next_seq = (float *)malloc(seq_hidden * sizeof(float));
            if (!next_seq) {
                snprintf(err, err_cap, "oom full prefill copy");
                goto fail;
            }
            memcpy(next_seq, st->full_states[ci].output_seq, seq_hidden * sizeof(float));
            fprintf(stderr, "qwen36_unified prefill full layer=%u ms=%.2f\n", full_layer, now_ms() - full_t0);
            fflush(stderr);
            free(state_seq);
            state_seq = next_seq;
            next_seq = NULL;
        }
#ifdef QWEN36_UNIFIED_HAVE_GPU
        else if (st->enable_full_prefill_gpu) {
            const uint32_t full_layer = st->cfg.cycles[ci].full_layer;
            double full_t0 = now_ms();
            fprintf(stderr, "qwen36_unified prefill full layer=%u rows=%u start[gpu]\n", full_layer, seq_len);
            fflush(stderr);
            if (!run_full_layer_prefill_gpu(full_layer, &st->model.layers[full_layer], &st->mf, &st->full_small_caches[ci], &st->full_states[ci], state_seq, seq_len)) {
                snprintf(err, err_cap, "gpu full prefill failed layer %u", full_layer);
                goto fail;
            }
            fprintf(stderr,
                    "qwen36_unified prefill full layer=%u returned[gpu] state_output_seq=%p state_seq_len=%u\n",
                    full_layer, (void *)st->full_states[ci].output_seq, st->full_states[ci].seq_len);
            fflush(stderr);
            next_seq = (float *)malloc(seq_hidden * sizeof(float));
            if (!next_seq) {
                snprintf(err, err_cap, "oom full prefill gpu copy");
                goto fail;
            }
            fprintf(stderr,
                    "qwen36_unified prefill full layer=%u copying[gpu] bytes=%zu from=%p to=%p\n",
                    full_layer, seq_hidden * sizeof(float), (void *)st->full_states[ci].output_seq, (void *)next_seq);
            fflush(stderr);
            memcpy(next_seq, st->full_states[ci].output_seq, seq_hidden * sizeof(float));
            fprintf(stderr, "qwen36_unified prefill full layer=%u copied[gpu]\n", full_layer);
            fflush(stderr);
            fprintf(stderr, "qwen36_unified prefill full layer=%u ms=%.2f[gpu]\n", full_layer, now_ms() - full_t0);
            fflush(stderr);
            free(state_seq);
            state_seq = next_seq;
            next_seq = NULL;
        }
#endif
    }
    st->owned_seq = state_seq;
    st->seq_len = seq_len;
    st->owned_cap = seq_len;
    st->prefilled = 1;
    fprintf(stderr, "qwen36_unified prefill fixtures=%u full_prefill_cpu=%d"
#ifdef QWEN36_UNIFIED_HAVE_GPU
            " full_prefill_gpu=%d"
#endif
            " seq_len=%u ms=%.2f\n",
            st->n_fx, st->enable_full_prefill_cpu,
#ifdef QWEN36_UNIFIED_HAVE_GPU
            st->enable_full_prefill_gpu,
#endif
            st->seq_len, now_ms() - t0);
    fflush(stderr);
    return 1;

fail:
    free(state_seq);
    free(next_seq);
    return 0;
}

static int session_step(unified_session_state *st, uint32_t token_id, char *err, size_t err_cap) {
    float *state_row = NULL;
    float *next_row = NULL;
    float *row_a = NULL;
    float *row_b = NULL;
    uint32_t ci, li;
    double t0;
    const int verbose = st->verbose_logs;
    if (!st->prefilled || st->n_fx == 0 || st->fxs[0].layer != 0) {
        snprintf(err, err_cap, "hybrid step requires blk0 first fixture");
        return 0;
    }
    row_a = (float *)malloc((size_t)st->hidden * sizeof(float));
    row_b = (float *)malloc((size_t)st->hidden * sizeof(float));
    if (!row_a || !row_b) {
        snprintf(err, err_cap, "oom step rows");
        return 0;
    }
    state_row = row_a;
    next_row = row_b;
    t0 = now_ms();
    if (!decode_q8_row(&st->gf, st->model.token_embd, token_id, st->hidden, state_row, err, err_cap)) goto fail;
    if (st->enable_full_prefill_cpu) {
        for (ci = 0; ci < st->cfg.n_cycles; ++ci) {
            uint32_t start = st->cycle_fx_offsets[ci];
            uint32_t end = st->cycle_fx_offsets[ci + 1u];
            for (li = start; li < end; ++li) {
                double layer_t0 = now_ms();
                if (verbose) {
                    fprintf(stderr, "qwen36_unified step layer=%u start\n", st->fxs[li].layer);
                    fflush(stderr);
                }
                if (!run_step_layer_dynamic(&st->fxs[li], &st->gf, &st->model.layers[st->fxs[li].layer], &st->layers[li], state_row, next_row, err, err_cap)) goto fail;
                if (verbose) {
                    fprintf(stderr, "qwen36_unified step layer=%u ms=%.2f\n", st->fxs[li].layer, now_ms() - layer_t0);
                    fflush(stderr);
                }
                { float *tmp = state_row; state_row = next_row; next_row = tmp; }
            }
            {
                const uint32_t full_layer = st->cfg.cycles[ci].full_layer;
                double full_t0 = now_ms();
                if (verbose) {
                    fprintf(stderr, "qwen36_unified step full layer=%u start\n", full_layer);
                    fflush(stderr);
                }
                if (!run_full_layer_step_cpu(&st->model.layers[full_layer], &st->full_caches[ci], &st->full_states[ci], &st->gf, state_row, next_row, err, err_cap)) goto fail;
                if (verbose) {
                    fprintf(stderr, "qwen36_unified step full layer=%u ms=%.2f\n", full_layer, now_ms() - full_t0);
                    fflush(stderr);
                }
                { float *tmp = state_row; state_row = next_row; next_row = tmp; }
            }
        }
    } else {
#ifdef QWEN36_UNIFIED_HAVE_GPU
        if (st->enable_full_prefill_gpu) {
            for (ci = 0; ci < st->cfg.n_cycles; ++ci) {
                uint32_t start = st->cycle_fx_offsets[ci];
                uint32_t end = st->cycle_fx_offsets[ci + 1u];
                for (li = start; li < end; ++li) {
                    double layer_t0 = now_ms();
                    if (ci < st->hybrid_gpu_cycles) {
                        if (verbose) {
                            fprintf(stderr, "qwen36_unified step layer=%u start[gpu-proj]\n", st->fxs[li].layer);
                            fflush(stderr);
                        }
                        if (!run_hybrid_layer_step_gpuproj(&st->fxs[li], &st->gf, &st->model.layers[st->fxs[li].layer], &st->mf, &st->layers[li], state_row, next_row, err, err_cap)) goto fail;
                        if (verbose) fprintf(stderr, "qwen36_unified step layer=%u ms=%.2f[gpu-proj]\n", st->fxs[li].layer, now_ms() - layer_t0);
                    } else {
                        if (verbose) {
                            fprintf(stderr, "qwen36_unified step layer=%u start\n", st->fxs[li].layer);
                            fflush(stderr);
                        }
                        if (!run_step_layer_dynamic(&st->fxs[li], &st->gf, &st->model.layers[st->fxs[li].layer], &st->layers[li], state_row, next_row, err, err_cap)) goto fail;
                        if (verbose) fprintf(stderr, "qwen36_unified step layer=%u ms=%.2f\n", st->fxs[li].layer, now_ms() - layer_t0);
                    }
                    if (verbose) fflush(stderr);
                    { float *tmp = state_row; state_row = next_row; next_row = tmp; }
                }
                {
                    const uint32_t full_layer = st->cfg.cycles[ci].full_layer;
                    double full_t0 = now_ms();
                    if (verbose) {
                        fprintf(stderr, "qwen36_unified step full layer=%u start[gpu]\n", full_layer);
                        fflush(stderr);
                    }
                    if (!run_full_layer_step_gpu(&st->model.layers[full_layer], &st->mf, &st->full_small_caches[ci], &st->full_states[ci], state_row, next_row)) {
                        snprintf(err, err_cap, "gpu full step failed layer %u", full_layer);
                        goto fail;
                    }
                    if (verbose) {
                        fprintf(stderr, "qwen36_unified step full layer=%u ms=%.2f[gpu]\n", full_layer, now_ms() - full_t0);
                        fflush(stderr);
                    }
                    { float *tmp = state_row; state_row = next_row; next_row = tmp; }
                }
            }
        } else
#endif
        {
        for (li = 0; li < st->n_fx; ++li) {
            double layer_t0 = now_ms();
            if (verbose) {
                fprintf(stderr, "qwen36_unified step layer=%u start\n", st->fxs[li].layer);
                fflush(stderr);
            }
            if (!run_step_layer_dynamic(&st->fxs[li], &st->gf, &st->model.layers[st->fxs[li].layer], &st->layers[li], state_row, next_row, err, err_cap)) goto fail;
            if (verbose) {
                fprintf(stderr, "qwen36_unified step layer=%u ms=%.2f\n", st->fxs[li].layer, now_ms() - layer_t0);
                fflush(stderr);
            }
            { float *tmp = state_row; state_row = next_row; next_row = tmp; }
        }
        }
    }
    if (!ensure_owned_decode_cap(st, st->seq_len + 1u, err, err_cap)) goto fail;
    memcpy(st->owned_seq + (size_t)st->seq_len * st->hidden, state_row, (size_t)st->hidden * sizeof(float));
    st->token_ids[st->seq_len] = token_id;
    st->seq_len += 1u;
    if (verbose) {
        fprintf(stderr, "qwen36_unified step fixtures=%u seq_len=%u token=%u ms=%.2f\n", st->n_fx, st->seq_len, token_id, now_ms() - t0);
        fflush(stderr);
    }
    free(row_a);
    free(row_b);
    return 1;

fail:
    free(row_a);
    free(row_b);
    return 0;
}

static void session_print_summary(const unified_session_state *st) {
    uint32_t i;
    fprintf(stderr, "qwen36_unified cycles=%u flattened_fixtures=%u first_layer=%u last_layer=%u mode=%s\n",
            st->cfg.n_cycles,
            st->n_fx,
            st->n_fx ? st->fxs[0].layer : 0u,
            st->n_fx ? st->fxs[st->n_fx - 1u].layer : 0u,
            st->enable_full_prefill_cpu ? "hybrid_plus_full_prefill_cpu" :
#ifdef QWEN36_UNIFIED_HAVE_GPU
            (st->enable_full_prefill_gpu ? "hybrid_plus_full_gpu" :
#endif
            "hybrid_only"
#ifdef QWEN36_UNIFIED_HAVE_GPU
            )
#endif
            );
    for (i = 0; i < st->cfg.n_cycles; ++i) {
        fprintf(stderr, "qwen36_unified cycle=%u fixtures=%u full_layer=%u\n",
                i, st->cfg.cycles[i].n_fixtures, st->cfg.cycles[i].full_layer);
    }
    fflush(stderr);
}

static void handle_info(const unified_session_state *st) {
    printf("INFO mode=%s cycles=%u fixtures=%u prefilled=%d seq_len=%u hidden=%u\n",
           st->enable_full_prefill_cpu ? "hybrid_plus_full_prefill_cpu" :
#ifdef QWEN36_UNIFIED_HAVE_GPU
           (st->enable_full_prefill_gpu ? "hybrid_plus_full_gpu" :
#endif
           "hybrid_only"
#ifdef QWEN36_UNIFIED_HAVE_GPU
           )
#endif
           ,
           st->cfg.n_cycles, st->n_fx, st->prefilled, st->seq_len, st->hidden);
    fflush(stdout);
}

static void handle_dump_hidden(const unified_session_state *st) {
    size_t n = (size_t)st->seq_len * st->hidden;
    size_t n_bytes = n * sizeof(float);
    if (!st->prefilled || !st->owned_seq) {
        printf("ERROR not prefilled\n");
        fflush(stdout);
        return;
    }
    printf("HIDDEN %zu %zu\n", n, n_bytes);
    fflush(stdout);
    (void)write_all(STDOUT_FILENO, st->owned_seq, n_bytes);
}

static void handle_dump_last(const unified_session_state *st) {
    const float *last = NULL;
    size_t n_bytes = 0;
    if (!st->prefilled || !st->owned_seq || st->seq_len == 0) {
        printf("ERROR not prefilled\n");
        fflush(stdout);
        return;
    }
    last = st->owned_seq + (size_t)(st->seq_len - 1u) * st->hidden;
    n_bytes = (size_t)st->hidden * sizeof(float);
    printf("LAST %u %zu\n", st->hidden, n_bytes);
    fflush(stdout);
    (void)write_all(STDOUT_FILENO, last, n_bytes);
}

static void handle_prefill_prefix_bin(unified_session_state *st, char *rest) {
    uint32_t seq_len = 0, hidden = 0;
    uint32_t *token_ids = NULL;
    float *input_seq = NULL;
    char err[256] = {0};
    if (sscanf(rest ? rest : "", "%u %u", &seq_len, &hidden) != 2 || seq_len == 0 || hidden == 0) {
        printf("ERROR bad PREFILL_PREFIX_BIN args\n");
        fflush(stdout);
        return;
    }
    token_ids = (uint32_t *)malloc((size_t)seq_len * sizeof(uint32_t));
    input_seq = (float *)malloc((size_t)seq_len * hidden * sizeof(float));
    if (!token_ids || !input_seq) {
        free(token_ids);
        free(input_seq);
        printf("ERROR oom prefill payload\n");
        fflush(stdout);
        return;
    }
    if (!read_all_fd(STDIN_FILENO, token_ids, (size_t)seq_len * sizeof(uint32_t)) ||
        !read_all_fd(STDIN_FILENO, input_seq, (size_t)seq_len * hidden * sizeof(float))) {
        free(token_ids);
        free(input_seq);
        printf("ERROR short prefill payload\n");
        fflush(stdout);
        return;
    }
    if (!session_prefill(st, token_ids, seq_len, hidden, input_seq, err, sizeof(err))) {
        free(token_ids);
        free(input_seq);
        printf("ERROR %s\n", err);
        fflush(stdout);
        return;
    }
    free(token_ids);
    free(input_seq);
    printf("PREFILL_OK %u %u\n", st->seq_len, st->hidden);
    fflush(stdout);
}

static void handle_step(unified_session_state *st, char *rest) {
    uint32_t token_id = 0;
    char err[256] = {0};
    if (sscanf(rest ? rest : "", "%u", &token_id) != 1) {
        printf("ERROR bad STEP args\n");
        fflush(stdout);
        return;
    }
    if (!session_step(st, token_id, err, sizeof(err))) {
        printf("ERROR %s\n", err);
        fflush(stdout);
        return;
    }
    printf("STEP_OK %u %u\n", st->seq_len, st->hidden);
    fflush(stdout);
}

int main(int argc, char **argv) {
    unified_session_state st;
    char err[512];
    char line[4096];
    memset(&st, 0, sizeof(st));
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL.gguf CONFIG.txt\n", argv[0]);
        return 1;
    }
    if (!parse_config(argv[2], &st.cfg, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        free_worker_config(&st.cfg);
        return 1;
    }
    if (!session_init_runtime(&st, argv[1], err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        session_close_runtime(&st);
        free_worker_config(&st.cfg);
        return 1;
    }
    setvbuf(stdin, NULL, _IONBF, 0);
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    session_print_summary(&st);
    printf("READY\n");
    fflush(stdout);

    while (fgets(line, sizeof(line), stdin)) {
        char *cmd = NULL;
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        if (len == 0) continue;
        cmd = strtok(line, " ");
        if (!cmd) continue;
        if (strcmp(cmd, "INFO") == 0) {
            handle_info(&st);
        } else if (strcmp(cmd, "PREFILL_PREFIX_BIN") == 0) {
            handle_prefill_prefix_bin(&st, strtok(NULL, ""));
        } else if (strcmp(cmd, "STEP") == 0) {
            handle_step(&st, strtok(NULL, ""));
        } else if (strcmp(cmd, "DUMP_HIDDEN") == 0) {
            handle_dump_hidden(&st);
        } else if (strcmp(cmd, "DUMP_LAST") == 0) {
            handle_dump_last(&st);
        } else if (strcmp(cmd, "RESET") == 0) {
            session_clear_decode_state(&st);
            printf("OK\n");
            fflush(stdout);
        } else if (strcmp(cmd, "QUIT") == 0) {
            printf("OK\n");
            fflush(stdout);
            break;
        } else {
            printf("ERROR unknown command\n");
            fflush(stdout);
        }
    }

    session_close_runtime(&st);
    free_worker_config(&st.cfg);
    return 0;
}
