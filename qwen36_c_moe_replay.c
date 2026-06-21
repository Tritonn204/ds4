#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "Q36MOEF2"
#define MAGIC_LEN 8

typedef struct moe_fixture {
    uint32_t layer;
    uint32_t hidden;
    uint32_t intermediate;
    uint32_t topk;
    float *layer_input;
    float *mixer_out;
    float *hidden_in;
    float *residual;
    float *hf_mlp;
    float *hf_layer;
    uint32_t *router_indices;
    float *router_scores;
    float shared_gate_pre;
    float *gate_sel;
    float *up_sel;
    float *down_sel;
    float *shared_gate;
    float *shared_up;
    float *shared_down;
} moe_fixture;

static bool read_exact(FILE *fp, void *buf, size_t n) {
    return fread(buf, 1, n, fp) == n;
}

static bool alloc_read_f32(FILE *fp, float **out, size_t count) {
    float *p = (float *)malloc(count * sizeof(float));
    if (!p) return false;
    if (!read_exact(fp, p, count * sizeof(float))) {
        free(p);
        return false;
    }
    *out = p;
    return true;
}

static bool alloc_read_u32(FILE *fp, uint32_t **out, size_t count) {
    uint32_t *p = (uint32_t *)malloc(count * sizeof(uint32_t));
    if (!p) return false;
    if (!read_exact(fp, p, count * sizeof(uint32_t))) {
        free(p);
        return false;
    }
    *out = p;
    return true;
}

static void fixture_free(moe_fixture *fx) {
    if (!fx) return;
    free(fx->layer_input);
    free(fx->mixer_out);
    free(fx->hidden_in);
    free(fx->residual);
    free(fx->hf_mlp);
    free(fx->hf_layer);
    free(fx->router_indices);
    free(fx->router_scores);
    free(fx->gate_sel);
    free(fx->up_sel);
    free(fx->down_sel);
    free(fx->shared_gate);
    free(fx->shared_up);
    free(fx->shared_down);
    memset(fx, 0, sizeof(*fx));
}

static bool fixture_load(const char *path, moe_fixture *fx) {
    FILE *fp = fopen(path, "rb");
    char magic[MAGIC_LEN];
    size_t expert_count;
    memset(fx, 0, sizeof(*fx));
    if (!fp) {
        fprintf(stderr, "open %s failed\n", path);
        return false;
    }
    if (!read_exact(fp, magic, MAGIC_LEN) || memcmp(magic, MAGIC, MAGIC_LEN) != 0) {
        fprintf(stderr, "bad fixture magic\n");
        fclose(fp);
        return false;
    }
    if (!read_exact(fp, &fx->layer, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->hidden, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->intermediate, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->topk, sizeof(uint32_t))) {
        fprintf(stderr, "bad fixture header\n");
        fclose(fp);
        return false;
    }
    expert_count = (size_t)fx->topk * fx->intermediate * fx->hidden;
    if (!alloc_read_f32(fp, &fx->layer_input, fx->hidden) ||
        !alloc_read_f32(fp, &fx->mixer_out, fx->hidden) ||
        !alloc_read_f32(fp, &fx->hidden_in, fx->hidden) ||
        !alloc_read_f32(fp, &fx->residual, fx->hidden) ||
        !alloc_read_f32(fp, &fx->hf_mlp, fx->hidden) ||
        !alloc_read_f32(fp, &fx->hf_layer, fx->hidden) ||
        !alloc_read_u32(fp, &fx->router_indices, fx->topk) ||
        !alloc_read_f32(fp, &fx->router_scores, fx->topk) ||
        !read_exact(fp, &fx->shared_gate_pre, sizeof(float)) ||
        !alloc_read_f32(fp, &fx->gate_sel, expert_count) ||
        !alloc_read_f32(fp, &fx->up_sel, expert_count) ||
        !alloc_read_f32(fp, &fx->down_sel, expert_count) ||
        !alloc_read_f32(fp, &fx->shared_gate, (size_t)fx->intermediate * fx->hidden) ||
        !alloc_read_f32(fp, &fx->shared_up, (size_t)fx->intermediate * fx->hidden) ||
        !alloc_read_f32(fp, &fx->shared_down, (size_t)fx->hidden * fx->intermediate)) {
        fprintf(stderr, "bad fixture payload\n");
        fclose(fp);
        fixture_free(fx);
        return false;
    }
    fclose(fp);
    return true;
}

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

static void matvec(const float *mat, const float *vec, float *out, uint32_t rows, uint32_t cols) {
    uint32_t r, c;
    for (r = 0; r < rows; ++r) {
        const float *row = mat + (size_t)r * cols;
        double sum = 0.0;
        for (c = 0; c < cols; ++c) sum += (double)row[c] * vec[c];
        out[r] = (float)sum;
    }
}

static double vec_rmse(const float *a, const float *b, uint32_t n) {
    uint32_t i;
    double acc = 0.0;
    for (i = 0; i < n; ++i) {
        double d = (double)a[i] - b[i];
        acc += d * d;
    }
    return sqrt(acc / n);
}

static double vec_mae(const float *a, const float *b, uint32_t n) {
    uint32_t i;
    double acc = 0.0;
    for (i = 0; i < n; ++i) {
        double d = fabs((double)a[i] - b[i]);
        acc += d;
    }
    return acc / n;
}

static double vec_cosine(const float *a, const float *b, uint32_t n) {
    uint32_t i;
    double dot = 0.0, an = 0.0, bn = 0.0;
    for (i = 0; i < n; ++i) {
        dot += (double)a[i] * b[i];
        an += (double)a[i] * a[i];
        bn += (double)b[i] * b[i];
    }
    if (an == 0.0 || bn == 0.0) return NAN;
    return dot / sqrt(an * bn);
}

int main(int argc, char **argv) {
    moe_fixture fx;
    float *gate = NULL, *up = NULL, *act = NULL, *down = NULL;
    float *routed = NULL, *shared_gate = NULL, *shared_up = NULL, *shared_act = NULL, *shared = NULL;
    float *residual = NULL, *mlp = NULL, *layer = NULL;
    uint32_t i, j;
    if (argc != 2) {
        fprintf(stderr, "usage: %s FIXTURE.bin\n", argv[0]);
        return 1;
    }
    if (!fixture_load(argv[1], &fx)) return 1;

    gate = (float *)calloc(fx.intermediate, sizeof(float));
    up = (float *)calloc(fx.intermediate, sizeof(float));
    act = (float *)calloc(fx.intermediate, sizeof(float));
    down = (float *)calloc(fx.hidden, sizeof(float));
    routed = (float *)calloc(fx.hidden, sizeof(float));
    shared_gate = (float *)calloc(fx.intermediate, sizeof(float));
    shared_up = (float *)calloc(fx.intermediate, sizeof(float));
    shared_act = (float *)calloc(fx.intermediate, sizeof(float));
    shared = (float *)calloc(fx.hidden, sizeof(float));
    residual = (float *)calloc(fx.hidden, sizeof(float));
    mlp = (float *)calloc(fx.hidden, sizeof(float));
    layer = (float *)calloc(fx.hidden, sizeof(float));
    if (!gate || !up || !act || !down || !routed || !shared_gate || !shared_up || !shared_act || !shared || !residual || !mlp || !layer) {
        fprintf(stderr, "oom\n");
        fixture_free(&fx);
        free(gate); free(up); free(act); free(down); free(routed);
        free(shared_gate); free(shared_up); free(shared_act); free(shared); free(residual); free(mlp); free(layer);
        return 1;
    }

    for (i = 0; i < fx.topk; ++i) {
        const float *gate_mat = fx.gate_sel + (size_t)i * fx.intermediate * fx.hidden;
        const float *up_mat = fx.up_sel + (size_t)i * fx.intermediate * fx.hidden;
        const float *down_mat = fx.down_sel + (size_t)i * fx.hidden * fx.intermediate;
        matvec(gate_mat, fx.hidden_in, gate, fx.intermediate, fx.hidden);
        matvec(up_mat, fx.hidden_in, up, fx.intermediate, fx.hidden);
        for (j = 0; j < fx.intermediate; ++j) act[j] = silu(gate[j]) * up[j];
        matvec(down_mat, act, down, fx.hidden, fx.intermediate);
        for (j = 0; j < fx.hidden; ++j) routed[j] += down[j] * fx.router_scores[i];
    }

    matvec(fx.shared_gate, fx.hidden_in, shared_gate, fx.intermediate, fx.hidden);
    matvec(fx.shared_up, fx.hidden_in, shared_up, fx.intermediate, fx.hidden);
    for (j = 0; j < fx.intermediate; ++j) shared_act[j] = silu(shared_gate[j]) * shared_up[j];
    matvec(fx.shared_down, shared_act, shared, fx.hidden, fx.intermediate);
    {
        const float shared_scale = 1.0f / (1.0f + expf(-fx.shared_gate_pre));
        for (j = 0; j < fx.hidden; ++j) {
            residual[j] = fx.layer_input[j] + fx.mixer_out[j];
            mlp[j] = routed[j] + shared[j] * shared_scale;
            layer[j] = residual[j] + mlp[j];
        }
    }

    printf("layer: %u\n", fx.layer);
    printf("topk: %u\n", fx.topk);
    printf("residual_rmse: %.8f\n", vec_rmse(residual, fx.residual, fx.hidden));
    printf("residual_mae: %.8f\n", vec_mae(residual, fx.residual, fx.hidden));
    printf("residual_cosine: %.8f\n", vec_cosine(residual, fx.residual, fx.hidden));
    printf("mlp_rmse: %.8f\n", vec_rmse(mlp, fx.hf_mlp, fx.hidden));
    printf("mlp_mae: %.8f\n", vec_mae(mlp, fx.hf_mlp, fx.hidden));
    printf("mlp_cosine: %.8f\n", vec_cosine(mlp, fx.hf_mlp, fx.hidden));
    printf("layer_rmse: %.8f\n", vec_rmse(layer, fx.hf_layer, fx.hidden));
    printf("layer_mae: %.8f\n", vec_mae(layer, fx.hf_layer, fx.hidden));
    printf("layer_cosine: %.8f\n", vec_cosine(layer, fx.hf_layer, fx.hidden));

    fixture_free(&fx);
    free(gate); free(up); free(act); free(down); free(routed);
    free(shared_gate); free(shared_up); free(shared_act); free(shared); free(residual); free(mlp); free(layer);
    return 0;
}
