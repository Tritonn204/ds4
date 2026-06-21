#include "qwen36_35a3b_iq2xxs.h"

#include <stdio.h>
#include <string.h>

static void usage(const char *prog) {
    fprintf(stderr, "usage: %s <model.gguf> [--summary]\n", prog);
}

int main(int argc, char **argv) {
    qwen36_gguf_file gf;
    qwen36_35a3b_iq2xxs_model model;
    char err[512];
    const char *path = NULL;
    int want_summary = 0;
    int i;

    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--summary") == 0) {
            want_summary = 1;
        } else if (!path) {
            path = argv[i];
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (!path) {
        usage(argv[0]);
        return 2;
    }
    if (!qwen36_gguf_open(&gf, path, err, sizeof(err))) {
        fprintf(stderr, "error: %s\n", err);
        return 1;
    }
    if (!qwen36_35a3b_iq2xxs_bind(&gf, &model, err, sizeof(err))) {
        fprintf(stderr, "contract mismatch: %s\n", err);
        qwen36_gguf_close(&gf);
        return 1;
    }
    printf("OK: exact Qwen3.6-35B-A3B UD_IQ2_XXS contract matched\n");
    if (want_summary) qwen36_35a3b_iq2xxs_dump_summary(&model, stdout);
    qwen36_gguf_close(&gf);
    return 0;
}
