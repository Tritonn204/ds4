#include "qwen36_35a3b_q8.h"

#include <stdio.h>
#include <string.h>

static void usage(FILE *fp) {
    fputs("usage: qwen36-35a3b-q8-check <model.gguf> [--summary]\n", fp);
}

int main(int argc, char **argv) {
    qwen36_gguf_file gf;
    qwen36_35a3b_q8_model model;
    char err[512];
    const char *path = NULL;
    int want_summary = 0;
    int i;

    if (argc < 2) {
        usage(stderr);
        return 2;
    }
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--summary") == 0) {
            want_summary = 1;
        } else if (!path) {
            path = argv[i];
        } else {
            usage(stderr);
            return 2;
        }
    }
    if (!path) {
        usage(stderr);
        return 2;
    }
    if (!qwen36_gguf_open(&gf, path, err, sizeof(err))) {
        fprintf(stderr, "qwen36-35a3b-q8-check: %s\n", err);
        return 1;
    }
    if (!qwen36_35a3b_q8_bind(&gf, &model, err, sizeof(err))) {
        fprintf(stderr, "qwen36-35a3b-q8-check: validation failed: %s\n", err);
        qwen36_gguf_close(&gf);
        return 1;
    }
    puts("OK: exact Qwen3.6-35B-A3B Q8_0 contract matched");
    if (want_summary) qwen36_35a3b_q8_dump_summary(&model, stdout);
    qwen36_gguf_close(&gf);
    return 0;
}
