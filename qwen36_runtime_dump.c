#include "qwen36_runtime.h"

#include <stdio.h>

static void usage(FILE *fp) {
    fputs("usage: qwen36-runtime-dump <model.gguf>\n", fp);
}

int main(int argc, char **argv) {
    qwen36_gguf_file gf;
    qwen36_runtime_plan plan;
    char err[512];

    if (argc != 2) {
        usage(stderr);
        return 2;
    }
    if (!qwen36_gguf_open(&gf, argv[1], err, sizeof(err))) {
        fprintf(stderr, "qwen36-runtime-dump: %s\n", err);
        return 1;
    }
    if (!qwen36_runtime_build(&gf, &plan, err, sizeof(err))) {
        fprintf(stderr, "qwen36-runtime-dump: unsupported contract: %s\n", err);
        qwen36_gguf_close(&gf);
        return 1;
    }
    qwen36_runtime_dump(&plan, stdout);
    qwen36_gguf_close(&gf);
    return 0;
}
