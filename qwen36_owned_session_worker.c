#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>

#define PREFIX_MAGIC "Q36PFX01"
#define PREFIX_MAGIC_LEN 8

typedef struct cycle_config {
    char **fixtures;
    uint32_t n_fixtures;
    uint32_t full_layer;
} cycle_config;

typedef struct child_proc {
    pid_t pid;
    int in_fd;
    int out_fd;
    int err_fd;
} child_proc;

typedef struct worker_config {
    char *prefix_worker_bin;
    char *prefix_fixture;
    char *hybrid_worker_bin;
    char *full_worker_bin;
    cycle_config *cycles;
    uint32_t n_cycles;
} worker_config;

typedef struct session_state {
    worker_config cfg;
    child_proc prefix_worker;
    child_proc *hybrid_workers;
    child_proc *full_workers;
    int prefix_worker_live;
    uint32_t seq_len;
    uint32_t hidden;
    uint32_t *token_ids;
    float *owned_seq;
} session_state;

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
    free(cfg->prefix_worker_bin);
    free(cfg->prefix_fixture);
    free(cfg->hybrid_worker_bin);
    free(cfg->full_worker_bin);
    for (i = 0; i < cfg->n_cycles; ++i) free_cycle(&cfg->cycles[i]);
    free(cfg->cycles);
    memset(cfg, 0, sizeof(*cfg));
}

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

static int read_all(int fd, void *buf, size_t n) {
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

static int read_line_fd(int fd, char *buf, size_t cap) {
    size_t pos = 0;
    while (pos + 1 < cap) {
        char ch = 0;
        ssize_t r = read(fd, &ch, 1);
        if (r < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (r == 0) break;
        if (ch == '\n') break;
        if (ch != '\r') buf[pos++] = ch;
    }
    buf[pos] = '\0';
    return pos > 0;
}

static void read_fd_available(int fd, char *buf, size_t cap) {
    size_t pos = 0;
    int flags;
    ssize_t r;
    if (cap == 0) return;
    buf[0] = '\0';
    if (fd < 0) return;
    flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return;
    if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) return;
    while (pos + 1 < cap) {
        r = read(fd, buf + pos, cap - pos - 1);
        if (r > 0) {
            pos += (size_t)r;
            continue;
        }
        break;
    }
    buf[pos] = '\0';
    (void)fcntl(fd, F_SETFL, flags);
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
        if (strcmp(cmd, "prefix_worker_bin") == 0) {
            cfg->prefix_worker_bin = xstrdup_local(rest);
        } else if (strcmp(cmd, "prefix_fixture") == 0) {
            cfg->prefix_fixture = xstrdup_local(rest);
        } else if (strcmp(cmd, "hybrid_worker_bin") == 0) {
            cfg->hybrid_worker_bin = xstrdup_local(rest);
        } else if (strcmp(cmd, "full_worker_bin") == 0) {
            cfg->full_worker_bin = xstrdup_local(rest);
        } else if (strcmp(cmd, "cycle") == 0) {
            char *layer_s = strtok(rest, " \t");
            char *fixtures_s = strtok(NULL, "");
            cycle_config *next_cycles;
            cycle_config *cy;
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
    if (cfg->prefix_worker_bin && !cfg->prefix_fixture) {
        snprintf(err, err_cap, "prefix worker requires prefix_fixture");
        return 0;
    }
    if (cfg->n_cycles == 0) {
        snprintf(err, err_cap, "no cycles configured");
        return 0;
    }
    return 1;
}

static int spawn_child(child_proc *cp, char *const argv[], char *err, size_t err_cap) {
    int in_pipe[2], out_pipe[2], err_pipe[2];
    char line[512];
    char stderr_buf[1024];
    int tries = 0;
    memset(cp, 0, sizeof(*cp));
    cp->pid = -1;
    cp->in_fd = cp->out_fd = cp->err_fd = -1;
    if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0 || pipe(err_pipe) != 0) {
        snprintf(err, err_cap, "pipe failed: %s", strerror(errno));
        return 0;
    }
    cp->pid = fork();
    if (cp->pid < 0) {
        snprintf(err, err_cap, "fork failed: %s", strerror(errno));
        return 0;
    }
    if (cp->pid == 0) {
        dup2(in_pipe[0], STDIN_FILENO);
        dup2(out_pipe[1], STDOUT_FILENO);
        dup2(err_pipe[1], STDERR_FILENO);
        close(in_pipe[1]); close(out_pipe[0]); close(err_pipe[0]);
        execvp(argv[0], argv);
        _exit(127);
    }
    close(in_pipe[0]); close(out_pipe[1]); close(err_pipe[1]);
    cp->in_fd = in_pipe[1];
    cp->out_fd = out_pipe[0];
    cp->err_fd = err_pipe[0];
    for (tries = 0; tries < 32; ++tries) {
        if (!read_line_fd(cp->out_fd, line, sizeof(line))) {
            read_fd_available(cp->err_fd, stderr_buf, sizeof(stderr_buf));
            if (stderr_buf[0] != '\0') {
                snprintf(err, err_cap, "worker failed to start: %s", stderr_buf);
            } else {
                snprintf(err, err_cap, "worker failed to start");
            }
            return 0;
        }
        if (strcmp(line, "READY") == 0) return 1;
        if (strncmp(line, "ERROR ", 6) == 0) {
            snprintf(err, err_cap, "worker startup error: %s", line);
            return 0;
        }
        if (strncmp(line, "ds4:", 4) == 0 || strncmp(line, "qwen36_", 7) == 0 || line[0] == '\0') {
            continue;
        }
        snprintf(err, err_cap, "worker not ready: %s", line);
        return 0;
    }
    snprintf(err, err_cap, "worker startup exceeded preamble budget");
    return 0;
}

static void close_child(child_proc *cp) {
    char line[256];
    if (!cp || cp->pid <= 0) return;
    if (cp->in_fd >= 0) {
        (void)write_all(cp->in_fd, "QUIT\n", 5);
    }
    if (cp->out_fd >= 0) {
        (void)read_line_fd(cp->out_fd, line, sizeof(line));
    }
    if (cp->in_fd >= 0) close(cp->in_fd);
    if (cp->out_fd >= 0) close(cp->out_fd);
    if (cp->err_fd >= 0) close(cp->err_fd);
    waitpid(cp->pid, NULL, 0);
    memset(cp, 0, sizeof(*cp));
    cp->pid = -1;
    cp->in_fd = cp->out_fd = cp->err_fd = -1;
}

static int child_cmd_line(child_proc *cp, const char *cmd, char *resp, size_t resp_cap) {
    if (!write_all(cp->in_fd, cmd, strlen(cmd)) || !read_line_fd(cp->out_fd, resp, resp_cap)) return 0;
    return 1;
}

static int child_reset(child_proc *cp, char *err, size_t err_cap) {
    char line[256];
    if (!child_cmd_line(cp, "RESET\n", line, sizeof(line))) {
        snprintf(err, err_cap, "reset failed");
        return 0;
    }
    if (strcmp(line, "OK") != 0) {
        snprintf(err, err_cap, "reset bad response: %s", line);
        return 0;
    }
    return 1;
}

static int child_dump_hidden(child_proc *cp, float **out_seq, uint32_t *out_seq_len, uint32_t hidden, char *err, size_t err_cap) {
    char line[256];
    size_t n_floats = 0, n_bytes = 0;
    float *buf = NULL;
    if (!child_cmd_line(cp, "DUMP_HIDDEN\n", line, sizeof(line))) {
        snprintf(err, err_cap, "dump hidden failed");
        return 0;
    }
    if (sscanf(line, "HIDDEN %zu %zu", &n_floats, &n_bytes) != 2 || n_bytes != n_floats * sizeof(float)) {
        snprintf(err, err_cap, "bad HIDDEN header: %s", line);
        return 0;
    }
    if (hidden == 0 || n_floats % hidden != 0) {
        snprintf(err, err_cap, "bad HIDDEN shape");
        return 0;
    }
    buf = (float *)malloc(n_bytes);
    if (!buf) {
        snprintf(err, err_cap, "oom hidden");
        return 0;
    }
    if (!read_all(cp->out_fd, buf, n_bytes)) {
        free(buf);
        snprintf(err, err_cap, "short HIDDEN payload");
        return 0;
    }
    *out_seq = buf;
    *out_seq_len = (uint32_t)(n_floats / hidden);
    return 1;
}

static int child_dump_last(child_proc *cp, float *out_row, uint32_t hidden, char *err, size_t err_cap) {
    char line[256];
    uint32_t got_hidden = 0;
    size_t n_bytes = 0;
    if (!child_cmd_line(cp, "DUMP_LAST\n", line, sizeof(line))) {
        snprintf(err, err_cap, "dump last failed");
        return 0;
    }
    if (sscanf(line, "LAST %u %zu", &got_hidden, &n_bytes) != 2 || got_hidden != hidden || n_bytes != (size_t)hidden * sizeof(float)) {
        snprintf(err, err_cap, "bad LAST header: %s", line);
        return 0;
    }
    if (!read_all(cp->out_fd, out_row, n_bytes)) {
        snprintf(err, err_cap, "short LAST payload");
        return 0;
    }
    return 1;
}

static int child_prefill_prefix_bin(child_proc *cp, const uint32_t *token_ids, uint32_t seq_len, uint32_t hidden, const float *input_seq, char *err, size_t err_cap) {
    char cmd[128], line[256];
    snprintf(cmd, sizeof(cmd), "PREFILL_PREFIX_BIN %u %u\n", seq_len, hidden);
    if (!write_all(cp->in_fd, cmd, strlen(cmd)) ||
        !write_all(cp->in_fd, token_ids, (size_t)seq_len * sizeof(uint32_t)) ||
        !write_all(cp->in_fd, input_seq, (size_t)seq_len * hidden * sizeof(float)) ||
        !read_line_fd(cp->out_fd, line, sizeof(line))) {
        snprintf(err, err_cap, "prefix prefill failed");
        return 0;
    }
    if (strncmp(line, "PREFILL_OK ", 11) != 0) {
        snprintf(err, err_cap, "prefix prefill bad response: %s", line);
        return 0;
    }
    return 1;
}

static int child_prefill_seq_bin(child_proc *cp, const float *seq, uint32_t seq_len, uint32_t hidden, char *err, size_t err_cap) {
    char cmd[128], line[256];
    snprintf(cmd, sizeof(cmd), "PREFILL_SEQ_BIN %u %u\n", seq_len, hidden);
    if (!write_all(cp->in_fd, cmd, strlen(cmd)) ||
        !write_all(cp->in_fd, seq, (size_t)seq_len * hidden * sizeof(float)) ||
        !read_line_fd(cp->out_fd, line, sizeof(line))) {
        snprintf(err, err_cap, "seq prefill failed");
        return 0;
    }
    if (strncmp(line, "PREFILL_OK ", 11) != 0) {
        snprintf(err, err_cap, "seq prefill bad response: %s", line);
        return 0;
    }
    return 1;
}

static int child_step_token(child_proc *cp, uint32_t token_id, char *err, size_t err_cap) {
    char cmd[128], line[256];
    snprintf(cmd, sizeof(cmd), "STEP %u\n", token_id);
    if (!child_cmd_line(cp, cmd, line, sizeof(line))) {
        snprintf(err, err_cap, "step token failed");
        return 0;
    }
    if (strncmp(line, "STEP_OK ", 8) != 0) {
        snprintf(err, err_cap, "step token bad response: %s", line);
        return 0;
    }
    return 1;
}

static int child_step_row_bin(child_proc *cp, const float *row, uint32_t hidden, char *err, size_t err_cap) {
    char cmd[128], line[256];
    snprintf(cmd, sizeof(cmd), "STEP_ROW_BIN %u\n", hidden);
    if (!write_all(cp->in_fd, cmd, strlen(cmd)) ||
        !write_all(cp->in_fd, row, (size_t)hidden * sizeof(float)) ||
        !read_line_fd(cp->out_fd, line, sizeof(line))) {
        snprintf(err, err_cap, "step row failed");
        return 0;
    }
    if (strncmp(line, "STEP_OK ", 8) != 0) {
        snprintf(err, err_cap, "step row bad response: %s", line);
        return 0;
    }
    return 1;
}

static int read_fixture_layer(const char *path, uint32_t *layer_out) {
    FILE *fp = fopen(path, "rb");
    char magic[8];
    uint32_t layer = 0;
    if (!fp) return 0;
    if (fread(magic, 1, 8, fp) != 8 || fread(&layer, sizeof(uint32_t), 1, fp) != 1) {
        fclose(fp);
        return 0;
    }
    fclose(fp);
    *layer_out = layer;
    return 1;
}

static int session_spawn_workers(session_state *st, const char *gguf, char *err, size_t err_cap) {
    uint32_t i, j;
    char *const *argv = NULL;
    (void)argv;
    st->hybrid_workers = (child_proc *)calloc(st->cfg.n_cycles, sizeof(child_proc));
    st->full_workers = (child_proc *)calloc(st->cfg.n_cycles, sizeof(child_proc));
    if (!st->hybrid_workers || !st->full_workers) {
        snprintf(err, err_cap, "oom worker arrays");
        return 0;
    }
    if (st->cfg.prefix_worker_bin) {
        fprintf(stderr, "spawn prefix worker\n");
        fflush(stderr);
        char *args[] = { st->cfg.prefix_worker_bin, (char *)gguf, st->cfg.prefix_fixture, NULL };
        if (!spawn_child(&st->prefix_worker, args, err, err_cap)) return 0;
        st->prefix_worker_live = 1;
    }
    for (i = 0; i < st->cfg.n_cycles; ++i) {
        cycle_config *cy = &st->cfg.cycles[i];
        char **args = (char **)calloc((size_t)cy->n_fixtures + 3, sizeof(char *));
        char layer_buf[32];
        if (!args) {
            snprintf(err, err_cap, "oom spawn args");
            return 0;
        }
        args[0] = st->cfg.hybrid_worker_bin;
        args[1] = (char *)gguf;
        for (j = 0; j < cy->n_fixtures; ++j) args[2 + j] = cy->fixtures[j];
        fprintf(stderr, "spawn hybrid worker cycle=%u fixtures=%u\n", i, cy->n_fixtures);
        fflush(stderr);
        if (!spawn_child(&st->hybrid_workers[i], args, err, err_cap)) {
            free(args);
            return 0;
        }
        free(args);
        snprintf(layer_buf, sizeof(layer_buf), "%u", cy->full_layer);
        {
            fprintf(stderr, "spawn full worker cycle=%u layer=%u\n", i, cy->full_layer);
            fflush(stderr);
            char *fargs[] = { st->cfg.full_worker_bin, (char *)gguf, "--layer", layer_buf, NULL };
            if (!spawn_child(&st->full_workers[i], fargs, err, err_cap)) return 0;
        }
    }
    return 1;
}

static void session_close_workers(session_state *st) {
    uint32_t i;
    if (st->prefix_worker_live) close_child(&st->prefix_worker);
    for (i = 0; i < st->cfg.n_cycles; ++i) {
        close_child(&st->hybrid_workers[i]);
        close_child(&st->full_workers[i]);
    }
    free(st->hybrid_workers);
    free(st->full_workers);
    st->hybrid_workers = NULL;
    st->full_workers = NULL;
    st->prefix_worker_live = 0;
}

static void session_reset_state(session_state *st) {
    uint32_t i;
    char err[256];
    if (st->prefix_worker_live) (void)child_reset(&st->prefix_worker, err, sizeof(err));
    for (i = 0; i < st->cfg.n_cycles; ++i) {
        (void)child_reset(&st->hybrid_workers[i], err, sizeof(err));
        (void)child_reset(&st->full_workers[i], err, sizeof(err));
    }
    free(st->token_ids);
    free(st->owned_seq);
    st->token_ids = NULL;
    st->owned_seq = NULL;
    st->seq_len = 0;
    st->hidden = 0;
}

static int session_prefill(session_state *st, const uint32_t *token_ids, uint32_t seq_len, uint32_t hidden, char *err, size_t err_cap) {
    uint32_t i;
    float *cur_seq = NULL;
    uint32_t cur_seq_len = 0;
    float *zero_seq = NULL;
    session_reset_state(st);
    st->hidden = hidden;
    st->seq_len = seq_len;
    st->token_ids = (uint32_t *)malloc((size_t)seq_len * sizeof(uint32_t));
    if (!st->token_ids) {
        snprintf(err, err_cap, "oom token ids");
        return 0;
    }
    memcpy(st->token_ids, token_ids, (size_t)seq_len * sizeof(uint32_t));
    zero_seq = (float *)calloc((size_t)seq_len * hidden, sizeof(float));
    if (!zero_seq) {
        snprintf(err, err_cap, "oom zero seq");
        return 0;
    }
    if (st->prefix_worker_live) {
        fprintf(stderr, "prefill prefix seq_len=%u\n", seq_len);
        fflush(stderr);
        if (!child_prefill_prefix_bin(&st->prefix_worker, token_ids, seq_len, hidden, zero_seq, err, err_cap) ||
            !child_dump_hidden(&st->prefix_worker, &cur_seq, &cur_seq_len, hidden, err, err_cap)) {
            free(zero_seq);
            return 0;
        }
    }
    for (i = 0; i < st->cfg.n_cycles; ++i) {
        cycle_config *cy = &st->cfg.cycles[i];
        child_proc *hy = &st->hybrid_workers[i];
        child_proc *fw = &st->full_workers[i];
        float *next_seq = NULL;
        uint32_t next_seq_len = 0;
        uint32_t first_layer = 0;
        if (!read_fixture_layer(cy->fixtures[0], &first_layer)) {
            snprintf(err, err_cap, "failed to read fixture header");
            free(zero_seq);
            free(cur_seq);
            return 0;
        }
        if (i == 0 && !st->prefix_worker_live && first_layer == 0) {
            fprintf(stderr, "prefill cycle=%u hybrid owns blk0 seq_len=%u\n", i, seq_len);
            fflush(stderr);
            if (!child_prefill_prefix_bin(hy, token_ids, seq_len, hidden, zero_seq, err, err_cap) ||
                !child_dump_hidden(hy, &next_seq, &next_seq_len, hidden, err, err_cap)) {
                free(zero_seq);
                free(cur_seq);
                return 0;
            }
        } else {
            if (!cur_seq) {
                snprintf(err, err_cap, "missing cycle input seq");
                free(zero_seq);
                return 0;
            }
            fprintf(stderr, "prefill cycle=%u hybrid seq_len=%u\n", i, cur_seq_len);
            fflush(stderr);
            if (!child_prefill_seq_bin(hy, cur_seq, cur_seq_len, hidden, err, err_cap) ||
                !child_dump_hidden(hy, &next_seq, &next_seq_len, hidden, err, err_cap)) {
                free(zero_seq);
                free(cur_seq);
                return 0;
            }
        }
        free(cur_seq);
        cur_seq = next_seq;
        cur_seq_len = next_seq_len;
        fprintf(stderr, "prefill cycle=%u full layer=%u seq_len=%u\n", i, cy->full_layer, cur_seq_len);
        fflush(stderr);
        if (!child_prefill_seq_bin(fw, cur_seq, cur_seq_len, hidden, err, err_cap) ||
            !child_dump_hidden(fw, &next_seq, &next_seq_len, hidden, err, err_cap)) {
            free(zero_seq);
            free(cur_seq);
            return 0;
        }
        free(cur_seq);
        cur_seq = next_seq;
        cur_seq_len = next_seq_len;
    }
    free(zero_seq);
    st->owned_seq = cur_seq;
    return 1;
}

static int session_step(session_state *st, uint32_t token_id, char *err, size_t err_cap) {
    uint32_t i;
    float *new_owned = NULL;
    float *row = NULL;
    float *tmp_row = NULL;
    uint32_t first_layer = 0;
    if (st->seq_len == 0 || st->hidden == 0) {
        snprintf(err, err_cap, "not prefilled");
        return 0;
    }
    row = (float *)malloc((size_t)st->hidden * sizeof(float));
    tmp_row = (float *)malloc((size_t)st->hidden * sizeof(float));
    new_owned = (float *)realloc(st->owned_seq, (size_t)(st->seq_len + 1) * st->hidden * sizeof(float));
    st->token_ids = (uint32_t *)realloc(st->token_ids, (size_t)(st->seq_len + 1) * sizeof(uint32_t));
    if (!row || !tmp_row || !new_owned || !st->token_ids) {
        snprintf(err, err_cap, "oom step buffers");
        free(row);
        free(tmp_row);
        return 0;
    }
    st->owned_seq = new_owned;
    st->token_ids[st->seq_len] = token_id;

    if (st->prefix_worker_live) {
        fprintf(stderr, "step prefix next_token=%u seq_len=%u\n", token_id, st->seq_len + 1u);
        fflush(stderr);
        if (!child_step_token(&st->prefix_worker, token_id, err, err_cap) ||
            !child_dump_last(&st->prefix_worker, row, st->hidden, err, err_cap)) {
            free(row);
            free(tmp_row);
            return 0;
        }
    }
    for (i = 0; i < st->cfg.n_cycles; ++i) {
        cycle_config *cy = &st->cfg.cycles[i];
        child_proc *hy = &st->hybrid_workers[i];
        child_proc *fw = &st->full_workers[i];
        if (!read_fixture_layer(cy->fixtures[0], &first_layer)) {
            snprintf(err, err_cap, "failed to read fixture header");
            free(row);
            free(tmp_row);
            return 0;
        }
        if (i == 0 && !st->prefix_worker_live && first_layer == 0) {
            fprintf(stderr, "step cycle=%u hybrid owns blk0 token=%u\n", i, token_id);
            fflush(stderr);
            if (!child_step_token(hy, token_id, err, err_cap) ||
                !child_dump_last(hy, row, st->hidden, err, err_cap)) {
                free(row);
                free(tmp_row);
                return 0;
            }
        } else {
            fprintf(stderr, "step cycle=%u hybrid row\n", i);
            fflush(stderr);
            if (!child_step_row_bin(hy, row, st->hidden, err, err_cap) ||
                !child_dump_last(hy, tmp_row, st->hidden, err, err_cap)) {
                free(row);
                free(tmp_row);
                return 0;
            }
            memcpy(row, tmp_row, (size_t)st->hidden * sizeof(float));
        }
        fprintf(stderr, "step cycle=%u full layer=%u\n", i, cy->full_layer);
        fflush(stderr);
        if (!child_step_row_bin(fw, row, st->hidden, err, err_cap) ||
            !child_dump_last(fw, tmp_row, st->hidden, err, err_cap)) {
            free(row);
            free(tmp_row);
            return 0;
        }
        memcpy(row, tmp_row, (size_t)st->hidden * sizeof(float));
    }
    memcpy(st->owned_seq + (size_t)st->seq_len * st->hidden, row, (size_t)st->hidden * sizeof(float));
    st->seq_len += 1;
    free(row);
    free(tmp_row);
    return 1;
}

static void handle_prefill_prefix_bin(session_state *st, char *rest) {
    uint32_t seq_len = 0, hidden = 0;
    uint32_t *token_ids = NULL;
    float *ignored_seq = NULL;
    char err[512];
    if (sscanf(rest ? rest : "", "%u %u", &seq_len, &hidden) != 2) {
        printf("ERROR bad PREFILL_PREFIX_BIN args\n");
        fflush(stdout);
        return;
    }
    fprintf(stderr, "handle prefill: alloc seq_len=%u hidden=%u\n", seq_len, hidden);
    fflush(stderr);
    token_ids = (uint32_t *)malloc((size_t)seq_len * sizeof(uint32_t));
    ignored_seq = (float *)malloc((size_t)seq_len * hidden * sizeof(float));
    if (!token_ids || !ignored_seq) {
        printf("ERROR oom prefill buffers\n");
        fflush(stdout);
        free(token_ids);
        free(ignored_seq);
        return;
    }
    fprintf(stderr, "handle prefill: reading token ids\n");
    fflush(stderr);
    if (!read_all(STDIN_FILENO, token_ids, (size_t)seq_len * sizeof(uint32_t))) {
        printf("ERROR short prefill token payload\n");
        fflush(stdout);
        free(token_ids);
        free(ignored_seq);
        return;
    }
    fprintf(stderr, "handle prefill: reading input seq bytes=%zu\n", (size_t)seq_len * hidden * sizeof(float));
    fflush(stderr);
    if (!read_all(STDIN_FILENO, ignored_seq, (size_t)seq_len * hidden * sizeof(float))) {
        printf("ERROR short prefill seq payload\n");
        fflush(stdout);
        free(token_ids);
        free(ignored_seq);
        return;
    }
    fprintf(stderr, "handle prefill: payload read complete\n");
    fflush(stderr);
    free(ignored_seq);
    if (!session_prefill(st, token_ids, seq_len, hidden, err, sizeof(err))) {
        printf("ERROR %s\n", err);
        fflush(stdout);
        free(token_ids);
        return;
    }
    free(token_ids);
    printf("PREFILL_OK %u %u\n", st->seq_len, st->hidden);
    fflush(stdout);
}

static void handle_step(session_state *st, char *rest) {
    uint32_t token_id = 0;
    char err[512];
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

static void handle_dump_hidden(session_state *st) {
    size_t n = (size_t)st->seq_len * st->hidden;
    size_t bytes = n * sizeof(float);
    if (!st->owned_seq) {
        printf("ERROR not prefilled\n");
        fflush(stdout);
        return;
    }
    printf("HIDDEN %zu %zu\n", n, bytes);
    fflush(stdout);
    fwrite(st->owned_seq, sizeof(float), n, stdout);
    fflush(stdout);
}

static void handle_dump_last(session_state *st) {
    if (!st->owned_seq || st->seq_len == 0) {
        printf("ERROR not prefilled\n");
        fflush(stdout);
        return;
    }
    printf("LAST %u %zu\n", st->hidden, (size_t)st->hidden * sizeof(float));
    fflush(stdout);
    fwrite(st->owned_seq + (size_t)(st->seq_len - 1) * st->hidden, sizeof(float), st->hidden, stdout);
    fflush(stdout);
}

int main(int argc, char **argv) {
    session_state st;
    char err[512];
    char line[4096];
    memset(&st, 0, sizeof(st));
    setvbuf(stdin, NULL, _IONBF, 0);
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL.gguf CONFIG.txt\n", argv[0]);
        return 1;
    }
    if (!parse_config(argv[2], &st.cfg, err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        free_worker_config(&st.cfg);
        return 1;
    }
    if (!session_spawn_workers(&st, argv[1], err, sizeof(err))) {
        fprintf(stderr, "%s\n", err);
        session_close_workers(&st);
        free_worker_config(&st.cfg);
        return 1;
    }
    printf("READY\n");
    fflush(stdout);
    while (fgets(line, sizeof(line), stdin)) {
        char *cmd = NULL;
        char *rest = NULL;
        size_t len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
        if (len == 0) continue;
        cmd = strtok(line, " ");
        if (!cmd) continue;
        rest = cmd + strlen(cmd) + 1;
        if (rest >= line + len) rest = line + len;
        if (strcmp(cmd, "PREFILL_PREFIX_BIN") == 0) {
            handle_prefill_prefix_bin(&st, rest);
        } else if (strcmp(cmd, "STEP") == 0) {
            handle_step(&st, rest);
        } else if (strcmp(cmd, "DUMP_HIDDEN") == 0) {
            handle_dump_hidden(&st);
        } else if (strcmp(cmd, "DUMP_LAST") == 0) {
            handle_dump_last(&st);
        } else if (strcmp(cmd, "RESET") == 0) {
            session_reset_state(&st);
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
    session_reset_state(&st);
    session_close_workers(&st);
    free_worker_config(&st.cfg);
    return 0;
}
