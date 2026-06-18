#include "qwen36_gguf.h"

#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#define QWEN36_GGUF_MAGIC 0x46554747u

typedef struct qwen36_reader {
    FILE *fp;
} qwen36_reader;

static char *dup_cstr(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (!p) return NULL;
    memcpy(p, s, n);
    return p;
}

static void set_err(char *err, size_t err_cap, const char *fmt, ...) {
    va_list ap;
    if (!err || err_cap == 0) return;
    va_start(ap, fmt);
    vsnprintf(err, err_cap, fmt, ap);
    va_end(ap);
}

static bool read_exact(qwen36_reader *r, void *buf, size_t n, char *err, size_t err_cap) {
    if (fread(buf, 1, n, r->fp) != n) {
        set_err(err, err_cap, "unexpected EOF");
        return false;
    }
    return true;
}

static bool read_u32(qwen36_reader *r, uint32_t *out, char *err, size_t err_cap) {
    return read_exact(r, out, sizeof(*out), err, err_cap);
}

static bool read_u64(qwen36_reader *r, uint64_t *out, char *err, size_t err_cap) {
    return read_exact(r, out, sizeof(*out), err, err_cap);
}

static bool read_string(qwen36_reader *r, char **out, char *err, size_t err_cap) {
    uint64_t len = 0;
    char *s = NULL;
    if (!read_u64(r, &len, err, err_cap)) return false;
    if (len > SIZE_MAX - 1) {
        set_err(err, err_cap, "string too large");
        return false;
    }
    s = (char *)malloc((size_t)len + 1);
    if (!s) {
        set_err(err, err_cap, "out of memory");
        return false;
    }
    if (!read_exact(r, s, (size_t)len, err, err_cap)) {
        free(s);
        return false;
    }
    s[len] = '\0';
    *out = s;
    return true;
}

static size_t scalar_size(uint32_t type) {
    switch (type) {
    case QWEN36_GGUF_VALUE_UINT8: return sizeof(uint8_t);
    case QWEN36_GGUF_VALUE_INT8: return sizeof(int8_t);
    case QWEN36_GGUF_VALUE_UINT16: return sizeof(uint16_t);
    case QWEN36_GGUF_VALUE_INT16: return sizeof(int16_t);
    case QWEN36_GGUF_VALUE_UINT32: return sizeof(uint32_t);
    case QWEN36_GGUF_VALUE_INT32: return sizeof(int32_t);
    case QWEN36_GGUF_VALUE_FLOAT32: return sizeof(float);
    case QWEN36_GGUF_VALUE_BOOL: return sizeof(uint8_t);
    case QWEN36_GGUF_VALUE_UINT64: return sizeof(uint64_t);
    case QWEN36_GGUF_VALUE_INT64: return sizeof(int64_t);
    case QWEN36_GGUF_VALUE_FLOAT64: return sizeof(double);
    default: return 0;
    }
}

static bool read_array(qwen36_reader *r, qwen36_gguf_array *arr, char *err, size_t err_cap) {
    uint64_t i;
    memset(arr, 0, sizeof(*arr));
    if (!read_u32(r, &arr->item_type, err, err_cap)) return false;
    if (!read_u64(r, &arr->len, err, err_cap)) return false;
    if (arr->len > SIZE_MAX) {
        set_err(err, err_cap, "array too large");
        return false;
    }
    switch (arr->item_type) {
    case QWEN36_GGUF_VALUE_INT32:
        arr->data.i32 = (int32_t *)calloc((size_t)arr->len, sizeof(int32_t));
        if (!arr->data.i32) {
            set_err(err, err_cap, "out of memory");
            return false;
        }
        return read_exact(r, arr->data.i32, (size_t)arr->len * sizeof(int32_t), err, err_cap);
    case QWEN36_GGUF_VALUE_STRING:
        arr->data.str = (char **)calloc((size_t)arr->len, sizeof(char *));
        if (!arr->data.str) {
            set_err(err, err_cap, "out of memory");
            return false;
        }
        for (i = 0; i < arr->len; i++) {
            if (!read_string(r, &arr->data.str[i], err, err_cap)) return false;
        }
        return true;
    default:
        {
            size_t sz = scalar_size(arr->item_type);
            if (sz == 0) {
                set_err(err, err_cap, "unsupported array type %u", arr->item_type);
                return false;
            }
            arr->data.raw = calloc((size_t)arr->len, sz);
            if (!arr->data.raw) {
                set_err(err, err_cap, "out of memory");
                return false;
            }
            return read_exact(r, arr->data.raw, (size_t)arr->len * sz, err, err_cap);
        }
    }
}

static bool read_value(qwen36_reader *r, qwen36_gguf_kv *kv, char *err, size_t err_cap) {
    switch (kv->type) {
    case QWEN36_GGUF_VALUE_UINT8: return read_exact(r, &kv->v.u8, sizeof(kv->v.u8), err, err_cap);
    case QWEN36_GGUF_VALUE_INT8: return read_exact(r, &kv->v.i8, sizeof(kv->v.i8), err, err_cap);
    case QWEN36_GGUF_VALUE_UINT16: return read_exact(r, &kv->v.u16, sizeof(kv->v.u16), err, err_cap);
    case QWEN36_GGUF_VALUE_INT16: return read_exact(r, &kv->v.i16, sizeof(kv->v.i16), err, err_cap);
    case QWEN36_GGUF_VALUE_UINT32: return read_exact(r, &kv->v.u32, sizeof(kv->v.u32), err, err_cap);
    case QWEN36_GGUF_VALUE_INT32: return read_exact(r, &kv->v.i32, sizeof(kv->v.i32), err, err_cap);
    case QWEN36_GGUF_VALUE_FLOAT32: return read_exact(r, &kv->v.f32, sizeof(kv->v.f32), err, err_cap);
    case QWEN36_GGUF_VALUE_BOOL: return read_exact(r, &kv->v.b, sizeof(uint8_t), err, err_cap);
    case QWEN36_GGUF_VALUE_UINT64: return read_exact(r, &kv->v.u64, sizeof(kv->v.u64), err, err_cap);
    case QWEN36_GGUF_VALUE_INT64: return read_exact(r, &kv->v.i64, sizeof(kv->v.i64), err, err_cap);
    case QWEN36_GGUF_VALUE_FLOAT64: return read_exact(r, &kv->v.f64, sizeof(kv->v.f64), err, err_cap);
    case QWEN36_GGUF_VALUE_STRING: return read_string(r, &kv->v.str, err, err_cap);
    case QWEN36_GGUF_VALUE_ARRAY: return read_array(r, &kv->v.arr, err, err_cap);
    default:
        set_err(err, err_cap, "unsupported metadata type %u", kv->type);
        return false;
    }
}

bool qwen36_gguf_open(qwen36_gguf_file *out, const char *path, char *err, size_t err_cap) {
    qwen36_reader r;
    uint32_t magic = 0;
    uint64_t i = 0;
    long end = 0;

    memset(out, 0, sizeof(*out));
    r.fp = fopen(path, "rb");
    if (!r.fp) {
        set_err(err, err_cap, "open %s: %s", path, strerror(errno));
        return false;
    }
    if (fseek(r.fp, 0, SEEK_END) != 0 || (end = ftell(r.fp)) < 0 || fseek(r.fp, 0, SEEK_SET) != 0) {
        set_err(err, err_cap, "failed to stat file");
        fclose(r.fp);
        return false;
    }
    out->file_size = (uint64_t)end;
    out->path = dup_cstr(path);
    if (!out->path) {
        set_err(err, err_cap, "out of memory");
        fclose(r.fp);
        return false;
    }
    if (!read_u32(&r, &magic, err, err_cap) || magic != QWEN36_GGUF_MAGIC) {
        if (magic != QWEN36_GGUF_MAGIC) set_err(err, err_cap, "not a GGUF file");
        qwen36_gguf_close(out);
        fclose(r.fp);
        return false;
    }
    if (!read_u32(&r, &out->version, err, err_cap) || out->version != 3) {
        if (out->version != 3) set_err(err, err_cap, "unsupported GGUF version %u", out->version);
        qwen36_gguf_close(out);
        fclose(r.fp);
        return false;
    }
    if (!read_u64(&r, &out->tensor_count, err, err_cap) ||
        !read_u64(&r, &out->kv_count, err, err_cap)) {
        qwen36_gguf_close(out);
        fclose(r.fp);
        return false;
    }
    if (out->kv_count > SIZE_MAX / sizeof(*out->kvs) ||
        out->tensor_count > SIZE_MAX / sizeof(*out->tensors)) {
        set_err(err, err_cap, "GGUF table too large");
        qwen36_gguf_close(out);
        fclose(r.fp);
        return false;
    }
    out->kvs = (qwen36_gguf_kv *)calloc((size_t)out->kv_count, sizeof(*out->kvs));
    out->tensors = (qwen36_gguf_tensor *)calloc((size_t)out->tensor_count, sizeof(*out->tensors));
    if ((!out->kvs && out->kv_count != 0) || (!out->tensors && out->tensor_count != 0)) {
        set_err(err, err_cap, "out of memory");
        qwen36_gguf_close(out);
        fclose(r.fp);
        return false;
    }
    for (i = 0; i < out->kv_count; i++) {
        qwen36_gguf_kv *kv = &out->kvs[i];
        if (!read_string(&r, &kv->key, err, err_cap) ||
            !read_u32(&r, &kv->type, err, err_cap) ||
            !read_value(&r, kv, err, err_cap)) {
            qwen36_gguf_close(out);
            fclose(r.fp);
            return false;
        }
    }
    for (i = 0; i < out->tensor_count; i++) {
        qwen36_gguf_tensor *t = &out->tensors[i];
        uint32_t d = 0;
        if (!read_string(&r, &t->name, err, err_cap) ||
            !read_u32(&r, &t->ndim, err, err_cap)) {
            qwen36_gguf_close(out);
            fclose(r.fp);
            return false;
        }
        if (t->ndim > QWEN36_GGUF_MAX_DIMS) {
            set_err(err, err_cap, "tensor %s ndim %u exceeds max %u", t->name, t->ndim, QWEN36_GGUF_MAX_DIMS);
            qwen36_gguf_close(out);
            fclose(r.fp);
            return false;
        }
        for (d = 0; d < t->ndim; d++) {
            if (!read_u64(&r, &t->dims[d], err, err_cap)) {
                qwen36_gguf_close(out);
                fclose(r.fp);
                return false;
            }
        }
        if (!read_u32(&r, &t->type, err, err_cap) ||
            !read_u64(&r, &t->rel_offset, err, err_cap)) {
            qwen36_gguf_close(out);
            fclose(r.fp);
            return false;
        }
    }
    fclose(r.fp);
    return true;
}

void qwen36_gguf_close(qwen36_gguf_file *gf) {
    uint64_t i = 0;
    if (!gf) return;
    free(gf->path);
    for (i = 0; i < gf->kv_count; i++) {
        qwen36_gguf_kv *kv = &gf->kvs[i];
        free(kv->key);
        if (kv->type == QWEN36_GGUF_VALUE_STRING) free(kv->v.str);
        if (kv->type == QWEN36_GGUF_VALUE_ARRAY) {
            uint64_t j;
            if (kv->v.arr.item_type == QWEN36_GGUF_VALUE_STRING) {
                for (j = 0; j < kv->v.arr.len; j++) free(kv->v.arr.data.str[j]);
            }
            free(kv->v.arr.data.raw);
        }
    }
    for (i = 0; i < gf->tensor_count; i++) free(gf->tensors[i].name);
    free(gf->kvs);
    free(gf->tensors);
    memset(gf, 0, sizeof(*gf));
}

const qwen36_gguf_kv *qwen36_gguf_find_kv(const qwen36_gguf_file *gf, const char *key) {
    uint64_t i;
    for (i = 0; i < gf->kv_count; i++) {
        if (strcmp(gf->kvs[i].key, key) == 0) return &gf->kvs[i];
    }
    return NULL;
}

const qwen36_gguf_tensor *qwen36_gguf_find_tensor(const qwen36_gguf_file *gf, const char *name) {
    uint64_t i;
    for (i = 0; i < gf->tensor_count; i++) {
        if (strcmp(gf->tensors[i].name, name) == 0) return &gf->tensors[i];
    }
    return NULL;
}

const char *qwen36_gguf_type_name(uint32_t type) {
    switch (type) {
    case QWEN36_GGUF_TYPE_F32: return "f32";
    case QWEN36_GGUF_TYPE_F16: return "f16";
    case QWEN36_GGUF_TYPE_Q8_0: return "q8_0";
    case 10: return "q2_k";
    case 12: return "q4_k";
    case 13: return "q5_k";
    case 14: return "q6_k";
    case 16: return "iq2_xxs";
    default: return "unknown";
    }
}

void qwen36_gguf_print_tensor(FILE *fp, const qwen36_gguf_tensor *t) {
    uint32_t i;
    fprintf(fp, "%s type=%s dims=[", t->name, qwen36_gguf_type_name(t->type));
    for (i = 0; i < t->ndim; i++) {
        if (i != 0) fputs(", ", fp);
        fprintf(fp, "%llu", (unsigned long long)t->dims[i]);
    }
    fputs("]", fp);
}
