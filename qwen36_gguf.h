#ifndef QWEN36_GGUF_H
#define QWEN36_GGUF_H

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define QWEN36_GGUF_MAX_DIMS 4

enum {
    QWEN36_GGUF_TYPE_F32  = 0,
    QWEN36_GGUF_TYPE_F16  = 1,
    QWEN36_GGUF_TYPE_Q8_0 = 8,
};

enum {
    QWEN36_GGUF_VALUE_UINT8   = 0,
    QWEN36_GGUF_VALUE_INT8    = 1,
    QWEN36_GGUF_VALUE_UINT16  = 2,
    QWEN36_GGUF_VALUE_INT16   = 3,
    QWEN36_GGUF_VALUE_UINT32  = 4,
    QWEN36_GGUF_VALUE_INT32   = 5,
    QWEN36_GGUF_VALUE_FLOAT32 = 6,
    QWEN36_GGUF_VALUE_BOOL    = 7,
    QWEN36_GGUF_VALUE_STRING  = 8,
    QWEN36_GGUF_VALUE_ARRAY   = 9,
    QWEN36_GGUF_VALUE_UINT64  = 10,
    QWEN36_GGUF_VALUE_INT64   = 11,
    QWEN36_GGUF_VALUE_FLOAT64 = 12,
};

typedef struct qwen36_gguf_array {
    uint32_t item_type;
    uint64_t len;
    union {
        int32_t  *i32;
        uint32_t *u32;
        uint64_t *u64;
        float    *f32;
        char    **str;
        void     *raw;
    } data;
} qwen36_gguf_array;

typedef struct qwen36_gguf_kv {
    char    *key;
    uint32_t type;
    union {
        bool                b;
        uint8_t             u8;
        int8_t              i8;
        uint16_t            u16;
        int16_t             i16;
        uint32_t            u32;
        int32_t             i32;
        uint64_t            u64;
        int64_t             i64;
        float               f32;
        double              f64;
        char               *str;
        qwen36_gguf_array   arr;
    } v;
} qwen36_gguf_kv;

typedef struct qwen36_gguf_tensor {
    char    *name;
    uint32_t ndim;
    uint64_t dims[QWEN36_GGUF_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
} qwen36_gguf_tensor;

typedef struct qwen36_gguf_file {
    char               *path;
    uint32_t            version;
    uint64_t            file_size;
    uint64_t            tensor_count;
    uint64_t            kv_count;
    qwen36_gguf_kv     *kvs;
    qwen36_gguf_tensor *tensors;
} qwen36_gguf_file;

bool qwen36_gguf_open(qwen36_gguf_file *out, const char *path, char *err, size_t err_cap);
void qwen36_gguf_close(qwen36_gguf_file *gf);

const qwen36_gguf_kv *qwen36_gguf_find_kv(const qwen36_gguf_file *gf, const char *key);
const qwen36_gguf_tensor *qwen36_gguf_find_tensor(const qwen36_gguf_file *gf, const char *name);
const char *qwen36_gguf_type_name(uint32_t type);
void qwen36_gguf_print_tensor(FILE *fp, const qwen36_gguf_tensor *t);

#endif
