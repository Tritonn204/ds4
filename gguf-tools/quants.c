/*
 * DS4 quantization facade.
 *
 * These are the small GGUF quantization pieces needed by our DeepSeek V4
 * Flash recipes: float conversion, q8_0, q2_K, q4_K, q5_K, q6_K, iq2_xxs,
 * and iq2_s.  The code is
 * local C and deliberately narrow; other GGUF type IDs are named for metadata
 * compatibility, but cannot be emitted by this tool.
 *
 * The quantized block layouts and search procedures are derived from the
 * MIT-licensed GGML/llama.cpp quantizers.  Keep changes conservative: byte
 * layout compatibility is more important here than generality.
 */

#include "quants.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static pthread_mutex_t ds4q_init_mutex = PTHREAD_MUTEX_INITIALIZER;

typedef struct {
    const char *name;
    int64_t block_size;
    size_t type_size;
    bool can_quantize;
    bool requires_imatrix;
} ds4q_traits;

#define QK_K 256
#define K_SCALE_SIZE 12
#define DS4Q_GROUP_MAX_EPS 1e-15f
#define DS4Q_GROUP_MAX_EPS_IQ2_S 1e-8f
#define DS4Q_MIN(a, b) ((a) < (b) ? (a) : (b))
#define DS4Q_MAX(a, b) ((a) > (b) ? (a) : (b))

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[K_SCALE_SIZE];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
} ds4q_block_q5_K;

typedef struct {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t  scales[QK_K / 16];
    uint16_t d;
} ds4q_block_q6_K;

typedef struct {
    uint16_t d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t scales[QK_K / 32];
} ds4q_block_iq2_s;

#define DS4Q_IQ3S_N_SCALE (QK_K / 64)
typedef struct {
    uint16_t d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t signs[QK_K / 8];
    uint8_t scales[DS4Q_IQ3S_N_SCALE];
} ds4q_block_iq3_s;

static const ds4q_traits ds4q_type_traits[DS4Q_TYPE_COUNT] = {
    [DS4Q_TYPE_F32]     = { "f32",      1,   4, false, false },
    [DS4Q_TYPE_F16]     = { "f16",      1,   2, false, false },
    [DS4Q_TYPE_Q4_0]    = { "q4_0",    32,  18, false, false },
    [DS4Q_TYPE_Q4_1]    = { "q4_1",    32,  20, false, false },
    [DS4Q_TYPE_Q5_0]    = { "q5_0",    32,  22, false, false },
    [DS4Q_TYPE_Q5_1]    = { "q5_1",    32,  24, false, false },
    [DS4Q_TYPE_Q8_0]    = { "q8_0",    32,  34, true,  false },
    [DS4Q_TYPE_Q8_1]    = { "q8_1",    32,  36, false, false },
    [DS4Q_TYPE_Q2_K]    = { "q2_K",  QK_K,  84, true,  false },
    [DS4Q_TYPE_Q3_K]    = { "q3_K",  QK_K, 110, false, false },
    [DS4Q_TYPE_Q4_K]    = { "q4_K",  QK_K, 144, true,  false },
    [DS4Q_TYPE_Q5_K]    = { "q5_K",  QK_K, 176, true,  false },
    [DS4Q_TYPE_Q6_K]    = { "q6_K",  QK_K, 210, true,  false },
    [DS4Q_TYPE_Q8_K]    = { "q8_K",  QK_K, 292, false, false },
    [DS4Q_TYPE_IQ2_XXS] = { "iq2_xxs", QK_K,  66, true,  true  },
    [DS4Q_TYPE_IQ2_XS]  = { "iq2_xs",  QK_K,  74, false, true  },
    [DS4Q_TYPE_IQ3_XXS] = { "iq3_xxs", QK_K,  98, false, false },
    [DS4Q_TYPE_IQ1_S]   = { "iq1_s",   QK_K,  50, false, true  },
    [DS4Q_TYPE_IQ4_NL]  = { "iq4_nl",     32,  18, false, false },
    [DS4Q_TYPE_IQ3_S]   = { "iq3_s",   QK_K, 110, true,  false },
    [DS4Q_TYPE_IQ2_S]   = { "iq2_s",   QK_K,  82, true,  false },
    [DS4Q_TYPE_IQ4_XS]  = { "iq4_xs",  QK_K, 136, false, false },
    [DS4Q_TYPE_I8]      = { "i8",          1,   1, false, false },
    [DS4Q_TYPE_I16]     = { "i16",         1,   2, false, false },
    [DS4Q_TYPE_I32]     = { "i32",         1,   4, false, false },
    [DS4Q_TYPE_I64]     = { "i64",         1,   8, false, false },
    [DS4Q_TYPE_F64]     = { "f64",         1,   8, false, false },
    [DS4Q_TYPE_IQ1_M]   = { "iq1_m",   QK_K,  56, false, false },
    [DS4Q_TYPE_BF16]    = { "bf16",        1,   2, false, false },
    [DS4Q_TYPE_TQ1_0]   = { "tq1_0",   QK_K,  54, false, false },
    [DS4Q_TYPE_TQ2_0]   = { "tq2_0",   QK_K,  66, false, false },
    [DS4Q_TYPE_MXFP4]   = { "mxfp4",      32,  17, false, false },
    [DS4Q_TYPE_NVFP4]   = { "nvfp4",      64,  36, false, false },
    [DS4Q_TYPE_Q1_0]    = { "q1_0",      128,  18, false, false },
};

static float ds4q_f32_from_bits(uint32_t bits) {
    union {
        uint32_t u;
        float f;
    } v = { .u = bits };
    return v.f;
}

static uint32_t ds4q_f32_to_bits(float f) {
    union {
        float f;
        uint32_t u;
    } v = { .f = f };
    return v.u;
}

static uint16_t ds4q_f32_to_f16(float f) {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = ds4q_f32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) bias = UINT32_C(0x71000000);

    base = ds4q_f32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t out = ds4q_f32_to_bits(base);
    const uint32_t exp_bits = (out >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = out & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (uint16_t)((sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
}

static int ds4q_nearest_int(float fval) {
    if (!isfinite(fval)) return 0;
    if (fabsf(fval) > 4194303.f) return (int)lrintf(fval);
    float val = fval + 12582912.f;
    int i;
    memcpy(&i, &val, sizeof(i));
    return (i & 0x007fffff) - 0x00400000;
}

static float ds4q_make_qkx2_quants(int n, int nmax, const float *x, const float *weights,
                                   uint8_t *L, float *the_min, uint8_t *Laux,
                                   float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max - min < DS4Q_GROUP_MAX_EPS) {
        memset(L, 0, (size_t)n);
        *the_min = -min;
        return 0.0f;
    }
    float iscale = nmax / (max - min);
    float scale = 1 / iscale;
    float best_error = 0;
    for (int i = 0; i < n; i++) {
        int l = ds4q_nearest_int(iscale * (x[i] - min));
        L[i] = DS4Q_MAX(0, DS4Q_MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        best_error += weights[i] * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; is++) {
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; i++) {
            int l = ds4q_nearest_int(iscale * (x[i] - min));
            l = DS4Q_MAX(0, DS4Q_MIN(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; i++) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                cur_error += weights[i] * diff;
            }
            if (cur_error < best_error) {
                memcpy(L, Laux, (size_t)n);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

static float ds4q_make_qkx3_quants(int n, int nmax, const float *x, const float *weights,
                                   uint8_t *L, float *the_min, uint8_t *Laux,
                                   float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights ? weights[0] : x[0] * x[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights ? weights[i] : x[i] * x[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max - min < DS4Q_GROUP_MAX_EPS) {
        memset(L, 0, (size_t)n);
        *the_min = -min;
        return 0.0f;
    }
    float iscale = nmax / (max - min);
    float scale = 1 / iscale;
    float best_mad = 0;
    for (int i = 0; i < n; i++) {
        int l = ds4q_nearest_int(iscale * (x[i] - min));
        L[i] = DS4Q_MAX(0, DS4Q_MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights ? weights[i] : x[i] * x[i];
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; is++) {
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; i++) {
            int l = ds4q_nearest_int(iscale * (x[i] - min));
            l = DS4Q_MAX(0, DS4Q_MIN(nmax, l));
            Laux[i] = l;
            float w = weights ? weights[i] : x[i] * x[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0;
            for (int i = 0; i < n; i++) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights ? weights[i] : x[i] * x[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                memcpy(L, Laux, (size_t)n);
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

static float ds4q_make_qp_quants(int n, int nmax, const float *x, uint8_t *L, const float *quant_weights) {
    float max = 0;
    for (int i = 0; i < n; i++) max = DS4Q_MAX(max, x[i]);
    if (max < DS4Q_GROUP_MAX_EPS) {
        memset(L, 0, (size_t)n);
        return 0.0f;
    }
    float iscale = nmax / max;
    for (int i = 0; i < n; i++) L[i] = ds4q_nearest_int(iscale * x[i]);
    float scale = 1 / iscale;
    float best_mse = 0;
    for (int i = 0; i < n; i++) {
        float diff = x[i] - scale * L[i];
        best_mse += quant_weights[i] * diff * diff;
    }
    for (int is = -4; is <= 4; is++) {
        if (is == 0) continue;
        float iscale_is = (0.1f * is + nmax) / max;
        float scale_is = 1 / iscale_is;
        float mse = 0;
        for (int i = 0; i < n; i++) {
            int l = ds4q_nearest_int(iscale_is * x[i]);
            l = DS4Q_MIN(nmax, l);
            float diff = x[i] - scale_is * l;
            mse += quant_weights[i] * diff * diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    float sumlx = 0, suml2 = 0;
    for (int i = 0; i < n; i++) {
        int l = ds4q_nearest_int(iscale * x[i]);
        l = DS4Q_MIN(nmax, l);
        L[i] = l;
        float w = quant_weights[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    for (int itry = 0; itry < 5; itry++) {
        int n_changed = 0;
        for (int i = 0; i < n; i++) {
            float w = quant_weights[i];
            float slx = sumlx - w * x[i] * L[i];
            float sl2 = suml2 - w * L[i] * L[i];
            if (slx > 0 && sl2 > 0) {
                int new_l = ds4q_nearest_int(x[i] * sl2 / slx);
                new_l = DS4Q_MIN(nmax, new_l);
                if (new_l != L[i]) {
                    slx += w * x[i] * new_l;
                    sl2 += w * new_l * new_l;
                    if (slx * slx * suml2 > sumlx * sumlx * sl2) {
                        L[i] = new_l;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed++;
                    }
                }
            }
        }
        if (!n_changed) break;
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

static float ds4q_make_qx_quants(int n, int nmax, const float *x, int8_t *L, int rmse_type,
                                 const float *qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    if (amax < DS4Q_GROUP_MAX_EPS) {
        memset(L, 0, (size_t)n);
        return 0.0f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = ds4q_nearest_int(iscale * x[i]);
            L[i] = nmax + DS4Q_MAX(-nmax, DS4Q_MIN(nmax - 1, l));
        }
        return 1 / iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = ds4q_nearest_int(iscale * x[i]);
        l = DS4Q_MAX(-nmax, DS4Q_MIN(nmax - 1, l));
        L[i] = l + nmax;
        float w = qw ? qw[i]
                     : rmse_type == 1 ? x[i] * x[i]
                     : rmse_type == 2 ? 1.0f
                     : rmse_type == 3 ? fabsf(x[i])
                                      : sqrtf(fabsf(x[i]));
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    float scale = suml2 ? sumlx / suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f * (scale + 1 / iscale) : 1 / iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) continue;
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = 0;
        suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = ds4q_nearest_int(iscale * x[i]);
            l = DS4Q_MAX(-nmax, DS4Q_MIN(nmax - 1, l));
            float w = qw ? qw[i]
                         : rmse_type == 1 ? x[i] * x[i]
                         : rmse_type == 2 ? 1.0f
                         : rmse_type == 3 ? fabsf(x[i])
                                          : sqrtf(fabsf(x[i]));
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }
        if (suml2 > 0 && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; ++i) {
                int l = ds4q_nearest_int(iscale * x[i]);
                L[i] = nmax + DS4Q_MAX(-nmax, DS4Q_MIN(nmax - 1, l));
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

static void ds4q_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static void ds4q_write_q5_k_block_ref(const float *x, uint8_t *dst) {
    ds4q_block_q5_K *b = (ds4q_block_q5_K *)dst;
    uint8_t L[QK_K];
    float mins[QK_K / 32];
    float scales[QK_K / 32];
    float weights[32];
    uint8_t Laux[32];

    memset(b, 0, sizeof(*b));
    float max_scale = 0;
    float max_min = 0;
    for (int j = 0; j < QK_K / 32; ++j) {
        float sum_x2 = 0;
        for (int l = 0; l < 32; ++l) sum_x2 += x[32 * j + l] * x[32 * j + l];
        float av_x = sqrtf(sum_x2 / 32);
        for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32 * j + l]);
        scales[j] = ds4q_make_qkx2_quants(32, 31, x + 32 * j, weights, L + 32 * j,
                                          &mins[j], Laux, -0.5f, 0.1f, 15, false);
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min) max_min = mins[j];
    }

    float inv_scale = max_scale > 0 ? 63.0f / max_scale : 0.0f;
    float inv_min = max_min > 0 ? 63.0f / max_min : 0.0f;
    for (int j = 0; j < QK_K / 32; ++j) {
        uint8_t ls = ds4q_nearest_int(inv_scale * scales[j]);
        uint8_t lm = ds4q_nearest_int(inv_min * mins[j]);
        ls = DS4Q_MIN(63, ls);
        lm = DS4Q_MIN(63, lm);
        if (j < 4) {
            b->scales[j] = ls;
            b->scales[j + 4] = lm;
        } else {
            b->scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            b->scales[j - 4] |= ((ls >> 4) << 6);
            b->scales[j - 0] |= ((lm >> 4) << 6);
        }
    }
    b->d = ds4q_f32_to_f16(max_scale / 63.0f);
    b->dmin = ds4q_f32_to_f16(max_min / 63.0f);

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; ++j) {
        ds4q_get_scale_min_k4(j, b->scales, &sc, &m);
        const float d = ds4q_f16_to_f32(b->d) * sc;
        if (!d) continue;
        const float dm = ds4q_f16_to_f32(b->dmin) * m;
        for (int ii = 0; ii < 32; ++ii) {
            int l = ds4q_nearest_int((x[32 * j + ii] + dm) / d);
            l = DS4Q_MAX(0, DS4Q_MIN(31, l));
            L[32 * j + ii] = l;
        }
    }

    memset(b->qh, 0, sizeof(b->qh));
    uint8_t *qh = b->qh;
    uint8_t *ql = b->qs;
    uint8_t m1 = 1, m2 = 2;
    for (int n = 0; n < QK_K; n += 64) {
        for (int j = 0; j < 32; ++j) {
            int l1 = L[n + j];
            if (l1 > 15) {
                l1 -= 16;
                qh[j] |= m1;
            }
            int l2 = L[n + j + 32];
            if (l2 > 15) {
                l2 -= 16;
                qh[j] |= m2;
            }
            ql[j] = (uint8_t)(l1 | (l2 << 4));
        }
        m1 <<= 2;
        m2 <<= 2;
        ql += 32;
    }
}

static void ds4q_write_q5_k_block_weighted(const float *x, uint8_t *dst, const float *quant_weights) {
    ds4q_block_q5_K *b = (ds4q_block_q5_K *)dst;
    uint8_t L[QK_K];
    uint8_t Laux[32];
    uint8_t Ls[QK_K / 32];
    uint8_t Lm[QK_K / 32];
    float mins[QK_K / 32];
    float scales[QK_K / 32];
    float sw[QK_K / 32];
    float weights[32];

    memset(b, 0, sizeof(*b));
    float sum_x2 = 0;
    for (int l = 0; l < QK_K; ++l) sum_x2 += x[l] * x[l];
    float sigma2 = 2 * sum_x2 / QK_K;
    float av_x = sqrtf(sigma2);

    for (int j = 0; j < QK_K / 32; ++j) {
        if (quant_weights) {
            const float *qw = quant_weights + 32 * j;
            for (int l = 0; l < 32; ++l) weights[l] = qw[l] * sqrtf(sigma2 + x[32 * j + l] * x[32 * j + l]);
        } else {
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32 * j + l]);
        }
        float sumw = 0;
        for (int l = 0; l < 32; ++l) sumw += weights[l];
        sw[j] = sumw;
        scales[j] = ds4q_make_qkx3_quants(32, 31, x + 32 * j, weights, L + 32 * j,
                                          &mins[j], Laux, -0.9f, 0.05f, 36, false);
    }

    float d_block = ds4q_make_qp_quants(QK_K / 32, 63, scales, Ls, sw);
    float m_block = ds4q_make_qp_quants(QK_K / 32, 63, mins, Lm, sw);
    for (int j = 0; j < QK_K / 32; ++j) {
        uint8_t ls = Ls[j];
        uint8_t lm = Lm[j];
        ls = DS4Q_MIN(63, ls);
        lm = DS4Q_MIN(63, lm);
        if (j < 4) {
            b->scales[j] = ls;
            b->scales[j + 4] = lm;
        } else {
            b->scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            b->scales[j - 4] |= ((ls >> 4) << 6);
            b->scales[j - 0] |= ((lm >> 4) << 6);
        }
    }
    b->d = ds4q_f32_to_f16(d_block);
    b->dmin = ds4q_f32_to_f16(m_block);

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; ++j) {
        ds4q_get_scale_min_k4(j, b->scales, &sc, &m);
        const float d = ds4q_f16_to_f32(b->d) * sc;
        if (!d) continue;
        const float dm = ds4q_f16_to_f32(b->dmin) * m;
        for (int ii = 0; ii < 32; ++ii) {
            int l = ds4q_nearest_int((x[32 * j + ii] + dm) / d);
            l = DS4Q_MAX(0, DS4Q_MIN(31, l));
            L[32 * j + ii] = l;
        }
    }

    memset(b->qh, 0, sizeof(b->qh));
    uint8_t *qh = b->qh;
    uint8_t *ql = b->qs;
    uint8_t m1 = 1, m2 = 2;
    for (int n = 0; n < QK_K; n += 64) {
        for (int j = 0; j < 32; ++j) {
            int l1 = L[n + j];
            if (l1 > 15) {
                l1 -= 16;
                qh[j] |= m1;
            }
            int l2 = L[n + j + 32];
            if (l2 > 15) {
                l2 -= 16;
                qh[j] |= m2;
            }
            ql[j] = (uint8_t)(l1 | (l2 << 4));
        }
        m1 <<= 2;
        m2 <<= 2;
        ql += 32;
    }
}

static size_t ds4q_quantize_q5_k(const float *src, void *dst, int64_t start,
                                 int64_t nrows, int64_t ncols, const float *quant_weights) {
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_Q5_K, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t blocks_per_row = ncols / QK_K;

    for (int64_t row = 0; row < nrows; ++row) {
        const float *xrow = src + start + (size_t)row * (size_t)ncols;
        for (int64_t b = 0; b < blocks_per_row; ++b) {
            uint8_t *block = out + (size_t)row * row_size + (size_t)b * ds4q_type_traits[DS4Q_TYPE_Q5_K].type_size;
            const float *x = xrow + (size_t)b * QK_K;
            if (quant_weights) {
                ds4q_write_q5_k_block_weighted(x, block, quant_weights + (size_t)b * QK_K);
            } else {
                ds4q_write_q5_k_block_ref(x, block);
            }
        }
    }
    return (size_t)nrows * row_size;
}

static void ds4q_write_q6_k_block_ref(const float *x, uint8_t *dst) {
    ds4q_block_q6_K *b = (ds4q_block_q6_K *)dst;
    int8_t L[QK_K];
    float scales[QK_K / 16];

    memset(b, 0, sizeof(*b));
    float max_scale = 0;
    float max_abs_scale = 0;
    for (int ib = 0; ib < QK_K / 16; ++ib) {
        const float scale = ds4q_make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, NULL);
        scales[ib] = scale;
        const float abs_scale = fabsf(scale);
        if (abs_scale > max_abs_scale) {
            max_abs_scale = abs_scale;
            max_scale = scale;
        }
    }

    if (max_abs_scale < DS4Q_GROUP_MAX_EPS) {
        b->d = ds4q_f32_to_f16(0.0f);
        return;
    }

    float iscale = -128.0f / max_scale;
    b->d = ds4q_f32_to_f16(1 / iscale);
    for (int ib = 0; ib < QK_K / 16; ++ib) {
        b->scales[ib] = (int8_t)DS4Q_MIN(127, ds4q_nearest_int(iscale * scales[ib]));
    }

    for (int j = 0; j < QK_K / 16; ++j) {
        float d = ds4q_f16_to_f32(b->d) * b->scales[j];
        if (!d) continue;
        for (int ii = 0; ii < 16; ++ii) {
            int l = ds4q_nearest_int(x[16 * j + ii] / d);
            l = DS4Q_MAX(-32, DS4Q_MIN(31, l));
            L[16 * j + ii] = (int8_t)(l + 32);
        }
    }

    uint8_t *ql = b->ql;
    uint8_t *qh = b->qh;
    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; ++l) {
            const uint8_t q1 = (uint8_t)L[j + l + 0] & 0xF;
            const uint8_t q2 = (uint8_t)L[j + l + 32] & 0xF;
            const uint8_t q3 = (uint8_t)L[j + l + 64] & 0xF;
            const uint8_t q4 = (uint8_t)L[j + l + 96] & 0xF;
            ql[l + 0] = q1 | (q3 << 4);
            ql[l + 32] = q2 | (q4 << 4);
            qh[l] = (uint8_t)(((uint8_t)L[j + l] >> 4) |
                              (((uint8_t)L[j + l + 32] >> 4) << 2) |
                              (((uint8_t)L[j + l + 64] >> 4) << 4) |
                              (((uint8_t)L[j + l + 96] >> 4) << 6));
        }
        ql += 64;
        qh += 32;
    }
}

static void ds4q_write_q6_k_block_weighted(const float *x, uint8_t *dst, const float *quant_weights) {
    ds4q_block_q6_K *b = (ds4q_block_q6_K *)dst;
    int8_t L[QK_K];
    float scales[QK_K / 16];

    memset(b, 0, sizeof(*b));
    float max_scale = 0;
    float max_abs_scale = 0;
    for (int ib = 0; ib < QK_K / 16; ++ib) {
        float scale = quant_weights
            ? ds4q_make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, quant_weights + 16 * ib)
            : ds4q_make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, NULL);
        scales[ib] = scale;
        const float abs_scale = fabsf(scale);
        if (abs_scale > max_abs_scale) {
            max_abs_scale = abs_scale;
            max_scale = scale;
        }
    }

    if (max_abs_scale < DS4Q_GROUP_MAX_EPS) {
        b->d = ds4q_f32_to_f16(0.0f);
        return;
    }

    float iscale = -128.0f / max_scale;
    b->d = ds4q_f32_to_f16(1 / iscale);
    for (int ib = 0; ib < QK_K / 16; ++ib) {
        b->scales[ib] = (int8_t)DS4Q_MIN(127, ds4q_nearest_int(iscale * scales[ib]));
    }

    for (int j = 0; j < QK_K / 16; ++j) {
        float d = ds4q_f16_to_f32(b->d) * b->scales[j];
        if (!d) continue;
        for (int ii = 0; ii < 16; ++ii) {
            int l = ds4q_nearest_int(x[16 * j + ii] / d);
            l = DS4Q_MAX(-32, DS4Q_MIN(31, l));
            L[16 * j + ii] = (int8_t)(l + 32);
        }
    }

    uint8_t *ql = b->ql;
    uint8_t *qh = b->qh;
    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; ++l) {
            const uint8_t q1 = (uint8_t)L[j + l + 0] & 0xF;
            const uint8_t q2 = (uint8_t)L[j + l + 32] & 0xF;
            const uint8_t q3 = (uint8_t)L[j + l + 64] & 0xF;
            const uint8_t q4 = (uint8_t)L[j + l + 96] & 0xF;
            ql[l + 0] = q1 | (q3 << 4);
            ql[l + 32] = q2 | (q4 << 4);
            qh[l] = (uint8_t)(((uint8_t)L[j + l] >> 4) |
                              (((uint8_t)L[j + l + 32] >> 4) << 2) |
                              (((uint8_t)L[j + l + 64] >> 4) << 4) |
                              (((uint8_t)L[j + l + 96] >> 4) << 6));
        }
        ql += 64;
        qh += 32;
    }
}

static size_t ds4q_quantize_q6_k(const float *src, void *dst, int64_t start,
                                 int64_t nrows, int64_t ncols, const float *quant_weights) {
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_Q6_K, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t blocks_per_row = ncols / QK_K;

    for (int64_t row = 0; row < nrows; ++row) {
        const float *xrow = src + start + (size_t)row * (size_t)ncols;
        for (int64_t b = 0; b < blocks_per_row; ++b) {
            uint8_t *block = out + (size_t)row * row_size + (size_t)b * ds4q_type_traits[DS4Q_TYPE_Q6_K].type_size;
            const float *x = xrow + (size_t)b * QK_K;
            if (quant_weights) {
                ds4q_write_q6_k_block_weighted(x, block, quant_weights + (size_t)b * QK_K);
            } else {
                ds4q_write_q6_k_block_ref(x, block);
            }
        }
    }
    return (size_t)nrows * row_size;
}

static size_t ds4q_quantize_q8_0(const float *src, void *dst, int64_t start,
                                 int64_t nrows, int64_t ncols) {
    const int64_t qk = 32;
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_Q8_0, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t nblocks = nrows * (ncols / qk);

    for (int64_t b = 0; b < nblocks; b++) {
        const float *x = src + start + (size_t)b * qk;
        float amax = 0.0f;
        for (int j = 0; j < qk; j++) {
            const float av = fabsf(x[j]);
            if (av > amax) amax = av;
        }

        const float d = amax / 127.0f;
        const float id = d ? 1.0f / d : 0.0f;
        const uint16_t hd = ds4q_f32_to_f16(d);
        memcpy(out, &hd, sizeof(hd));

        int8_t *qs = (int8_t *)(out + sizeof(hd));
        for (int j = 0; j < qk; j++) qs[j] = (int8_t)roundf(x[j] * id);
        out += sizeof(hd) + qk;
    }
    return (size_t)nrows * row_size;
}

static void ds4q_write_q4_k_block_ref(const float *x, uint8_t *y) {
    enum { scales_off = 4, qs_off = 16 };
    uint8_t L[QK_K];
    uint8_t Laux[32];
    float weights[32];
    float mins[QK_K / 32];
    float scales[QK_K / 32];
    uint8_t *scales_out = y + scales_off;
    uint8_t *qs_out = y + qs_off;

    float max_scale = 0;
    float max_min = 0;
    for (int j = 0; j < QK_K / 32; j++) {
        float sum_x2 = 0;
        for (int l = 0; l < 32; l++) sum_x2 += x[32 * j + l] * x[32 * j + l];
        float av_x = sqrtf(sum_x2 / 32);
        for (int l = 0; l < 32; l++) weights[l] = av_x + fabsf(x[32 * j + l]);
        scales[j] = ds4q_make_qkx2_quants(32, 15, x + 32 * j, weights, L + 32 * j,
                                           &mins[j], Laux, -1.0f, 0.1f, 20, false);
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min) max_min = mins[j];
    }

    float inv_scale = max_scale > 0 ? 63.0f / max_scale : 0.0f;
    float inv_min = max_min > 0 ? 63.0f / max_min : 0.0f;
    for (int j = 0; j < QK_K / 32; j++) {
        uint8_t ls = ds4q_nearest_int(inv_scale * scales[j]);
        uint8_t lm = ds4q_nearest_int(inv_min * mins[j]);
        ls = DS4Q_MIN(63, ls);
        lm = DS4Q_MIN(63, lm);
        if (j < 4) {
            scales_out[j] = ls;
            scales_out[j + 4] = lm;
        } else {
            scales_out[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            scales_out[j - 4] |= ((ls >> 4) << 6);
            scales_out[j - 0] |= ((lm >> 4) << 6);
        }
    }

    uint16_t d = ds4q_f32_to_f16(max_scale / 63.0f);
    uint16_t dmin = ds4q_f32_to_f16(max_min / 63.0f);
    memcpy(y + 0, &d, sizeof(d));
    memcpy(y + 2, &dmin, sizeof(dmin));

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; j++) {
        ds4q_get_scale_min_k4(j, scales_out, &sc, &m);
        const float dd = ds4q_f16_to_f32(d) * sc;
        if (!dd) continue;
        const float dm = ds4q_f16_to_f32(dmin) * m;
        for (int ii = 0; ii < 32; ii++) {
            int l = ds4q_nearest_int((x[32 * j + ii] + dm) / dd);
            l = DS4Q_MAX(0, DS4Q_MIN(15, l));
            L[32 * j + ii] = l;
        }
    }

    uint8_t *q = qs_out;
    for (int j = 0; j < QK_K; j += 64) {
        for (int l = 0; l < 32; l++) q[l] = L[j + l] | (L[j + l + 32] << 4);
        q += 32;
    }
}

static void ds4q_write_q4_k_block_weighted(const float *x, uint8_t *y, const float *quant_weights) {
    enum { scales_off = 4, qs_off = 16 };
    uint8_t L[QK_K];
    uint8_t Laux[32];
    uint8_t Ls[QK_K / 32];
    uint8_t Lm[QK_K / 32];
    float weights[32];
    float sw[QK_K / 32];
    float mins[QK_K / 32];
    float scales[QK_K / 32];
    uint8_t *scales_out = y + scales_off;
    uint8_t *qs_out = y + qs_off;

    float sum_x2 = 0;
    for (int l = 0; l < QK_K; l++) sum_x2 += x[l] * x[l];
    float sigma2 = 2 * sum_x2 / QK_K;
    float av_x = sqrtf(sigma2);

    for (int j = 0; j < QK_K / 32; j++) {
        if (quant_weights) {
            const float *qw = quant_weights + 32 * j;
            for (int l = 0; l < 32; l++) weights[l] = qw[l] * sqrtf(sigma2 + x[32 * j + l] * x[32 * j + l]);
        } else {
            for (int l = 0; l < 32; l++) weights[l] = av_x + fabsf(x[32 * j + l]);
        }
        float sumw = 0;
        for (int l = 0; l < 32; l++) sumw += weights[l];
        sw[j] = sumw;
        scales[j] = ds4q_make_qkx3_quants(32, 15, x + 32 * j, weights, L + 32 * j,
                                           &mins[j], Laux, -0.9f, 0.05f, 36, false);
    }

    float d_block = ds4q_make_qp_quants(QK_K / 32, 63, scales, Ls, sw);
    float m_block = ds4q_make_qp_quants(QK_K / 32, 63, mins, Lm, sw);
    for (int j = 0; j < QK_K / 32; j++) {
        uint8_t ls = Ls[j];
        uint8_t lm = Lm[j];
        if (j < 4) {
            scales_out[j] = ls;
            scales_out[j + 4] = lm;
        } else {
            scales_out[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            scales_out[j - 4] |= ((ls >> 4) << 6);
            scales_out[j - 0] |= ((lm >> 4) << 6);
        }
    }

    uint16_t d = ds4q_f32_to_f16(d_block);
    uint16_t dmin = ds4q_f32_to_f16(m_block);
    memcpy(y + 0, &d, sizeof(d));
    memcpy(y + 2, &dmin, sizeof(dmin));

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; j++) {
        ds4q_get_scale_min_k4(j, scales_out, &sc, &m);
        const float dd = ds4q_f16_to_f32(d) * sc;
        if (!dd) continue;
        const float dm = ds4q_f16_to_f32(dmin) * m;
        for (int ii = 0; ii < 32; ii++) {
            int l = ds4q_nearest_int((x[32 * j + ii] + dm) / dd);
            l = DS4Q_MAX(0, DS4Q_MIN(15, l));
            L[32 * j + ii] = l;
        }
    }

    uint8_t *q = qs_out;
    for (int j = 0; j < QK_K; j += 64) {
        for (int l = 0; l < 32; l++) q[l] = L[j + l] | (L[j + l + 32] << 4);
        q += 32;
    }
}

static size_t ds4q_quantize_q4_k(const float *src, void *dst, int64_t start,
                                 int64_t nrows, int64_t ncols, const float *quant_weights) {
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_Q4_K, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t blocks_per_row = ncols / QK_K;

    for (int64_t row = 0; row < nrows; row++) {
        const float *xrow = src + start + (size_t)row * (size_t)ncols;
        for (int64_t b = 0; b < blocks_per_row; b++) {
            uint8_t *block = out + (size_t)row * row_size + (size_t)b * ds4q_type_traits[DS4Q_TYPE_Q4_K].type_size;
            const float *x = xrow + (size_t)b * QK_K;
            if (quant_weights) {
                ds4q_write_q4_k_block_weighted(x, block, quant_weights + (size_t)b * QK_K);
            } else {
                ds4q_write_q4_k_block_ref(x, block);
            }
        }
    }
    return (size_t)nrows * row_size;
}

static void ds4q_write_q2_k_block_ref(const float *x, uint8_t *y) {
    enum { scales_off = 0, qs_off = 16, d_off = 80, dmin_off = 82 };
    const float q4scale = 15.0f;
    uint8_t L[QK_K];
    uint8_t Laux[16];
    float weights[16];
    float mins[QK_K / 16];
    float scales[QK_K / 16];
    uint8_t *scales_out = y + scales_off;
    uint8_t *qs_out = y + qs_off;

    float max_scale = 0;
    float max_min = 0;
    for (int j = 0; j < QK_K / 16; j++) {
        for (int l = 0; l < 16; l++) weights[l] = fabsf(x[16 * j + l]);
        scales[j] = ds4q_make_qkx2_quants(16, 3, x + 16 * j, weights, L + 16 * j,
                                           &mins[j], Laux, -0.5f, 0.1f, 15, true);
        if (scales[j] > max_scale) max_scale = scales[j];
        if (mins[j] > max_min) max_min = mins[j];
    }

    uint16_t hd, hmin;
    if (max_scale > 0) {
        float iscale = q4scale / max_scale;
        for (int j = 0; j < QK_K / 16; j++) scales_out[j] = ds4q_nearest_int(iscale * scales[j]);
        hd = ds4q_f32_to_f16(max_scale / q4scale);
    } else {
        memset(scales_out, 0, QK_K / 16);
        hd = ds4q_f32_to_f16(0.0f);
    }
    if (max_min > 0) {
        float iscale = q4scale / max_min;
        for (int j = 0; j < QK_K / 16; j++) scales_out[j] |= ds4q_nearest_int(iscale * mins[j]) << 4;
        hmin = ds4q_f32_to_f16(max_min / q4scale);
    } else {
        hmin = ds4q_f32_to_f16(0.0f);
    }
    memcpy(y + d_off, &hd, sizeof(hd));
    memcpy(y + dmin_off, &hmin, sizeof(hmin));

    for (int j = 0; j < QK_K / 16; j++) {
        const float d = ds4q_f16_to_f32(hd) * (scales_out[j] & 0xF);
        if (!d) continue;
        const float dm = ds4q_f16_to_f32(hmin) * (scales_out[j] >> 4);
        for (int ii = 0; ii < 16; ii++) {
            int l = ds4q_nearest_int((x[16 * j + ii] + dm) / d);
            l = DS4Q_MAX(0, DS4Q_MIN(3, l));
            L[16 * j + ii] = l;
        }
    }

    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; l++) {
            qs_out[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) |
                                (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
        }
    }
}

static void ds4q_write_q2_k_block_weighted(const float *x, uint8_t *y, const float *quant_weights) {
    enum { scales_off = 0, qs_off = 16, d_off = 80, dmin_off = 82 };
    uint8_t L[QK_K];
    uint8_t Laux[16];
    float mins[QK_K / 16];
    float scales[QK_K / 16];
    float sw[QK_K / 16];
    float weight[16];
    uint8_t Ls[QK_K / 16], Lm[QK_K / 16];
    uint8_t *scales_out = y + scales_off;
    uint8_t *qs_out = y + qs_off;

    memset(sw, 0, sizeof(sw));
    float sumx2 = 0;
    for (int j = 0; j < QK_K; j++) sumx2 += x[j] * x[j];
    float sigma2 = sumx2 / QK_K;
    for (int j = 0; j < QK_K / 16; j++) {
        const float *qw = quant_weights + 16 * j;
        for (int l = 0; l < 16; l++) weight[l] = qw[l] * sqrtf(sigma2 + x[16 * j + l] * x[16 * j + l]);
        for (int l = 0; l < QK_K / 16; l++) sw[j] += weight[l];
        scales[j] = ds4q_make_qkx3_quants(16, 3, x + 16 * j, weight, L + 16 * j,
                                           &mins[j], Laux, -0.9f, 0.05f, 36, false);
    }

    float dm = ds4q_make_qp_quants(QK_K / 16, 15, scales, Ls, sw);
    float mm = ds4q_make_qp_quants(QK_K / 16, 15, mins, Lm, sw);
    uint16_t hd = ds4q_f32_to_f16(dm);
    uint16_t hmin = ds4q_f32_to_f16(mm);
    memcpy(y + d_off, &hd, sizeof(hd));
    memcpy(y + dmin_off, &hmin, sizeof(hmin));
    dm = ds4q_f16_to_f32(hd);
    mm = ds4q_f16_to_f32(hmin);

    for (int j = 0; j < QK_K / 16; j++) scales_out[j] = Ls[j] | (Lm[j] << 4);

    for (int j = 0; j < QK_K / 16; j++) {
        const float d = dm * (scales_out[j] & 0xF);
        if (!d) continue;
        const float m = mm * (scales_out[j] >> 4);
        for (int ii = 0; ii < 16; ii++) {
            int l = ds4q_nearest_int((x[16 * j + ii] + m) / d);
            l = DS4Q_MAX(0, DS4Q_MIN(3, l));
            L[16 * j + ii] = l;
        }
    }

    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; l++) {
            qs_out[j / 4 + l] = L[j + l] | (L[j + l + 32] << 2) |
                                (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
        }
    }
}

static size_t ds4q_quantize_q2_k(const float *src, void *dst, int64_t start,
                                 int64_t nrows, int64_t ncols, const float *quant_weights) {
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_Q2_K, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t blocks_per_row = ncols / QK_K;

    for (int64_t row = 0; row < nrows; row++) {
        const float *xrow = src + start + (size_t)row * (size_t)ncols;
        for (int64_t b = 0; b < blocks_per_row; b++) {
            uint8_t *block = out + (size_t)row * row_size + (size_t)b * ds4q_type_traits[DS4Q_TYPE_Q2_K].type_size;
            const float *x = xrow + (size_t)b * QK_K;
            if (quant_weights) {
                ds4q_write_q2_k_block_weighted(x, block, quant_weights + (size_t)b * QK_K);
            } else {
                ds4q_write_q2_k_block_ref(x, block);
            }
        }
    }
    return (size_t)nrows * row_size;
}

typedef struct {
    uint64_t *grid;
    int *map;
    uint16_t *neighbours;
} ds4q_iq2_data;

typedef struct {
    uint32_t *grid;
    int *map;
    uint16_t *neighbours;
} ds4q_iq3_data;

static ds4q_iq2_data ds4q_iq2_xxs_data;
static ds4q_iq2_data ds4q_iq2_s_data;
static ds4q_iq3_data ds4q_iq3_s_data;

static int ds4q_iq2_compare_func(const void *left, const void *right) {
    const int *l = (const int *)left;
    const int *r = (const int *)right;
    return l[0] < r[0] ? -1 :
           l[0] > r[0] ?  1 :
           l[1] < r[1] ? -1 :
           l[1] > r[1] ?  1 : 0;
}

static int ds4q_iq3_compare_func(const void *left, const void *right) {
    const int *l = (const int *)left;
    const int *r = (const int *)right;
    return l[0] < r[0] ? -1 :
           l[0] > r[0] ?  1 :
           l[1] < r[1] ? -1 :
           l[1] > r[1] ?  1 : 0;
}

/*
 * IQ2_XXS quantizes a 256-value row block as eight 32-value groups.  Each
 * group stores four 8-value grid indices plus four 7-bit sign masks; the
 * single f16 block scale is refined by 4-bit per-group scale nibbles.
 *
 * The grid is tiny, but not every possible 2-bit 8-tuple is allowed.  During
 * initialization we build the direct map for allowed tuples and a nearest-grid
 * list for the missing ones, matching the GGML search exactly.
 */
static void ds4q_iq2_init(ds4q_iq2_data *data, const uint16_t *kgrid,
                          int grid_size, int neighbour_shells) {
    if (data->grid) return;
    pthread_mutex_lock(&ds4q_init_mutex);
    if (data->grid) {
        pthread_mutex_unlock(&ds4q_init_mutex);
        return;
    }

    enum { map_size = 43692 };

    uint64_t *grid = malloc((size_t)grid_size * sizeof(grid[0]));
    int *map = malloc((size_t)map_size * sizeof(map[0]));
    int *dist2 = malloc((size_t)2 * grid_size * sizeof(dist2[0]));
    assert(grid && map && dist2);

    for (int k = 0; k < grid_size; k++) {
        int8_t *pos = (int8_t *)(grid + k);
        for (int i = 0; i < 8; i++) {
            int l = (kgrid[k] >> (2 * i)) & 3;
            pos[i] = 2 * l + 1;
        }
    }

    for (int i = 0; i < map_size; i++) map[i] = -1;
    for (int i = 0; i < grid_size; i++) map[kgrid[i]] = i;

    int8_t pos[8];
    int num_neighbors = 0;
    int num_not_in_map = 0;
    for (int i = 0; i < map_size; i++) {
        if (map[i] >= 0) continue;
        num_not_in_map++;
        for (int k = 0; k < 8; k++) pos[k] = 2 * ((i >> (2 * k)) & 3) + 1;
        for (int j = 0; j < grid_size; j++) {
            const int8_t *pg = (const int8_t *)(grid + j);
            int d2 = 0;
            for (int k = 0; k < 8; k++) d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            dist2[2 * j + 0] = d2;
            dist2[2 * j + 1] = j;
        }
        qsort(dist2, grid_size, 2 * sizeof(int), ds4q_iq2_compare_func);
        int d2 = dist2[0], have = 1;
        for (int j = 0; j < grid_size; j++) {
            if (dist2[2 * j] > d2) {
                if (have == neighbour_shells) break;
                d2 = dist2[2 * j];
                have++;
            }
            num_neighbors++;
        }
    }

    uint16_t *neighbours = malloc((size_t)(num_neighbors + num_not_in_map) * sizeof(neighbours[0]));
    assert(neighbours);
    int counter = 0;
    for (int i = 0; i < map_size; i++) {
        if (map[i] >= 0) continue;
        for (int k = 0; k < 8; k++) pos[k] = 2 * ((i >> (2 * k)) & 3) + 1;
        for (int j = 0; j < grid_size; j++) {
            const int8_t *pg = (const int8_t *)(grid + j);
            int d2 = 0;
            for (int k = 0; k < 8; k++) d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            dist2[2 * j + 0] = d2;
            dist2[2 * j + 1] = j;
        }
        qsort(dist2, grid_size, 2 * sizeof(int), ds4q_iq2_compare_func);
        map[i] = -(counter + 1);
        int d2 = dist2[0], have = 1;
        uint16_t *start = &neighbours[counter++];
        int n = 0;
        for (int j = 0; j < grid_size; j++) {
            if (dist2[2 * j] > d2) {
                if (have == neighbour_shells) break;
                d2 = dist2[2 * j];
                have++;
            }
            neighbours[counter++] = (uint16_t)dist2[2 * j + 1];
            n++;
        }
        *start = (uint16_t)n;
    }

    free(dist2);
    data->map = map;
    data->neighbours = neighbours;
    data->grid = grid;
    pthread_mutex_unlock(&ds4q_init_mutex);
}

static void ds4q_iq2_xxs_init(void) {
    static const uint16_t kgrid[256] = {
            0,     2,     5,     8,    10,    17,    20,    32,    34,    40,    42,    65,    68,    80,    88,    97,
          100,   128,   130,   138,   162,   257,   260,   272,   277,   320,   388,   408,   512,   514,   546,   642,
         1025,  1028,  1040,  1057,  1060,  1088,  1090,  1096,  1120,  1153,  1156,  1168,  1188,  1280,  1282,  1288,
         1312,  1350,  1385,  1408,  1425,  1545,  1552,  1600,  1668,  1700,  2048,  2053,  2056,  2068,  2088,  2113,
         2116,  2128,  2130,  2184,  2308,  2368,  2562,  2580,  4097,  4100,  4112,  4129,  4160,  4192,  4228,  4240,
         4245,  4352,  4360,  4384,  4432,  4442,  4480,  4644,  4677,  5120,  5128,  5152,  5157,  5193,  5248,  5400,
         5474,  5632,  5654,  6145,  6148,  6160,  6208,  6273,  6400,  6405,  6560,  6737,  8192,  8194,  8202,  8260,
         8289,  8320,  8322,  8489,  8520,  8704,  8706,  9217,  9220,  9232,  9280,  9302,  9472,  9537,  9572,  9872,
        10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
        16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
        17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
        20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
        22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
        25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
        33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
        37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
    };

    ds4q_iq2_init(&ds4q_iq2_xxs_data, kgrid, 256, 2);
}

static void ds4q_iq2_s_init(void) {
    static const uint16_t kgrid[1024] = {
            0,     2,     5,     8,    10,    17,    20,    22,    25,    32,    34,    37,    40,    65,    68,    70,
           73,    80,    82,    85,    88,    97,   100,   102,   105,   128,   130,   133,   136,   145,   148,   160,
          165,   170,   257,   260,   262,   265,   272,   274,   277,   280,   289,   292,   320,   322,   325,   328,
          337,   340,   342,   345,   352,   357,   360,   385,   388,   400,   402,   405,   417,   420,   512,   514,
          517,   520,   529,   532,   544,   554,   577,   580,   582,   585,   592,   597,   640,   645,   650,   660,
          674,  1025,  1028,  1030,  1033,  1040,  1042,  1045,  1048,  1057,  1060,  1062,  1065,  1088,  1090,  1093,
         1096,  1098,  1105,  1108,  1110,  1113,  1120,  1122,  1125,  1153,  1156,  1158,  1161,  1168,  1173,  1176,
         1185,  1188,  1280,  1282,  1285,  1288,  1290,  1297,  1300,  1302,  1305,  1312,  1317,  1320,  1345,  1348,
         1350,  1353,  1360,  1362,  1365,  1368,  1377,  1380,  1408,  1410,  1413,  1416,  1425,  1428,  1440,  1537,
         1540,  1542,  1545,  1552,  1557,  1600,  1605,  1608,  1617,  1620,  1632,  1665,  1668,  1680,  2048,  2050,
         2053,  2056,  2065,  2068,  2070,  2073,  2080,  2085,  2090,  2113,  2116,  2118,  2121,  2128,  2130,  2133,
         2136,  2145,  2148,  2176,  2181,  2196,  2218,  2305,  2308,  2320,  2322,  2325,  2328,  2337,  2368,  2373,
         2376,  2385,  2388,  2400,  2433,  2448,  2560,  2577,  2580,  2594,  2600,  2602,  2640,  2713,  4097,  4100,
         4102,  4105,  4112,  4114,  4117,  4120,  4129,  4132,  4134,  4160,  4162,  4165,  4168,  4177,  4180,  4182,
         4185,  4192,  4194,  4197,  4200,  4225,  4228,  4230,  4240,  4245,  4248,  4257,  4260,  4352,  4354,  4357,
         4360,  4362,  4369,  4372,  4374,  4377,  4384,  4386,  4389,  4392,  4417,  4420,  4422,  4425,  4432,  4434,
         4437,  4440,  4449,  4452,  4480,  4482,  4485,  4488,  4497,  4500,  4609,  4612,  4617,  4624,  4629,  4641,
         4644,  4672,  4677,  4689,  4692,  4737,  4740,  4752,  5120,  5122,  5125,  5128,  5137,  5140,  5142,  5145,
         5152,  5157,  5160,  5185,  5188,  5190,  5193,  5200,  5202,  5205,  5208,  5217,  5220,  5248,  5250,  5253,
         5256,  5265,  5268,  5280,  5377,  5380,  5382,  5385,  5392,  5394,  5397,  5400,  5409,  5412,  5440,  5442,
         5445,  5448,  5457,  5460,  5472,  5505,  5508,  5520,  5632,  5637,  5640,  5649,  5652,  5664,  5697,  5700,
         5712,  5760,  5802,  6145,  6148,  6150,  6153,  6160,  6165,  6168,  6177,  6208,  6210,  6213,  6216,  6225,
         6228,  6240,  6273,  6276,  6400,  6402,  6405,  6408,  6417,  6420,  6432,  6465,  6468,  6480,  6505,  6562,
         6660,  6672,  6720,  6742,  8192,  8194,  8197,  8200,  8209,  8212,  8214,  8217,  8224,  8229,  8234,  8257,
         8260,  8272,  8274,  8277,  8292,  8320,  8330,  8340,  8362,  8449,  8452,  8464,  8466,  8469,  8481,  8512,
         8514,  8517,  8529,  8532,  8544,  8577,  8580,  8592,  8704,  8714,  8738,  8744,  8746,  8772,  8784,  8840,
         8842,  8872,  9217,  9220,  9222,  9225,  9232,  9237,  9240,  9249,  9252,  9280,  9282,  9285,  9288,  9297,
         9300,  9312,  9345,  9348,  9360,  9472,  9477,  9480,  9489,  9492,  9504,  9537,  9540,  9552,  9574,  9600,
         9729,  9732,  9744,  9792,  9817, 10240, 10245, 10257, 10260, 10305, 10308, 10320, 10378, 10410, 10497, 10500,
        10512, 10645, 10762, 10786, 10852, 10888, 10890, 16385, 16388, 16390, 16393, 16400, 16402, 16405, 16408, 16410,
        16417, 16420, 16422, 16448, 16450, 16453, 16456, 16458, 16465, 16468, 16470, 16473, 16480, 16482, 16485, 16513,
        16516, 16528, 16533, 16536, 16545, 16548, 16640, 16642, 16645, 16648, 16657, 16660, 16662, 16665, 16672, 16674,
        16677, 16705, 16708, 16710, 16713, 16720, 16722, 16725, 16728, 16737, 16740, 16768, 16770, 16773, 16776, 16785,
        16788, 16800, 16897, 16900, 16912, 16914, 16917, 16920, 16932, 16960, 16965, 16968, 16977, 16980, 16992, 17025,
        17028, 17408, 17410, 17413, 17416, 17418, 17425, 17428, 17430, 17433, 17440, 17442, 17445, 17448, 17473, 17476,
        17478, 17481, 17488, 17490, 17493, 17496, 17505, 17508, 17536, 17538, 17541, 17544, 17553, 17556, 17568, 17665,
        17668, 17670, 17673, 17680, 17682, 17685, 17688, 17697, 17700, 17728, 17730, 17733, 17736, 17745, 17748, 17760,
        17770, 17793, 17796, 17808, 17920, 17922, 17925, 17928, 17937, 17940, 17952, 17985, 17988, 18000, 18048, 18085,
        18433, 18436, 18441, 18448, 18450, 18453, 18456, 18465, 18468, 18496, 18498, 18501, 18504, 18513, 18516, 18528,
        18564, 18576, 18688, 18690, 18693, 18696, 18705, 18708, 18720, 18753, 18756, 18768, 18816, 18838, 18945, 18948,
        18960, 19008, 20480, 20482, 20485, 20488, 20497, 20500, 20502, 20505, 20512, 20514, 20517, 20520, 20545, 20548,
        20550, 20553, 20560, 20562, 20565, 20568, 20577, 20580, 20608, 20610, 20613, 20616, 20625, 20628, 20737, 20740,
        20742, 20745, 20752, 20754, 20757, 20760, 20769, 20772, 20800, 20802, 20805, 20808, 20817, 20820, 20832, 20865,
        20868, 20880, 20992, 20997, 21000, 21009, 21012, 21024, 21057, 21060, 21072, 21097, 21120, 21505, 21508, 21510,
        21513, 21520, 21522, 21525, 21528, 21537, 21540, 21568, 21570, 21573, 21576, 21585, 21588, 21600, 21633, 21636,
        21648, 21760, 21762, 21765, 21768, 21777, 21780, 21792, 21825, 21828, 21840, 21888, 22017, 22020, 22032, 22054,
        22080, 22528, 22530, 22533, 22536, 22545, 22548, 22560, 22593, 22596, 22608, 22618, 22656, 22785, 22788, 22800,
        22848, 23040, 23065, 23173, 23208, 24577, 24580, 24582, 24592, 24594, 24597, 24600, 24609, 24612, 24640, 24645,
        24648, 24657, 24660, 24672, 24708, 24720, 24832, 24834, 24837, 24840, 24849, 24852, 24864, 24897, 24900, 24912,
        24960, 24985, 25092, 25104, 25152, 25174, 25249, 25600, 25605, 25608, 25617, 25620, 25632, 25665, 25668, 25680,
        25728, 25857, 25860, 25872, 25920, 25930, 25960, 26002, 26112, 26260, 26625, 26628, 26640, 26725, 26776, 26880,
        26922, 27202, 27297, 32768, 32770, 32773, 32776, 32785, 32788, 32793, 32800, 32805, 32833, 32836, 32848, 32850,
        32853, 32856, 32865, 32896, 32901, 32913, 32916, 33025, 33028, 33033, 33040, 33042, 33045, 33048, 33057, 33060,
        33088, 33090, 33093, 33096, 33105, 33108, 33153, 33156, 33168, 33193, 33280, 33285, 33290, 33297, 33300, 33345,
        33348, 33360, 33793, 33796, 33798, 33801, 33808, 33810, 33813, 33816, 33825, 33856, 33858, 33861, 33864, 33873,
        33876, 33888, 33921, 33924, 33936, 34048, 34050, 34053, 34056, 34065, 34068, 34080, 34113, 34116, 34128, 34176,
        34186, 34305, 34308, 34320, 34345, 34368, 34816, 34821, 34833, 34836, 34881, 34884, 34896, 34978, 35073, 35076,
        35136, 35173, 35362, 35416, 35418, 35458, 35490, 36865, 36868, 36873, 36880, 36882, 36885, 36888, 36900, 36928,
        36930, 36933, 36936, 36945, 36948, 36960, 36993, 36996, 37008, 37120, 37125, 37137, 37140, 37185, 37188, 37200,
        37210, 37377, 37380, 37392, 37440, 37542, 37888, 37890, 37893, 37896, 37905, 37908, 37920, 37953, 37956, 37968,
        38016, 38038, 38145, 38148, 38160, 38208, 38296, 38305, 38400, 38470, 38500, 38913, 38916, 38928, 38950, 38976,
        39081, 39168, 39241, 39250, 39568, 40960, 40965, 40970, 40980, 40994, 41002, 41025, 41028, 41040, 41122, 41130,
        41280, 41317, 41474, 41482, 41506, 41512, 41514, 41602, 41608, 41610, 41640, 41985, 41988, 42000, 42048, 42121,
        42148, 42240, 42265, 42577, 43018, 43048, 43170, 43348, 43398, 43528, 43530, 43552, 43554, 43560, 43656, 43690,
    };

    ds4q_iq2_init(&ds4q_iq2_s_data, kgrid, 1024, 1);
}

static void ds4q_iq3_init(ds4q_iq3_data *data, const uint16_t *kgrid,
                          int grid_size, int neighbour_shells) {
    if (data->grid) return;
    pthread_mutex_lock(&ds4q_init_mutex);
    if (data->grid) {
        pthread_mutex_unlock(&ds4q_init_mutex);
        return;
    }

    enum { map_size = 4096 };

    uint32_t *grid = malloc((size_t)grid_size * sizeof(grid[0]));
    int *map = malloc((size_t)map_size * sizeof(map[0]));
    int *dist2 = malloc((size_t)2 * grid_size * sizeof(dist2[0]));
    assert(grid && map && dist2);

    for (int k = 0; k < grid_size; ++k) {
        int8_t *pos = (int8_t *)(grid + k);
        for (int i = 0; i < 4; ++i) {
            int l = (kgrid[k] >> (3 * i)) & 7;
            pos[i] = 2 * l + 1;
        }
    }

    for (int i = 0; i < map_size; ++i) map[i] = -1;
    for (int i = 0; i < grid_size; ++i) {
        uint32_t aux32 = grid[i];
        uint8_t *aux8 = (uint8_t *)&aux32;
        uint16_t index = 0;
        for (int k = 0; k < 4; ++k) {
            uint16_t q = (uint16_t)((aux8[k] - 1) / 2);
            index |= (uint16_t)(q << (3 * k));
        }
        map[index] = i;
    }

    int8_t pos[4];
    int num_neighbors = 0;
    int num_not_in_map = 0;
    for (int i = 0; i < map_size; ++i) {
        if (map[i] >= 0) continue;
        ++num_not_in_map;
        for (int k = 0; k < 4; ++k) pos[k] = 2 * ((i >> (3 * k)) & 7) + 1;
        for (int j = 0; j < grid_size; ++j) {
            const int8_t *pg = (const int8_t *)(grid + j);
            int d2 = 0;
            for (int k = 0; k < 4; ++k) d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            dist2[2 * j + 0] = d2;
            dist2[2 * j + 1] = j;
        }
        qsort(dist2, grid_size, 2 * sizeof(int), ds4q_iq3_compare_func);
        int d2 = dist2[0], have = 1;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2 * j] > d2) {
                if (have == neighbour_shells) break;
                d2 = dist2[2 * j];
                ++have;
            }
            ++num_neighbors;
        }
    }

    uint16_t *neighbours = malloc((size_t)(num_neighbors + num_not_in_map) * sizeof(neighbours[0]));
    assert(neighbours);
    int counter = 0;
    for (int i = 0; i < map_size; ++i) {
        if (map[i] >= 0) continue;
        for (int k = 0; k < 4; ++k) pos[k] = 2 * ((i >> (3 * k)) & 7) + 1;
        for (int j = 0; j < grid_size; ++j) {
            const int8_t *pg = (const int8_t *)(grid + j);
            int d2 = 0;
            for (int k = 0; k < 4; ++k) d2 += (pg[k] - pos[k]) * (pg[k] - pos[k]);
            dist2[2 * j + 0] = d2;
            dist2[2 * j + 1] = j;
        }
        qsort(dist2, grid_size, 2 * sizeof(int), ds4q_iq3_compare_func);
        map[i] = -(counter + 1);
        int d2 = dist2[0], have = 1;
        uint16_t *start = &neighbours[counter++];
        int n = 0;
        for (int j = 0; j < grid_size; ++j) {
            if (dist2[2 * j] > d2) {
                if (have == neighbour_shells) break;
                d2 = dist2[2 * j];
                ++have;
            }
            neighbours[counter++] = (uint16_t)dist2[2 * j + 1];
            ++n;
        }
        *start = (uint16_t)n;
    }

    free(dist2);
    data->grid = grid;
    data->map = map;
    data->neighbours = neighbours;
    pthread_mutex_unlock(&ds4q_init_mutex);
}

static void ds4q_iq3_s_init(void) {
    static const uint16_t kgrid[512] = {
            0,     1,     2,     5,     7,     8,     9,    10,    12,    14,    16,    17,    21,    27,    32,    34,
           37,    39,    41,    43,    48,    50,    57,    60,    63,    64,    65,    66,    68,    72,    73,    77,
           80,    83,    87,    89,    93,   100,   113,   117,   122,   128,   129,   133,   135,   136,   139,   142,
          145,   149,   152,   156,   162,   165,   167,   169,   171,   184,   187,   195,   201,   205,   208,   210,
          217,   219,   222,   228,   232,   234,   247,   249,   253,   256,   267,   271,   273,   276,   282,   288,
          291,   297,   312,   322,   324,   336,   338,   342,   347,   353,   357,   359,   374,   379,   390,   393,
          395,   409,   426,   441,   448,   450,   452,   464,   466,   470,   475,   488,   492,   512,   513,   514,
          516,   520,   521,   523,   525,   527,   528,   530,   537,   540,   542,   556,   558,   561,   570,   576,
          577,   579,   582,   584,   588,   593,   600,   603,   609,   616,   618,   632,   638,   640,   650,   653,
          655,   656,   660,   666,   672,   675,   685,   688,   698,   705,   708,   711,   712,   715,   721,   727,
          728,   732,   737,   754,   760,   771,   773,   778,   780,   793,   795,   802,   806,   808,   812,   833,
          840,   843,   849,   856,   858,   873,   912,   916,   919,   932,   934,   961,   963,   968,   970,   977,
          989,   993,  1010,  1016,  1024,  1025,  1027,  1029,  1031,  1032,  1034,  1036,  1038,  1041,  1043,  1047,
         1048,  1050,  1057,  1059,  1061,  1064,  1066,  1079,  1080,  1083,  1085,  1088,  1090,  1096,  1099,  1103,
         1106,  1109,  1113,  1116,  1122,  1129,  1153,  1156,  1159,  1169,  1171,  1176,  1183,  1185,  1195,  1199,
         1209,  1212,  1216,  1218,  1221,  1225,  1234,  1236,  1241,  1243,  1250,  1256,  1270,  1281,  1287,  1296,
         1299,  1306,  1309,  1313,  1338,  1341,  1348,  1353,  1362,  1375,  1376,  1387,  1400,  1408,  1410,  1415,
         1425,  1453,  1457,  1477,  1481,  1494,  1496,  1507,  1512,  1538,  1545,  1547,  1549,  1551,  1554,  1561,
         1563,  1565,  1570,  1572,  1575,  1577,  1587,  1593,  1601,  1603,  1605,  1612,  1617,  1619,  1632,  1648,
         1658,  1662,  1664,  1674,  1680,  1690,  1692,  1704,  1729,  1736,  1740,  1745,  1747,  1751,  1752,  1761,
         1763,  1767,  1773,  1787,  1795,  1801,  1806,  1810,  1817,  1834,  1840,  1844,  1857,  1864,  1866,  1877,
         1882,  1892,  1902,  1915,  1934,  1953,  1985,  1987,  2000,  2002,  2013,  2048,  2052,  2058,  2064,  2068,
         2071,  2074,  2081,  2088,  2104,  2114,  2119,  2121,  2123,  2130,  2136,  2141,  2147,  2153,  2157,  2177,
         2179,  2184,  2189,  2193,  2203,  2208,  2223,  2226,  2232,  2244,  2249,  2251,  2256,  2258,  2265,  2269,
         2304,  2306,  2324,  2335,  2336,  2361,  2373,  2375,  2385,  2418,  2443,  2460,  2480,  2504,  2509,  2520,
         2531,  2537,  2562,  2568,  2572,  2578,  2592,  2596,  2599,  2602,  2614,  2620,  2625,  2627,  2629,  2634,
         2641,  2650,  2682,  2688,  2697,  2707,  2712,  2718,  2731,  2754,  2759,  2760,  2775,  2788,  2793,  2805,
         2811,  2817,  2820,  2832,  2842,  2854,  2890,  2902,  2921,  2923,  2978,  3010,  3012,  3026,  3081,  3083,
         3085,  3097,  3099,  3120,  3136,  3152,  3159,  3188,  3210,  3228,  3234,  3245,  3250,  3256,  3264,  3276,
         3281,  3296,  3349,  3363,  3378,  3392,  3395,  3420,  3440,  3461,  3488,  3529,  3531,  3584,  3588,  3591,
         3600,  3602,  3614,  3616,  3628,  3634,  3650,  3657,  3668,  3683,  3685,  3713,  3716,  3720,  3726,  3729,
         3736,  3753,  3778,  3802,  3805,  3819,  3841,  3845,  3851,  3856,  3880,  3922,  3938,  3970,  3993,  4032,
    };

    ds4q_iq3_init(&ds4q_iq3_s_data, kgrid, 512, 3);
}

static int ds4q_iq3_find_best_neighbour(const uint16_t *neighbours, const uint32_t *grid,
                                        const float *xval, const float *weight,
                                        float scale, int8_t *L) {
    int num_neighbors = neighbours[0];
    assert(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t *pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 4; ++i) {
            float q = pg[i];
            float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            grid_index = neighbours[j];
        }
    }
    assert(grid_index >= 0);
    const int8_t *pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 4; ++i) L[i] = (int8_t)((pg[i] - 1) / 2);
    return grid_index;
}

static int ds4q_iq2_find_best_neighbour(const uint16_t *neighbours, const uint64_t *grid,
                                        const float *xval, const float *weight,
                                        float scale, uint8_t *L) {
    int num_neighbors = neighbours[0];
    assert(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; j++) {
        const int8_t *pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 8; i++) {
            float q = pg[i];
            float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2;
            grid_index = neighbours[j];
        }
    }
    assert(grid_index >= 0);
    const int8_t *pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 8; i++) L[i] = (uint8_t)((pg[i] - 1) / 2);
    return grid_index;
}

static void ds4q_write_iq2_xxs_block(const float *x, uint8_t *y, const float *quant_weights) {
    enum { d_off = 0, qs_off = 2, block_size = 32, k_max_q = 3 };
    assert(quant_weights);

    uint32_t q2[2 * (QK_K / block_size)];
    float scales[QK_K / block_size];
    float weight[block_size];
    float xval[block_size];
    uint8_t L[block_size];
    uint8_t Laux[block_size];
    float waux[block_size];
    uint8_t block_signs[4];

    uint16_t hd = ds4q_f32_to_f16(0.0f);
    memcpy(y + d_off, &hd, sizeof(hd));
    memset(q2, 0, sizeof(q2));

    const uint64_t *grid = ds4q_iq2_xxs_data.grid;
    const int *map = ds4q_iq2_xxs_data.map;
    const uint16_t *neighbours = ds4q_iq2_xxs_data.neighbours;
    assert(grid && map && neighbours);

    float sumx2 = 0;
    for (int i = 0; i < QK_K; i++) sumx2 += x[i] * x[i];
    float sigma2 = sumx2 / QK_K;
    float max_scale = 0;

    for (int ib = 0; ib < QK_K / block_size; ib++) {
        const float *xb = x + block_size * ib;
        const float *qw = quant_weights + block_size * ib;
        for (int i = 0; i < block_size; i++) {
            weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
            waux[i] = sqrtf(weight[i]);
        }
        for (int k = 0; k < 4; k++) {
            int nflip = 0;
            uint8_t s = 0;
            for (int i = 0; i < 8; i++) {
                float v = xb[8 * k + i];
                if (v >= 0) {
                    xval[8 * k + i] = v;
                } else {
                    xval[8 * k + i] = -v;
                    nflip++;
                    s |= (uint8_t)(1u << i);
                }
            }
            if (nflip % 2) {
                int imin = 0;
                float min = weight[8 * k] * xb[8 * k] * xb[8 * k];
                for (int i = 1; i < 8; i++) {
                    float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                    if (ax < min) {
                        min = ax;
                        imin = i;
                    }
                }
                xval[8 * k + imin] = -xval[8 * k + imin];
                s ^= (uint8_t)(1u << imin);
            }
            block_signs[k] = s & 127;
        }

        float max = xval[0];
        for (int i = 1; i < block_size; i++) max = DS4Q_MAX(max, xval[i]);
        if (max < DS4Q_GROUP_MAX_EPS) {
            scales[ib] = 0;
            memset(L, 0, sizeof(L));
            continue;
        }

        float scale = ds4q_make_qp_quants(block_size, k_max_q + 1, xval, L, weight);
        float eff_max = scale * k_max_q;
        if (eff_max <= 0) {
            scales[ib] = 0;
            memset(L, 0, sizeof(L));
            continue;
        }

        float best = 0;
        for (int is = -6; is <= 6; is++) {
            float id = (2 * k_max_q - 1 + is * 0.1f) / eff_max;
            float this_scale = 1 / id;
            for (int k = 0; k < 4; k++) {
                uint16_t u = 0;
                for (int i = 0; i < 8; i++) {
                    int l = ds4q_nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                    l = DS4Q_MAX(0, DS4Q_MIN(k_max_q - 1, l));
                    Laux[8 * k + i] = (uint8_t)l;
                    u |= (uint16_t)(l << (2 * i));
                }
                int grid_index = map[u];
                if (grid_index < 0) {
                    const uint16_t *nbs = neighbours - map[u] - 1;
                    ds4q_iq2_find_best_neighbour(nbs, grid, xval + 8 * k, waux + 8 * k,
                                                 this_scale, Laux + 8 * k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < block_size; i++) {
                float w = weight[i];
                float q = 2 * Laux[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                memcpy(L, Laux, sizeof(L));
            }
        }

        if (scale > 0) {
            float id = 1 / scale;
            for (int k = 0; k < 4; k++) {
                uint16_t u = 0;
                for (int i = 0; i < 8; i++) {
                    int l = ds4q_nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                    l = DS4Q_MAX(0, DS4Q_MIN(k_max_q - 1, l));
                    u |= (uint16_t)(l << (2 * i));
                }
                int grid_index = map[u];
                if (grid_index < 0) {
                    const uint16_t *nbs = neighbours - map[u] - 1;
                    grid_index = ds4q_iq2_find_best_neighbour(nbs, grid, xval + 8 * k,
                                                              waux + 8 * k, scale, L + 8 * k);
                }
                const int8_t *pg = (const int8_t *)(grid + grid_index);
                for (int i = 0; i < 8; i++) L[8 * k + i] = (uint8_t)((pg[i] - 1) / 2);
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < block_size; i++) {
                float w = weight[i];
                float q = 2 * L[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0) scale = sumqx / sumq2;
        }

        if (scale < 0) {
            scale = -scale;
            for (int k = 0; k < 4; k++) block_signs[k] = (~block_signs[k]) & 127;
        }

        for (int k = 0; k < 4; k++) {
            uint16_t u = 0;
            for (int i = 0; i < 8; i++) u |= (uint16_t)(L[8 * k + i] << (2 * i));
            int grid_index = map[u];
            assert(grid_index >= 0);
            q2[2 * ib + 0] |= (uint32_t)grid_index << (8 * k);
            q2[2 * ib + 1] |= (uint32_t)block_signs[k] << (7 * k);
        }
        assert(scale >= 0);
        scales[ib] = scale;
        max_scale = DS4Q_MAX(max_scale, scale);
    }

    if (!max_scale) {
        memset(y + qs_off, 0, QK_K / 4);
        return;
    }

    float d = max_scale / 31;
    hd = ds4q_f32_to_f16(d);
    memcpy(y + d_off, &hd, sizeof(hd));
    float id = 1 / d;
    for (int ib = 0; ib < QK_K / block_size; ib++) {
        int l = ds4q_nearest_int(0.5f * (id * scales[ib] - 1));
        l = DS4Q_MAX(0, DS4Q_MIN(15, l));
        q2[2 * ib + 1] |= (uint32_t)l << 28;
    }
    memcpy(y + qs_off, q2, QK_K / 4);
}

static size_t ds4q_quantize_iq2_xxs(const float *src, void *dst, int64_t start,
                                    int64_t nrows, int64_t ncols, const float *quant_weights) {
    assert(quant_weights);
    ds4q_iq2_xxs_init();
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_IQ2_XXS, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t blocks_per_row = ncols / QK_K;

    for (int64_t row = 0; row < nrows; row++) {
        const float *xrow = src + start + (size_t)row * (size_t)ncols;
        for (int64_t b = 0; b < blocks_per_row; b++) {
            uint8_t *block = out + (size_t)row * row_size + (size_t)b * ds4q_type_traits[DS4Q_TYPE_IQ2_XXS].type_size;
            ds4q_write_iq2_xxs_block(xrow + (size_t)b * QK_K, block,
                                     quant_weights + (size_t)b * QK_K);
        }
    }
    return (size_t)nrows * row_size;
}

static void ds4q_write_iq2_s_block(const float *x, uint8_t *dst,
                                   const float *quant_weights) {
    ds4q_block_iq2_s *y = (ds4q_block_iq2_s *)dst;
    memset(y, 0, sizeof(*y));
    y->d = ds4q_f32_to_f16(0.0f);

    const uint64_t *grid = ds4q_iq2_s_data.grid;
    const int *map = ds4q_iq2_s_data.map;
    const uint16_t *neighbours = ds4q_iq2_s_data.neighbours;
    assert(grid && map && neighbours);

    float scales[QK_K / 16];
    float weight[16];
    float xval[16];
    int8_t L[16];
    int8_t Laux[16];
    float waux[16];
    bool is_on_grid[2];
    bool is_on_grid_aux[2];
    uint8_t block_signs[2];

    float sumx2 = 0;
    for (int i = 0; i < QK_K; ++i) sumx2 += x[i] * x[i];
    const float sigma2 = 2 * sumx2 / QK_K;
    float max_scale = 0;

    for (int ib = 0; ib < QK_K / 16; ++ib) {
        const float *xb = x + 16 * ib;
        if (quant_weights) {
            const float *qw = quant_weights + 16 * ib;
            for (int i = 0; i < 16; ++i) {
                weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
            }
        } else {
            for (int i = 0; i < 16; ++i) {
                weight[i] = 0.25f * sigma2 + xb[i] * xb[i];
            }
        }
        for (int i = 0; i < 16; ++i) waux[i] = sqrtf(weight[i]);
        for (int k = 0; k < 2; ++k) {
            uint8_t s = 0;
            for (int i = 0; i < 8; ++i) {
                float v = xb[8 * k + i];
                if (v >= 0) {
                    xval[8 * k + i] = v;
                } else {
                    xval[8 * k + i] = -v;
                    s |= (uint8_t)(1u << i);
                }
            }
            block_signs[k] = s;
        }
        float max = xval[0];
        for (int i = 1; i < 16; ++i) max = DS4Q_MAX(max, xval[i]);
        memset(L, 0, sizeof(L));
        if (max < DS4Q_GROUP_MAX_EPS_IQ2_S) {
            scales[ib] = 0;
            continue;
        }
        float best = 0;
        float scale = max / 5.0f;
        is_on_grid[0] = true;
        is_on_grid[1] = true;
        for (int is = -9; is <= 9; ++is) {
            float id = (5.0f + is * 0.1f) / max;
            float this_scale = 1 / id;
            for (int k = 0; k < 2; ++k) {
                for (int i = 0; i < 8; ++i) {
                    int l = ds4q_nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                    Laux[8 * k + i] = (int8_t)DS4Q_MAX(0, DS4Q_MIN(2, l));
                }
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) u |= (uint16_t)(Laux[8 * k + i] << (2 * i));
                int grid_index = map[u];
                is_on_grid_aux[k] = true;
                if (grid_index < 0) {
                    is_on_grid_aux[k] = false;
                    const uint16_t *nbs = neighbours - map[u] - 1;
                    grid_index = ds4q_iq2_find_best_neighbour(nbs, grid, xval + 8 * k,
                                                              waux + 8 * k, this_scale,
                                                              (uint8_t *)(Laux + 8 * k));
                    (void)grid_index;
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 16; ++i) {
                float w = weight[i];
                float q = 2 * Laux[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                memcpy(L, Laux, sizeof(L));
                is_on_grid[0] = is_on_grid_aux[0];
                is_on_grid[1] = is_on_grid_aux[1];
            }
        }

        int n_not_on_grid = (!is_on_grid[0]) + (!is_on_grid[1]);
        if (n_not_on_grid > 0 && scale > 0) {
            float id = 1 / scale;
            for (int k = 0; k < 2; ++k) {
                if (is_on_grid[k]) continue;
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) {
                    int l = ds4q_nearest_int(0.5f * (id * xval[8 * k + i] - 1));
                    l = DS4Q_MAX(0, DS4Q_MIN(2, l));
                    u |= (uint16_t)(l << (2 * i));
                    L[8 * k + i] = (int8_t)l;
                }
                int grid_index = map[u];
                if (grid_index < 0) {
                    const uint16_t *nbs = neighbours - map[u] - 1;
                    ds4q_iq2_find_best_neighbour(nbs, grid, xval + 8 * k, waux + 8 * k,
                                                 scale, (uint8_t *)(L + 8 * k));
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 16; ++i) {
                float w = weight[i];
                float q = 2 * L[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0) scale = sumqx / sumq2;
        }

        if (scale < 0) {
            scale = -scale;
            block_signs[0] = (uint8_t)~block_signs[0];
            block_signs[1] = (uint8_t)~block_signs[1];
        }

        for (int k = 0; k < 2; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 8; ++i) u |= (uint16_t)(L[8 * k + i] << (2 * i));
            int grid_index = map[u];
            assert(grid_index >= 0);
            const int i8 = 2 * ib + k;
            y->qs[i8] = (uint8_t)(grid_index & 255);
            y->qh[i8 / 4] |= (uint8_t)((grid_index >> 8) << (2 * (i8 % 4)));
            y->qs[QK_K / 8 + i8] = block_signs[k];
        }
        assert(scale >= 0);
        scales[ib] = scale;
        max_scale = DS4Q_MAX(max_scale, scale);
    }

    if (!max_scale) {
        return;
    }

    float d = max_scale / 31;
    y->d = ds4q_f32_to_f16(d * 0.9875f);
    float id = 1 / d;
    for (int ib = 0; ib < QK_K / 16; ++ib) {
        int l = ds4q_nearest_int(0.5f * (id * scales[ib] - 1));
        l = DS4Q_MAX(0, DS4Q_MIN(15, l));
        if ((ib & 1) == 0) y->scales[ib / 2] = (uint8_t)l;
        else y->scales[ib / 2] |= (uint8_t)(l << 4);
    }
}

static size_t ds4q_quantize_iq2_s(const float *src, void *dst, int64_t start,
                                  int64_t nrows, int64_t ncols,
                                  const float *quant_weights) {
    ds4q_iq2_s_init();
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_IQ2_S, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t blocks_per_row = ncols / QK_K;

    for (int64_t row = 0; row < nrows; ++row) {
        const float *xrow = src + start + (size_t)row * (size_t)ncols;
        for (int64_t b = 0; b < blocks_per_row; ++b) {
            uint8_t *block = out + (size_t)row * row_size +
                             (size_t)b * ds4q_type_traits[DS4Q_TYPE_IQ2_S].type_size;
            const float *weights = quant_weights ? quant_weights + (size_t)b * QK_K : NULL;
            ds4q_write_iq2_s_block(xrow + (size_t)b * QK_K, block, weights);
        }
    }
    return (size_t)nrows * row_size;
}

static void ds4q_write_iq3_s_block(const float *x, uint8_t *dst,
                                   const float *quant_weights) {
    ds4q_block_iq3_s *y = (ds4q_block_iq3_s *)dst;
    memset(y, 0, sizeof(*y));
    y->d = ds4q_f32_to_f16(0.0f);

    const uint32_t *grid = ds4q_iq3_s_data.grid;
    const int *map = ds4q_iq3_s_data.map;
    const uint16_t *neighbours = ds4q_iq3_s_data.neighbours;
    assert(grid && map && neighbours);

    float scales[QK_K / 32];
    float weight[32];
    float xval[32];
    int8_t L[32];
    int8_t Laux[32];
    float waux[32];
    bool is_on_grid[8];
    bool is_on_grid_aux[8];
    uint8_t block_signs[4];

    float sumx2 = 0;
    for (int i = 0; i < QK_K; ++i) sumx2 += x[i] * x[i];
    const float sigma2 = 2 * sumx2 / QK_K;
    float max_scale = 0;

    uint8_t *qs = y->qs;
    uint8_t *qh = y->qh;
    uint8_t *signs = y->signs;

    for (int ib = 0; ib < QK_K / 32; ++ib) {
        const float *xb = x + 32 * ib;
        if (quant_weights) {
            const float *qw = quant_weights + 32 * ib;
            for (int i = 0; i < 32; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
        } else {
            for (int i = 0; i < 32; ++i) weight[i] = xb[i] * xb[i];
        }
        for (int i = 0; i < 32; ++i) waux[i] = sqrtf(weight[i]);
        for (int k = 0; k < 4; ++k) {
            uint8_t s = 0;
            for (int i = 0; i < 8; ++i) {
                float v = xb[8 * k + i];
                if (v >= 0) {
                    xval[8 * k + i] = v;
                } else {
                    xval[8 * k + i] = -v;
                    s |= (uint8_t)(1u << i);
                }
            }
            block_signs[k] = s;
        }
        float max = xval[0];
        for (int i = 1; i < 32; ++i) max = DS4Q_MAX(max, xval[i]);
        memset(L, 0, sizeof(L));
        if (!max) {
            scales[ib] = 0;
            qs += 8;
            signs += 4;
            continue;
        }

        float best = 0;
        float scale = max / 15.0f;
        for (int k = 0; k < 8; ++k) is_on_grid[k] = false;
        for (int is = -9; is <= 9; ++is) {
            float id = (15.0f + is * 0.2f) / max;
            float this_scale = 1 / id;
            for (int k = 0; k < 8; ++k) {
                for (int i = 0; i < 4; ++i) {
                    int l = ds4q_nearest_int(0.5f * (id * xval[4 * k + i] - 1));
                    Laux[4 * k + i] = (int8_t)DS4Q_MAX(0, DS4Q_MIN(7, l));
                }
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) u |= (uint16_t)(Laux[4 * k + i] << (3 * i));
                int grid_index = map[u];
                is_on_grid_aux[k] = true;
                if (grid_index < 0) {
                    is_on_grid_aux[k] = false;
                    const uint16_t *nbs = neighbours - map[u] - 1;
                    ds4q_iq3_find_best_neighbour(nbs, grid, xval + 4 * k, waux + 4 * k,
                                                 this_scale, Laux + 4 * k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 32; ++i) {
                float w = weight[i];
                float q = 2 * Laux[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best = scale * sumqx;
                memcpy(L, Laux, sizeof(L));
                for (int k = 0; k < 8; ++k) is_on_grid[k] = is_on_grid_aux[k];
            }
        }

        int n_not_on_grid = 0;
        for (int k = 0; k < 8; ++k) if (!is_on_grid[k]) ++n_not_on_grid;
        if (n_not_on_grid > 0 && scale > 0) {
            float id = 1 / scale;
            for (int k = 0; k < 8; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) {
                    int l = ds4q_nearest_int(0.5f * (id * xval[4 * k + i] - 1));
                    l = DS4Q_MAX(0, DS4Q_MIN(7, l));
                    u |= (uint16_t)(l << (3 * i));
                }
                int grid_index = map[u];
                if (grid_index < 0) {
                    const uint16_t *nbs = neighbours - map[u] - 1;
                    grid_index = ds4q_iq3_find_best_neighbour(nbs, grid, xval + 4 * k,
                                                              waux + 4 * k, scale, L + 4 * k);
                }
                const int8_t *pg = (const int8_t *)(grid + grid_index);
                for (int i = 0; i < 4; ++i) L[4 * k + i] = (int8_t)((pg[i] - 1) / 2);
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 32; ++i) {
                float w = weight[i];
                float q = 2 * L[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0) scale = sumqx / sumq2;
        }

        if (scale < 0) {
            scale = -scale;
            for (int k = 0; k < 4; ++k) block_signs[k] = (uint8_t)~block_signs[k];
        }

        for (int k = 0; k < 8; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 4; ++i) u |= (uint16_t)(L[4 * k + i] << (3 * i));
            int grid_index = map[u];
            assert(grid_index >= 0);
            qs[k] = (uint8_t)(grid_index & 255);
            qh[(ib * 8 + k) / 8] |= (uint8_t)((grid_index >> 8) << ((ib * 8 + k) % 8));
        }
        qs += 8;
        for (int k = 0; k < 4; ++k) signs[k] = block_signs[k];
        signs += 4;

        assert(scale >= 0);
        scales[ib] = scale;
        max_scale = DS4Q_MAX(max_scale, scale);
    }

    if (!max_scale) return;

    float d = max_scale / 31;
    y->d = ds4q_f32_to_f16(d * 1.033f);
    float id = 1 / d;
    for (int ib = 0; ib < QK_K / 32; ib += 2) {
        int l1 = ds4q_nearest_int(0.5f * (id * scales[ib + 0] - 1));
        l1 = DS4Q_MAX(0, DS4Q_MIN(15, l1));
        int l2 = ds4q_nearest_int(0.5f * (id * scales[ib + 1] - 1));
        l2 = DS4Q_MAX(0, DS4Q_MIN(15, l2));
        y->scales[ib / 2] = (uint8_t)(l1 | (l2 << 4));
    }
}

static size_t ds4q_quantize_iq3_s(const float *src, void *dst, int64_t start,
                                  int64_t nrows, int64_t ncols,
                                  const float *quant_weights) {
    ds4q_iq3_s_init();
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_IQ3_S, ncols);
    const int64_t start_row = start / ncols;
    uint8_t *out = (uint8_t *)dst + (size_t)start_row * row_size;
    const int64_t blocks_per_row = ncols / QK_K;

    for (int64_t row = 0; row < nrows; ++row) {
        const float *xrow = src + start + (size_t)row * (size_t)ncols;
        for (int64_t b = 0; b < blocks_per_row; ++b) {
            uint8_t *block = out + (size_t)row * row_size +
                             (size_t)b * ds4q_type_traits[DS4Q_TYPE_IQ3_S].type_size;
            const float *weights = quant_weights ? quant_weights + (size_t)b * QK_K : NULL;
            ds4q_write_iq3_s_block(xrow + (size_t)b * QK_K, block, weights);
        }
    }
    return (size_t)nrows * row_size;
}

const char *ds4q_type_name(ds4q_type type) {
    if (type < 0 || type >= DS4Q_TYPE_COUNT) return NULL;
    return ds4q_type_traits[type].name;
}

bool ds4q_can_quantize(ds4q_type type) {
    if (type < 0 || type >= DS4Q_TYPE_COUNT) return false;
    return ds4q_type_traits[type].can_quantize;
}

int64_t ds4q_block_size(ds4q_type type) {
    if (type < 0 || type >= DS4Q_TYPE_COUNT) return 0;
    return ds4q_type_traits[type].block_size;
}

size_t ds4q_row_size(ds4q_type type, int64_t ne) {
    if (type < 0 || type >= DS4Q_TYPE_COUNT) return 0;
    const ds4q_traits *tr = &ds4q_type_traits[type];
    if (tr->block_size <= 0 || tr->type_size == 0 || ne % tr->block_size != 0) return 0;
    return tr->type_size * (size_t)(ne / tr->block_size);
}

bool ds4q_requires_imatrix(ds4q_type type) {
    if (type < 0 || type >= DS4Q_TYPE_COUNT) return false;
    return ds4q_type_traits[type].requires_imatrix;
}

void ds4q_quantize_init(ds4q_type type) {
    if (type == DS4Q_TYPE_IQ2_XXS) {
        ds4q_iq2_xxs_init();
        return;
    }
    if (type == DS4Q_TYPE_IQ2_S) {
        ds4q_iq2_s_init();
        return;
    }
    if (type == DS4Q_TYPE_IQ3_S) {
        ds4q_iq3_s_init();
    }
}

size_t ds4q_quantize_chunk(ds4q_type type, const float *src, void *dst,
                           int64_t start, int64_t nrows, int64_t ncols,
                           const float *imatrix) {
    if (type == DS4Q_TYPE_Q8_0) {
        (void)imatrix;
        return ds4q_quantize_q8_0(src, dst, start, nrows, ncols);
    }
    if (type == DS4Q_TYPE_Q2_K) {
        return ds4q_quantize_q2_k(src, dst, start, nrows, ncols, imatrix);
    }
    if (type == DS4Q_TYPE_Q4_K) {
        return ds4q_quantize_q4_k(src, dst, start, nrows, ncols, imatrix);
    }
    if (type == DS4Q_TYPE_Q5_K) {
        return ds4q_quantize_q5_k(src, dst, start, nrows, ncols, imatrix);
    }
    if (type == DS4Q_TYPE_Q6_K) {
        return ds4q_quantize_q6_k(src, dst, start, nrows, ncols, imatrix);
    }
    if (type == DS4Q_TYPE_IQ2_XXS) {
        return ds4q_quantize_iq2_xxs(src, dst, start, nrows, ncols, imatrix);
    }
    if (type == DS4Q_TYPE_IQ2_S) {
        return ds4q_quantize_iq2_s(src, dst, start, nrows, ncols, imatrix);
    }
    if (type == DS4Q_TYPE_IQ3_S) {
        return ds4q_quantize_iq3_s(src, dst, start, nrows, ncols, imatrix);
    }
    (void)src;
    (void)dst;
    (void)start;
    (void)nrows;
    (void)ncols;
    (void)imatrix;
    assert(!"unsupported DS4 quantization target");
    return 0;
}

float ds4q_f16_to_f32(uint16_t bits) {
    const uint32_t w = (uint32_t)bits << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;
    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = 0x1.0p-112f;
    const float normalized_value = ds4q_f32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = ds4q_f32_from_bits((two_w >> 17) | magic_mask) - magic_bias;
    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? ds4q_f32_to_bits(denormalized_value) : ds4q_f32_to_bits(normalized_value));
    return ds4q_f32_from_bits(result);
}

float ds4q_bf16_to_f32(uint16_t bits) {
    return ds4q_f32_from_bits((uint32_t)bits << 16);
}

void ds4q_f32_to_f16_row(const float *src, uint16_t *dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] = ds4q_f32_to_f16(src[i]);
}

void ds4q_f32_to_bf16_row(const float *src, uint16_t *dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        uint32_t bits = ds4q_f32_to_bits(src[i]);
        if ((bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000)) {
            dst[i] = (uint16_t)((bits >> 16) | 64);
        } else {
            dst[i] = (uint16_t)((bits + (UINT32_C(0x7fff) + ((bits >> 16) & 1))) >> 16);
        }
    }
}
