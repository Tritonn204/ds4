#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC "Q36DLF01"
#define MAGIC_LEN 8

typedef struct dec_fixture {
    uint32_t layer, seq_len, num_v_heads, num_k_heads, head_k_dim, head_v_dim, topk, inter, hidden;
    float *layer_input, *q, *k, *v, *beta, *g, *z_last;
    float *mixer_out_ref, *residual_after_mixer_ref, *post_attn_ln, *mlp_out_ref, *layer_output_ref;
    float *router_indices_f32, *router_scores, *gate_sel, *up_sel, *down_sel;
    float *gate_shexp, *up_shexp, *down_shexp, *gate_inp_shexp, *norm_w, *out_w;
} dec_fixture;

static int read_exact(FILE *fp, void *buf, size_t n) { return fread(buf, 1, n, fp) == n; }
static int alloc_read_f32(FILE *fp, float **out, size_t count) {
    float *p = (float *)malloc(count * sizeof(float));
    if (!p) return 0;
    if (!read_exact(fp, p, count * sizeof(float))) { free(p); return 0; }
    *out = p;
    return 1;
}
static void fixture_free(dec_fixture *fx) {
    if (!fx) return;
    free(fx->layer_input); free(fx->q); free(fx->k); free(fx->v); free(fx->beta); free(fx->g); free(fx->z_last);
    free(fx->mixer_out_ref); free(fx->residual_after_mixer_ref); free(fx->post_attn_ln); free(fx->mlp_out_ref);
    free(fx->layer_output_ref); free(fx->router_indices_f32); free(fx->router_scores); free(fx->gate_sel); free(fx->up_sel);
    free(fx->down_sel); free(fx->gate_shexp); free(fx->up_shexp); free(fx->down_shexp); free(fx->gate_inp_shexp);
    free(fx->norm_w); free(fx->out_w); memset(fx, 0, sizeof(*fx));
}
static int fixture_load(const char *path, dec_fixture *fx) {
    FILE *fp = fopen(path, "rb");
    char magic[MAGIC_LEN];
    memset(fx, 0, sizeof(*fx));
    if (!fp) return 0;
    if (!read_exact(fp, magic, MAGIC_LEN) || memcmp(magic, MAGIC, MAGIC_LEN) != 0) { fclose(fp); return 0; }
    if (!read_exact(fp, &fx->layer, sizeof(uint32_t)) || !read_exact(fp, &fx->seq_len, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->num_v_heads, sizeof(uint32_t)) || !read_exact(fp, &fx->num_k_heads, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->head_k_dim, sizeof(uint32_t)) || !read_exact(fp, &fx->head_v_dim, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->topk, sizeof(uint32_t)) || !read_exact(fp, &fx->inter, sizeof(uint32_t)) ||
        !read_exact(fp, &fx->hidden, sizeof(uint32_t))) { fclose(fp); return 0; }
#define R(name,count) if (!alloc_read_f32(fp, &fx->name, (count))) { fclose(fp); fixture_free(fx); return 0; }
    R(layer_input, fx->hidden); R(q, (size_t)fx->seq_len * fx->num_v_heads * fx->head_k_dim); R(k, (size_t)fx->seq_len * fx->num_v_heads * fx->head_k_dim);
    R(v, (size_t)fx->seq_len * fx->num_v_heads * fx->head_v_dim); R(beta, (size_t)fx->seq_len * fx->num_v_heads); R(g, (size_t)fx->seq_len * fx->num_v_heads);
    R(z_last, (size_t)fx->num_v_heads * fx->head_v_dim); R(mixer_out_ref, fx->hidden); R(residual_after_mixer_ref, fx->hidden); R(post_attn_ln, fx->hidden);
    R(mlp_out_ref, fx->hidden); R(layer_output_ref, fx->hidden); R(router_indices_f32, fx->topk); R(router_scores, fx->topk);
    R(gate_sel, (size_t)fx->topk * fx->inter * fx->hidden); R(up_sel, (size_t)fx->topk * fx->inter * fx->hidden); R(down_sel, (size_t)fx->topk * fx->hidden * fx->inter);
    R(gate_shexp, (size_t)fx->inter * fx->hidden); R(up_shexp, (size_t)fx->inter * fx->hidden); R(down_shexp, (size_t)fx->hidden * fx->inter);
    R(gate_inp_shexp, fx->hidden); R(norm_w, fx->head_v_dim); R(out_w, (size_t)fx->hidden * fx->num_v_heads * fx->head_v_dim);
#undef R
    fclose(fp); return 1;
}

static inline float siluf_local(float x) { return x / (1.0f + expf(-x)); }
static void matvec(const float *mat, const float *vec, float *out, uint32_t rows, uint32_t cols) {
    uint32_t r, c; for (r = 0; r < rows; ++r) { const float *row = mat + (size_t)r * cols; double sum = 0.0; for (c = 0; c < cols; ++c) sum += (double)row[c] * vec[c]; out[r] = (float)sum; }
}
static double vec_rmse(const float *a, const float *b, size_t n) { size_t i; double acc=0.0; for(i=0;i<n;++i){ double d=(double)a[i]-b[i]; acc+=d*d;} return sqrt(acc/(double)n); }
static double vec_cosine(const float *a, const float *b, size_t n) { size_t i; double dot=0.0,an=0.0,bn=0.0; for(i=0;i<n;++i){ dot+=(double)a[i]*b[i]; an+=(double)a[i]*a[i]; bn+=(double)b[i]*b[i]; } if(an==0.0||bn==0.0) return NAN; return dot/sqrt(an*bn); }

static void run_layer(const dec_fixture *fx, const float *input_override, float *resid_after_mixer, float *mlp_out, float *final_out) {
    float *state = calloc((size_t)fx->num_v_heads * fx->head_k_dim * fx->head_v_dim, sizeof(float));
    float *core = calloc((size_t)fx->seq_len * fx->num_v_heads * fx->head_v_dim, sizeof(float));
    float *out_in = calloc((size_t)fx->num_v_heads * fx->head_v_dim, sizeof(float));
    float *mixer = calloc(fx->hidden, sizeof(float));
    float *gate = calloc(fx->inter, sizeof(float));
    float *up = calloc(fx->inter, sizeof(float));
    float *act = calloc(fx->inter, sizeof(float));
    float *down = calloc(fx->hidden, sizeof(float));
    float *shared_gate = calloc(fx->inter, sizeof(float));
    float *shared_up = calloc(fx->inter, sizeof(float));
    float *shared_act = calloc(fx->inter, sizeof(float));
    float *shared = calloc(fx->hidden, sizeof(float));
    uint32_t t,h,hd,vd,i;
    const float *input = input_override ? input_override : fx->layer_input;
    for (t = 0; t < fx->seq_len; ++t) {
        for (h = 0; h < fx->num_v_heads; ++h) {
            float gexp = expf(fx->g[t * fx->num_v_heads + h]), beta_t = fx->beta[t * fx->num_v_heads + h], qnorm=0.0f, knorm=0.0f;
            size_t qbase=((size_t)t*fx->num_v_heads+h)*fx->head_k_dim, vbase=((size_t)t*fx->num_v_heads+h)*fx->head_v_dim, sbase=(size_t)h*fx->head_k_dim*fx->head_v_dim;
            for(hd=0;hd<fx->head_k_dim;++hd){ qnorm+=fx->q[qbase+hd]*fx->q[qbase+hd]; knorm+=fx->k[qbase+hd]*fx->k[qbase+hd]; }
            qnorm=1.0f/sqrtf(qnorm+1e-6f); knorm=1.0f/sqrtf(knorm+1e-6f);
            for(hd=0;hd<fx->head_k_dim*fx->head_v_dim;++hd) state[sbase+hd]*=gexp;
            {
                float delta[128];
                for(vd=0;vd<fx->head_v_dim;++vd){ float kv_mem=0.0f; for(hd=0;hd<fx->head_k_dim;++hd) kv_mem += state[sbase + (size_t)hd*fx->head_v_dim + vd] * (fx->k[qbase+hd]*knorm); delta[vd]=(fx->v[vbase+vd]-kv_mem)*beta_t; }
                for(hd=0;hd<fx->head_k_dim;++hd){ float kval=fx->k[qbase+hd]*knorm; for(vd=0;vd<fx->head_v_dim;++vd) state[sbase+(size_t)hd*fx->head_v_dim+vd] += kval*delta[vd]; }
                for(vd=0;vd<fx->head_v_dim;++vd){ float outv=0.0f; for(hd=0;hd<fx->head_k_dim;++hd) outv += state[sbase+(size_t)hd*fx->head_v_dim+vd] * (fx->q[qbase+hd]*qnorm/sqrtf((float)fx->head_k_dim)); core[vbase+vd]=outv; }
            }
        }
    }
    for(h=0; h<fx->num_v_heads; ++h){ double var=0.0; size_t base=(size_t)h*fx->head_v_dim; for(vd=0; vd<fx->head_v_dim; ++vd){ double cv=core[((size_t)(fx->seq_len-1)*fx->num_v_heads+h)*fx->head_v_dim+vd]; var += cv*cv; } var/=(double)fx->head_v_dim; for(vd=0; vd<fx->head_v_dim; ++vd){ float cv=core[((size_t)(fx->seq_len-1)*fx->num_v_heads+h)*fx->head_v_dim+vd]; float normed=(float)(cv/sqrt(var+1e-6)); out_in[base+vd]=normed*fx->norm_w[vd]*siluf_local(fx->z_last[base+vd]); } }
    matvec(fx->out_w, out_in, mixer, fx->hidden, fx->num_v_heads*fx->head_v_dim);
    for(i=0;i<fx->hidden;++i) resid_after_mixer[i]=input[i]+mixer[i];
    for(i=0;i<fx->topk;++i){ const float *gate_mat=fx->gate_sel+(size_t)i*fx->inter*fx->hidden; const float *up_mat=fx->up_sel+(size_t)i*fx->inter*fx->hidden; const float *down_mat=fx->down_sel+(size_t)i*fx->hidden*fx->inter; matvec(gate_mat, fx->post_attn_ln, gate, fx->inter, fx->hidden); matvec(up_mat, fx->post_attn_ln, up, fx->inter, fx->hidden); for(vd=0;vd<fx->inter;++vd) act[vd]=siluf_local(gate[vd])*up[vd]; matvec(down_mat, act, down, fx->hidden, fx->inter); for(vd=0;vd<fx->hidden;++vd) mlp_out[vd]+=down[vd]*fx->router_scores[i]; }
    matvec(fx->gate_shexp, fx->post_attn_ln, shared_gate, fx->inter, fx->hidden); matvec(fx->up_shexp, fx->post_attn_ln, shared_up, fx->inter, fx->hidden); for(vd=0;vd<fx->inter;++vd) shared_act[vd]=siluf_local(shared_gate[vd])*shared_up[vd]; matvec(fx->down_shexp, shared_act, shared, fx->hidden, fx->inter);
    { float scale_in=0.0f, s; for(vd=0;vd<fx->hidden;++vd) scale_in += fx->post_attn_ln[vd]*fx->gate_inp_shexp[vd]; s=1.0f/(1.0f+expf(-scale_in)); for(vd=0;vd<fx->hidden;++vd){ mlp_out[vd]+=shared[vd]*s; final_out[vd]=resid_after_mixer[vd]+mlp_out[vd]; } }
    free(state); free(core); free(out_in); free(mixer); free(gate); free(up); free(act); free(down); free(shared_gate); free(shared_up); free(shared_act); free(shared);
}

int main(int argc, char **argv) {
    dec_fixture a,b; float *res0,*mlp0,*out0,*res1,*mlp1,*out1; int ok;
    if (argc != 3) { fprintf(stderr, "usage: %s LAYER0.bin LAYER1.bin\n", argv[0]); return 1; }
    ok = fixture_load(argv[1], &a) && fixture_load(argv[2], &b);
    if (!ok) { fprintf(stderr, "failed to load fixtures\n"); return 1; }
    res0=calloc(a.hidden,sizeof(float)); mlp0=calloc(a.hidden,sizeof(float)); out0=calloc(a.hidden,sizeof(float));
    res1=calloc(b.hidden,sizeof(float)); mlp1=calloc(b.hidden,sizeof(float)); out1=calloc(b.hidden,sizeof(float));
    run_layer(&a, NULL, res0, mlp0, out0);
    run_layer(&b, out0, res1, mlp1, out1);
    printf("layer0_output_rmse: %.8f\n", vec_rmse(out0, a.layer_output_ref, a.hidden));
    printf("layer0_output_cosine: %.8f\n", vec_cosine(out0, a.layer_output_ref, a.hidden));
    printf("layer1_input_handoff_rmse: %.8f\n", vec_rmse(out0, b.layer_input, b.hidden));
    printf("layer1_input_handoff_cosine: %.8f\n", vec_cosine(out0, b.layer_input, b.hidden));
    printf("layer1_output_rmse: %.8f\n", vec_rmse(out1, b.layer_output_ref, b.hidden));
    printf("layer1_output_cosine: %.8f\n", vec_cosine(out1, b.layer_output_ref, b.hidden));
    fixture_free(&a); fixture_free(&b); free(res0); free(mlp0); free(out0); free(res1); free(mlp1); free(out1);
    return 0;
}
