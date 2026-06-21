#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

struct args_t {
    std::string model_path;
    std::string prompt;
    std::string prompt_file;
    std::string force_tokens_csv;
    int n_predict = 24;
    int top_k = 10;
    int ctx_size = 4096;
    int threads = 8;
};

struct token_view {
    llama_token id;
    std::vector<unsigned char> bytes;
    float logit;
    float logprob;
};

struct step_view {
    token_view argmax;
    token_view chosen;
    std::vector<token_view> top;
    bool forced;
};

static void null_log_callback(ggml_log_level, const char *, void *) {
}

static void usage(const char * prog) {
    std::fprintf(stderr,
            "usage: %s --model model.gguf [--prompt TEXT | --prompt-file FILE] [--force-tokens-csv ids] [--n-predict N] [--top-k K] [--ctx-size N] [--threads N]\n",
            prog);
}

static bool parse_args(int argc, char ** argv, args_t & out) {
    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        if (std::strcmp(arg, "--model") == 0 && i + 1 < argc) {
            out.model_path = argv[++i];
        } else if (std::strcmp(arg, "--prompt") == 0 && i + 1 < argc) {
            out.prompt = argv[++i];
        } else if (std::strcmp(arg, "--prompt-file") == 0 && i + 1 < argc) {
            out.prompt_file = argv[++i];
        } else if (std::strcmp(arg, "--force-tokens-csv") == 0 && i + 1 < argc) {
            out.force_tokens_csv = argv[++i];
        } else if (std::strcmp(arg, "--n-predict") == 0 && i + 1 < argc) {
            out.n_predict = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--top-k") == 0 && i + 1 < argc) {
            out.top_k = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--ctx-size") == 0 && i + 1 < argc) {
            out.ctx_size = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--threads") == 0 && i + 1 < argc) {
            out.threads = std::atoi(argv[++i]);
        } else {
            return false;
        }
    }
    return !out.model_path.empty() && (!out.prompt.empty() || !out.prompt_file.empty());
}

static void parse_force_tokens_csv(const std::string & s, std::vector<llama_token> & out) {
    size_t start = 0;
    while (start < s.size()) {
        size_t end = s.find(',', start);
        std::string item = s.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!item.empty()) out.push_back((llama_token) std::strtol(item.c_str(), nullptr, 10));
        if (end == std::string::npos) break;
        start = end + 1;
    }
}

static bool read_file(const std::string & path, std::string & out) {
    FILE * fp = std::fopen(path.c_str(), "rb");
    long size;
    size_t nread;
    if (!fp) return false;
    if (std::fseek(fp, 0, SEEK_END) != 0) {
        std::fclose(fp);
        return false;
    }
    size = std::ftell(fp);
    if (size < 0) {
        std::fclose(fp);
        return false;
    }
    if (std::fseek(fp, 0, SEEK_SET) != 0) {
        std::fclose(fp);
        return false;
    }
    out.resize((size_t) size);
    nread = std::fread(out.data(), 1, out.size(), fp);
    std::fclose(fp);
    if (nread != out.size()) return false;
    return true;
}

static void json_escape(std::ostream & os, const std::string & s) {
    os << '"';
    for (unsigned char c : s) {
        switch (c) {
            case '\\': os << "\\\\"; break;
            case '"':  os << "\\\""; break;
            case '\b': os << "\\b"; break;
            case '\f': os << "\\f"; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    os << buf;
                } else {
                    os << (char) c;
                }
                break;
        }
    }
    os << '"';
}

static std::vector<unsigned char> token_bytes(const llama_vocab * vocab, llama_token id) {
    std::vector<char> buf(64);
    int32_t n = llama_token_to_piece(vocab, id, buf.data(), (int32_t) buf.size(), 0, true);
    while (n < 0) {
        buf.resize((size_t) (-n));
        n = llama_token_to_piece(vocab, id, buf.data(), (int32_t) buf.size(), 0, true);
    }
    return std::vector<unsigned char>(buf.begin(), buf.begin() + n);
}

static void print_bytes_json(std::ostream & os, const std::vector<unsigned char> & bytes) {
    os << '[';
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (i) os << ',';
        os << (unsigned int) bytes[i];
    }
    os << ']';
}

static void print_token_json(std::ostream & os, const token_view & tok) {
    os << "{\"id\":" << tok.id << ",\"bytes\":";
    print_bytes_json(os, tok.bytes);
    os << ",\"logit\":" << tok.logit << ",\"logprob\":" << tok.logprob << '}';
}

static std::vector<token_view> top_tokens(const llama_vocab * vocab, const float * logits, int n_vocab, int top_k) {
    float max_logit = logits[0];
    double sumexp = 0.0;
    std::vector<token_view> top;
    top.reserve((size_t) top_k);

    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    for (int i = 0; i < n_vocab; ++i) {
        sumexp += std::exp((double) logits[i] - max_logit);
        if ((int) top.size() < top_k) {
            top.push_back(token_view{(llama_token) i, token_bytes(vocab, i), logits[i], 0.0f});
            continue;
        }
        size_t worst = 0;
        for (size_t j = 1; j < top.size(); ++j) {
            if (top[j].logit < top[worst].logit) worst = j;
        }
        if (logits[i] > top[worst].logit) {
            top[worst] = token_view{(llama_token) i, token_bytes(vocab, i), logits[i], 0.0f};
        }
    }
    const double log_denom = std::log(sumexp);
    std::sort(top.begin(), top.end(), [](const token_view & a, const token_view & b) {
        return a.logit > b.logit;
    });
    for (token_view & tok : top) {
        tok.logprob = (float) (((double) tok.logit - max_logit) - log_denom);
    }
    return top;
}

static token_view token_from_logits(const llama_vocab * vocab, const float * logits, int n_vocab, llama_token id) {
    float max_logit = logits[0];
    double sumexp = 0.0;
    int i;

    for (i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    for (i = 0; i < n_vocab; ++i) {
        sumexp += std::exp((double) logits[i] - max_logit);
    }

    return token_view{
        id,
        token_bytes(vocab, id),
        logits[id],
        (float) (((double) logits[id] - max_logit) - std::log(sumexp))
    };
}

static void print_token_list_json(std::ostream & os, const llama_vocab * vocab, const std::vector<llama_token> & ids) {
    os << '[';
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) os << ',';
        token_view tok{ids[i], token_bytes(vocab, ids[i]), 0.0f, 0.0f};
        print_token_json(os, tok);
    }
    os << ']';
}

int main(int argc, char ** argv) {
    args_t args;
    llama_log_set(null_log_callback, nullptr);
    ggml_backend_load_all();

    if (!parse_args(argc, argv, args)) {
        usage(argv[0]);
        return 2;
    }
    if (!args.prompt_file.empty() && !read_file(args.prompt_file, args.prompt)) {
        std::fprintf(stderr, "failed to read prompt file: %s\n", args.prompt_file.c_str());
        return 1;
    }
    std::vector<llama_token> forced_tokens;
    if (!args.force_tokens_csv.empty()) {
        parse_force_tokens_csv(args.force_tokens_csv, forced_tokens);
    }

    llama_model_params mparams = llama_model_default_params();
    llama_context_params cparams = llama_context_default_params();
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    const llama_vocab * vocab;
    std::vector<llama_token> prompt_tokens;
    std::vector<llama_token> generated;
    std::vector<step_view> steps;

    mparams.n_gpu_layers = 0;
    model = llama_model_load_from_file(args.model_path.c_str(), mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model: %s\n", args.model_path.c_str());
        return 1;
    }

    vocab = llama_model_get_vocab(model);
    {
        const bool add_bos = llama_vocab_get_add_bos(vocab);
        const int n_prompt = -llama_tokenize(vocab, args.prompt.c_str(), (int32_t) args.prompt.size(), nullptr, 0, add_bos, true);
        if (n_prompt <= 0) {
            std::fprintf(stderr, "failed to size prompt tokenization\n");
            llama_model_free(model);
            return 1;
        }
        prompt_tokens.resize((size_t) n_prompt);
        if (llama_tokenize(vocab, args.prompt.c_str(), (int32_t) args.prompt.size(),
                    prompt_tokens.data(), (int32_t) prompt_tokens.size(), add_bos, true) < 0) {
            std::fprintf(stderr, "failed to tokenize prompt\n");
            llama_model_free(model);
            return 1;
        }
    }

    cparams.n_ctx = (uint32_t) std::max(args.ctx_size, (int) prompt_tokens.size() + args.n_predict + 16);
    cparams.n_batch = (uint32_t) prompt_tokens.size();
    cparams.n_threads = args.threads;
    cparams.n_threads_batch = args.threads;
    cparams.no_perf = true;

    ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t) prompt_tokens.size());
    llama_token next = LLAMA_TOKEN_NULL;
    const int n_vocab = llama_vocab_n_tokens(vocab);

    for (int step = 0; step < args.n_predict; ++step) {
        const float * logits;
        std::vector<token_view> top;
        step_view sv;
        if (llama_decode(ctx, batch) != 0) {
            std::fprintf(stderr, "llama_decode failed at step %d\n", step);
            llama_free(ctx);
            llama_model_free(model);
            return 1;
        }
        logits = llama_get_logits(ctx);
        top = top_tokens(vocab, logits, n_vocab, args.top_k);
        if (top.empty()) break;
        sv.argmax = top[0];
        sv.top = top;
        sv.forced = step < (int) forced_tokens.size();
        next = sv.forced ? forced_tokens[(size_t) step] : sv.argmax.id;
        sv.chosen = sv.forced ? token_from_logits(vocab, logits, n_vocab, next) : sv.argmax;
        steps.push_back(sv);
        if (llama_vocab_is_eog(vocab, next)) break;
        generated.push_back(next);
        batch = llama_batch_get_one(&generated.back(), 1);
    }

    std::cout << '{';
    std::cout << "\"model_path\":";
    json_escape(std::cout, args.model_path);
    std::cout << ",\"prompt\":";
    json_escape(std::cout, args.prompt);
    std::cout << ",\"prompt_tokens\":";
    print_token_list_json(std::cout, vocab, prompt_tokens);
    std::cout << ",\"steps\":[";
    for (size_t i = 0; i < steps.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << "{\"step\":" << i << ",\"argmax\":";
        print_token_json(std::cout, steps[i].argmax);
        std::cout << ",\"forced\":" << (steps[i].forced ? "true" : "false");
        std::cout << ",\"chosen\":";
        print_token_json(std::cout, steps[i].chosen);
        std::cout << ",\"top\":[";
        for (size_t j = 0; j < steps[i].top.size(); ++j) {
            if (j) std::cout << ',';
            print_token_json(std::cout, steps[i].top[j]);
        }
        std::cout << "]}";
    }
    std::cout << ']';
    std::cout << ",\"generated_tokens\":";
    print_token_list_json(std::cout, vocab, generated);
    std::cout << "}\n";

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
