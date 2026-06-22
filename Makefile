CC ?= cc
CXX ?= c++
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
NATIVE_CPU_FLAG ?= -mcpu=native
else
NATIVE_CPU_FLAG ?= -march=native
endif

DEBUG_FLAGS ?= -g
CFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c99
CXXFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -std=c++17
OBJCFLAGS ?= -O3 -ffast-math $(DEBUG_FLAGS) $(NATIVE_CPU_FLAG) -Wall -Wextra -fobjc-arc

LDLIBS ?= -lm -pthread
METAL_SRCS := $(wildcard metal/*.metal)
ROCM_SRCS := $(wildcard rocm/*.cuh)

ifeq ($(UNAME_S),Darwin)
METAL_LDLIBS := $(LDLIBS) -framework Foundation -framework Metal
CORE_OBJS = ds4.o ds4_distributed.o ds4_ssd.o ds4_metal.o
CPU_CORE_OBJS = ds4_cpu.o ds4_distributed.o ds4_ssd.o
else
CFLAGS += -D_GNU_SOURCE -fno-finite-math-only
CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?=
ifneq ($(strip $(CUDA_ARCH)),)
NVCC_ARCH_FLAGS := -arch=$(CUDA_ARCH)
endif
NVCCFLAGS ?= -O3 -g -lineinfo --use_fast_math $(NVCC_ARCH_FLAGS) -Xcompiler $(NATIVE_CPU_FLAG) -Xcompiler -pthread
CORE_OBJS = ds4.o ds4_distributed.o ds4_ssd.o ds4_cuda.o
CPU_CORE_OBJS = ds4_cpu.o ds4_distributed.o ds4_ssd.o
CUDA_LDLIBS ?= -lm -Xcompiler -pthread -L$(CUDA_HOME)/targets/sbsa-linux/lib -L$(CUDA_HOME)/lib64 -lcudart -lcublas
HIPCC ?= $(shell command -v hipcc 2>/dev/null || echo /opt/rocm/bin/hipcc)
ROCM_ARCH ?= gfx1100
ROCM_CFLAGS ?= -O3 -ffast-math -g -fno-finite-math-only -pthread -D__HIP_PLATFORM_AMD__ -Wno-unused-command-line-argument --offload-arch=$(ROCM_ARCH)
ROCM_LDLIBS ?= -lm -pthread -lhipblas -lhipblaslt
ROCWMMA_INC ?=
ifneq ($(strip $(ROCWMMA_INC)),)
ROCM_CFLAGS += -I$(ROCWMMA_INC)
endif
DS4_LINK ?= $(NVCC) $(NVCCFLAGS)
DS4_LINK_LIBS ?= $(CUDA_LDLIBS)
METAL_LDLIBS := $(LDLIBS)
endif

LLAMA_CPP_DIR ?= /mnt/f/git/llama.cpp
LLAMA_CPP_BUILD ?= $(LLAMA_CPP_DIR)/build
LLAMA_CPP_INC ?= -I$(LLAMA_CPP_DIR)/include -I$(LLAMA_CPP_DIR)/ggml/include
LLAMA_CPP_LIBDIR ?= $(LLAMA_CPP_BUILD)/bin
LLAMA_CPP_LDLIBS ?= -L$(LLAMA_CPP_LIBDIR) -Wl,-rpath,$(LLAMA_CPP_LIBDIR) -lllama -lggml -lm -pthread -ldl

.PHONY: all help clean test cpu cuda cuda-spark cuda-generic cuda-regression strix-halo rocm qwen36-check qwen36-check-q4xl qwen36-check-iq2xxs qwen36-check-v0 qwen36-plan qwen36-runtime-dump qwen36-oracle qwen36-v0-quantize qwen36-v0-quant-lib qwen36-c-moe-replay qwen36-c-linear-replay qwen36-c-linear-conv-replay qwen36-c-linear-core-replay qwen36-c-linear-norm-replay qwen36-c-linear-layer-stub qwen36-c-linear-layer-full qwen36-c-linear-layer-weight qwen36-c-root qwen36-c-root-q8 qwen36-c-decoder-layer qwen36-c-decoder-layer-weight qwen36-c-two-decoder-layers qwen36-c-two-decoder-layers-weight qwen36-c-decoder-chain-weight qwen36-c-prefix-q8-chain-dynamic qwen36-c-full-layer-q8-dynamic qwen36-direct-hybrid-contract-check qwen36-gpu-oracle-scaffold qwen36-gpu-blk0-ffn-q8-oracle qwen36-gpu-blk0-dynamic-q8-oracle qwen36-gpu-hybrid-layer-q8-dynamic qwen36-gpu-prefix-q8-chain-dynamic qwen36-gpu-full-layer-q8-dynamic qwen36-fixture-blk0-worker

ifeq ($(UNAME_S),Darwin)
all: ds4 ds4-server ds4-bench ds4-eval ds4-agent

help:
	@echo "DS4 build targets:"
	@echo "  make              Build Metal ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make cpu          Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make qwen36-check Build the narrow Qwen3.6-35B-A3B Q8_0 contract checker"
	@echo "  make qwen36-check-q4xl Build the narrow Qwen3.6-35B-A3B UD_Q4_K_XL contract checker"
	@echo "  make qwen36-check-iq2xxs Build the narrow Qwen3.6-35B-A3B UD_IQ2_XXS contract checker"
	@echo "  make qwen36-check-v0 Build the narrow Qwen3.6-35B-A3B DS4Style_v0 contract checker"
	@echo "  make qwen36-plan Build the narrow Qwen3.6 plan dump tool"
	@echo "  make qwen36-runtime-dump Build the narrow Qwen3.6 runtime skeleton dumper"
	@echo "  make qwen36-oracle Build the llama.cpp-backed Qwen3.6 token/logit oracle"
	@echo "  make qwen36-v0-quantize Build the narrow Qwen3.6 v0 quantizer scaffold"
	@echo "  make qwen36-v0-quant-lib Build the local shared quantization bridge for Python export tooling"
	@echo "  make test         Build and run tests"
	@echo "  make clean        Remove build outputs"

ds4: ds4_cli.o ds4_help.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_cli.o ds4_help.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_server.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-bench: ds4_bench.o ds4_help.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_bench.o ds4_help.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-eval: ds4_eval.o ds4_help.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_eval.o ds4_help.o $(CORE_OBJS) $(METAL_LDLIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(CC) $(CFLAGS) -o $@ ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

cuda-regression:
	@echo "cuda-regression requires a CUDA build"
else
all: help

help:
	@echo "DS4 build targets:"
	@echo "  make cuda-spark          Build CUDA for DGX Spark / GB10"
	@echo "  make cuda-generic        Build CUDA for a generic local CUDA GPU"
	@echo "  make cuda CUDA_ARCH=sm_N Build CUDA with an explicit nvcc -arch value"
	@echo "  make rocm                Build ROCm for a generic AMD GPU (default gfx1100)"
	@echo "  make strix-halo          Build ROCm for Strix Halo / gfx1151"
	@echo "       optional: ROCWMMA_INC=/path/to/rocwmma/library/include"
	@echo "  make cpu                 Build CPU-only ./ds4, ./ds4-server, ./ds4-bench, ./ds4-eval, and ./ds4-agent"
	@echo "  make qwen36-check        Build the narrow Qwen3.6-35B-A3B Q8_0 contract checker"
	@echo "  make qwen36-check-q4xl   Build the narrow Qwen3.6-35B-A3B UD_Q4_K_XL contract checker"
	@echo "  make qwen36-check-iq2xxs Build the narrow Qwen3.6-35B-A3B UD_IQ2_XXS contract checker"
	@echo "  make qwen36-check-v0     Build the narrow Qwen3.6-35B-A3B DS4Style_v0 contract checker"
	@echo "  make qwen36-plan         Build the narrow Qwen3.6 plan dump tool"
	@echo "  make qwen36-runtime-dump Build the narrow Qwen3.6 runtime skeleton dumper"
	@echo "  make qwen36-oracle       Build the llama.cpp-backed Qwen3.6 token/logit oracle"
	@echo "  make qwen36-v0-quantize  Build the narrow Qwen3.6 v0 quantizer scaffold"
	@echo "  make qwen36-v0-quant-lib Build the local shared quantization bridge for Python export tooling"
	@echo "  make qwen36-c-moe-replay Build the narrow Qwen3.6 C MoE replay validator"
	@echo "  make qwen36-c-linear-replay Build the narrow Qwen3.6 C linear-attention projection validator"
	@echo "  make qwen36-c-linear-conv-replay Build the narrow Qwen3.6 C linear-attention conv validator"
	@echo "  make qwen36-c-linear-core-replay Build the narrow Qwen3.6 C DeltaNet core-boundary validator"
	@echo "  make qwen36-c-linear-norm-replay Build the narrow Qwen3.6 C gated RMS norm validator"
	@echo "  make qwen36-c-linear-layer-stub Build the narrow Qwen3.6 C stubbed full linear-attention layer validator"
	@echo "  make qwen36-c-linear-layer-full Build the narrow Qwen3.6 C full linear-attention layer validator"
	@echo "  make qwen36-c-linear-layer-weight Build the narrow Qwen3.6 C weight-driven linear-attention layer validator"
	@echo "  make qwen36-c-root Build the narrow Qwen3.6 C root-surface validator"
	@echo "  make qwen36-c-root-q8 Build the narrow Qwen3.6 C root validator with direct GGUF Q8 row reads"
	@echo "  make qwen36-c-decoder-layer Build the narrow Qwen3.6 C full decoder-layer validator"
	@echo "  make qwen36-c-decoder-layer-weight Build the narrow Qwen3.6 C weight-driven decoder-layer validator"
	@echo "  make qwen36-c-two-decoder-layers Build the narrow Qwen3.6 C two-layer decoder-chain validator"
	@echo "  make qwen36-c-two-decoder-layers-weight Build the narrow Qwen3.6 C two-layer weight-driven decoder-chain validator"
	@echo "  make qwen36-c-decoder-chain-weight Build the narrow Qwen3.6 C generic weight-driven decoder-chain validator"
	@echo "  make qwen36-c-prefix-q8-chain-dynamic Build the narrow Qwen3.6 dynamic prefix chain for token_embd+blk0..N"
	@echo "  make qwen36-c-full-layer-q8-dynamic Build the narrow Qwen3.6 dynamic full-attention+MoE layer runner"
	@echo "  make qwen36-direct-hybrid-contract-check Compare direct GGUF hybrid tensors against a traced fixture"
	@echo "  make qwen36-gpu-oracle-scaffold Build the narrow Qwen3.6 HIP/ROCm oracle scaffold"
	@echo "  make qwen36-gpu-blk0-ffn-q8-oracle Build the narrow Qwen3.6 GPU blk.0 FFN closure oracle"
	@echo "  make qwen36-gpu-blk0-dynamic-q8-oracle Build the narrow Qwen3.6 dynamic blk.0 GPU oracle scaffold"
	@echo "  make qwen36-gpu-hybrid-layer-q8-dynamic Build the generic Qwen3.6 GPU hybrid-layer runner"
	@echo "  make qwen36-gpu-prefix-q8-chain-dynamic Build the GPU-owned Qwen3.6 hybrid prefix chain"
	@echo "  make qwen36-gpu-full-layer-q8-dynamic Build the narrow Qwen3.6 GPU full-attention+MoE layer runner"
	@echo "  make qwen36-fixture-blk0-worker Build the persistent fixture-backed blk.0 worker"
	@echo "  make test                Build and run tests"
	@echo "  make clean               Remove build outputs"

cuda-spark:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=

cuda-generic:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH=native

cuda:
	@if [ -z "$(strip $(CUDA_ARCH))" ]; then \
		echo "error: specify CUDA_ARCH, for example: make cuda CUDA_ARCH=sm_120"; \
		echo "       or use make cuda-spark / make cuda-generic"; \
		exit 2; \
	fi
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent CUDA_ARCH="$(CUDA_ARCH)"

strix-halo:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent \
		CORE_OBJS="ds4.o ds4_distributed.o ds4_ssd.o ds4_rocm.o" \
		CFLAGS="$(CFLAGS) -DDS4_ROCM_BUILD" \
		ROCM_ARCH=gfx1151 \
		DS4_LINK="$(HIPCC) $(ROCM_CFLAGS)" \
		DS4_LINK_LIBS="$(ROCM_LDLIBS)"

rocm:
	$(MAKE) -B ds4 ds4-server ds4-bench ds4-eval ds4-agent \
		CORE_OBJS="ds4.o ds4_distributed.o ds4_ssd.o ds4_rocm.o" \
		CFLAGS="$(CFLAGS) -DDS4_ROCM_BUILD" \
		DS4_LINK="$(HIPCC) $(ROCM_CFLAGS)" \
		DS4_LINK_LIBS="$(ROCM_LDLIBS)"

ds4: ds4_cli.o ds4_help.o linenoise.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-server: ds4_server.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-bench: ds4_bench.o ds4_help.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-eval: ds4_eval.o ds4_help.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

ds4-agent: ds4_agent.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
	$(DS4_LINK) -o $@ $^ $(DS4_LINK_LIBS)

cpu: ds4_cli_cpu.o ds4_server_cpu.o ds4_bench_cpu.o ds4_eval_cpu.o ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o rax.o $(CPU_CORE_OBJS)
	$(CC) $(CFLAGS) -o ds4 ds4_cli_cpu.o ds4_help.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-server ds4_server_cpu.o ds4_help.o ds4_kvstore.o rax.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-bench ds4_bench_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-eval ds4_eval_cpu.o ds4_help.o $(CPU_CORE_OBJS) $(LDLIBS)
	$(CC) $(CFLAGS) -o ds4-agent ds4_agent_cpu.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CPU_CORE_OBJS) $(LDLIBS)

qwen36-check: qwen36-35a3b-q8-check

qwen36-check-q4xl: qwen36-35a3b-q4xl-check

qwen36-check-iq2xxs: qwen36-35a3b-iq2xxs-check

qwen36-check-v0: qwen36-35a3b-v0-check

qwen36-plan: qwen36-plan-dump

qwen36-runtime-dump: qwen36-runtime-dump-bin

qwen36-oracle: qwen36-llama-oracle

qwen36-v0-quantize: qwen36-v0-quantize-bin

qwen36-v0-quant-lib: gguf-tools/libds4q.so


cuda-regression: tests/cuda_long_context_smoke
	./tests/cuda_long_context_smoke
endif

ds4.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -c -o $@ ds4.c

ds4_ssd.o: ds4_ssd.c ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_ssd.c

ds4_cli.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_cli.c

ds4_distributed.o: ds4_distributed.c ds4_distributed.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_distributed.c

ds4_help.o: ds4_help.c ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_help.c

ds4_server.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -c -o $@ ds4_server.c

ds4_bench.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_bench.c

ds4_eval.o: ds4_eval.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -c -o $@ ds4_eval.c

ds4_agent.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -c -o $@ ds4_agent.c

ds4_web.o: ds4_web.c ds4_web.h
	$(CC) $(CFLAGS) -c -o $@ ds4_web.c

ds4_kvstore.o: ds4_kvstore.c ds4_kvstore.h ds4.h ds4_ssd.h
	$(CC) $(CFLAGS) -c -o $@ ds4_kvstore.c

ds4_test.o: tests/ds4_test.c ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_test.c

ds4_agent_test.o: tests/ds4_agent_test.c ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -Wno-unused-function -c -o $@ tests/ds4_agent_test.c

tests/cuda_long_context_smoke.o: tests/cuda_long_context_smoke.c ds4_gpu.h
	$(CC) $(CFLAGS) -I. -c -o $@ tests/cuda_long_context_smoke.c

rax.o: rax.c rax.h rax_malloc.h
	$(CC) $(CFLAGS) -c -o $@ rax.c

linenoise.o: linenoise.c linenoise.h
	$(CC) $(CFLAGS) -c -o $@ linenoise.c

ds4_cpu.o: ds4.c ds4.h ds4_ssd.h ds4_distributed.h ds4_gpu.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4.c

ds4_cli_cpu.o: ds4_cli.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_cli.c

ds4_server_cpu.o: ds4_server.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h rax.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_server.c

ds4_bench_cpu.o: ds4_bench.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_bench.c

ds4_eval_cpu.o: ds4_eval.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_eval.c

ds4_agent_cpu.o: ds4_agent.c ds4.h ds4_ssd.h ds4_distributed.h ds4_help.h ds4_kvstore.h ds4_web.h linenoise.h
	$(CC) $(CFLAGS) -DDS4_NO_GPU -c -o $@ ds4_agent.c

qwen36_gguf.o: qwen36_gguf.c qwen36_gguf.h
	$(CC) $(CFLAGS) -c -o $@ qwen36_gguf.c

qwen36_35a3b_q8.o: qwen36_35a3b_q8.c qwen36_35a3b_q8.h qwen36_gguf.h
	$(CC) $(CFLAGS) -c -o $@ qwen36_35a3b_q8.c

qwen36_35a3b_q8_check.o: qwen36_35a3b_q8_check.c qwen36_35a3b_q8.h qwen36_gguf.h
	$(CC) $(CFLAGS) -c -o $@ qwen36_35a3b_q8_check.c

qwen36_35a3b_q4xl.o: qwen36_35a3b_q4xl.c qwen36_35a3b_q4xl.h qwen36_gguf.h
	$(CC) $(CFLAGS) -c -o $@ qwen36_35a3b_q4xl.c

qwen36_35a3b_iq2xxs.o: qwen36_35a3b_iq2xxs.c qwen36_35a3b_iq2xxs.h qwen36_gguf.h
	$(CC) $(CFLAGS) -c -o $@ qwen36_35a3b_iq2xxs.c

qwen36_35a3b_v0.o: qwen36_35a3b_v0.c qwen36_35a3b_v0.h qwen36_gguf.h
	$(CC) $(CFLAGS) -c -o $@ qwen36_35a3b_v0.c

qwen36_35a3b_q4xl_check.o: qwen36_35a3b_q4xl_check.c qwen36_35a3b_q4xl.h qwen36_gguf.h
	$(CC) $(CFLAGS) -c -o $@ qwen36_35a3b_q4xl_check.c

qwen36_35a3b_iq2xxs_check.o: qwen36_35a3b_iq2xxs_check.c qwen36_35a3b_iq2xxs.h qwen36_gguf.h
	$(CC) $(CFLAGS) -c -o $@ qwen36_35a3b_iq2xxs_check.c

qwen36_35a3b_v0_check.o: qwen36_35a3b_v0_check.c qwen36_35a3b_v0.h qwen36_gguf.h
	$(CC) $(CFLAGS) -c -o $@ qwen36_35a3b_v0_check.c

qwen36_plan_dump.o: qwen36_plan_dump.c qwen36_35a3b_q8.h qwen36_35a3b_q4xl.h qwen36_gguf.h
	$(CC) $(CFLAGS) -c -o $@ qwen36_plan_dump.c

qwen36_runtime.o: qwen36_runtime.c qwen36_runtime.h qwen36_35a3b_q8.h qwen36_35a3b_q4xl.h qwen36_gguf.h
	$(CC) $(CFLAGS) -c -o $@ qwen36_runtime.c

qwen36_runtime_dump.o: qwen36_runtime_dump.c qwen36_runtime.h qwen36_35a3b_q8.h qwen36_35a3b_q4xl.h qwen36_gguf.h
	$(CC) $(CFLAGS) -c -o $@ qwen36_runtime_dump.c

qwen36_c_moe_replay.o: qwen36_c_moe_replay.c
	$(CC) $(CFLAGS) -c -o $@ qwen36_c_moe_replay.c

qwen36_c_linear_replay.o: qwen36_c_linear_replay.c
	$(CC) $(CFLAGS) -c -o $@ qwen36_c_linear_replay.c

qwen36_c_linear_conv_replay.o: qwen36_c_linear_conv_replay.c
	$(CC) $(CFLAGS) -c -o $@ qwen36_c_linear_conv_replay.c

qwen36_c_linear_core_replay.o: qwen36_c_linear_core_replay.c
	$(CC) $(CFLAGS) -c -o $@ qwen36_c_linear_core_replay.c

qwen36_c_linear_norm_replay.o: qwen36_c_linear_norm_replay.c
	$(CC) $(CFLAGS) -c -o $@ qwen36_c_linear_norm_replay.c

qwen36_c_linear_layer_stub_replay.o: qwen36_c_linear_layer_stub_replay.c
	$(CC) $(CFLAGS) -c -o $@ qwen36_c_linear_layer_stub_replay.c

qwen36_c_linear_layer_full_replay.o: qwen36_c_linear_layer_full_replay.c
	$(CC) $(CFLAGS) -c -o $@ qwen36_c_linear_layer_full_replay.c

qwen36_c_decoder_layer_replay.o: qwen36_c_decoder_layer_replay.c
	$(CC) $(CFLAGS) -c -o $@ qwen36_c_decoder_layer_replay.c

qwen36_gpu_blk0_dynamic_q8_oracle.o: qwen36_gpu_blk0_dynamic_q8_oracle.c
	$(CC) $(CFLAGS) -c -o $@ qwen36_gpu_blk0_dynamic_q8_oracle.c

qwen36_fixture_blk0_worker.o: qwen36_fixture_blk0_worker.c qwen36_35a3b_q8.h qwen36_gguf.h
	$(CC) $(CFLAGS) -c -o $@ qwen36_fixture_blk0_worker.c

qwen36_llama_oracle.o: qwen36_llama_oracle.cpp
	$(CXX) $(CXXFLAGS) $(LLAMA_CPP_INC) -c -o $@ qwen36_llama_oracle.cpp

qwen36_v0_quantize.o: gguf-tools/qwen36-v0-quantize.c qwen36_35a3b_q8.h qwen36_gguf.h gguf-tools/quants.h
	$(CC) $(CFLAGS) -c -o $@ gguf-tools/qwen36-v0-quantize.c

gguf_tools_quants.o: gguf-tools/quants.c gguf-tools/quants.h
	$(CC) $(CFLAGS) -c -o $@ gguf-tools/quants.c

gguf-tools/libds4q.so: gguf-tools/quants.c gguf-tools/quants.h
	$(CC) $(CFLAGS) -shared -fPIC -o $@ gguf-tools/quants.c $(LDLIBS)

qwen36-35a3b-q8-check: qwen36_35a3b_q8_check.o qwen36_35a3b_q8.o qwen36_gguf.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-35a3b-q4xl-check: qwen36_35a3b_q4xl_check.o qwen36_35a3b_q4xl.o qwen36_gguf.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-35a3b-iq2xxs-check: qwen36_35a3b_iq2xxs_check.o qwen36_35a3b_iq2xxs.o qwen36_gguf.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-35a3b-v0-check: qwen36_35a3b_v0_check.o qwen36_35a3b_v0.o qwen36_gguf.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-plan-dump: qwen36_plan_dump.o qwen36_35a3b_q8.o qwen36_35a3b_q4xl.o qwen36_gguf.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-runtime-dump-bin: qwen36_runtime_dump.o qwen36_runtime.o qwen36_35a3b_q8.o qwen36_35a3b_q4xl.o qwen36_gguf.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-llama-oracle: qwen36_llama_oracle.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LLAMA_CPP_LDLIBS)

qwen36-v0-quantize-bin: qwen36_v0_quantize.o qwen36_35a3b_q8.o qwen36_gguf.o gguf_tools_quants.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-c-moe-replay: qwen36_c_moe_replay.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-c-linear-replay: qwen36_c_linear_replay.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-c-linear-conv-replay: qwen36_c_linear_conv_replay.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-c-linear-core-replay: qwen36_c_linear_core_replay.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-c-linear-norm-replay: qwen36_c_linear_norm_replay.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-c-linear-layer-stub: qwen36_c_linear_layer_stub_replay.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-c-linear-layer-full: qwen36_c_linear_layer_full_replay.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-c-linear-layer-weight: qwen36_c_linear_layer_weight_replay.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-c-root: qwen36_c_root_replay.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-c-root-q8: qwen36_c_root_q8_selected_logits.o qwen36_35a3b_q8.o qwen36_gguf.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-q8-attn-qkv-audit: qwen36_q8_attn_qkv_audit.o qwen36_35a3b_q8.o qwen36_gguf.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-c-decoder-layer: qwen36_c_decoder_layer_replay.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-c-decoder-layer-weight: qwen36_c_decoder_layer_weight_replay.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-c-two-decoder-layers: qwen36_c_two_decoder_layers_replay.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-c-two-decoder-layers-weight: qwen36_c_two_decoder_layers_weight_replay.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-c-decoder-chain-weight: qwen36_c_decoder_chain_weight_replay.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-c-prefix-q8-chain: qwen36_c_prefix_q8_chain_replay.o qwen36_35a3b_q8.o qwen36_gguf.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-c-prefix-q8-chain-dynamic: qwen36_c_prefix_q8_chain_dynamic.o qwen36_35a3b_q8.o qwen36_gguf.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-fixture-blk0-worker: qwen36_fixture_blk0_worker.o qwen36_35a3b_q8.o qwen36_gguf.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-c-full-layer-q8-dynamic: qwen36_c_full_layer_q8_dynamic.o qwen36_35a3b_q8.o qwen36_gguf.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-direct-hybrid-contract-check: qwen36_direct_hybrid_contract_check.o qwen36_35a3b_q8.o qwen36_gguf.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

qwen36-gpu-oracle-scaffold: qwen36_gpu_oracle_scaffold.o qwen36_runtime.o qwen36_35a3b_q8.o qwen36_35a3b_q4xl.o qwen36_gguf.o ds4.o ds4_distributed.o ds4_ssd.o ds4_rocm.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

qwen36-gpu-blk0-ffn-q8-oracle: qwen36_gpu_blk0_ffn_q8_oracle.o qwen36_35a3b_q8.o qwen36_gguf.o ds4.o ds4_distributed.o ds4_ssd.o ds4_rocm.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

qwen36-gpu-blk0-dynamic-q8-oracle: qwen36_gpu_blk0_dynamic_q8_oracle.o qwen36_35a3b_q8.o qwen36_gguf.o ds4.o ds4_distributed.o ds4_ssd.o ds4_rocm.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

qwen36-gpu-hybrid-layer-q8-dynamic: qwen36_gpu_blk0_dynamic_q8_oracle.o qwen36_35a3b_q8.o qwen36_gguf.o ds4.o ds4_distributed.o ds4_ssd.o ds4_rocm.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

qwen36-gpu-prefix-q8-chain-dynamic: qwen36_gpu_blk0_dynamic_q8_oracle.o qwen36_35a3b_q8.o qwen36_gguf.o ds4.o ds4_distributed.o ds4_ssd.o ds4_rocm.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

qwen36-gpu-full-layer-q8-dynamic: qwen36_gpu_full_layer_q8_dynamic.o qwen36_35a3b_q8.o qwen36_gguf.o ds4.o ds4_distributed.o ds4_ssd.o ds4_rocm.o
	$(HIPCC) $(ROCM_CFLAGS) -o $@ $^ $(ROCM_LDLIBS)

ds4_metal.o: ds4_metal.m ds4_gpu.h $(METAL_SRCS)
	$(CC) $(OBJCFLAGS) -c -o $@ ds4_metal.m

ds4_cuda.o: ds4_cuda.cu ds4_gpu.h ds4_iq2_tables_cuda.inc
	$(NVCC) $(NVCCFLAGS) -c -o $@ ds4_cuda.cu

ds4_rocm.o: ds4_rocm.cu ds4_gpu.h ds4_iq2_tables_cuda.inc $(ROCM_SRCS)
	$(HIPCC) $(ROCM_CFLAGS) -c -o $@ ds4_rocm.cu

tests/cuda_long_context_smoke: tests/cuda_long_context_smoke.o ds4_cuda.o
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(CUDA_LDLIBS)

ds4_test: ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_test.o ds4_help.o ds4_kvstore.o rax.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

ds4_agent_test: ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS)
ifeq ($(UNAME_S),Darwin)
	$(CC) $(CFLAGS) -o $@ ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(METAL_LDLIBS)
else
	$(NVCC) $(NVCCFLAGS) -o $@ ds4_agent_test.o ds4_help.o ds4_web.o ds4_kvstore.o linenoise.o $(CORE_OBJS) $(CUDA_LDLIBS)
endif

test: ds4_test ds4_agent_test ds4-eval q4k-dot-test
	./ds4-eval --self-test-extractors
	./ds4_agent_test
	./ds4_test

q4k-dot-test: tests/test_q4k_dot.c
	$(CC) -O2 -Wall -Wextra -std=c99 -o tests/test_q4k_dot tests/test_q4k_dot.c -lm -pthread
	./tests/test_q4k_dot

clean:
	rm -f ds4 ds4-server ds4-bench ds4-eval ds4-agent ds4_cpu ds4_native ds4_server_test ds4_test ds4_agent_test qwen36-35a3b-q8-check qwen36-35a3b-q4xl-check qwen36-35a3b-iq2xxs-check qwen36-35a3b-v0-check qwen36-plan-dump qwen36-runtime-dump-bin qwen36-v0-quantize-bin qwen36-c-moe-replay qwen36-c-linear-replay qwen36-c-linear-conv-replay qwen36-c-linear-core-replay qwen36-c-linear-norm-replay qwen36-c-linear-layer-stub qwen36-c-linear-layer-full qwen36-c-linear-layer-weight qwen36-c-root qwen36-c-root-q8 qwen36-c-decoder-layer qwen36-c-decoder-layer-weight qwen36-c-two-decoder-layers qwen36-c-two-decoder-layers-weight qwen36-c-decoder-chain-weight qwen36-fixture-blk0-worker gguf-tools/libds4q.so tests/test_q4k_dot *.o tests/cuda_long_context_smoke tests/cuda_long_context_smoke.o
