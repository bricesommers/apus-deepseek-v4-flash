# apus — root Makefile
# Milestone M2: tokenizer (c/tok.h) + message encoding (c/encoding.h) + tests.

UNAME   := $(shell uname)
# M12a-1: clang on macOS, gcc on Linux/x86_64 (same warning set).
ifeq ($(UNAME),Darwin)
CC      := clang
else
CC      := gcc
endif
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra
ifneq ($(UNAME),Darwin)
# Linux (M12a-1): under -std=c11 glibc hides pread/posix_memalign/strdup/
# clock_gettime/posix_fadvise behind feature-test macros; _GNU_SOURCE
# exposes them. -ffp-contract=off pins FP mul+add contraction off so the
# scalar kernels' documented rounding sequences survive -O2 on any future
# -march (x86-64 baseline has no FMA, so it is a no-op today).
# -fno-tree-vectorize -fno-tree-slp-vectorize: works around a Rosetta
# linux/amd64-emulation mistranslation of gcc -O2 auto-vectorized SSE2
# code (test-m8 SIGTRAPs with "rosetta error: could not find free space
# for allocation"; ASan/UBSan clean, macOS -O2 clean, -O1 clean).
# Numerics are unaffected: FP reductions are never reassociated without
# -ffast-math and elementwise loops are per-element identical either way,
# so the scalar kernels produce the same bits with or without these flags.
# M12a-2 (AVX2) hand-writes the vector kernels and can revisit this.
CFLAGS  += -D_GNU_SOURCE -ffp-contract=off -fno-tree-vectorize -fno-tree-slp-vectorize
endif
LDLIBS  := -lm
# M9b: Accelerate.framework (system vecLib/AMX BLAS) for the batch-M prefill
# GEMM dispatch (c/blas.h). macOS system framework, ships with the OS.
ifeq ($(UNAME),Darwin)
LDLIBS  += -framework Accelerate
endif
ASAN_CFLAGS := -std=c11 -O1 -g -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer

M2   := tests/m2
BIN  := $(M2)/bin
PY   := .venv/bin/python

M3   := tests/m3
BIN3 := $(M3)/bin

M4A  := tests/m4a
BIN4 := $(M4A)/bin

M4B  := tests/m4b
M4C  := tests/m4c
BIN4C := $(M4C)/bin

M5   := tests/m5
BIN5 := $(M5)/bin

M6A  := tests/m6a
BIN6 := $(M6A)/bin

M6B  := tests/m6b
BIN6B := $(M6B)/bin

M7A  := tests/m7a

M6C  := tests/m6c
BIN6C := $(M6C)/bin

M8   := tests/m8
BIN8 := $(M8)/bin

M11A := tests/m11a

M2_DEPS := c/json.h c/tok.h c/uni_tables.h
M3_DEPS := c/fp4.h
M4A_DEPS := c/fp4.h c/fp8.h c/blas.h c/mhc.h
M4C_DEPS := c/fp4.h c/fp8.h c/blas.h c/mhc.h c/json.h c/st.h c/attn.h c/moe.h c/layer.h c/pool.h
M5_DEPS := $(M4C_DEPS) c/model.h c/sample.h
M6_DEPS := $(M5_DEPS) c/compat.h c/cache.h
M6B_DEPS := $(M6_DEPS) c/pilot.h
APUS_DEPS := $(M6B_DEPS) c/tok.h c/encoding.h c/uni_tables.h c/dspark.h
# (defined after M5_DEPS so the := below actually captures the kernel headers)
M6C_DEPS := $(M5_DEPS) c/pool.h
M8_DEPS := $(M5_DEPS) c/mtp.h

.PHONY: all test-m2 asan-m2 ubsan-m2 golden golden-exhaustive \
        test-m3 bench-m3 ubsan-m3 golden-m3 \
        test-m4a ubsan-m4a golden-m4a golden-m4b test-m4c ubsan-m4c clean \
        test-m5 ubsan-m5 golden-m5 apus \
        test-m6a ubsan-m6a bench-m6a golden-m6a \
        test-m6b ubsan-m6b golden-m6b \
        test-m6c ubsan-m6c \
        test-m7a ubsan-m7a golden-m7a \
        test-m7b ubsan-m7b bench-m7b \
        test-m8 ubsan-m8 golden-m8 \
        golden-m11a check-m11a \
        test-m11b ubsan-m11b \
        test-m9a ubsan-m9a bench-m9a \
        test-m9b ubsan-m9b bench-m9b \
        test-m9c ubsan-m9c \
        test-m9d ubsan-m9d bench-m9d \
        test-m9e ubsan-m9e bench-m9e \
        test-m12a2 ubsan-m12a2 bench-m12a2

all: $(BIN)/test_tok $(BIN)/test_encoding $(BIN3)/test_fp4 $(BIN3)/bench_fp4 \
     $(BIN4)/test_fp8 $(BIN4)/test_mhc

$(BIN):
	mkdir -p $(BIN)

$(BIN)/test_tok: $(M2)/test_tok.c $(M2_DEPS) | $(BIN)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS)

$(BIN)/test_encoding: $(M2)/test_encoding.c $(M2_DEPS) c/encoding.h | $(BIN)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS)

$(BIN)/test_tok_asan: $(M2)/test_tok.c $(M2_DEPS) | $(BIN)
	$(CC) $(ASAN_CFLAGS) -Ic -o $@ $< $(LDLIBS)

$(BIN)/test_encoding_asan: $(M2)/test_encoding.c $(M2_DEPS) c/encoding.h | $(BIN)
	$(CC) $(ASAN_CFLAGS) -Ic -o $@ $< $(LDLIBS)

golden:
	$(PY) $(M2)/gen_golden.py

golden-exhaustive:
	$(PY) $(M2)/gen_golden.py --exhaustive

test-m2: all golden
	./$(BIN)/test_tok
	./$(BIN)/test_encoding

asan-m2: $(BIN)/test_tok_asan $(BIN)/test_encoding_asan golden
	./$(BIN)/test_tok_asan
	./$(BIN)/test_encoding_asan

# UBSan-only variant (Apple's ASan runtime hangs in dyld init on some macOS
# versions; see tests/m2/README notes in the M2 report)
ubsan-m2: CFLAGS = -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer
ubsan-m2: $(BIN)/test_tok_ubsan $(BIN)/test_encoding_ubsan golden
	./$(BIN)/test_tok_ubsan
	./$(BIN)/test_encoding_ubsan

$(BIN)/test_tok_ubsan: $(M2)/test_tok.c $(M2_DEPS) | $(BIN)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS)

$(BIN)/test_encoding_ubsan: $(M2)/test_encoding.c $(M2_DEPS) c/encoding.h | $(BIN)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS)

# --- M3: MXFP4 kernel (c/fp4.h) -------------------------------------------

$(BIN3):
	mkdir -p $(BIN3)

$(BIN3)/test_fp4: $(M3)/test_fp4.c $(M3_DEPS) | $(BIN3)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS)

$(BIN3)/bench_fp4: $(M3)/bench_fp4.c $(M3_DEPS) | $(BIN3)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS)

$(BIN3)/test_fp4_ubsan: $(M3)/test_fp4.c $(M3_DEPS) | $(BIN3)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS)

golden-m3:
	$(PY) $(M3)/gen_golden.py

test-m3: $(BIN3)/test_fp4 golden-m3
	./$(BIN3)/test_fp4

# UBSan-only, like ubsan-m2 (Apple ASan runtime broken on this machine)
ubsan-m3: $(BIN3)/test_fp4_ubsan golden-m3
	./$(BIN3)/test_fp4_ubsan

bench-m3: $(BIN3)/bench_fp4
	./$(BIN3)/bench_fp4

# --- M4a: FP8 dense kernel (c/fp8.h) + mHC (c/mhc.h) ----------------------

$(BIN4):
	mkdir -p $(BIN4)

$(BIN4)/test_fp8: $(M4A)/test_fp8.c $(M4A_DEPS) | $(BIN4)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS)

$(BIN4)/test_mhc: $(M4A)/test_mhc.c $(M4A_DEPS) | $(BIN4)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS)

$(BIN4)/test_fp8_ubsan: $(M4A)/test_fp8.c $(M4A_DEPS) | $(BIN4)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS)

$(BIN4)/test_mhc_ubsan: $(M4A)/test_mhc.c $(M4A_DEPS) | $(BIN4)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS)

golden-m4a:
	$(PY) $(M4A)/gen_golden.py

test-m4a: $(BIN4)/test_fp8 $(BIN4)/test_mhc golden-m4a
	./$(BIN4)/test_fp8
	./$(BIN4)/test_mhc

# UBSan-only, like ubsan-m2/m3 (Apple ASan runtime broken on this machine)
ubsan-m4a: $(BIN4)/test_fp8_ubsan $(BIN4)/test_mhc_ubsan golden-m4a
	./$(BIN4)/test_fp8_ubsan
	./$(BIN4)/test_mhc_ubsan

# --- M4c: single-layer forward (c/st.h + c/attn.h + c/moe.h + c/layer.h) ---
# Verified against the M4b golden fixtures (tests/m4b/fixtures, gitignored;
# regenerate with golden-m4b — required from a clean checkout, e.g. CI).

$(BIN4C):
	mkdir -p $(BIN4C)

$(BIN4C)/test_layer: $(M4C)/test_layer.c $(M4C_DEPS) | $(BIN4C)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS)

$(BIN4C)/test_layer_ubsan: $(M4C)/test_layer.c $(M4C_DEPS) | $(BIN4C)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS)

golden-m4b:
	$(PY) $(M4B)/run_oracle.py

test-m4c: $(BIN4C)/test_layer golden-m4b
	./$(BIN4C)/test_layer

# UBSan-only, like ubsan-m2/m3/m4a (Apple ASan runtime broken on this machine)
ubsan-m4c: $(BIN4C)/test_layer_ubsan golden-m4b
	./$(BIN4C)/test_layer_ubsan

# --- M5: full-model forward (c/model.h + c/sample.h + c/apus.c) ------------
# Verified against the synthetic full mini-model fixtures (tests/m5/fixtures,
# gitignored; regenerate with golden-m5 — a test-m5 dependency).

$(BIN5):
	mkdir -p $(BIN5)

$(BIN5)/test_full: $(M5)/test_full.c $(M5_DEPS) | $(BIN5)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS)

$(BIN5)/test_full_ubsan: $(M5)/test_full.c $(M5_DEPS) | $(BIN5)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS)

golden-m5:
	$(PY) $(M5)/gen_fixtures.py

test-m5: $(BIN5)/test_full golden-m5
	./$(BIN5)/test_full

# UBSan-only, like the other milestones (Apple ASan runtime broken here)
ubsan-m5: $(BIN5)/test_full_ubsan golden-m5
	./$(BIN5)/test_full_ubsan

# --- M6a: expert-store tiering (c/cache.h + c/st.h lazy path + c/compat.h) -
# Verified on tests/m6a/fixtures (6 layers x 64 experts, coalesced slabs;
# regenerate with golden-m6a). LDLIBS gains -lpthread for the I/O pool.

$(BIN6):
	mkdir -p $(BIN6)

$(BIN6)/test_store: $(M6A)/test_store.c $(M6_DEPS) | $(BIN6)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN6)/test_invariance: $(M6A)/test_invariance.c $(M6_DEPS) | $(BIN6)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN6)/bench_m6a: $(M6A)/bench_m6a.c $(M6_DEPS) | $(BIN6)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN6)/test_store_ubsan: $(M6A)/test_store.c $(M6_DEPS) | $(BIN6)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN6)/test_invariance_ubsan: $(M6A)/test_invariance.c $(M6_DEPS) | $(BIN6)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

golden-m6a:
	$(PY) $(M6A)/gen_fixtures.py

test-m6a: $(BIN6)/test_store $(BIN6)/test_invariance golden-m6a
	./$(BIN6)/test_store
	./$(BIN6)/test_invariance

# UBSan-only, like the other milestones
ubsan-m6a: $(BIN6)/test_store_ubsan $(BIN6)/test_invariance_ubsan golden-m6a
	./$(BIN6)/test_store_ubsan
	./$(BIN6)/test_invariance_ubsan

bench-m6a: $(BIN6)/bench_m6a
	./$(BIN6)/bench_m6a

# --- M6b: router-lookahead prefetch pilot (c/pilot.h) + locality tooling ---
# Verified on tests/m6b/fixtures (6 layers x 64 experts, 3 hash layers;
# regenerate with golden-m6b).

$(BIN6B):
	mkdir -p $(BIN6B)

$(BIN6B)/test_pilot: $(M6B)/test_pilot.c $(M6B_DEPS) | $(BIN6B)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN6B)/test_invariance: $(M6B)/test_invariance.c $(M6B_DEPS) | $(BIN6B)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN6B)/test_recall: $(M6B)/test_recall.c $(M6B_DEPS) | $(BIN6B)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN6B)/test_pilot_ubsan: $(M6B)/test_pilot.c $(M6B_DEPS) | $(BIN6B)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN6B)/test_invariance_ubsan: $(M6B)/test_invariance.c $(M6B_DEPS) | $(BIN6B)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN6B)/test_recall_ubsan: $(M6B)/test_recall.c $(M6B_DEPS) | $(BIN6B)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

golden-m6b:
	$(PY) $(M6B)/gen_fixtures.py

test-m6b: $(BIN6B)/test_pilot $(BIN6B)/test_invariance $(BIN6B)/test_recall golden-m6b
	./$(BIN6B)/test_pilot
	./$(BIN6B)/test_invariance
	mkdir -p $(M6B)/tmp
	./$(BIN6B)/test_recall $(M6B)/tmp/recall_runtime.ndjson $(M6B)/tmp/recall_measure.ndjson > $(M6B)/tmp/recall_line.txt
	cat $(M6B)/tmp/recall_line.txt
	$(PY) $(M6B)/check_recall.py $(M6B)/tmp/recall_runtime.ndjson $(M6B)/tmp/recall_measure.ndjson $(M6B)/tmp/recall_line.txt

# UBSan-only, like the other milestones
ubsan-m6b: $(BIN6B)/test_pilot_ubsan $(BIN6B)/test_invariance_ubsan $(BIN6B)/test_recall_ubsan golden-m6b
	./$(BIN6B)/test_pilot_ubsan
	./$(BIN6B)/test_invariance_ubsan
	mkdir -p $(M6B)/tmp
	./$(BIN6B)/test_recall_ubsan $(M6B)/tmp/recall_runtime.ndjson $(M6B)/tmp/recall_measure.ndjson > $(M6B)/tmp/recall_line.txt
	$(PY) $(M6B)/check_recall.py $(M6B)/tmp/recall_runtime.ndjson $(M6B)/tmp/recall_measure.ndjson $(M6B)/tmp/recall_line.txt

# engine CLI binary
bin:
	mkdir -p bin

ifdef metal
apus: bin/apus_metal
else
apus: bin/apus
endif

bin/apus: c/apus.c $(APUS_DEPS) | bin
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

# --- M7b: optional Metal backend (c/backend_metal.mm + c/backend_metal.h) ---
# Dense compute (FP8 attention/shared-expert linears, router gate, wo_a,
# BF16 LM head) on the Apple GPU, FP32 shaders, zero-copy unified-memory
# buffers. `make metal=1 apus` (or `make bin/apus_metal`). CPU stays the
# default; bin/apus is behaviorally untouched. macOS-only (M12a-1: the
# targets are stubbed with a clear error on Linux).

ifeq ($(UNAME),Darwin)

M7B    := tests/m7b
BIN7B  := $(M7B)/bin
M7B_DEPS := $(APUS_DEPS) c/backend_metal.h

METAL_CXX      := clang++
METAL_CXXFLAGS := -std=c++17 -O2 -Wall -Wextra
METAL_LDLIBS   := -framework Foundation -framework Metal

bin/apus_metal: c/apus.c c/backend_metal.mm $(M7B_DEPS) | bin
	$(CC) $(CFLAGS) -Ic -c c/apus.c -o bin/apus_metal_main.o
	$(METAL_CXX) $(METAL_CXXFLAGS) -Ic -c c/backend_metal.mm -o bin/backend_metal.o
	$(METAL_CXX) -o $@ bin/apus_metal_main.o bin/backend_metal.o $(LDLIBS) -lpthread $(METAL_LDLIBS)

$(BIN7B):
	mkdir -p $(BIN7B)

$(BIN7B)/backend_metal.o: c/backend_metal.mm c/backend_metal.h | $(BIN7B)
	$(METAL_CXX) $(METAL_CXXFLAGS) -Ic -c c/backend_metal.mm -o $@

$(BIN7B)/test_kernels: $(M7B)/test_kernels.c $(BIN7B)/backend_metal.o $(M7B_DEPS)
	$(CC) $(CFLAGS) -Ic -c $(M7B)/test_kernels.c -o $(BIN7B)/test_kernels_c.o
	$(METAL_CXX) -o $@ $(BIN7B)/test_kernels_c.o $(BIN7B)/backend_metal.o $(LDLIBS) $(METAL_LDLIBS)

$(BIN7B)/test_model: $(M7B)/test_model.c $(BIN7B)/backend_metal.o $(M7B_DEPS)
	$(CC) $(CFLAGS) -Ic -c $(M7B)/test_model.c -o $(BIN7B)/test_model_c.o
	$(METAL_CXX) -o $@ $(BIN7B)/test_model_c.o $(BIN7B)/backend_metal.o $(LDLIBS) $(METAL_LDLIBS)

$(BIN7B)/bench_metal: $(M7B)/bench_metal.c $(BIN7B)/backend_metal.o $(M7B_DEPS)
	$(CC) $(CFLAGS) -Ic -c $(M7B)/bench_metal.c -o $(BIN7B)/bench_metal_c.o
	$(METAL_CXX) -o $@ $(BIN7B)/bench_metal_c.o $(BIN7B)/backend_metal.o $(LDLIBS) $(METAL_LDLIBS)

test-m7b: $(BIN7B)/test_kernels $(BIN7B)/test_model bin/apus_metal golden-m7a
	./$(BIN7B)/test_kernels
	./$(BIN7B)/test_model
	APUS_BIN=$(CURDIR)/bin/apus_metal APUS_METAL=1 $(PY) $(M7A)/test_server.py

bench-m7b: $(BIN7B)/bench_metal
	./$(BIN7B)/bench_metal

# UBSan on the C side (the .mm stays unsanitized, like ASan-less m2..m7a)
$(BIN7B)/test_kernels_ubsan: $(M7B)/test_kernels.c $(BIN7B)/backend_metal.o $(M7B_DEPS)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -c $(M7B)/test_kernels.c -o $(BIN7B)/test_kernels_ubsan_c.o
	$(METAL_CXX) -fsanitize=undefined -o $@ $(BIN7B)/test_kernels_ubsan_c.o $(BIN7B)/backend_metal.o $(LDLIBS) $(METAL_LDLIBS)

$(BIN7B)/test_model_ubsan: $(M7B)/test_model.c $(BIN7B)/backend_metal.o $(M7B_DEPS)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -c $(M7B)/test_model.c -o $(BIN7B)/test_model_ubsan_c.o
	$(METAL_CXX) -fsanitize=undefined -o $@ $(BIN7B)/test_model_ubsan_c.o $(BIN7B)/backend_metal.o $(LDLIBS) $(METAL_LDLIBS)

$(BIN7B)/apus_metal_ubsan: c/apus.c $(BIN7B)/backend_metal.o $(M7B_DEPS)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -c c/apus.c -o $(BIN7B)/apus_metal_ubsan_main.o
	$(METAL_CXX) -fsanitize=undefined -o $@ $(BIN7B)/apus_metal_ubsan_main.o $(BIN7B)/backend_metal.o $(LDLIBS) -lpthread $(METAL_LDLIBS)

ubsan-m7b: $(BIN7B)/test_kernels_ubsan $(BIN7B)/test_model_ubsan $(BIN7B)/apus_metal_ubsan golden-m7a
	./$(BIN7B)/test_kernels_ubsan
	./$(BIN7B)/test_model_ubsan
	APUS_BIN=$(CURDIR)/$(BIN7B)/apus_metal_ubsan APUS_METAL=1 $(PY) $(M7A)/test_server.py

else  # !Darwin: Metal is macOS-only (M12a-1)

bin/apus_metal test-m7b bench-m7b ubsan-m7b:
	@echo "error: the Metal backend (m7b) is macOS-only, not available on $(UNAME)" >&2; exit 1

endif

# --- M6c: decode threading (c/pool.h) + threaded/NEON kernels --------------
# Pool correctness, thread-count independence (bitwise digests across
# APUS_THREADS=1/4/8), bf16/f32 NEON reorder bounds, scratch arena.

$(BIN6C):
	mkdir -p $(BIN6C)

$(BIN6C)/test_m6c: $(M6C)/test_m6c.c $(M6C_DEPS) | $(BIN6C)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN6C)/test_m6c_ubsan: $(M6C)/test_m6c.c $(M6C_DEPS) | $(BIN6C)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

test-m6c: $(BIN6C)/test_m6c
	APUS_THREADS=1 ./$(BIN6C)/test_m6c > $(BIN6C)/out_t1.txt
	APUS_THREADS=4 ./$(BIN6C)/test_m6c > $(BIN6C)/out_t4.txt
	APUS_THREADS=8 ./$(BIN6C)/test_m6c > $(BIN6C)/out_t8.txt
	diff $(BIN6C)/out_t1.txt $(BIN6C)/out_t4.txt
	diff $(BIN6C)/out_t1.txt $(BIN6C)/out_t8.txt
	cat $(BIN6C)/out_t4.txt

# UBSan-only, like the other milestones
ubsan-m6c: $(BIN6C)/test_m6c_ubsan
	APUS_THREADS=1 ./$(BIN6C)/test_m6c_ubsan > $(BIN6C)/out_u1.txt
	APUS_THREADS=4 ./$(BIN6C)/test_m6c_ubsan > $(BIN6C)/out_u4.txt
	diff $(BIN6C)/out_u1.txt $(BIN6C)/out_u4.txt
	cat $(BIN6C)/out_u4.txt

# --- M8: MTP speculative decoding (c/mtp.h + loader/store extensions) ------
# Verified on tests/m8/fixtures (the M5 mini-model + an MTP block;
# regenerate with golden-m8). Hard gate: spec vs non-spec streams BITWISE
# (greedy and seeded-sampled); rollback state digests bitwise; MTP forward
# vs oracle goldens under the m5 tolerancing policy. The Makefile diffs
# stdout across APUS_THREADS=1/4/8 (thread independence), like m6c.

$(BIN8):
	mkdir -p $(BIN8)

$(BIN8)/test_m8: $(M8)/test_m8.c $(M8_DEPS) | $(BIN8)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN8)/test_m8_ubsan: $(M8)/test_m8.c $(M8_DEPS) | $(BIN8)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

golden-m8:
	$(PY) $(M8)/gen_fixtures.py

test-m8: $(BIN8)/test_m8 golden-m8
	APUS_THREADS=1 ./$(BIN8)/test_m8 > $(BIN8)/out_t1.txt
	APUS_THREADS=4 ./$(BIN8)/test_m8 > $(BIN8)/out_t4.txt
	APUS_THREADS=8 ./$(BIN8)/test_m8 > $(BIN8)/out_t8.txt
	diff $(BIN8)/out_t1.txt $(BIN8)/out_t4.txt
	diff $(BIN8)/out_t1.txt $(BIN8)/out_t8.txt
	cat $(BIN8)/out_t4.txt

# UBSan-only, like the other milestones (Apple ASan runtime broken here)
ubsan-m8: $(BIN8)/test_m8_ubsan golden-m8
	APUS_THREADS=1 ./$(BIN8)/test_m8_ubsan > $(BIN8)/out_u1.txt
	APUS_THREADS=4 ./$(BIN8)/test_m8_ubsan > $(BIN8)/out_u4.txt
	diff $(BIN8)/out_u1.txt $(BIN8)/out_u4.txt
	cat $(BIN8)/out_u4.txt

# --- M11a: DSpark oracle + fixtures (numpy only; the C port is M11b) -----
# tests/m11a/fixtures (regenerate with golden-m11a): the m5-style mini-model
# extended with 3 synthetic DSpark stages (real mtp.* naming), goldens for
# the draft forwards, spec episodes, forced draft patterns and rollback
# digests. check-m11a runs the oracle self-consistency suite (determinism,
# spec==non-spec streams bitwise, rollback, legality). See tests/m11a/README.

golden-m11a:
	$(PY) $(M11A)/gen_fixtures.py

check-m11a: golden-m11a
	$(PY) $(M11A)/check_oracle.py

# --- M11b: DSpark speculative decoding in C (c/dspark.h) -------------------
# Verified against the M11a fixtures. Hard gates: spec vs non-spec streams
# BITWISE (greedy + seeded-sampled); rollback state digests (canonical
# oracle_dspark.state_digest layout) spec == non-spec; forced draft patterns
# with exact accept counts; stage-ring catch-up (D13). Golden comparisons
# under the m5/m8 margin policy (tests/m11a/README tolerancing guidance).
# The Makefile diffs stdout across APUS_THREADS=1/4/8 (thread independence).

M11B   := tests/m11b
BIN11B := $(M11B)/bin
M11B_DEPS := $(M8_DEPS) c/dspark.h

$(BIN11B):
	mkdir -p $(BIN11B)

$(BIN11B)/test_m11b: $(M11B)/test_m11b.c $(M11B_DEPS) | $(BIN11B)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN11B)/test_m11b_ubsan: $(M11B)/test_m11b.c $(M11B_DEPS) | $(BIN11B)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

test-m11b: $(BIN11B)/test_m11b golden-m11a
	APUS_THREADS=1 ./$(BIN11B)/test_m11b > $(BIN11B)/out_t1.txt
	APUS_THREADS=4 ./$(BIN11B)/test_m11b > $(BIN11B)/out_t4.txt
	APUS_THREADS=8 ./$(BIN11B)/test_m11b > $(BIN11B)/out_t8.txt
	diff $(BIN11B)/out_t1.txt $(BIN11B)/out_t4.txt
	diff $(BIN11B)/out_t1.txt $(BIN11B)/out_t8.txt
	cat $(BIN11B)/out_t4.txt

# UBSan-only, like the other milestones (Apple ASan runtime broken here)
ubsan-m11b: $(BIN11B)/test_m11b_ubsan golden-m11a
	APUS_THREADS=1 ./$(BIN11B)/test_m11b_ubsan > $(BIN11B)/out_u1.txt
	APUS_THREADS=4 ./$(BIN11B)/test_m11b_ubsan > $(BIN11B)/out_u4.txt
	diff $(BIN11B)/out_u1.txt $(BIN11B)/out_u4.txt
	cat $(BIN11B)/out_u4.txt

# --- M7a: OpenAI-compatible server (apus serve NDJSON + tools/server.py) ---
# End-to-end on the scripted tests/m7a fixtures (regenerate with golden-m7a;
# see tests/m7a/README.md for the protocol and endpoint coverage).

golden-m7a:
	$(PY) $(M7A)/gen_fixtures.py

test-m7a: bin/apus golden-m7a
	$(PY) $(M7A)/test_server.py

bin/apus_ubsan: c/apus.c $(APUS_DEPS) | bin
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

# UBSan-only, like the other milestones (Apple ASan runtime broken here)
ubsan-m7a: bin/apus_ubsan golden-m7a
	APUS_BIN=$(CURDIR)/bin/apus_ubsan $(PY) $(M7A)/test_server.py

M9B   := tests/m9b
BIN9B := $(M9B)/bin
M9B_DEPS := c/fp4.h c/fp8.h c/blas.h c/pool.h

$(BIN9B):
	mkdir -p $(BIN9B)

$(BIN9B)/test_m9b: $(M9B)/test_m9b.c $(M9B_DEPS) | $(BIN9B)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN9B)/test_m9b_ubsan: $(M9B)/test_m9b.c $(M9B_DEPS) | $(BIN9B)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN9B)/bench_m9b: $(M9B)/bench_m9b.c $(M9B_DEPS) | $(BIN9B)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

test-m9b: $(BIN9B)/test_m9b
	APUS_THREADS=1 ./$(BIN9B)/test_m9b > $(BIN9B)/out_t1.txt
	APUS_THREADS=4 ./$(BIN9B)/test_m9b > $(BIN9B)/out_t4.txt
	APUS_THREADS=8 ./$(BIN9B)/test_m9b > $(BIN9B)/out_t8.txt
	diff $(BIN9B)/out_t1.txt $(BIN9B)/out_t4.txt
	diff $(BIN9B)/out_t1.txt $(BIN9B)/out_t8.txt
	cat $(BIN9B)/out_t4.txt

# UBSan-only, like the other milestones
ubsan-m9b: $(BIN9B)/test_m9b_ubsan
	APUS_THREADS=1 ./$(BIN9B)/test_m9b_ubsan > $(BIN9B)/out_u1.txt
	APUS_THREADS=4 ./$(BIN9B)/test_m9b_ubsan > $(BIN9B)/out_u4.txt
	diff $(BIN9B)/out_u1.txt $(BIN9B)/out_u4.txt
	cat $(BIN9B)/out_u4.txt

bench-m9b: $(BIN9B)/bench_m9b
	./$(BIN9B)/bench_m9b

# --- M9c: expert-I/O pipelining (c/cache.h hot jobs, c/pilot.h union) ------
# Demand-boost queue ordering, batched prefill union lookahead equivalence,
# wait instrumentation, pilot-ON quality invariance (bitwise), and the
# thread-count-independence digest (diffed across APUS_THREADS=1/4/8).

M9C   := tests/m9c
BIN9C := $(M9C)/bin
M9C_DEPS := $(M6B_DEPS) c/sample.h

$(BIN9C):
	mkdir -p $(BIN9C)

$(BIN9C)/test_m9c: $(M9C)/test_m9c.c $(M9C_DEPS) | $(BIN9C)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN9C)/test_m9c_ubsan: $(M9C)/test_m9c.c $(M9C_DEPS) | $(BIN9C)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN9C)/run_model: $(M9C)/run_model.c $(APUS_DEPS) c/mtp.h | $(BIN9C)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

test-m9c: $(BIN9C)/test_m9c golden-m6b
	APUS_THREADS=1 ./$(BIN9C)/test_m9c > $(BIN9C)/out_t1.txt
	APUS_THREADS=4 ./$(BIN9C)/test_m9c > $(BIN9C)/out_t4.txt
	APUS_THREADS=8 ./$(BIN9C)/test_m9c > $(BIN9C)/out_t8.txt
	diff $(BIN9C)/out_t1.txt $(BIN9C)/out_t4.txt
	diff $(BIN9C)/out_t1.txt $(BIN9C)/out_t8.txt
	cat $(BIN9C)/out_t4.txt

# UBSan-only, like the other milestones
ubsan-m9c: $(BIN9C)/test_m9c_ubsan golden-m6b
	APUS_THREADS=1 ./$(BIN9C)/test_m9c_ubsan > $(BIN9C)/out_u1.txt
	APUS_THREADS=4 ./$(BIN9C)/test_m9c_ubsan > $(BIN9C)/out_u4.txt
	diff $(BIN9C)/out_u1.txt $(BIN9C)/out_u4.txt
	cat $(BIN9C)/out_u4.txt

# --- M9a: fp8/fp4 NEON GEMM ILP reorder (c/fp8.h, c/fp4.h, Metal mirror) ---
# Canonical accumulation order proofs (bitwise scalar model), exhaustive E4M3
# conversion, FMLAL-vs-anchor bitwise, mt==neon bitwise, esc bounds, and the
# thread-count-independence digest (diffed across APUS_THREADS=1/4/8).

M9A   := tests/m9a
BIN9A := $(M9A)/bin
M9A_DEPS := c/fp4.h c/fp8.h c/blas.h c/pool.h

$(BIN9A):
	mkdir -p $(BIN9A)

$(BIN9A)/test_m9a: $(M9A)/test_m9a.c $(M9A_DEPS) | $(BIN9A)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN9A)/test_m9a_ubsan: $(M9A)/test_m9a.c $(M9A_DEPS) | $(BIN9A)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN9A)/bench_m9a: $(M9A)/bench_m9a.c $(M9A_DEPS) | $(BIN9A)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

test-m9a: $(BIN9A)/test_m9a
	APUS_THREADS=1 ./$(BIN9A)/test_m9a > $(BIN9A)/out_t1.txt
	APUS_THREADS=4 ./$(BIN9A)/test_m9a > $(BIN9A)/out_t4.txt
	APUS_THREADS=8 ./$(BIN9A)/test_m9a > $(BIN9A)/out_t8.txt
	diff $(BIN9A)/out_t1.txt $(BIN9A)/out_t4.txt
	diff $(BIN9A)/out_t1.txt $(BIN9A)/out_t8.txt
	cat $(BIN9A)/out_t4.txt

# UBSan-only, like the other milestones
ubsan-m9a: $(BIN9A)/test_m9a_ubsan
	APUS_THREADS=1 ./$(BIN9A)/test_m9a_ubsan > $(BIN9A)/out_u1.txt
	APUS_THREADS=4 ./$(BIN9A)/test_m9a_ubsan > $(BIN9A)/out_u4.txt
	diff $(BIN9A)/out_u1.txt $(BIN9A)/out_u4.txt
	cat $(BIN9A)/out_u4.txt

bench-m9a: $(BIN9A)/bench_m9a
	./$(BIN9A)/bench_m9a

# --- M9d: prefill compute utilization (dense/woa BLAS dispatch + row pooling)
# f32/bf16/woa BLAS vs FP64 truth (reorder-class esc bounds), dispatch
# boundary bitwise checks, model-level s=300 (>cutoff) digest diffed across
# APUS_THREADS=1/4/8 (thread-count independence of every new path).

M9D   := tests/m9d
BIN9D := $(M9D)/bin
M9D_DEPS := $(M5_DEPS) c/blas.h c/sample.h

$(BIN9D):
	mkdir -p $(BIN9D)

$(BIN9D)/test_m9d: $(M9D)/test_m9d.c $(M9D_DEPS) | $(BIN9D)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN9D)/test_m9d_ubsan: $(M9D)/test_m9d.c $(M9D_DEPS) | $(BIN9D)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN9D)/bench_m9d: $(M9D)/bench_m9d.c $(M9D_DEPS) | $(BIN9D)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

test-m9d: $(BIN9D)/test_m9d golden-m6b
	APUS_THREADS=1 ./$(BIN9D)/test_m9d > $(BIN9D)/out_t1.txt
	APUS_THREADS=4 ./$(BIN9D)/test_m9d > $(BIN9D)/out_t4.txt
	APUS_THREADS=8 ./$(BIN9D)/test_m9d > $(BIN9D)/out_t8.txt
	diff $(BIN9D)/out_t1.txt $(BIN9D)/out_t4.txt
	diff $(BIN9D)/out_t1.txt $(BIN9D)/out_t8.txt
	cat $(BIN9D)/out_t4.txt

# UBSan-only, like the other milestones
ubsan-m9d: $(BIN9D)/test_m9d_ubsan golden-m6b
	APUS_THREADS=1 ./$(BIN9D)/test_m9d_ubsan > $(BIN9D)/out_u1.txt
	APUS_THREADS=4 ./$(BIN9D)/test_m9d_ubsan > $(BIN9D)/out_u4.txt
	diff $(BIN9D)/out_u1.txt $(BIN9D)/out_u4.txt
	cat $(BIN9D)/out_u4.txt

bench-m9d: $(BIN9D)/bench_m9d
	./$(BIN9D)/bench_m9d

# --- M9e: routed-expert dispatch grouping (c/fp4.h grouped GEMM, c/moe.h) --
# Grouped GEMM bitwise == per-entry mt (incl. the M=1 decode pin vs the M9a
# kernel body), MoE group/solo mixes bitwise vs per-token M=1, and the
# thread-count-independence digest (diffed across APUS_THREADS=1/4/8).

M9E   := tests/m9e
BIN9E := $(M9E)/bin
M9E_DEPS := c/fp4.h c/fp8.h c/blas.h c/mhc.h c/st.h c/attn.h c/moe.h c/pool.h c/json.h

$(BIN9E):
	mkdir -p $(BIN9E)

$(BIN9E)/test_m9e: $(M9E)/test_m9e.c $(M9E_DEPS) | $(BIN9E)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN9E)/test_m9e_ubsan: $(M9E)/test_m9e.c $(M9E_DEPS) | $(BIN9E)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN9E)/bench_m9e: $(M9E)/bench_m9e.c c/fp4.h c/pool.h c/blas.h | $(BIN9E)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

test-m9e: $(BIN9E)/test_m9e
	APUS_THREADS=1 ./$(BIN9E)/test_m9e > $(BIN9E)/out_t1.txt
	APUS_THREADS=4 ./$(BIN9E)/test_m9e > $(BIN9E)/out_t4.txt
	APUS_THREADS=8 ./$(BIN9E)/test_m9e > $(BIN9E)/out_t8.txt
	diff $(BIN9E)/out_t1.txt $(BIN9E)/out_t4.txt
	diff $(BIN9E)/out_t1.txt $(BIN9E)/out_t8.txt
	cat $(BIN9E)/out_t4.txt

# UBSan-only, like the other milestones
ubsan-m9e: $(BIN9E)/test_m9e_ubsan
	APUS_THREADS=1 ./$(BIN9E)/test_m9e_ubsan > $(BIN9E)/out_u1.txt
	APUS_THREADS=4 ./$(BIN9E)/test_m9e_ubsan > $(BIN9E)/out_u4.txt
	diff $(BIN9E)/out_u1.txt $(BIN9E)/out_u4.txt
	cat $(BIN9E)/out_u4.txt

bench-m9e: $(BIN9E)/bench_m9e
	./$(BIN9E)/bench_m9e

# --- M12a-2: AVX2 x86 kernels (c/x86.h + dispatch in fp4/fp8/attn/model) ---
# Bitwise-vs-scalar hard gates, exhaustive E4M3/FP4 expand proofs, FP64
# truth (esc class), M-independence, the "AVX2 path taken" probe, and the
# thread-count-independence digest (diffed across APUS_THREADS=1/4/8).
# Off x86-64 the suite is a trivial pass (uniform target list).

M12   := tests/m12
BIN12 := $(M12)/bin
M12_DEPS := $(M5_DEPS) c/blas.h c/sample.h c/x86.h

$(BIN12):
	mkdir -p $(BIN12)

$(BIN12)/test_m12a2: $(M12)/test_m12a2.c $(M12_DEPS) | $(BIN12)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN12)/test_m12a2_ubsan: $(M12)/test_m12a2.c $(M12_DEPS) | $(BIN12)
	$(CC) -std=c11 -O1 -g -Wall -Wextra -fsanitize=undefined -fno-omit-frame-pointer -ffp-contract=off $(if $(filter-out Darwin,$(UNAME)),-D_GNU_SOURCE) -Ic -o $@ $< $(LDLIBS) -lpthread

$(BIN12)/bench_m12a2: $(M12)/bench_m12a2.c $(M12_DEPS) | $(BIN12)
	$(CC) $(CFLAGS) -Ic -o $@ $< $(LDLIBS) -lpthread

test-m12a2: $(BIN12)/test_m12a2
	APUS_THREADS=1 ./$(BIN12)/test_m12a2 > $(BIN12)/out_t1.txt
	APUS_THREADS=4 ./$(BIN12)/test_m12a2 > $(BIN12)/out_t4.txt
	APUS_THREADS=8 ./$(BIN12)/test_m12a2 > $(BIN12)/out_t8.txt
	diff $(BIN12)/out_t1.txt $(BIN12)/out_t4.txt
	diff $(BIN12)/out_t1.txt $(BIN12)/out_t8.txt
	cat $(BIN12)/out_t4.txt

# UBSan-only, like the other milestones (-ffp-contract=off pinned: the
# sanitizer flags override CFLAGS, dropping the Linux default flags)
ubsan-m12a2: $(BIN12)/test_m12a2_ubsan
	APUS_THREADS=1 ./$(BIN12)/test_m12a2_ubsan > $(BIN12)/out_u1.txt
	APUS_THREADS=4 ./$(BIN12)/test_m12a2_ubsan > $(BIN12)/out_u4.txt
	diff $(BIN12)/out_u1.txt $(BIN12)/out_u4.txt
	cat $(BIN12)/out_u4.txt

bench-m12a2: $(BIN12)/bench_m12a2
	./$(BIN12)/bench_m12a2

clean:
	rm -rf $(BIN) $(M2)/golden $(BIN3) $(M3)/golden $(BIN4) $(M4A)/golden $(BIN4C) $(BIN5) $(BIN6) $(M6A)/tmp $(BIN6B) $(M6B)/tmp $(M7A)/fixtures $(BIN7B) $(BIN9A) $(BIN9B) $(BIN11B) bin
