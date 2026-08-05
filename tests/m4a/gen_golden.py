#!/usr/bin/env python3
"""M4a golden generator — numpy ports of the reference FP8 blockwise GEMM
and mHC (hc_split_sinkhorn + hc_pre/hc_post/hc_head) semantics.

Ports (numpy only, no torch/tilelang):

  FP8 (reference/inference/kernel.py):
  * act_quant with scale_fmt="ue8m0" (kernel.py:40-125): per-128-along-K
    amax, floor 1e-4, scale = 2^ceil(log2(amax/448)) via the FP32 bit trick,
    clamp +-448, RNE to E4M3. (Same rule as the FP4 path's activations.)
  * fp8_gemm (kernel.py:203-254): per-128-K block FP32 dot of E4M3 codes,
    then total += dot * (scale_a * scale_b) — the scale PRODUCT is formed
    first (kernel.py:242-249), unlike the FP4 path's (dot*sa)*sb.
    Weight scales are UE8M0 per 128x128 tile, indexed [o//128, kb].
    Evaluated in float64 as the "exact" target; esc = sum |dot*sa*sb| per
    output is the FP32 error scale.

  mHC (model.py Block.hc_pre/hc_post:673-686, ParallelHead.hc_head:728-735,
  kernel.py hc_split_sinkhorn_kernel:371-438), all float64:
  * rsqrt = rsqrt(mean(x^2) + norm_eps); mixes = (hc_fn @ x) * rsqrt
    (rsqrt applied AFTER the matmul, model.py:678).
  * pre = sigmoid(m*s0 + b) + eps; post = 2*sigmoid(m*s1 + b);
    comb logits = m*s2 + b (mix layout: [pre(4) | post(4) | comb(16)]).
  * Sinkhorn-20 EXACT order (kernel.py:398-423):
      1. row softmax (max-subtracted), then PER-ELEMENT + eps;
      2. col normalize with eps IN THE DENOMINATOR;
      3. 19 x (row normalize /(sum+eps); col normalize /(sum+eps)).
    All 40 stages are dumped for iteration-for-iteration checking.
  * collapse y = sum_j pre[j]*X[j]; apply Y[j] = post[j]*f + sum_k comb[k,j]*R[k]
    (comb indexed [residual k][output j], model.py:685 — the M4a golden was
    originally generated with the transposed convention, matching the
    pre-M5 c/mhc.h bug; both are fixed at M5);
    hc_head: pre-only sigmoid gate + collapse (no Sinkhorn).

Fixtures (tests/m4a/golden/):
  fp8_manifest.txt            M=, O=, K=
  fp8_w_codes.bin             uint8  [O, K]           E4M3 weight codes
  fp8_w_scales.bin            uint8  [ceil(O/128), ceil(K/128)]  UE8M0
  fp8_act_x.bin               float32 [M, K]
  fp8_act_codes.bin           uint8  [M, K]
  fp8_act_scales.bin          float32 [M, ceil(K/128)]
  fp8_out.bin                 float64 [M, O]
  fp8_esc.bin                 float64 [M, O]   error scale
  mhc_manifest.txt            T=, D=, N=, T2=, ITERS=, NORM_EPS=, HC_EPS=
  mhc_x4.bin                  float32 [T, N, D]   states
  mhc_fn.bin                  float32 [24, N*D]
  mhc_scale.bin               float32 [3]
  mhc_base.bin                float32 [24]
  mhc_mixes.bin               float64 [T, 24]
  mhc_pre.bin / mhc_post.bin  float64 [T, N]
  mhc_comb.bin                float64 [T, N, N]
  mhc_ypre.bin                float64 [T, D]
  mhc_f.bin                   float32 [T, D]    fake sublayer output
  mhc_ypost.bin               float64 [T, N, D]
  mhc_hfn.bin                 float32 [N, N*D]
  mhc_hscale.bin              float32 [1]
  mhc_hbase.bin               float32 [N]
  mhc_yhead.bin               float64 [T, D]
  mhc_sk_logits.bin           float32 [T2, N, N]
  mhc_sk_stages.bin           float64 [T2, 40, N, N]  every normalization stage
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "golden")

NORM_EPS = 1e-6
HC_EPS = 1e-6
ITERS = 20


# --- shared helpers (same ports as tests/m3/gen_golden.py) ------------------

def ceil_log2_pow2_f32(p):
    """2^ceil(log2(p)) for positive normal float32 p, via the reference
    fast_log2_ceil/fast_pow2 bit trick (kernel.py:22-33)."""
    p = np.asarray(p, dtype=np.float32)
    b = p.view(np.uint32)
    e = ((b >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int64) - 127
    e += ((b & np.uint32(0x7FFFFF)) != 0).astype(np.int64)
    scale = ((e + 127).astype(np.uint32) << np.uint32(23)).view(np.float32)
    return scale, e


def build_e4m3_table():
    t = np.zeros(256, dtype=np.float64)
    for c in range(256):
        e = (c >> 3) & 0xF
        m = c & 7
        if e == 0xF and m == 7:
            t[c] = np.nan
        elif e == 0:
            t[c] = m * 2.0 ** -9
        else:
            t[c] = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        if c & 0x80:
            t[c] = -t[c]
    return t


E4M3_TABLE = build_e4m3_table()
E4M3_POS = E4M3_TABLE[:127]


def e4m3_quant_rne(x):
    """Brute-force RNE quantize float32 -> e4m3 codes, clamped to +-448."""
    y = np.clip(np.asarray(x, dtype=np.float32), -448.0, 448.0)
    sign = np.signbit(y)
    a = np.abs(y).astype(np.float64)
    idx = np.searchsorted(E4M3_POS, a)
    hi = np.clip(idx, 0, 126)
    lo = np.clip(idx - 1, 0, 126)
    d_lo = a - E4M3_POS[lo]
    d_hi = E4M3_POS[hi] - a
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    code = np.where(pick_hi, hi, lo).astype(np.uint8)
    return code | (sign.astype(np.uint8) << np.uint8(7))


def act_quant_ue8m0(x, block=128):
    """Port of act_quant(round_scale=True): x float32 [M, K] ->
    codes u8 [M, K], scales float32 [M, ceil(K/block)]."""
    M, K = x.shape
    nb = (K + block - 1) // block
    codes = np.empty((M, K), dtype=np.uint8)
    scales = np.empty((M, nb), dtype=np.float32)
    for b in range(nb):
        lo, hi = b * block, min(K, (b + 1) * block)
        xb = x[:, lo:hi]
        amax = np.abs(xb).max(axis=1)
        amax = np.maximum(amax, np.float32(1e-4))
        p = (amax * np.float32(1.0 / 448.0)).astype(np.float32)
        scale, _ = ceil_log2_pow2_f32(p)
        scales[:, b] = scale
        y = np.clip(xb / scale[:, None], -448.0, 448.0)
        codes[:, lo:hi] = e4m3_quant_rne(y)
    return codes, scales


def ue8m0_to_f64(b):
    """UE8M0 byte -> value, float64 (2^127 fits f64; byte 255 -> 2^128)."""
    return np.exp2(b.astype(np.int64) - 127)


# --- FP8 GEMM ---------------------------------------------------------------

def fp8_gemm_f64(act_codes, act_scales, w_codes, w_scales):
    """fp8_gemm semantics in float64. Returns (out, esc)."""
    M, K = act_codes.shape
    O = w_codes.shape[0]
    nb = (K + 127) // 128
    ad = E4M3_TABLE[act_codes]                       # [M, K] f64
    wd = E4M3_TABLE[w_codes]                         # [O, K] f64
    sb = ue8m0_to_f64(w_scales)                      # [ceil(O/128), nb] f64
    out = np.zeros((M, O), dtype=np.float64)
    esc = np.zeros((M, O), dtype=np.float64)
    oidx = np.arange(O) // 128
    for kb in range(nb):
        lo, hi = kb * 128, min(K, (kb + 1) * 128)
        dot = ad[:, lo:hi] @ wd[:, lo:hi].T          # [M, O] f64
        sc = act_scales[:, kb].astype(np.float64)[:, None] * sb[oidx, kb][None, :]
        out += dot * sc
        # honest FP32 error scale: intra-block cancellation can drive
        # |dot| far below sum|products|, which is what accumulation
        # rounding actually scales with
        esc += (np.abs(ad[:, lo:hi]) @ np.abs(wd[:, lo:hi]).T) * sc
    return out, esc


def gen_fp8(rng):
    M, O, K = 8, 320, 384     # 3 full K blocks; O = 2 full + 1 partial 128-tile

    # weight codes: random E4M3 codes (no NaN codes), structured corners
    mag = rng.integers(0, 127, size=(O, K), dtype=np.uint8)
    sgn = rng.integers(0, 2, size=(O, K), dtype=np.uint8) << np.uint8(7)
    w_codes = mag | sgn
    w_codes[0, 0:128] = 0x00                    # all-zero tile row segment
    w_codes[5, :] = 0x00                        # all-zero weight row
    w_codes[7, 128:256] = 0x7E                  # +448 saturation segment
    w_codes[9, 256:384] = 0xFE                  # -448 saturation segment
    w_codes[11, 0:128] = 0x01                   # min subnormal 2^-9
    w_codes[256:320, 256:384] = 0x00            # all-zero tile (see scale 254)

    # weight scales [ceil(O/128), ceil(K/128)] ue8m0
    oh, nb = (O + 127) // 128, (K + 127) // 128
    w_scales = rng.integers(112, 140, size=(oh, nb)).astype(np.uint8)
    w_scales[0, 1] = 1                          # near-denormal 2^-126
    w_scales[1, 0] = 0                          # 2^-127 (FP32 subnormal)
    w_scales[1, 2] = 200                        # 2^73, large but FP32-finite
    w_scales[2, 2] = 254                        # 2^127 on an all-zero tile -> 0

    # activations
    act_x = (rng.standard_normal((M, K)) * 3.0).astype(np.float32)
    act_x[1, :] = 0.0                           # all-zero row (amax floor)
    act_x[2, :] *= 1.0e4                        # huge row
    act_x[3, :] *= 1.0e-6                       # tiny row
    act_x[4, 128:] = 0.0                        # zero trailing blocks
    act_codes, act_scales = act_quant_ue8m0(act_x)

    out, esc = fp8_gemm_f64(act_codes, act_scales, w_codes, w_scales)

    def dump(name, arr):
        arr.tofile(os.path.join(OUT, name))

    dump("fp8_w_codes.bin", w_codes)
    dump("fp8_w_scales.bin", w_scales)
    dump("fp8_act_x.bin", act_x)
    dump("fp8_act_codes.bin", act_codes)
    dump("fp8_act_scales.bin", act_scales.astype(np.float32))
    dump("fp8_out.bin", out)
    dump("fp8_esc.bin", esc)
    with open(os.path.join(OUT, "fp8_manifest.txt"), "w") as f:
        f.write(f"M={M}\nO={O}\nK={K}\n")
    print(f"fp8 golden: M={M} O={O} K={K}, out range "
          f"[{out.min():.6g}, {out.max():.6g}]")


# --- mHC --------------------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sinkhorn_stages(c, n, iters, eps):
    """Exact kernel.py:398-423 order, float64; returns list of stages:
    stage 0 = after row softmax (+eps per element), stage 1 = after the
    first col normalization, stages 2.. alternating row/col normalization."""
    c = c.copy()
    stages = []
    # 1. row softmax with max subtraction, per-element + eps
    mx = c.max(axis=-1, keepdims=True)
    e = np.exp(c - mx)
    rs = e.sum(axis=-1, keepdims=True)
    c = e / rs + eps
    stages.append(c.copy())
    # 2. col normalize, eps in denominator
    cs = c.sum(axis=-2, keepdims=True)
    c = c / (cs + eps)
    stages.append(c.copy())
    # 3. (iters-1) x (row norm; col norm), eps in denominators
    for _ in range(iters - 1):
        rs = c.sum(axis=-1, keepdims=True)
        c = c / (rs + eps)
        stages.append(c.copy())
        cs = c.sum(axis=-2, keepdims=True)
        c = c / (cs + eps)
        stages.append(c.copy())
    return stages


def gen_mhc(rng):
    T, D, N = 6, 64, 4
    nmix = (2 + N) * N                     # 24
    nx = N * D

    x4 = (rng.standard_normal((T, N, D)) * 1.5).astype(np.float32)
    x4[1] = 0.0                            # zero state -> rsqrt(0 + eps)
    x4[2] *= 100.0                         # huge state
    x4[3] *= 1.0e-3                        # tiny state

    fn = (rng.standard_normal((nmix, nx)) * 0.05).astype(np.float32)
    scale = (rng.standard_normal(3) * 0.4 + 1.0).astype(np.float32)
    base = (rng.standard_normal(nmix) * 0.8).astype(np.float32)
    base[0] = 30.0                         # sigmoid saturation (pre -> 1)
    base[N] = -30.0                        # post -> 0
    base[2 * N + 3] = 25.0                 # comb logit dominance

    # reference semantics in float64 (rsqrt AFTER the matmul)
    x = x4.reshape(T, nx).astype(np.float64)
    rsqrt = 1.0 / np.sqrt((x ** 2).mean(axis=-1) + NORM_EPS)
    mixes = (x @ fn.astype(np.float64).T) * rsqrt[:, None]

    pre = sigmoid(mixes[:, :N] * scale[0] + base[:N]) + HC_EPS
    post = 2.0 * sigmoid(mixes[:, N:2 * N] * scale[1] + base[N:2 * N])
    logits = (mixes[:, 2 * N:].reshape(T, N, N) * scale[2]
              + base[2 * N:].reshape(N, N))
    comb = np.stack([sinkhorn_stages(logits[t], N, ITERS, HC_EPS)[-1]
                     for t in range(T)])

    ypre = np.einsum("tj,tjd->td", pre, x4.astype(np.float64))

    f = (rng.standard_normal((T, D)) * 2.0).astype(np.float32)
    ypost = (post[:, :, None] * f.astype(np.float64)[:, None, :]
             + np.einsum("tkj,tkd->tjd", comb, x4.astype(np.float64)))

    hfn = (rng.standard_normal((N, nx)) * 0.05).astype(np.float32)
    hscale = (rng.standard_normal(1) * 0.3 + 1.0).astype(np.float32)
    hbase = (rng.standard_normal(N) * 0.5).astype(np.float32)
    hmixes = (x @ hfn.astype(np.float64).T) * rsqrt[:, None]
    hpre = sigmoid(hmixes * hscale[0] + hbase) + HC_EPS
    yhead = np.einsum("tj,tjd->td", hpre, x4.astype(np.float64))

    # standalone Sinkhorn stage fixtures: wide logit range incl. extremes
    T2 = 8
    sk_logits = (rng.standard_normal((T2, N, N)) * 8.0).astype(np.float32)
    sk_logits[1] = 0.0                     # uniform -> uniform
    sk_logits[2] = 50.0                    # all-large equal (max subtraction)
    sk_logits[3, 0, 0] = 80.0              # one dominant logit
    sk_logits[4] = -60.0                   # all-very-negative equal
    stages = np.stack([
        np.stack(sinkhorn_stages(sk_logits[t].astype(np.float64),
                                 N, ITERS, HC_EPS))
        for t in range(T2)])               # [T2, 40, N, N]

    def dump(name, arr):
        arr.tofile(os.path.join(OUT, name))

    dump("mhc_x4.bin", x4)
    dump("mhc_fn.bin", fn)
    dump("mhc_scale.bin", scale)
    dump("mhc_base.bin", base)
    dump("mhc_mixes.bin", mixes)
    dump("mhc_pre.bin", pre)
    dump("mhc_post.bin", post)
    dump("mhc_comb.bin", comb)
    dump("mhc_ypre.bin", ypre)
    dump("mhc_f.bin", f)
    dump("mhc_ypost.bin", ypost)
    dump("mhc_hfn.bin", hfn)
    dump("mhc_hscale.bin", hscale)
    dump("mhc_hbase.bin", hbase)
    dump("mhc_yhead.bin", yhead)
    dump("mhc_sk_logits.bin", sk_logits)
    dump("mhc_sk_stages.bin", stages)
    with open(os.path.join(OUT, "mhc_manifest.txt"), "w") as fh:
        fh.write(f"T={T}\nD={D}\nN={N}\nT2={T2}\nITERS={ITERS}\n"
                 f"NSTAGES={1 + 1 + 2 * (ITERS - 1)}\n"
                 f"NORM_EPS={NORM_EPS!r}\nHC_EPS={HC_EPS!r}\n")
    rs = comb.sum(axis=-1)
    cs = comb.sum(axis=-2)
    print(f"mhc golden: T={T} D={D} N={N}; comb row-sum range "
          f"[{rs.min():.9f}, {rs.max():.9f}], col-sum range "
          f"[{cs.min():.9f}, {cs.max():.9f}]")


def main():
    rng = np.random.default_rng(20260729)
    os.makedirs(OUT, exist_ok=True)
    gen_fp8(rng)
    gen_mhc(rng)


if __name__ == "__main__":
    main()
