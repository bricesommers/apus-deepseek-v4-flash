#!/usr/bin/env python3
"""tools/oracle.py — M4b golden-reference oracle for the apus C forward pass.

A numpy-only port of ONE DeepSeek-V4-Flash transformer Block, exactly
following reference/inference/model.py (normative) with the tilelang kernel
semantics ported from reference/inference/kernel.py. Runs in two modes:

  * "f64" — the same algorithm with all arithmetic in float64 (no bf16
    rounding). This is the numerical "truth".
  * "f32" — dtype-faithful mode: bf16 activations where the reference is
    bf16, fp32 where fp32, and every QAT quant simulation (FP8-E4M3
    activation quant per-128/per-64, FP4-E2M1 act quant per-32, MXFP4/FP8
    weight dequant) applied at the same points as the reference. This mode
    is the golden target the C single-layer forward (M4c) is verified
    against.

Lossy quantization steps (activation quant, weight storage formats,
Hadamard-before-quant, index-score bf16 rounding) are ALGORITHMIC and are
applied in both modes; the modes differ only in arithmetic precision and
bf16 rounding of activations.

Runnable layer types (fixtures use layers.0/1/2 respectively):
  * "swa" — pure sliding-window attention (compress_ratio=0), hash routing
    via gate.tid2eid (mirrors real layers 0-2, ARCHITECTURE §3.2).
  * "csa" — ratio-4 overlapping compressor + lightning indexer.
  * "hca" — ratio-128 non-overlapping compressor, dense compressed
    attention, no indexer.

The module also generates the M4b fixtures (deterministic seeded weights in
apus-container-style safetensors + golden I/O + named intermediates) and
contains a loader that reads the fixture safetensors back through the real
tensor naming/format scheme, so the oracle itself consumes exactly what the
C loader will consume.

Reference line numbers cited as model.py:N / kernel.py:N refer to
reference/inference/model.py and reference/inference/kernel.py.
"""

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests", "m1"))
sys.path.insert(0, os.path.join(ROOT, "tests", "m3"))
import stutil                       # manual safetensors writer (M1)
import gen_golden as m3             # pinned quant ports (M3 hard gate)

# ---------------------------------------------------------------------------
# Small fixture config (see tests/m4b/README.md for the rationale).
# ---------------------------------------------------------------------------

SMALL_CFG = {
    "dim": 256,
    "n_heads": 4,
    "head_dim": 128,
    "rope_head_dim": 64,
    "q_lora_rank": 128,
    "o_groups": 2,
    "o_lora_rank": 64,
    "window_size": 128,
    "moe_inter_dim": 256,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "n_activated_experts": 3,
    "score_func": "sqrtsoftplus",
    "route_scale": 1.5,
    "swiglu_limit": 10.0,
    "index_n_heads": 4,
    "index_head_dim": 64,
    "index_topk": 8,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "compress_rope_theta": 160000.0,
    "original_seq_len": 65536,
    "rope_factor": 16.0,
    "beta_fast": 32,
    "beta_slow": 1,
    "vocab_size": 512,          # fixture-only: rows of tid2eid / input id range
    "max_pos": 512,             # freqs table length for fixtures
    # layer list: (fixture name, compress_ratio, hash_routed)
    "layers": [
        {"name": "swa", "compress_ratio": 0, "hash": True},
        {"name": "csa", "compress_ratio": 4, "hash": False},
        {"name": "hca", "compress_ratio": 128, "hash": False},
    ],
}

MASTER_SEED = 20260729

# ---------------------------------------------------------------------------
# bf16 helpers (numpy has no bf16; values are carried as f32 with a rounded
# mantissa, stored on disk as raw uint16).
# ---------------------------------------------------------------------------


def bf16_round(x):
    """Round-to-nearest-even to bf16, returned as float32 holding a bf16 value."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    bias = ((u >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    u = (u + bias) & np.uint32(0xFFFF0000)
    return u.view(np.float32)


def f32_to_bf16_bytes(x):
    return (bf16_round(x).view(np.uint32) >> np.uint32(16)).astype(np.uint16).tobytes()


def bf16_bytes_to_f32(b):
    return (np.frombuffer(b, dtype=np.uint16).astype(np.uint32)
            << np.uint32(16)).view(np.float32).copy()


def _dt(f64):
    return np.float64 if f64 else np.float32


def _B(x, f64):
    """bf16 rounding boundary: applied in f32-faithful mode only."""
    return x if f64 else bf16_round(x)


# ---------------------------------------------------------------------------
# Quant simulations (QAT) — algorithmic, applied in BOTH modes.
# Ports of kernel.py act_quant (40-125) / fp4_quant_kernel (128-200).
# ---------------------------------------------------------------------------


def fp8_qat(x, group, f64):
    """Inplace FP8-E4M3 quant->dequant simulation (kernel.py act_quant with
    round_scale=True / ue8m0, inplace=True). Per-group amax (floor 1e-4),
    pow2 scale, clamp +-448, RNE codes."""
    xf = np.ascontiguousarray(x, dtype=np.float32)
    shp = xf.shape
    K = shp[-1]
    assert K % group == 0, (K, group)
    codes, scales = m3.act_quant_ue8m0(xf.reshape(-1, K), group)
    y = m3.E4M3_TABLE[codes].astype(np.float32) * np.repeat(scales, group, axis=1)
    return y.reshape(shp).astype(_dt(f64))


def fp4_qat(x, f64):
    """Inplace FP4-E2M1 quant->dequant simulation (kernel.py fp4_act_quant,
    per-32 along last dim, amax floor 6*2^-126, pow2 scale, clamp +-6)."""
    xf = np.ascontiguousarray(x, dtype=np.float32)
    shp = xf.shape
    K = shp[-1]
    assert K % 32 == 0, K
    xb = xf.reshape(-1, K // 32, 32)
    amax = np.maximum(np.abs(xb).max(axis=2), m3.AMAX_FLOOR_FP4)
    scale, _ = m3.ceil_log2_pow2_f32((amax * np.float32(1.0 / 6.0)).astype(np.float32))
    y = np.clip(xb / scale[:, :, None], -6.0, 6.0).astype(np.float32)
    codes = m3.e2m1_quant_rne(y)
    out = m3.FP4_LUT[codes].astype(np.float32) * scale[:, :, None]
    return out.reshape(shp).astype(_dt(f64))


# ---------------------------------------------------------------------------
# RoPE (model.py:199-244) and Hadamard (model.py:247-251).
# ---------------------------------------------------------------------------


def precompute_freqs(dim, seqlen, original_seq_len, base, factor, beta_fast,
                     beta_slow, f64):
    """Literal port of precompute_freqs_cis (model.py:199-229), returned as
    (cos, sin) [seqlen, dim//2]. f32 mode computes in float32 like torch."""
    dt = _dt(f64)

    def find_correction_dim(num_rot):
        return dim * math.log(original_seq_len / (num_rot * 2 * math.pi)) / (2 * math.log(base))

    i = np.arange(0, dim, 2, dtype=dt)
    freqs = 1.0 / (base ** (i / dt(dim)))
    if original_seq_len > 0:
        low = max(math.floor(find_correction_dim(beta_fast)), 0)
        high = min(math.ceil(find_correction_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = np.clip((np.arange(dim // 2, dtype=dt) - low) / (high - low), 0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = np.arange(seqlen, dtype=dt)
    ang = np.outer(t, freqs)                      # [seqlen, dim//2]
    return np.cos(ang).astype(dt), np.sin(ang).astype(dt)


def apply_rope(x, cos, sin, inverse, f64):
    """apply_rotary_emb (model.py:232-244): interleaved pairs (even, odd),
    complex multiply; inverse = conjugate. x [..., d] with d = len*2."""
    dt = _dt(f64)
    x1 = x[..., 0::2].astype(dt)
    x2 = x[..., 1::2].astype(dt)
    c = cos.astype(dt)
    s = -sin.astype(dt) if inverse else sin.astype(dt)
    # broadcast cos/sin [t, d/2] against x [..., t?, heads?, d/2]: the token
    # axis of cos/sin aligns with the leading axes of x1; insert head axes.
    while c.ndim < x1.ndim:
        c = np.expand_dims(c, -2)
        s = np.expand_dims(s, -2)
    out = np.empty_like(x, dtype=dt)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return _B(out, f64)


_H_CACHE = {}


def hadamard_matrix(n):
    if n not in _H_CACHE:
        H = np.array([[1.0]])
        while H.shape[0] < n:
            H = np.block([[H, H], [H, -H]])
        _H_CACHE[n] = H
    return _H_CACHE[n]


def hadamard_rotate(x, f64):
    """rotate_activation (model.py:247-251): Sylvester Hadamard, scale d^-0.5.
    NOTE: fast_hadamard_transform's sign/order convention is not runnable
    here; the Sylvester convention is orthonormal so q.k is preserved
    pre-quant. See tests/m4b/README.md ambiguities."""
    n = x.shape[-1]
    H = hadamard_matrix(n).astype(_dt(f64))
    return _B(x.astype(_dt(f64)) @ H * (n ** -0.5), f64)


# ---------------------------------------------------------------------------
# Linear layers. Weights arrive DEQUANTIZED to f32/f64 by the loader; the
# activation-side QAT quant is applied here, mirroring model.py:108-120
# (linear -> act_quant -> fp4_gemm/fp8_gemm) and kernel.py accumulation.
# ---------------------------------------------------------------------------


def fp8_linear(x, w_codes, w_scales, f64):
    """FP8 E4M3 blockwise-128x128 dense GEMM (kernel.py:203-273):
    per-128-K-block dot of raw codes, accum += dot * (scale_a * scale_b).
    Weight scales are pow2, so folding them into w is exact; scale_a is
    applied per K-block like the kernel (kernel.py:242-249). Output bf16."""
    dt = _dt(f64)
    shp = x.shape
    K = shp[-1]
    xb = _B(x, f64)                               # gemm input is bf16
    xf = np.ascontiguousarray(xb, dtype=np.float32).reshape(-1, K)
    codes_a, sa = m3.act_quant_ue8m0(xf, 128)
    a = m3.E4M3_TABLE[codes_a].astype(dt)         # act codes, scale separate
    w = (m3.E4M3_TABLE[w_codes].astype(dt)
         * np.repeat(np.repeat(w_scales.astype(dt), 128, axis=0), 128, axis=1))
    O = w.shape[0]
    out = np.zeros((a.shape[0], O), dtype=dt)
    for kb in range(K // 128):
        dot = a[:, kb * 128:(kb + 1) * 128] @ w[:, kb * 128:(kb + 1) * 128].T
        out += dot * sa[:, kb].astype(dt)[:, None]
    return _B(out, f64).reshape(*shp[:-1], O)


def fp4_linear(x, w_packed, w_scales, f64):
    """MXFP4 expert GEMM (kernel.py:441-515): activation FP8-quant per-128;
    per-32-K-block dot of (fp8 act vals) x (fp4 LUT vals), then
    accum += (dot * scale_a[kb//4]) * scale_b[kb]. Output bf16."""
    dt = _dt(f64)
    shp = x.shape
    K = shp[-1]
    xb = _B(x, f64)
    xf = np.ascontiguousarray(xb, dtype=np.float32).reshape(-1, K)
    codes_a, sa = m3.act_quant_ue8m0(xf, 128)
    a = m3.E4M3_TABLE[codes_a].astype(dt)         # act codes (scale separate)
    wl = np.empty((w_packed.shape[0], K), dtype=dt)
    wl[:, 0::2] = m3.FP4_LUT[w_packed & np.uint8(0x0F)].astype(dt)
    wl[:, 1::2] = m3.FP4_LUT[w_packed >> np.uint8(4)].astype(dt)
    sb = np.exp2(w_scales.astype(np.int64) - 127).astype(dt)   # [O, K/32]
    out = np.zeros((xf.shape[0], w_packed.shape[0]), dtype=dt)
    for kb in range(K // 32):
        dot = a[:, kb * 32:(kb + 1) * 32] @ wl[:, kb * 32:(kb + 1) * 32].T
        out += (dot * sa[:, (kb * 32) // 128].astype(dt)[:, None]) * sb[:, kb][None, :]
    return _B(out, f64).reshape(*shp[:-1], wl.shape[0])


def bf16_linear(x, w, f64):
    """Plain bf16 matmul: fp32-accumulate, bf16 out (torch semantics)."""
    dt = _dt(f64)
    y = _B(x, f64).astype(dt) @ w.astype(dt).T
    return _B(y, f64)


def f32_linear(x, w, f64):
    """fp32 matmul, no output rounding (compressor wkv/wgate, gate)."""
    dt = _dt(f64)
    return x.astype(dt) @ w.astype(dt).T


# ---------------------------------------------------------------------------
# Norms, activations, Sinkhorn.
# ---------------------------------------------------------------------------


def rms_norm(x, w, eps, f64):
    """RMSNorm (model.py:183-196): fp32 internal, weight mul, cast back."""
    dt = _dt(f64)
    xf = x.astype(dt)
    var = (xf * xf).mean(-1, keepdims=True)
    y = w.astype(dt) * (xf * (1.0 / np.sqrt(var + eps)))
    return _B(y, f64)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    """torch F.softplus (beta=1, threshold=20)."""
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def softmax(x, axis):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def hc_split_sinkhorn(mixes, scale, base, hc, iters, eps, f64):
    """Port of kernel.py:371-438 (hc_split_sinkhorn): pre/post/comb split,
    row-softmax + eps, col-norm, then (iters-1) x (row-norm, col-norm)."""
    dt = _dt(f64)
    mixes = mixes.astype(dt)
    pre = sigmoid(mixes[..., :hc] * dt(scale[0]) + base[:hc].astype(dt)) + eps
    post = 2.0 * sigmoid(mixes[..., hc:2 * hc] * dt(scale[1]) + base[hc:2 * hc].astype(dt))
    comb = (mixes[..., 2 * hc:].reshape(*mixes.shape[:-1], hc, hc)
            * dt(scale[2]) + base[2 * hc:].astype(dt).reshape(hc, hc))
    # row softmax + eps (kernel.py:401-408)
    comb = comb - comb.max(axis=-1, keepdims=True)
    comb = np.exp(comb)
    comb = comb / comb.sum(axis=-1, keepdims=True) + eps
    # col norm (kernel.py:410-413)
    comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(axis=-1, keepdims=True) + eps)
        comb = comb / (comb.sum(axis=-2, keepdims=True) + eps)
    return pre, post, comb


# ---------------------------------------------------------------------------
# Index helpers (model.py:254-276).
# ---------------------------------------------------------------------------


def window_topk_idxs(win, seqlen, start_pos):
    """get_window_topk_idxs (model.py:254-265), batch squeezed out."""
    if start_pos >= win - 1:
        sp = start_pos % win
        m = np.concatenate([np.arange(sp + 1, win), np.arange(0, sp + 1)])
        return m[None, :].repeat(seqlen, axis=0)
    elif start_pos > 0:
        m = np.full(win, -1, dtype=np.int64)
        m[:start_pos + 1] = np.arange(start_pos + 1)
        return m[None, :].repeat(seqlen, axis=0)
    base = np.arange(seqlen)[:, None]
    m = np.clip(base - win + 1, 0, None) + np.arange(min(seqlen, win))[None, :]
    return np.where(m > base, -1, m)


def compress_topk_idxs(ratio, seqlen, start_pos, offset):
    """get_compress_topk_idxs (model.py:268-276), HCA dense selection."""
    if start_pos > 0:
        m = np.arange(0, (start_pos + 1) // ratio) + offset
        return m[None, :].repeat(seqlen, axis=0)
    nb = seqlen // ratio
    m = np.arange(nb)[None, :].repeat(seqlen, axis=0)
    mask = m >= (np.arange(1, seqlen + 1)[:, None] // ratio)
    return np.where(mask, -1, m + offset)


def topk_stable(row, k):
    """torch.topk descending; stable ties -> lower index first (ambiguity A6)."""
    k = min(k, row.shape[-1])
    return np.argsort(-row, kind="stable", axis=-1)[..., :k]


# ---------------------------------------------------------------------------
# sparse_attn (kernel.py:276-368): index-gather attention with attention
# sink. Gathered-order serial softmax; probs rounded to bf16 before P.V
# (acc_s_cast, kernel.py:340). Sink enters the denominator only
# (kernel.py:345-348).
# ---------------------------------------------------------------------------


def sparse_attn(q, kv, sink, idxs, scale, f64):
    """q [s,h,d] (bf16 values), kv [n,d], sink [h] f32, idxs [s,k] (-1 = skip).
    Returns o [s,h,d] bf16-rounded (f32 mode)."""
    dt = _dt(f64)
    s, h, d = q.shape
    o = np.zeros((s, h, d), dtype=dt)
    for t in range(s):
        ids = idxs[t]
        ids = ids[ids >= 0]
        if ids.size == 0:
            continue                                # denom: exp(sink-(-inf))=inf -> 0
        k = kv[ids].astype(dt)
        sc = (q[t].astype(dt) @ k.T) * dt(scale)    # [h, n]
        m = sc.max(axis=1, keepdims=True)
        p = np.exp(sc - m)
        p = _B(p, f64)                              # acc_s_cast to bf16
        acc = p.astype(dt) @ k
        sum_exp = p.sum(axis=1) + np.exp(sink.astype(dt) - m[:, 0])
        o[t] = _B(acc / sum_exp[:, None], f64)
    return o


# ---------------------------------------------------------------------------
# Compressor (model.py:279-377) with the decode-time state machine.
# ---------------------------------------------------------------------------


class CompressorState:
    def __init__(self, ratio, overlap, d, f64):
        coff = 1 + overlap
        dt = _dt(f64)
        self.kv = np.zeros((coff * ratio, coff * d), dtype=dt)
        self.sc = np.full((coff * ratio, coff * d), -np.inf, dtype=dt)
        self.cache = np.zeros((0, d), dtype=np.float32)  # bf16-valued entries


def _overlap_transform(t, fill):
    """model.py:307-314. t [nb, ratio, 2d] -> [nb, 2*ratio, d]."""
    nb, ratio, cd = t.shape
    d = cd // 2
    out = np.full((nb, 2 * ratio, d), fill, dtype=t.dtype)
    out[:, ratio:] = t[:, :, d:]
    out[1:, :ratio] = t[:-1, :, :d]
    return out


def compressor_forward(cp, x, start_pos, st: CompressorState, cos, sin,
                       f64, interm=None, tag=""):
    """Port of Compressor.forward (model.py:316-377). x [s, dim] bf16 values.
    Mutates st; returns compressed kv [nb, d] produced THIS call or None."""
    ratio, overlap, d, rd = cp["ratio"], cp["overlap"], cp["d"], cp["rd"]
    rotate = cp["rotate"]
    coff = 1 + overlap
    dt = _dt(f64)
    s = x.shape[0]
    xf = x.astype(dt)                               # x.float() (line 322)
    kv = f32_linear(xf, cp["wkv"], f64)             # [s, coff*d]
    score = f32_linear(xf, cp["wgate"], f64)
    ape = cp["ape"].astype(dt)

    if start_pos == 0:
        should = s >= ratio
        rem = s % ratio
        cutoff = s - rem
        off = ratio if overlap else 0
        if overlap and cutoff >= ratio:             # lines 330-332
            st.kv[:ratio] = kv[cutoff - ratio:cutoff]
            st.sc[:ratio] = score[cutoff - ratio:cutoff] + ape
        if rem > 0:                                 # lines 333-336
            st.kv[off:off + rem] = kv[cutoff:]
            st.sc[off:off + rem] = score[cutoff:] + ape[:rem]
        kv_c = kv[:cutoff].reshape(-1, ratio, coff * d)
        sc_c = score[:cutoff].reshape(-1, ratio, coff * d) + ape
        if overlap:                                 # lines 339-341
            kv_c = _overlap_transform(kv_c, 0.0)
            sc_c = _overlap_transform(sc_c, -np.inf)
        out = (kv_c * softmax(sc_c, axis=1)).sum(axis=1)      # [nb, d] f32
        rope_pos = np.arange(0, cutoff, ratio)
    else:
        assert s == 1, "decode path expects one token"
        should = (start_pos + 1) % ratio == 0       # line 344
        score1 = score[0] + ape[start_pos % ratio]  # line 345
        if overlap:                                 # lines 346-354
            st.kv[ratio + start_pos % ratio] = kv[0]
            st.sc[ratio + start_pos % ratio] = score1
            if should:
                kvs = np.concatenate([st.kv[:ratio, :d], st.kv[ratio:, d:]], axis=0)
                scs = np.concatenate([st.sc[:ratio, :d], st.sc[ratio:, d:]], axis=0)
                out = (kvs * softmax(scs, axis=0)).sum(axis=0, keepdims=True)
                st.kv[:ratio] = st.kv[ratio:]
                st.sc[:ratio] = st.sc[ratio:]
            else:
                out = None
        else:                                       # lines 355-359
            st.kv[start_pos % ratio] = kv[0]
            st.sc[start_pos % ratio] = score1
            if should:
                out = (st.kv * softmax(st.sc, axis=0)).sum(axis=0, keepdims=True)
            else:
                out = None
        rope_pos = np.array([start_pos + 1 - ratio])

    if not should:
        return None
    out = rms_norm(_B(out, f64), cp["norm"], cp["eps"], f64)   # line 362
    out = out.copy()
    out[..., -rd:] = apply_rope(out[..., -rd:], cos[rope_pos], sin[rope_pos],
                                False, f64)                     # lines 363-367
    if rotate:                                      # lines 368-370 (indexer)
        out = hadamard_rotate(out, f64)
        out = fp4_qat(out, f64).astype(dt)
        out = _B(out, f64)
    else:                                           # line 372
        out[..., :-rd] = fp8_qat(out[..., :-rd], 64, f64)
        out = _B(out, f64)
    st.cache = np.concatenate([st.cache, out.astype(np.float32)], axis=0)
    if interm is not None and tag:
        interm[tag] = out.copy()
    return out


# ---------------------------------------------------------------------------
# Indexer (model.py:380-433).
# ---------------------------------------------------------------------------


def indexer_forward(P, x, qr, start_pos, st, f64, interm):
    """Returns topk idxs [s, k] into the combined (window ++ compressed) kv,
    -1 where masked. Mutates the indexer compressor state."""
    dt = _dt(f64)
    s = x.shape[0]
    ih, idm, rd = P["idx_heads"], P["idx_dim"], P["rope_head_dim"]
    ratio = P["ratio"]
    end_pos = start_pos + s
    cos, sin = P["cos"], P["sin"]
    q = fp8_linear(qr, *P["idx_wq_b"], f64).reshape(s, ih, idm)      # line 411
    q = q.copy()
    q[..., -rd:] = apply_rope(q[..., -rd:], cos[start_pos:start_pos + s],
                              sin[start_pos:start_pos + s], False, f64)  # 413
    q = hadamard_rotate(q, f64)                                        # 414
    q = _B(fp4_qat(q, f64), f64)                                       # 416
    compressor_forward(P["idx_comp"], x, start_pos, st.idx_comp, cos, sin,
                       f64, interm, "idx_comp_kv")                     # 417
    w = bf16_linear(x, P["idx_weights_proj"], f64)
    w = _B(w * dt(P["idx_dim"] ** -0.5 * P["idx_heads"] ** -0.5), f64)  # 418
    nb = end_pos // ratio
    kvc = st.idx_comp.cache[:nb].astype(dt)
    sc = np.einsum("shd,td->sht", q.astype(dt), kvc)                   # 420
    sc = _B(sc, f64)
    sc = _B(np.maximum(sc, 0) * w[:, :, None], f64).sum(axis=1, dtype=dt)  # 421
    sc = _B(sc, f64)
    if interm is not None:
        interm["idx_scores"] = sc.copy()
    if start_pos == 0:                                                 # 424-426
        mask = (np.arange(nb)[None, :]
                >= (np.arange(1, s + 1)[:, None] // ratio))
        sc = sc + np.where(mask, -np.inf, 0.0)
    idx = topk_stable(sc, min(P["idx_topk"], nb))                      # 427
    offset = s if start_pos == 0 else P["window"]                      # 509
    if start_pos == 0:                                                 # 428-430
        ill = idx >= (np.arange(1, s + 1)[:, None] // ratio)
        idx = np.where(ill, -1, idx + offset)
    else:
        idx = idx + offset                                             # 432
    if interm is not None:
        interm["idx_topk"] = idx.astype(np.int32)
    return idx


# ---------------------------------------------------------------------------
# Attention (model.py:436-543).
# ---------------------------------------------------------------------------


class LayerState:
    def __init__(self, cfg, layer, f64):
        self.pos = 0
        d, win = cfg["head_dim"], cfg["window_size"]
        self.win = np.zeros((win, d), dtype=np.float32)   # ring, bf16 values
        ratio = layer["compress_ratio"]
        self.ratio = ratio
        if ratio:
            overlap = ratio == 4
            self.comp = CompressorState(ratio, overlap, d, f64)
            if ratio == 4:
                self.idx_comp = CompressorState(ratio, True,
                                                cfg["index_head_dim"], f64)


def attention_forward(P, x, start_pos, st: LayerState, f64, interm):
    """Port of Attention.forward (model.py:484-543). x [s, dim]; returns
    [s, dim] attention sublayer output."""
    dt = _dt(f64)
    s = x.shape[0]
    h, d, rd = P["n_heads"], P["head_dim"], P["rope_head_dim"]
    win, ratio = P["window"], P["ratio"]
    cos, sin = P["cos"], P["sin"]
    fc, fs = cos[start_pos:start_pos + s], sin[start_pos:start_pos + s]

    qr = rms_norm(fp8_linear(x, *P["wq_a"], f64), P["q_norm"], P["eps"], f64)  # 496
    q = fp8_linear(qr, *P["wq_b"], f64).reshape(s, h, d)                       # 497
    # per-head weight-free RMSNorm (line 498)
    qf = q.astype(dt)
    q = _B(qf * (1.0 / np.sqrt((qf * qf).mean(-1, keepdims=True) + P["eps"])), f64)
    q = q.copy()
    q[..., -rd:] = apply_rope(q[..., -rd:], fc, fs, False, f64)                # 499
    if interm is not None:
        interm["q"] = q.copy()

    kv = rms_norm(fp8_linear(x, *P["wkv"], f64), P["kv_norm"], P["eps"], f64)  # 502-503
    kv = kv.copy()
    kv[..., -rd:] = apply_rope(kv[..., -rd:], fc, fs, False, f64)              # 504
    kv[..., :-rd] = fp8_qat(kv[..., :-rd], 64, f64)                            # 506
    kv = _B(kv, f64)
    if interm is not None:
        interm["win_kv"] = kv.copy()

    win_idx = window_topk_idxs(win, s, start_pos)                              # 507
    if ratio:
        offset = s if start_pos == 0 else win                                  # 509
        if P["has_indexer"]:
            comp_idx = indexer_forward(P, x, qr, start_pos, st, f64, interm)   # 511
        else:
            comp_idx = compress_topk_idxs(ratio, s, start_pos, offset)         # 513
        idxs = np.concatenate([win_idx, comp_idx], axis=-1)                    # 514
    else:
        idxs = win_idx

    if start_pos == 0:                                        # lines 518-528
        if s <= win:
            st.win[:s] = kv
        else:
            cut = s % win
            st.win[cut:win] = kv[s - win:s - win + (win - cut)]
            st.win[:cut] = kv[s - cut:]
        if ratio:
            kv_c = compressor_forward(P["comp"], x, 0, st.comp, cos, sin,
                                      f64, interm, "comp_kv")                 # 525
            attn_kv = (np.concatenate([kv, kv_c], axis=0)
                       if kv_c is not None else kv)
        else:
            attn_kv = kv
    else:                                                     # lines 529-533
        assert s == 1
        st.win[start_pos % win] = kv[0]
        if ratio:
            compressor_forward(P["comp"], x, start_pos, st.comp, cos, sin,
                               f64, interm, "comp_kv")                        # 532
        attn_kv = np.concatenate([st.win, st.comp.cache], axis=0) \
            if ratio else st.win
    o = sparse_attn(q, attn_kv, P["attn_sink"], idxs, d ** -0.5, f64)          # 528/533
    o = o.copy()
    o[..., -rd:] = apply_rope(o[..., -rd:], fc, fs, True, f64)                 # 534
    if interm is not None:
        interm["attn_out"] = o.copy()

    # grouped low-rank o-proj (lines 537-542); wo_a used as bf16 (line 462)
    G, o_lora = P["o_groups"], P["o_lora"]
    og = o.reshape(s, G, h * d // G)
    wo_a = P["wo_a"].reshape(G, o_lora, h * d // G)
    y = np.stack([_B(og[:, g, :].astype(dt) @ wo_a[g].astype(dt).T, f64)
                  for g in range(G)], axis=1)                                  # 541
    out = fp8_linear(_B(y.reshape(s, G * o_lora), f64), *P["wo_b"], f64)       # 542
    if interm is not None:
        interm["o_out"] = out.copy()
    return out


# ---------------------------------------------------------------------------
# MoE (model.py:546-644).
# ---------------------------------------------------------------------------


def gate_forward(P, x, input_ids, f64, interm):
    """Gate (model.py:564-584): sqrtsoftplus scores, bias for selection only,
    weights from unbiased scores, normalized x route_scale; hash override."""
    dt = _dt(f64)
    scores = f32_linear(x.astype(dt), P["gate_w"], f64)          # line 565
    sp = np.sqrt(softplus(scores))                               # line 571
    if interm is not None:
        interm["router_scores"] = sp.copy()
    if P["hash"]:
        idx = P["tid2eid"][input_ids]                            # line 577
    else:
        biased = sp + P["gate_bias"].astype(dt)                  # line 575
        if interm is not None:
            interm["router_scores_biased"] = biased.copy()
        idx = topk_stable(biased, P["topk"])                     # line 579
    w = np.take_along_axis(sp, idx, axis=1)                      # line 580
    w = w / w.sum(axis=-1, keepdims=True)                        # line 582
    w = w * dt(P["route_scale"])                                 # line 583
    if interm is not None:
        interm["router_idx"] = idx.astype(np.int32)
        interm["router_w"] = w.copy()
    return w, idx


def expert_forward(x, w1, w2, w3, kind, limit, f64):
    """Expert SwiGLU (model.py:596-606): fp32 compute, swiglu_limit clamps."""
    dt = _dt(f64)
    lin = fp4_linear if kind == "fp4" else fp8_linear
    g = lin(x, *w1, f64).astype(dt)
    u = lin(x, *w3, f64).astype(dt)
    if limit > 0:                                                # lines 600-602
        u = np.clip(u, -limit, limit)
        g = np.minimum(g, limit)
    h = (g * sigmoid(g)) * u                                     # line 603
    return lin(_B(h, f64), *w2, f64)                             # line 606


def moe_forward(P, x, input_ids, f64, interm):
    """MoE (model.py:629-644): fp32 accumulation over routed + shared."""
    dt = _dt(f64)
    w, idx = gate_forward(P, x, input_ids, f64, interm)
    y = np.zeros_like(x, dtype=dt)
    for e in range(P["n_routed"]):
        tok, slot = np.where(idx == e)
        if tok.size == 0:
            continue
        out = expert_forward(x[tok], *P["experts"][e], "fp4", P["limit"], f64)
        y[tok] += w[tok, slot, None].astype(dt) * out.astype(dt)  # line 640
    if interm is not None:
        interm["moe_routed"] = y.copy()
    shared = expert_forward(x, *P["shared"], "fp8", P["limit"], f64)
    if interm is not None:
        interm["moe_shared"] = shared.copy()
    y = _B(y + shared.astype(dt), f64)                            # line 643-644
    if interm is not None:
        interm["moe_out"] = y.copy()
    return y


# ---------------------------------------------------------------------------
# Block with mHC wiring (model.py:647-700).
# ---------------------------------------------------------------------------


def hc_pre(h, fn, scale, base, cfg, f64, interm, tag):
    """model.py:673-681: flatten, rsqrt of unnormalized x, mixes, sinkhorn."""
    dt = _dt(f64)
    s = h.shape[0]
    hc = cfg["hc_mult"]
    xf = h.reshape(s, hc * cfg["dim"]).astype(dt)
    rsqrt = 1.0 / np.sqrt((xf * xf).mean(-1, keepdims=True) + cfg["norm_eps"])
    mixes = (xf @ fn.astype(dt).T) * rsqrt                       # line 678
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, hc,
                                        cfg["hc_sinkhorn_iters"],
                                        cfg["hc_eps"], f64)
    if interm is not None:
        interm[tag + "_pre"] = pre.copy()
        interm[tag + "_post"] = post.copy()
        interm[tag + "_comb"] = comb.copy()
    y = (pre[..., None] * xf.reshape(s, hc, cfg["dim"])).sum(axis=1)  # line 680
    return _B(y, f64), post, comb


def hc_post(x, residual, post, comb, f64):
    """model.py:683-686: y_j = post_j * x + sum_i comb[i,j] * res_i."""
    dt = _dt(f64)
    y = (post[..., None].astype(dt) * x[:, None, :].astype(dt)
         + np.einsum("sij,sid->sjd", comb.astype(dt), residual.astype(dt)))
    return _B(y, f64)


def block_forward(P, cfg, h, input_ids, start_pos, st, f64, interm=None):
    """Block.forward (model.py:688-700). h [s, hc, dim]; returns same."""
    if interm is None:
        interm = {}
    x, post, comb = hc_pre(h, P["hc_attn_fn"], P["hc_attn_scale"],
                           P["hc_attn_base"], cfg, f64, interm, "attn_hc")
    x = rms_norm(x, P["attn_norm"], cfg["norm_eps"], f64)        # line 691
    if "attn_norm_out" not in interm:
        interm["attn_norm_out"] = x.copy()
    x = attention_forward(P, x, start_pos, st, f64, interm)      # line 692
    h = hc_post(x, h, post, comb, f64)                           # line 693
    if "post_attn_h" not in interm:
        interm["post_attn_h"] = h.copy()

    x, post, comb = hc_pre(h, P["hc_ffn_fn"], P["hc_ffn_scale"],
                           P["hc_ffn_base"], cfg, f64, interm, "ffn_hc")
    x = rms_norm(x, P["ffn_norm"], cfg["norm_eps"], f64)         # line 697
    interm["ffn_norm_out"] = x.copy()
    x = moe_forward(P, x, input_ids, f64, interm)                # line 698
    h = hc_post(x, h, post, comb, f64)                           # line 699
    return h, interm


# ===========================================================================
# Fixture generation: seeded weights -> real-format safetensors -> loader.
# ===========================================================================


def _e8m0_bytes_from_f32(scale_f32):
    """pow2 scale f32 -> UE8M0 byte (exponent + 127)."""
    e = ((scale_f32.view(np.uint32) >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int64)
    return (e).astype(np.uint8)  # biased exponent IS the byte


def fp8_store(W):
    """Quantize a raw f32 matrix [O, K] to FP8-E4M3 blockwise 128x128 with
    UE8M0 scales (amax floor 1e-4, pow2 via the kernel bit trick — same rule
    family as act_quant; see README ambiguities A9). Returns
    (codes_bytes, scale_bytes, shapes)."""
    W = np.ascontiguousarray(W, dtype=np.float32)
    O, K = W.shape
    assert O % 128 == 0 and K % 128 == 0, (O, K)
    blocks = W.reshape(O // 128, 128, K // 128, 128).transpose(0, 2, 1, 3)
    amax = np.maximum(np.abs(blocks).max(axis=(2, 3)), np.float32(1e-4))
    scale, _ = m3.ceil_log2_pow2_f32((amax * np.float32(1.0 / 448.0)).astype(np.float32))
    y = np.clip(blocks / scale[:, :, None, None], -448.0, 448.0)
    codes = m3.e4m3_quant_rne(y.astype(np.float32))
    codes = codes.transpose(0, 2, 1, 3).reshape(O, K)
    return codes.tobytes(), _e8m0_bytes_from_f32(scale).tobytes(), (O, K), (O // 128, K // 128)


def fp4_store(W):
    """Raw f32 [O, K] -> MXFP4 packed I8 + UE8M0 scales (M3-pinned rule)."""
    packed, scales = m3.fp4_quant(np.ascontiguousarray(W, dtype=np.float32))
    return packed.tobytes(), scales.tobytes(), packed.shape, scales.shape


def gen_layer_tensors(cfg, layer, layer_idx, rng, prefix=None):
    """Generate one layer's tensors as (name, dtype, shape, payload) records
    using the REAL checkpoint naming scheme (ARCHITECTURE §3.5). prefix
    overrides the "layers.{idx}" namespace (M8: "mtp.0")."""
    L = prefix if prefix is not None else f"layers.{layer_idx}"
    dim = cfg["dim"]
    h, d, rd = cfg["n_heads"], cfg["head_dim"], cfg["rope_head_dim"]
    ql, G, ol = cfg["q_lora_rank"], cfg["o_groups"], cfg["o_lora_rank"]
    inter, E, topk = cfg["moe_inter_dim"], cfg["n_routed_experts"], cfg["n_activated_experts"]
    ratio = layer["compress_ratio"]
    recs = []

    def rnd(*shape, std=None):
        k = shape[-1]
        return (rng.standard_normal(shape)
                * (std if std is not None else 1.0 / math.sqrt(k))).astype(np.float32)

    def fp8(name, W):
        cb, sb, csh, ssh = fp8_store(W)
        recs.append((f"{L}.{name}.weight", "F8_E4M3", csh, cb))
        recs.append((f"{L}.{name}.scale", "F8_E8M0", ssh, sb))

    def bf16(name, W):
        recs.append((f"{L}.{name}.weight", "BF16", W.shape, f32_to_bf16_bytes(W)))

    def f32raw(name, W):
        recs.append((f"{L}.{name}", "F32", W.shape,
                     np.ascontiguousarray(W, np.float32).tobytes()))

    # attention dense (FP8)
    fp8("attn.wq_a", rnd(ql, dim))
    fp8("attn.wq_b", rnd(h * d, ql))
    fp8("attn.wkv", rnd(d, dim))
    fp8("attn.wo_a", rnd(G * ol, h * d // G))
    fp8("attn.wo_b", rnd(dim, G * ol))
    bf16("attn.q_norm", (1.0 + 0.05 * rng.standard_normal(ql)).astype(np.float32))
    bf16("attn.kv_norm", (1.0 + 0.05 * rng.standard_normal(d)).astype(np.float32))
    f32raw("attn.attn_sink", (0.5 * rng.standard_normal(h)).astype(np.float32))
    bf16("attn_norm", (1.0 + 0.05 * rng.standard_normal(dim)).astype(np.float32))
    bf16("ffn_norm", (1.0 + 0.05 * rng.standard_normal(dim)).astype(np.float32))

    def compressor_recs(prefix, cdim):
        coff = 1 + (ratio == 4)
        bf16(f"{prefix}.wkv", rnd(coff * cdim, dim))
        bf16(f"{prefix}.wgate", rnd(coff * cdim, dim))
        f32raw(f"{prefix}.ape", (0.05 * rng.standard_normal((ratio, coff * cdim))).astype(np.float32))
        bf16(f"{prefix}.norm", (1.0 + 0.05 * rng.standard_normal(cdim)).astype(np.float32))

    if ratio:
        compressor_recs("attn.compressor", d)
        if ratio == 4:
            ih, idm = cfg["index_n_heads"], cfg["index_head_dim"]
            fp8("attn.indexer.wq_b", rnd(ih * idm, ql))
            bf16("attn.indexer.weights_proj", rnd(ih, dim))
            compressor_recs("attn.indexer.compressor", idm)

    # MoE
    bf16("ffn.gate", (0.05 * rng.standard_normal((E, dim))).astype(np.float32))
    if layer["hash"]:
        perm = np.stack([rng.permutation(E)[:topk]
                         for _ in range(cfg["vocab_size"])]).astype(np.int64)
        recs.append((f"{L}.ffn.gate.tid2eid", "I64", perm.shape, perm.tobytes()))
    else:
        f32raw("ffn.gate.bias", (0.1 * rng.standard_normal(E)).astype(np.float32))
    for e in range(E):
        for wname, shape in (("w1", (inter, dim)), ("w2", (dim, inter)),
                             ("w3", (inter, dim))):
            pb, sb, psh, ssh = fp4_store(rnd(*shape))
            recs.append((f"{L}.ffn.experts.{e}.{wname}.weight", "I8", psh, pb))
            recs.append((f"{L}.ffn.experts.{e}.{wname}.scale", "F8_E8M0", ssh, sb))
    for wname, shape in (("w1", (inter, dim)), ("w2", (dim, inter)),
                         ("w3", (inter, dim))):
        fp8(f"ffn.shared_experts.{wname}", rnd(*shape))

    # mHC (all FP32)
    mix_hc = (2 + cfg["hc_mult"]) * cfg["hc_mult"]
    hc_dim = cfg["hc_mult"] * dim
    for sub in ("attn", "ffn"):
        f32raw(f"hc_{sub}_fn", rnd(mix_hc, hc_dim))
        f32raw(f"hc_{sub}_base", (0.25 * rng.standard_normal(mix_hc)).astype(np.float32))
        f32raw(f"hc_{sub}_scale", (1.0 + 0.1 * rng.standard_normal(3)).astype(np.float32))
    return recs


def write_weights(cfg, out_dir):
    """Generate all layer weights and write apus-style shards + index."""
    os.makedirs(out_dir, exist_ok=True)
    dense_recs, expert_recs = [], []
    for li, layer in enumerate(cfg["layers"]):
        rng = np.random.default_rng(MASTER_SEED + 1000 * li)
        for rec in gen_layer_tensors(cfg, layer, li, rng):
            (expert_recs if ".experts." in rec[0] else dense_recs).append(rec)
    shards = []
    if dense_recs:
        shards.append(("apus-00001.safetensors", dense_recs))
    if expert_recs:
        shards.append(("apus-00002.safetensors", expert_recs))
    weight_map = {}
    total = 0
    for fname, recs in shards:
        stutil.write_shard(os.path.join(out_dir, fname), recs)
        for name, dtype, shape, payload in recs:
            weight_map[name] = fname
            total += len(payload)
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=1)
    return weight_map


# ---------------------------------------------------------------------------
# Loader: reads the fixture safetensors back into oracle params. Exercises
# the real naming/format path the C loader (M4a/M4c) must implement.
# ---------------------------------------------------------------------------


class ShardSet:
    def __init__(self, weights_dir):
        with open(os.path.join(weights_dir, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self.dir = weights_dir
        self._cache = {}

    def raw(self, name):
        fname = self.weight_map[name]
        if fname not in self._cache:
            self._cache[fname] = stutil.read_tensor_bytes(os.path.join(self.dir, fname))
        return self._cache[fname][name]

    def meta(self, name):
        import struct as _st
        fname = self.weight_map[name]
        header, _ = stutil.read_shard(os.path.join(self.dir, fname))
        return header[name]["dtype"], header[name]["shape"]

    def f32(self, name):
        dtype, shape = self.meta(name)
        b = self.raw(name)
        if dtype == "F32":
            return np.frombuffer(b, np.float32).reshape(shape).copy()
        if dtype == "BF16":
            return bf16_bytes_to_f32(b).reshape(shape)
        raise ValueError(f"{name}: {dtype}")

    def i64(self, name):
        dtype, shape = self.meta(name)
        assert dtype == "I64", (name, dtype)
        return np.frombuffer(self.raw(name), np.int64).reshape(shape).copy()

    def fp8(self, name):
        """-> (codes u8 [O,K], scales f32 [O/128,K/128]) with scales as pow2 f32."""
        dt, csh = self.meta(name + ".weight")
        assert dt == "F8_E4M3", (name, dt)
        codes = np.frombuffer(self.raw(name + ".weight"), np.uint8).reshape(csh)
        dt, ssh = self.meta(name + ".scale")
        assert dt == "F8_E8M0", (name, dt)
        sb = np.frombuffer(self.raw(name + ".scale"), np.uint8).reshape(ssh)
        scales = np.exp2(sb.astype(np.int32) - 127).astype(np.float32)
        return codes.copy(), scales

    def fp4(self, name):
        """-> (packed u8 [O,K/2], scale bytes u8 [O,K/32])."""
        dt, psh = self.meta(name + ".weight")
        assert dt == "I8", (name, dt)
        packed = np.frombuffer(self.raw(name + ".weight"), np.uint8).reshape(psh)
        dt, ssh = self.meta(name + ".scale")
        assert dt == "F8_E8M0", (name, dt)
        scales = np.frombuffer(self.raw(name + ".scale"), np.uint8).reshape(ssh)
        return packed.copy(), scales.copy()


def load_layer_params(shards: ShardSet, cfg, layer, layer_idx, f64,
                      prefix=None):
    """Build the oracle param dict for one layer from the safetensors.
    prefix overrides the "layers.{idx}" namespace (M8: "mtp.0")."""
    L = prefix if prefix is not None else f"layers.{layer_idx}"
    ratio = layer["compress_ratio"]
    rd = cfg["rope_head_dim"]
    if ratio:
        cos, sin = precompute_freqs(rd, cfg["max_pos"], cfg["original_seq_len"],
                                    cfg["compress_rope_theta"], cfg["rope_factor"],
                                    cfg["beta_fast"], cfg["beta_slow"], f64)
    else:
        cos, sin = precompute_freqs(rd, cfg["max_pos"], 0, cfg["rope_theta"],
                                    cfg["rope_factor"], cfg["beta_fast"],
                                    cfg["beta_slow"], f64)
    P = {
        "ratio": ratio, "has_indexer": ratio == 4,
        "n_heads": cfg["n_heads"], "head_dim": cfg["head_dim"],
        "rope_head_dim": rd, "window": cfg["window_size"],
        "o_groups": cfg["o_groups"], "o_lora": cfg["o_lora_rank"],
        "eps": cfg["norm_eps"], "cos": cos, "sin": sin,
        "wq_a": shards.fp8(f"{L}.attn.wq_a"),
        "wq_b": shards.fp8(f"{L}.attn.wq_b"),
        "wkv": shards.fp8(f"{L}.attn.wkv"),
        "wo_b": shards.fp8(f"{L}.attn.wo_b"),
        "q_norm": shards.f32(f"{L}.attn.q_norm.weight"),
        "kv_norm": shards.f32(f"{L}.attn.kv_norm.weight"),
        "attn_sink": shards.f32(f"{L}.attn.attn_sink"),
        # wo_a: FP8 in the checkpoint but used as bf16 (model.py:462) — the
        # dequantized values are rounded to bf16 here, matching the reference
        # load-then-cast (see README ambiguities A4).
        "wo_a": bf16_round(_fp8_dequant(shards, f"{L}.attn.wo_a")),
        "attn_norm": shards.f32(f"{L}.attn_norm.weight"),
        "ffn_norm": shards.f32(f"{L}.ffn_norm.weight"),
        "hc_attn_fn": shards.f32(f"{L}.hc_attn_fn"),
        "hc_attn_base": shards.f32(f"{L}.hc_attn_base"),
        "hc_attn_scale": shards.f32(f"{L}.hc_attn_scale"),
        "hc_ffn_fn": shards.f32(f"{L}.hc_ffn_fn"),
        "hc_ffn_base": shards.f32(f"{L}.hc_ffn_base"),
        "hc_ffn_scale": shards.f32(f"{L}.hc_ffn_scale"),
        "gate_w": shards.f32(f"{L}.ffn.gate.weight"),
        "hash": layer["hash"],
        "topk": cfg["n_activated_experts"],
        "route_scale": cfg["route_scale"],
        "n_routed": cfg["n_routed_experts"],
        "limit": cfg["swiglu_limit"],
        "experts": [(shards.fp4(f"{L}.ffn.experts.{e}.w1"),
                     shards.fp4(f"{L}.ffn.experts.{e}.w2"),
                     shards.fp4(f"{L}.ffn.experts.{e}.w3"))
                    for e in range(cfg["n_routed_experts"])],
        "shared": (shards.fp8(f"{L}.ffn.shared_experts.w1"),
                   shards.fp8(f"{L}.ffn.shared_experts.w2"),
                   shards.fp8(f"{L}.ffn.shared_experts.w3")),
    }
    if layer["hash"]:
        P["tid2eid"] = shards.i64(f"{L}.ffn.gate.tid2eid").astype(np.int64)
    else:
        P["gate_bias"] = shards.f32(f"{L}.ffn.gate.bias")

    def comp_params(prefix, cdim):
        return {
            "ratio": ratio, "overlap": ratio == 4, "d": cdim, "rd": rd,
            "rotate": prefix.endswith("indexer.compressor"),
            "eps": cfg["norm_eps"],
            "wkv": shards.f32(f"{L}.{prefix}.wkv.weight"),
            "wgate": shards.f32(f"{L}.{prefix}.wgate.weight"),
            "ape": shards.f32(f"{L}.{prefix}.ape"),
            "norm": shards.f32(f"{L}.{prefix}.norm.weight"),
        }

    if ratio:
        P["comp"] = comp_params("attn.compressor", cfg["head_dim"])
        if ratio == 4:
            P["idx_heads"] = cfg["index_n_heads"]
            P["idx_dim"] = cfg["index_head_dim"]
            P["idx_topk"] = cfg["index_topk"]
            P["idx_wq_b"] = shards.fp8(f"{L}.attn.indexer.wq_b")
            P["idx_weights_proj"] = shards.f32(f"{L}.attn.indexer.weights_proj.weight")
            P["idx_comp"] = comp_params("attn.indexer.compressor",
                                        cfg["index_head_dim"])
    return P


def _fp8_dequant(shards, name):
    codes, scales = shards.fp8(name)
    return (m3.E4M3_TABLE[codes].astype(np.float32)
            * np.repeat(np.repeat(scales, 128, 0), 128, 1))


# ===========================================================================
# Sequence runner + state serialization.
# ===========================================================================


def state_arrays(st: LayerState):
    """Flatten the decode-carried state to named arrays (f32)."""
    out = {"pos": np.array(st.pos, np.int64), "win_kv": st.win.astype(np.float32)}
    if st.ratio:
        out["comp_kv"] = st.comp.cache.astype(np.float32)
        out["comp_kv_state"] = st.comp.kv.astype(np.float32)
        out["comp_score_state"] = st.comp.sc.astype(np.float32)
        if st.ratio == 4:
            out["idx_kv"] = st.idx_comp.cache.astype(np.float32)
            out["idx_kv_state"] = st.idx_comp.kv.astype(np.float32)
            out["idx_score_state"] = st.idx_comp.sc.astype(np.float32)
    return out


def load_state_arrays(cfg, layer, arrays, f64):
    """Inverse of state_arrays (replay from serialized state)."""
    st = LayerState(cfg, layer, f64)
    st.pos = int(arrays["pos"])
    st.win = arrays["win_kv"].astype(np.float32)
    if st.ratio:
        st.comp.cache = arrays["comp_kv"].astype(np.float32)
        st.comp.kv = arrays["comp_kv_state"].astype(_dt(f64))
        st.comp.sc = arrays["comp_score_state"].astype(_dt(f64))
        if st.ratio == 4:
            st.idx_comp.cache = arrays["idx_kv"].astype(np.float32)
            st.idx_comp.kv = arrays["idx_kv_state"].astype(_dt(f64))
            st.idx_comp.sc = arrays["idx_score_state"].astype(_dt(f64))
    return st


def run_prefill(P, cfg, layer, h, ids, f64):
    """One-shot prefill; returns (out_h, interm, state)."""
    st = LayerState(cfg, layer, f64)
    out, interm = block_forward(P, cfg, h, ids, 0, st, f64)
    st.pos = h.shape[0]
    return out, interm, st


def run_decode_step(P, cfg, layer, h1, id1, st, f64):
    """One decode token at position st.pos; returns (out_h, interm)."""
    out, interm = block_forward(P, cfg, h1, id1, st.pos, st, f64)
    st.pos += 1
    return out, interm


# ===========================================================================
# Fixture driver.
# ===========================================================================

# (fixture layer name, sequence name, kind, length, decode steps)
SEQUENCES = [
    ("swa", "prefill_len6", "prefill", 6, 0),
    ("swa", "prefill_len140", "prefill", 140, 0),
    ("swa", "decode_from140", "decode", 140, 12),
    ("csa", "prefill_len6", "prefill", 6, 0),
    ("csa", "prefill_len199", "prefill", 199, 0),
    ("csa", "decode_from199", "decode", 199, 12),
    ("hca", "prefill_len130", "prefill", 130, 0),
    ("hca", "prefill_len250", "prefill", 250, 0),
    ("hca", "decode_from250", "decode", 250, 12),
]

# intermediates written per sequence/step (all are also kept in-memory)
INTERM_KEYS = [
    "attn_hc_pre", "attn_hc_post", "attn_hc_comb", "attn_norm_out",
    "q", "win_kv", "comp_kv", "idx_comp_kv", "idx_scores", "idx_topk",
    "attn_out", "o_out", "post_attn_h",
    "ffn_hc_pre", "ffn_hc_post", "ffn_hc_comb", "ffn_norm_out",
    "router_scores", "router_scores_biased", "router_idx", "router_w",
    "moe_routed", "moe_shared", "moe_out",
]


def _save_npy(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def _gen_input(cfg, seed, s):
    rng = np.random.default_rng(seed)
    h = bf16_round(rng.standard_normal((s, cfg["hc_mult"], cfg["dim"]))
                   .astype(np.float32))
    ids = rng.integers(0, cfg["vocab_size"], size=s).astype(np.int64)
    return h, ids


def _write_step_dir(d, h, ids, st_in, out32, out64, i32, i64, st_out):
    os.makedirs(d, exist_ok=True)
    _save_npy(os.path.join(d, "input_h.npy"), h.astype(np.float32))
    _save_npy(os.path.join(d, "input_ids.npy"), ids.astype(np.int64))
    _save_npy(os.path.join(d, "out_h_f32.npy"), out32.astype(np.float32))
    _save_npy(os.path.join(d, "out_h_f64.npy"), out64.astype(np.float64))
    if st_in is not None:
        for k, v in state_arrays(st_in).items():
            _save_npy(os.path.join(d, "state_in", k + ".npy"), v)
    if st_out is not None:
        for k, v in state_arrays(st_out).items():
            _save_npy(os.path.join(d, "state_out", k + ".npy"), v)
    for k in INTERM_KEYS:
        if k in i32:
            _save_npy(os.path.join(d, "interm", k + "_f32.npy"),
                      np.asarray(i32[k]))
        if k in i64:
            _save_npy(os.path.join(d, "interm", k + "_f64.npy"),
                      np.asarray(i64[k]))


def generate(fixtures_dir, clean=True, seed=MASTER_SEED):
    """Generate the full M4b fixture set (weights + golden I/O)."""
    cfg = dict(SMALL_CFG)
    if clean and os.path.isdir(fixtures_dir):
        import shutil
        shutil.rmtree(fixtures_dir)
    os.makedirs(fixtures_dir, exist_ok=True)
    with open(os.path.join(fixtures_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    weights_dir = os.path.join(fixtures_dir, "weights")
    write_weights(cfg, weights_dir)
    shards = ShardSet(weights_dir)

    layer_idx = {l["name"]: i for i, l in enumerate(cfg["layers"])}
    manifest = {"seed": seed, "config": "config.json",
                "weights": "weights/", "sequences": {}}

    for lname, sname, kind, length, steps in SEQUENCES:
        li = layer_idx[lname]
        layer = cfg["layers"][li]
        P32 = load_layer_params(shards, cfg, layer, li, f64=False)
        P64 = load_layer_params(shards, cfg, layer, li, f64=True)
        sdir = os.path.join(fixtures_dir, "golden", lname, sname)
        h, ids = _gen_input(cfg, seed + 31 * li + length, length)

        if kind == "prefill":
            o32, i32, st32 = run_prefill(P32, cfg, layer, h, ids, False)
            o64, i64, st64 = run_prefill(P64, cfg, layer, h, ids, True)
            _write_step_dir(sdir, h, ids, None, o32, o64, i32, i64, st32)
            manifest["sequences"][f"{lname}/{sname}"] = {
                "kind": "prefill", "len": length,
                "state_out": sorted(state_arrays(st32))}
        else:
            # establish the prefill state, then run `steps` decode tokens
            _, _, st32 = run_prefill(P32, cfg, layer, h, ids, False)
            _, _, st64 = run_prefill(P64, cfg, layer, h, ids, True)
            for k in range(steps):
                pos = st32.pos
                hk, idk = _gen_input(cfg, seed + 31 * li + 7000 + k, 1)
                st_in32 = load_state_arrays(cfg, layer, state_arrays(st32), False)
                o32, i32 = run_decode_step(P32, cfg, layer, hk, idk, st32, False)
                o64, i64 = run_decode_step(P64, cfg, layer, hk, idk, st64, True)
                _write_step_dir(os.path.join(sdir, f"step{k:02d}"), hk, idk,
                                st_in32, o32, o64, i32, i64, st32)
            manifest["sequences"][f"{lname}/{sname}"] = {
                "kind": "decode", "prefill_len": length, "steps": steps,
                "positions": list(range(length, length + steps))}
        print(f"  {lname}/{sname} done")

    with open(os.path.join(fixtures_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"fixtures written to {fixtures_dir}")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="apus M4b oracle fixture generator")
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "m4b", "fixtures"))
    ap.add_argument("--full", action="store_true",
                    help="generate the M5 full-model fixtures instead")
    ap.add_argument("--m8", action="store_true",
                    help="generate the M8 MTP/speculative-decoding fixtures")
    ap.add_argument("--no-clean", action="store_true")
    args = ap.parse_args(argv)
    if args.m8:
        if args.out == os.path.join(ROOT, "tests", "m4b", "fixtures"):
            args.out = os.path.join(ROOT, "tests", "m8", "fixtures")
        generate_m8(args.out, clean=not args.no_clean)
    elif args.full:
        if args.out == os.path.join(ROOT, "tests", "m4b", "fixtures"):
            args.out = os.path.join(ROOT, "tests", "m5", "fixtures")
        generate_full(args.out, clean=not args.no_clean)
    else:
        generate(args.out, clean=not args.no_clean)


# ===========================================================================
# M5 — full-model oracle: embed -> N blocks -> hc_head -> norm -> head ->
# logits -> sampling. Ports model.py Transformer.forward (802-808),
# ParallelHead.forward/hc_head (714-735). Reuses the M4b block port
# unchanged; the layer schedule and top-level weights are fixture-synthetic
# but use the real checkpoint naming.
# ===========================================================================

FULL_CFG = dict(SMALL_CFG)
FULL_CFG["layers"] = [
    {"name": "l0_swa", "compress_ratio": 0, "hash": True},
    {"name": "l1_csa", "compress_ratio": 4, "hash": False},
    {"name": "l2_hca", "compress_ratio": 128, "hash": False},
    {"name": "l3_csa", "compress_ratio": 4, "hash": False},   # repeated type
]

# Sampling fixture settings (documented in tests/m5/README.md):
# temperature/top_p per reference/generation_config.json style; the RNG is
# numpy PCG64 (np.random.default_rng) with this seed, ONE f64 uniform per
# sampled token, dumped to golden so the C side can replay the same draws.
SAMPLE_SEED = 20260805
SAMPLE_TEMP = 0.8
SAMPLE_TOP_P = 0.95


def gen_toplevel_tensors(cfg, rng):
    """Top-level tensors with real checkpoint naming: embed/norm/head BF16,
    hc_head_* F32."""
    dim, V, hc = cfg["dim"], cfg["vocab_size"], cfg["hc_mult"]

    def rnd(*shape, std=None):
        k = shape[-1]
        return (rng.standard_normal(shape)
                * (std if std is not None else 1.0 / math.sqrt(k))).astype(np.float32)

    recs = [
        ("embed.weight", "BF16", (V, dim),
         f32_to_bf16_bytes(rnd(V, dim, std=1.0))),   # h ~ N(0,1) like m4b inputs
        ("norm.weight", "BF16", (dim,),
         f32_to_bf16_bytes((1.0 + 0.05 * rng.standard_normal(dim)).astype(np.float32))),
        ("head.weight", "BF16", (V, dim), f32_to_bf16_bytes(rnd(V, dim))),
        ("hc_head_fn", "F32", (hc, hc * dim),
         np.ascontiguousarray(rnd(hc, hc * dim), np.float32).tobytes()),
        ("hc_head_scale", "F32", (1,),
         np.ascontiguousarray(1.0 + 0.1 * rng.standard_normal(1), np.float32).tobytes()),
        ("hc_head_base", "F32", (hc,),
         np.ascontiguousarray(0.25 * rng.standard_normal(hc), np.float32).tobytes()),
    ]
    return recs


def write_full_weights(cfg, out_dir):
    """write_weights + top-level tensors (all dense shard)."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(MASTER_SEED + 90000)
    dense_recs = gen_toplevel_tensors(cfg, rng)
    expert_recs = []
    for li, layer in enumerate(cfg["layers"]):
        lr = np.random.default_rng(MASTER_SEED + 1000 * li)
        for rec in gen_layer_tensors(cfg, layer, li, lr):
            (expert_recs if ".experts." in rec[0] else dense_recs).append(rec)
    shards = [("apus-00001.safetensors", dense_recs)]
    if expert_recs:
        shards.append(("apus-00002.safetensors", expert_recs))
    weight_map, total = {}, 0
    for fname, recs in shards:
        stutil.write_shard(os.path.join(out_dir, fname), recs)
        for name, dtype, shape, payload in recs:
            weight_map[name] = fname
            total += len(payload)
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=1)


# ---------------------------------------------------------------------------
# Full-model forward (model.py:802-808 + ParallelHead 714-735).
# ---------------------------------------------------------------------------


def hc_head_collapse(h, top, cfg, f64):
    """ParallelHead.hc_head (model.py:728-735): sigmoid-gated 4->1 collapse,
    NO Sinkhorn. bf16 output in f32 mode."""
    dt = _dt(f64)
    s, hc = h.shape[0], cfg["hc_mult"]
    xf = h.reshape(s, hc * cfg["dim"]).astype(dt)
    rsqrt = 1.0 / np.sqrt((xf * xf).mean(-1, keepdims=True) + cfg["norm_eps"])
    mixes = (xf @ top["hc_head_fn"].astype(dt).T) * rsqrt            # 731-732
    pre = sigmoid(mixes * dt(top["hc_head_scale"][0])
                  + top["hc_head_base"].astype(dt)) + cfg["hc_eps"]  # 733
    y = (pre[..., None] * xf.reshape(s, hc, cfg["dim"])).sum(axis=1)  # 734
    return _B(y, f64)                                                 # 735 .to(dtype)


def model_forward(Ps, top, cfg, ids, states, start_pos, f64):
    """Transformer.forward (model.py:802-808). ids [s]; states = per-layer
    LayerState list (mutated). Returns (logits [s, V], h [s, hc, dim]).
    Logits are computed for ALL s positions (the reference takes [:, -1]
    only; the per-position extension is the same math row-wise)."""
    dt = _dt(f64)
    ids = np.asarray(ids, dtype=np.int64)
    h = top["embed"][ids].astype(dt)                       # bf16-valued rows
    h = np.repeat(h[:, None, :], cfg["hc_mult"], axis=1)   # 804: expand 4x
    for i, P in enumerate(Ps):                             # 805-806
        h, _ = block_forward(P, cfg, h, ids, start_pos, states[i], f64)
    y = hc_head_collapse(h, top, cfg, f64)                 # 808 -> 720
    yn = rms_norm(y, top["norm"], cfg["norm_eps"], f64)    # 721 (bf16 out)
    logits = yn.astype(dt) @ top["head"].astype(dt).T      # 715 f32 linear
    return logits, h


def load_model_params(shards, cfg, f64):
    """Load every layer + top-level params from the fixture safetensors."""
    Ps = [load_layer_params(shards, cfg, layer, li, f64)
          for li, layer in enumerate(cfg["layers"])]
    top = {
        "embed": shards.f32("embed.weight"),      # bf16 values as f32
        "norm": shards.f32("norm.weight"),
        "head": shards.f32("head.weight"),
        "hc_head_fn": shards.f32("hc_head_fn"),
        "hc_head_scale": shards.f32("hc_head_scale"),
        "hc_head_base": shards.f32("hc_head_base"),
    }
    return Ps, top


def new_model_states(cfg, f64):
    return [LayerState(cfg, layer, f64) for layer in cfg["layers"]]


def run_model_prefill(Ps, top, cfg, ids, f64):
    states = new_model_states(cfg, f64)
    logits, h = model_forward(Ps, top, cfg, ids, states, 0, f64)
    for st in states:
        st.pos = len(ids)
    return logits, states


def run_model_decode(Ps, top, cfg, id1, states, f64):
    """One decode token at position states[0].pos."""
    logits, h = model_forward(Ps, top, cfg, [id1], states, states[0].pos, f64)
    for st in states:
        st.pos += 1
    return logits[0]


# ---------------------------------------------------------------------------
# Sampling (reference/generation_config.json defaults temp 1.0 top_p 1.0).
# These rules are the contract c/sample.h implements: stable descending
# sort, nucleus keep iff cumsum_before <= top_p (HF top-p shift rule),
# renormalize, draw = first CDF entry > u.
# ---------------------------------------------------------------------------


def probs_from_logits(logits, temp):
    z = logits.astype(np.float32) / np.float32(temp)
    z = z - z.max()
    e = np.exp(z)
    return (e / e.sum()).astype(np.float32)


def top_p_draw(probs, top_p, u):
    """Nucleus sample with explicit uniform u in [0,1). Returns
    (token, margin) where margin = min |u - cdf_boundary| (flip indicator)."""
    order = np.argsort(-probs, kind="stable")
    sp = probs[order].astype(np.float64)
    cum = np.cumsum(sp)
    keep = (cum - sp) <= np.float64(top_p)
    pk = np.where(keep, sp, 0.0)
    pk /= pk.sum()
    c = np.cumsum(pk)
    j = int(np.searchsorted(c, u, side="right"))
    if j >= len(order):
        j = len(order) - 1
    margin = float(np.min(np.abs(c - u)))
    return int(order[j]), margin


# ---------------------------------------------------------------------------
# M5 fixture driver.
# ---------------------------------------------------------------------------

FULL_SEQUENCES = {
    "prefills": [("prefill_len6", 6), ("prefill_len200", 200)],
    "decode": ("decode_from64", 64, 16),       # name, prompt len, steps
    "greedy": ("greedy_from24", 24, 24),       # name, prompt len, steps
    "sampled": ("sampled_from24", 24, 24),
}


def _gen_ids(cfg, seed, s):
    return np.random.default_rng(seed).integers(0, cfg["vocab_size"],
                                                size=s).astype(np.int64)


def generate_full(fixtures_dir, clean=True, seed=MASTER_SEED):
    """Generate the M5 full-model fixtures (weights + logits/token goldens)."""
    cfg = dict(FULL_CFG)
    if clean and os.path.isdir(fixtures_dir):
        import shutil
        shutil.rmtree(fixtures_dir)
    os.makedirs(fixtures_dir, exist_ok=True)
    with open(os.path.join(fixtures_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    weights_dir = os.path.join(fixtures_dir, "weights")
    write_full_weights(cfg, weights_dir)
    shards = ShardSet(weights_dir)
    Ps32, top32 = load_model_params(shards, cfg, False)
    Ps64, top64 = load_model_params(shards, cfg, True)
    V = cfg["vocab_size"]
    gdir = os.path.join(fixtures_dir, "golden")
    manifest = {"seed": seed, "config": "config.json", "weights": "weights/",
                "sampling": {"rng": "numpy PCG64 (np.random.default_rng), "
                                    "one f64 uniform per token",
                             "seed": SAMPLE_SEED, "temp": SAMPLE_TEMP,
                             "top_p": SAMPLE_TOP_P},
                "sequences": {}}
    div_rows = []

    def div(tag, a32, a64):
        a32 = np.asarray(a32, np.float64)
        a64 = np.asarray(a64, np.float64)
        d = np.abs(a32 - a64)
        row = (tag, float(d.max()), float(d.max() / np.abs(a64).max()),
               float(np.mean(np.argmax(a32, -1) != np.argmax(a64, -1))))
        div_rows.append(row)
        return row

    # --- prefills: per-position logits ---
    for sname, length in FULL_SEQUENCES["prefills"]:
        ids = _gen_ids(cfg, seed + 17 * length, length)
        l32, _ = run_model_prefill(Ps32, top32, cfg, ids, False)
        l64, _ = run_model_prefill(Ps64, top64, cfg, ids, True)
        # determinism self-check: rerun f32, must be bitwise
        l32b, _ = run_model_prefill(Ps32, top32, cfg, ids, False)
        assert np.array_equal(l32, l32b), f"{sname}: f32 not deterministic"
        d = os.path.join(gdir, sname)
        _save_npy(os.path.join(d, "input_ids.npy"), ids)
        _save_npy(os.path.join(d, "logits_f32.npy"), l32.astype(np.float32))
        _save_npy(os.path.join(d, "logits_f64.npy"), l64.astype(np.float64))
        div(sname, l32, l64)
        manifest["sequences"][sname] = {"kind": "prefill", "len": length}
        print(f"  {sname} done")

    # --- decode chain with fixed ids: per-step logits ---
    dname, plen, steps = FULL_SEQUENCES["decode"]
    pids = _gen_ids(cfg, seed + 23000, plen)
    dids = _gen_ids(cfg, seed + 24000, steps)
    _, st32 = run_model_prefill(Ps32, top32, cfg, pids, False)
    _, st64 = run_model_prefill(Ps64, top64, cfg, pids, True)
    ls32 = np.stack([run_model_decode(Ps32, top32, cfg, t, st32, False)
                     for t in dids])
    ls64 = np.stack([run_model_decode(Ps64, top64, cfg, t, st64, True)
                     for t in dids])
    d = os.path.join(gdir, dname)
    _save_npy(os.path.join(d, "prompt_ids.npy"), pids)
    _save_npy(os.path.join(d, "decode_ids.npy"), dids)
    _save_npy(os.path.join(d, "logits_f32.npy"), ls32.astype(np.float32))
    _save_npy(os.path.join(d, "logits_f64.npy"), ls64.astype(np.float64))
    div(dname, ls32, ls64)
    manifest["sequences"][dname] = {"kind": "decode", "prompt_len": plen,
                                    "steps": steps}
    print(f"  {dname} done")

    # --- greedy continuation (oracle f32 logits; gap = top1-top2 margin).
    # Chains are generated by ITERATIVE ONE-SHOT PREFILLS (not decode
    # steps): the oracle's f32 decode path has single-code-flip chunk noise
    # (tests/m4b/README §7) which cascades through 4 layers, while the C
    # engine is bitwise chunk-invariant (asserted in test_full.c), so
    # C-decode == C-prefill and both sides compare cleanly against one-shot
    # prefill goldens. Teacher-forced per-step logits are dumped so the C
    # side compares its argmax against the oracle's GIVEN THE SAME
    # CONTEXT — free-running streams diverge through the router-flip
    # cascade (see tests/m5/README.md). ---
    gname, plen, steps = FULL_SEQUENCES["greedy"]
    pids = _gen_ids(cfg, seed + 25000, plen)
    toks, gaps, tfl, tfl64 = [], [], [], []
    ctx32, ctx64 = list(pids), list(pids)
    for k in range(steps):
        l32, _ = run_model_prefill(Ps32, top32, cfg, ctx32, False)
        l64, _ = run_model_prefill(Ps64, top64, cfg, ctx64, True)
        logits = l32[-1]
        tfl.append(logits.copy())
        tfl64.append(l64[-1].copy())
        part = np.partition(logits, -2)
        gaps.append(float(part[-1] - part[-2]))
        toks.append(int(np.argmax(logits)))
        ctx32.append(toks[-1])
        ctx64.append(toks[-1])
    d = os.path.join(gdir, gname)
    _save_npy(os.path.join(d, "prompt_ids.npy"), pids)
    _save_npy(os.path.join(d, "tokens.npy"), np.asarray(toks, np.int64))
    _save_npy(os.path.join(d, "gap.npy"), np.asarray(gaps, np.float32))
    _save_npy(os.path.join(d, "logits_f32.npy"),
              np.stack(tfl).astype(np.float32))
    _save_npy(os.path.join(d, "logits_f64.npy"),
              np.stack(tfl64).astype(np.float64))
    div(gname, np.stack(tfl), np.stack(tfl64))
    manifest["sequences"][gname] = {"kind": "greedy", "prompt_len": plen,
                                    "steps": steps}
    print(f"  {gname} done")

    # --- sampled continuation (PCG64 uniforms dumped for C replay) ---
    sname, plen, steps = FULL_SEQUENCES["sampled"]
    pids = _gen_ids(cfg, seed + 26000, plen)
    uniforms = np.random.default_rng(SAMPLE_SEED).random(steps)
    toks, margins, tfl, tfl64 = [], [], [], []
    ctx32, ctx64 = list(pids), list(pids)
    for k in range(steps):
        l32, _ = run_model_prefill(Ps32, top32, cfg, ctx32, False)
        l64, _ = run_model_prefill(Ps64, top64, cfg, ctx64, True)
        logits = l32[-1]
        tfl.append(logits.copy())
        tfl64.append(l64[-1].copy())
        tok, margin = top_p_draw(probs_from_logits(logits, SAMPLE_TEMP),
                                 SAMPLE_TOP_P, float(uniforms[k]))
        toks.append(tok)
        margins.append(margin)
        ctx32.append(tok)
        ctx64.append(tok)
    d = os.path.join(gdir, sname)
    _save_npy(os.path.join(d, "prompt_ids.npy"), pids)
    _save_npy(os.path.join(d, "tokens.npy"), np.asarray(toks, np.int64))
    _save_npy(os.path.join(d, "uniforms.npy"), np.asarray(uniforms, np.float64))
    _save_npy(os.path.join(d, "margin.npy"), np.asarray(margins, np.float64))
    _save_npy(os.path.join(d, "logits_f32.npy"),
              np.stack(tfl).astype(np.float32))
    _save_npy(os.path.join(d, "logits_f64.npy"),
              np.stack(tfl64).astype(np.float64))
    div(sname, np.stack(tfl), np.stack(tfl64))
    manifest["sequences"][sname] = {"kind": "sampled", "prompt_len": plen,
                                    "steps": steps, "temp": SAMPLE_TEMP,
                                    "top_p": SAMPLE_TOP_P,
                                    "rng_seed": SAMPLE_SEED}
    print(f"  {sname} done")

    with open(os.path.join(fixtures_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    print("f32-vs-f64 logit divergence (self-check):")
    print(f"  {'sequence':<18} {'maxabs':>10} {'rel2scale':>10} {'argmax-flip':>12}")
    for tag, ma, rel, flip in div_rows:
        print(f"  {tag:<18} {ma:10.3g} {rel:10.3g} {flip:11.2%}")
    print(f"fixtures written to {fixtures_dir}")


# ---------------------------------------------------------------------------
# M8 fixture driver.
# ---------------------------------------------------------------------------

M8_PROMPT_LEN = 24
M8_CHAIN_DRAFTS = 3
M8_SPEC_STEPS = 24
M8_SPEC_DEPTH = 2


def generate_m8(fixtures_dir, clean=True, seed=MASTER_SEED):
    """Generate the M8 fixtures: the M5 mini-model + an MTP block, with
    goldens for the MTP forward (replay + chain) and a greedy spec-decode
    episode (draft/verify simulation)."""
    cfg = dict(M8_CFG)
    if clean and os.path.isdir(fixtures_dir):
        import shutil
        shutil.rmtree(fixtures_dir)
    os.makedirs(fixtures_dir, exist_ok=True)
    with open(os.path.join(fixtures_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)

    weights_dir = os.path.join(fixtures_dir, "weights")
    write_m8_weights(cfg, weights_dir)
    shards = ShardSet(weights_dir)
    Ps32, top32 = load_model_params(shards, cfg, False)
    Ps64, top64 = load_model_params(shards, cfg, True)
    Pm32 = load_mtp_params(shards, cfg, False)
    Pm64 = load_mtp_params(shards, cfg, True)
    V = cfg["vocab_size"]
    gdir = os.path.join(fixtures_dir, "golden")
    manifest = {"seed": seed, "config": "config.json", "weights": "weights/",
                "sequences": {}}
    div_rows = []

    def div(tag, a32, a64):
        a32 = np.asarray(a32, np.float64)
        a64 = np.asarray(a64, np.float64)
        d = np.abs(a32 - a64)
        row = (tag, float(d.max()), float(d.max() / np.abs(a64).max()),
               float(np.mean(np.argmax(a32, -1) != np.argmax(a64, -1))))
        div_rows.append(row)
        return row

    # --- mtp_prefill: main prefill + MTP true-pair replay goldens ---
    pids = _gen_ids(cfg, seed + 31000, M8_PROMPT_LEN)
    st32 = new_model_states(cfg, False)
    st64 = new_model_states(cfg, True)
    l32, h32 = model_forward(Ps32, top32, cfg, pids, st32, 0, False)
    l64, h64 = model_forward(Ps64, top64, cfg, pids, st64, 0, True)
    ms32, ms64 = new_mtp_state(cfg, False), new_mtp_state(cfg, True)
    ml32, hm32 = mtp_forward(Pm32, top32, cfg, h32, pids, 0, ms32, False)
    ml64, hm64 = mtp_forward(Pm64, top64, cfg, h64, pids, 0, ms64, True)
    d = os.path.join(gdir, "mtp_prefill")
    _save_npy(os.path.join(d, "input_ids.npy"), np.asarray(pids, np.int64))
    _save_npy(os.path.join(d, "logits_f32.npy"), ml32.astype(np.float32))
    _save_npy(os.path.join(d, "logits_f64.npy"), ml64.astype(np.float64))
    div("mtp_prefill", ml32, ml64)
    manifest["sequences"]["mtp_prefill"] = {
        "kind": "mtp_replay", "len": M8_PROMPT_LEN}
    print("  mtp_prefill done")

    # --- mtp_chain: greedy draft chain from the prefill end ---
    ch32, ch64 = [], []
    lg32, lg64 = [], []
    cur32, cur64 = ml32[-1], ml64[-1]
    hcur32, hcur64 = hm32[-1][None], hm64[-1][None]
    for i in range(M8_CHAIN_DRAFTS):
        ch32.append(int(np.argmax(cur32)))
        ch64.append(int(np.argmax(cur64)))
        lg32.append(np.asarray(cur32, np.float32))
        lg64.append(np.asarray(cur64, np.float64))
        if i + 1 < M8_CHAIN_DRAFTS:
            pos = M8_PROMPT_LEN + i
            cur32, hcur32 = mtp_forward(Pm32, top32, cfg, hcur32,
                                        [ch32[-1]], pos, ms32, False)
            cur64, hcur64 = mtp_forward(Pm64, top64, cfg, hcur64,
                                        [ch64[-1]], pos, ms64, True)
            cur32, cur64 = cur32[-1], cur64[-1]
            hcur32, hcur64 = hcur32[-1][None], hcur64[-1][None]
    d = os.path.join(gdir, "mtp_chain")
    _save_npy(os.path.join(d, "prompt_ids.npy"), np.asarray(pids, np.int64))
    _save_npy(os.path.join(d, "drafts_f32.npy"), np.asarray(ch32, np.int64))
    _save_npy(os.path.join(d, "drafts_f64.npy"), np.asarray(ch64, np.int64))
    _save_npy(os.path.join(d, "logits_f32.npy"), np.stack(lg32))
    _save_npy(os.path.join(d, "logits_f64.npy"), np.stack(lg64))
    div("mtp_chain", np.stack(lg32), np.stack(lg64))
    manifest["sequences"]["mtp_chain"] = {
        "kind": "mtp_chain", "prompt_len": M8_PROMPT_LEN,
        "drafts": M8_CHAIN_DRAFTS}
    print("  mtp_chain done")

    # --- spec_episode: greedy draft/verify simulation + non-spec chain ---
    pids2 = _gen_ids(cfg, seed + 32000, M8_PROMPT_LEN)
    toks, gaps = [], []
    ctx = list(pids2)
    for _ in range(M8_SPEC_STEPS):
        l32, _ = model_forward(Ps32, top32, cfg, ctx,
                               new_model_states(cfg, False), 0, False)
        logits = l32[-1]
        part = np.partition(logits, -2)
        gaps.append(float(part[-1] - part[-2]))
        toks.append(int(np.argmax(logits)))
        ctx.append(toks[-1])
    em32, slog32 = spec_simulate(Ps32, top32, Pm32, cfg, pids2,
                                 M8_SPEC_STEPS, M8_SPEC_DEPTH, False)
    em64, slog64 = spec_simulate(Ps64, top64, Pm64, cfg, pids2,
                                 M8_SPEC_STEPS, M8_SPEC_DEPTH, True)
    # the simulation must reproduce the non-spec chain up to the documented
    # near-tie class (oracle f32 chunk noise at small top1-top2 gaps)
    flips = [(i, t, s) for i, (t, s) in enumerate(zip(toks, em32)) if t != s]
    acc = [m for (_, _, m) in slog32]
    d = os.path.join(gdir, "spec_episode")
    _save_npy(os.path.join(d, "prompt_ids.npy"), np.asarray(pids2, np.int64))
    _save_npy(os.path.join(d, "tokens_nonspec.npy"), np.asarray(toks, np.int64))
    _save_npy(os.path.join(d, "gap.npy"), np.asarray(gaps, np.float32))
    _save_npy(os.path.join(d, "tokens_spec_f32.npy"), np.asarray(em32, np.int64))
    _save_npy(os.path.join(d, "tokens_spec_f64.npy"), np.asarray(em64, np.int64))
    _save_npy(os.path.join(d, "spec_log_f32.npy"),
              np.asarray([[q, m] + dr + [-1] * (M8_SPEC_DEPTH - len(dr))
                          for (q, dr, m) in slog32], np.int64))
    manifest["sequences"]["spec_episode"] = {
        "kind": "spec", "prompt_len": M8_PROMPT_LEN, "steps": M8_SPEC_STEPS,
        "depth": M8_SPEC_DEPTH,
        "nonspec_vs_spec_flips": len(flips),
        "oracle_accept_rate": sum(acc) / max(1, len(acc))}
    print(f"  spec_episode done (oracle accept rate {sum(acc)}/{len(acc)},"
          f" nonspec-vs-spec flips {len(flips)}"
          f"{'' if not flips else ' at ' + str([f[0] for f in flips])})")

    with open(os.path.join(fixtures_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    print("f32-vs-f64 logit divergence (self-check):")
    print(f"  {'sequence':<18} {'maxabs':>10} {'rel2scale':>10} {'argmax-flip':>12}")
    for tag, ma, rel, flip in div_rows:
        print(f"  {tag:<18} {ma:10.3g} {rel:10.3g} {flip:11.2%}")
    print(f"fixtures written to {fixtures_dir}")


if __name__ == "__main__":
    main()


# ===========================================================================
# M8 — MTP head + speculative decoding. Ports model.py MTPBlock (738-766):
#   e = enorm(embed(ids)) -> e_proj; x = hnorm(h) -> h_proj; bf16(sum) ->
#   Block (SWA ratio 0, bias-gate MoE) -> own hc_head (sigmoid, no
#   Sinkhorn) -> own norm -> SHARED head.
# The spec-decode simulation below pins the accept rule: a draft is
# accepted iff it equals the main model's own pick at that position; every
# emitted token is the main model's own draw, so the emitted stream equals
# non-speculative decoding by construction.
# ===========================================================================

M8_CFG = dict(FULL_CFG)
M8_CFG["mtp"] = [{"name": "mtp0", "compress_ratio": 0, "hash": False}]


def gen_mtp_tensors(cfg, rng):
    """MTP block tensors under the real "mtp.0" namespace: the standard
    block set (via gen_layer_tensors with prefix) + the e/h glue (FP8
    e_proj/h_proj, BF16 enorm/hnorm/norm) + own hc_head_* (F32)."""
    layer = cfg["mtp"][0]
    recs = gen_layer_tensors(cfg, layer, 0, rng, prefix="mtp.0")
    dim, hc = cfg["dim"], cfg["hc_mult"]

    def rnd(*shape, std=None):
        k = shape[-1]
        return (rng.standard_normal(shape)
                * (std if std is not None else 1.0 / math.sqrt(k))).astype(np.float32)

    for name in ("e_proj", "h_proj"):
        cb, sb, csh, ssh = fp8_store(rnd(dim, dim))
        recs.append((f"mtp.0.{name}.weight", "F8_E4M3", csh, cb))
        recs.append((f"mtp.0.{name}.scale", "F8_E8M0", ssh, sb))
    for name in ("enorm", "hnorm", "norm"):
        recs.append((f"mtp.0.{name}.weight", "BF16", (dim,),
                     f32_to_bf16_bytes(
                         (1.0 + 0.05 * rng.standard_normal(dim)).astype(np.float32))))
    recs.append(("mtp.0.hc_head_fn", "F32", (hc, hc * dim),
                 np.ascontiguousarray(rnd(hc, hc * dim), np.float32).tobytes()))
    recs.append(("mtp.0.hc_head_base", "F32", (hc,),
                 np.ascontiguousarray(0.25 * rng.standard_normal(hc), np.float32).tobytes()))
    recs.append(("mtp.0.hc_head_scale", "F32", (1,),
                 np.ascontiguousarray(1.0 + 0.1 * rng.standard_normal(1), np.float32).tobytes()))
    return recs


def write_m8_weights(cfg, out_dir):
    """write_full_weights + a third apus-mtp shard (real container layout:
    mtp tensors live in their own shard group)."""
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(MASTER_SEED + 90000)
    dense_recs = gen_toplevel_tensors(cfg, rng)
    expert_recs = []
    for li, layer in enumerate(cfg["layers"]):
        lr = np.random.default_rng(MASTER_SEED + 1000 * li)
        for rec in gen_layer_tensors(cfg, layer, li, lr):
            (expert_recs if ".experts." in rec[0] else dense_recs).append(rec)
    mrng = np.random.default_rng(MASTER_SEED + 5000)
    mtp_recs = gen_mtp_tensors(cfg, mrng)
    shards = [("apus-00001.safetensors", dense_recs),
              ("apus-00002.safetensors", expert_recs),
              ("apus-mtp-00001.safetensors", mtp_recs)]
    weight_map, total = {}, 0
    for fname, recs in shards:
        stutil.write_shard(os.path.join(out_dir, fname), recs)
        for name, dtype, shape, payload in recs:
            weight_map[name] = fname
            total += len(payload)
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=1)


def load_mtp_params(shards, cfg, f64):
    """Oracle params for the MTP block: standard block params (prefix
    "mtp.0") + the e/h glue + own hc_head_*."""
    layer = cfg["mtp"][0]
    P = load_layer_params(shards, cfg, layer, 0, f64, prefix="mtp.0")
    P["e_proj"] = shards.fp8("mtp.0.e_proj")
    P["h_proj"] = shards.fp8("mtp.0.h_proj")
    P["enorm"] = shards.f32("mtp.0.enorm.weight")
    P["hnorm"] = shards.f32("mtp.0.hnorm.weight")
    P["norm"] = shards.f32("mtp.0.norm.weight")
    P["hc_head_fn"] = shards.f32("mtp.0.hc_head_fn")
    P["hc_head_scale"] = shards.f32("mtp.0.hc_head_scale")
    P["hc_head_base"] = shards.f32("mtp.0.hc_head_base")
    return P


def mtp_forward(Pm, top, cfg, h, ids, start_pos, st, f64):
    """MTPBlock.forward (model.py:757-766). h [s, hc, dim] (previous mHC
    hidden), ids [s]. Mutates st (the MTP block's LayerState). Returns
    (logits [s, V], h_out [s, hc, dim]); the reference takes the last
    position's logits only — per-position rows are the same math."""
    dt = _dt(f64)
    dim, hc = cfg["dim"], cfg["hc_mult"]
    ids = np.asarray(ids, dtype=np.int64)
    s = len(ids)
    e = top["embed"][ids].astype(dt)                      # 760
    e = rms_norm(e, Pm["enorm"], cfg["norm_eps"], f64)    # 761
    ep = fp8_linear(e, Pm["e_proj"][0], Pm["e_proj"][1], f64)   # 763
    hn = rms_norm(h, Pm["hnorm"], cfg["norm_eps"], f64)   # 762
    hp = fp8_linear(hn.reshape(s * hc, dim),
                    Pm["h_proj"][0], Pm["h_proj"][1], f64).reshape(s, hc, dim)
    x = _B(ep[:, None, :] + hp, f64)                      # 763 (bf16 sum)
    x, _ = block_forward(Pm, cfg, x, ids, start_pos, st, f64)   # 764
    y = hc_head_collapse(x, Pm, cfg, f64)                 # 765 -> 720
    yn = rms_norm(y, Pm["norm"], cfg["norm_eps"], f64)    # 721
    logits = yn.astype(dt) @ top["head"].astype(dt).T     # 715
    return logits, x


def new_mtp_state(cfg, f64):
    return LayerState(cfg, cfg["mtp"][0], f64)


# ---------------------------------------------------------------------------
# Speculative decode simulation (numpy reference for the C engine's M8 loop;
# see c/mtp.h for the step-shape comment this mirrors).
# ---------------------------------------------------------------------------


def spec_simulate(Ps, top, Pm, cfg, prompt_ids, steps, depth, f64):
    """Greedy draft/verify simulation of the C ApusSpec loop (c/mtp.h),
    full-history style (fixture scale: every step recomputes one-shot
    prefills, like the m5 chain goldens — the C engine's batched forwards
    are chunk-invariant, so this is the same math modulo the documented
    oracle f32 chunk noise at near-ties).

    Returns (emitted, log) where log = [(q, drafts, matched), ...]."""
    ids = list(int(t) for t in prompt_ids)
    emitted = []
    log = []
    while len(emitted) < steps:
        q = len(ids)
        logits, h = model_forward(Ps, top, cfg, ids,
                                  new_model_states(cfg, f64), 0, f64)
        held = int(np.argmax(logits[-1]))
        # MTP true-pair replay over the full context + draft chain
        mst = new_mtp_state(cfg, f64)
        mlog, hm = mtp_forward(Pm, top, cfg, h, ids, 0, mst, f64)
        mst.pos = q
        drafts = [int(np.argmax(mlog[-1]))]
        for i in range(1, depth):
            lg, hm = mtp_forward(Pm, top, cfg, hm[-1][None],
                                 [drafts[-1]], q + i - 1, mst, f64)
            mst.pos = q + i
            drafts.append(int(np.argmax(lg[-1])))
        # verify batch [held, d2..dD] (one-shot prefill of the extension)
        batch = [held] + drafts[1:depth]
        lg, _ = model_forward(Ps, top, cfg, ids + batch,
                              new_model_states(cfg, f64), 0, f64)
        R = lg[q - 1:]                      # dists for positions q..q+len(batch)
        out = [held]
        matched = 0
        for j in range(1, len(batch)):
            x = int(np.argmax(R[j]))
            out.append(x)
            if drafts[j] == x:
                matched += 1
            else:
                break
        if matched == len(batch) - 1:       # full match: bonus token
            out.append(int(np.argmax(R[len(batch)])))
        log.append((q, list(drafts), matched))
        ids.extend(out)
        emitted.extend(out)
    return emitted[:steps], log
