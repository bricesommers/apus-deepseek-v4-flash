#!/usr/bin/env python3
"""M3 golden generator — numpy port of the reference FP4 semantics.

Ports (numpy only, no torch/tilelang):
  * fp4_quant_kernel (reference/inference/kernel.py:128-183):
      per-32-along-K amax, floor 6*2^-126, scale = 2^ceil(log2(amax/6))
      via the FP32 bit trick, clamp +-6, RNE to E2M1, pack low nibble = even K.
  * act_quant with scale_fmt="ue8m0" (kernel.py:40-102 + generate.py defaults):
      per-128-along-K amax, floor 1e-4, scale = 2^ceil(log2(amax/448)),
      clamp +-448, RNE to E4M3.
  * fp4_gemm (kernel.py:441-515): per-32-block FP8xFP4-code dot, then
      total += (dot * scale_a[kb//4]) * scale_b[kb], here evaluated in float64
      as the "exact" target; the C kernel must reproduce it within FP32 rounding.

Fixtures (tests/m3/golden/):
  manifest.txt   key=value dims
  w_packed.bin   uint8 [O, K/2]   (from fp4_quant port)
  w_scales.bin   uint8 [O, K/32]
  w_deq.bin      float32 [O, K]   (numpy dequant, bitwise check target)
  act_x.bin      float32 [M, K]   (raw activations)
  act_codes.bin  uint8 [M, K]     (act_quant port output)
  act_scales.bin float32 [M, K/128]
  out.bin        float64 [M, O]   (fp4_gemm semantics in f64)

The fixture weights are built to exercise the quant rule's corners: an
all-zero block (amax floor), an all-negative block, values beyond +-6
(saturation), and exact +-6 values.
"""

import os
import struct

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "golden")

FP4_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                   dtype=np.float32)
AMAX_FLOOR_FP4 = np.float32(6.0) * np.float32(2.0 ** -126)


def ceil_log2_pow2_f32(p):
    """2^ceil(log2(p)) for positive normal float32 p, via the reference
    fast_log2_ceil/fast_pow2 bit trick (kernel.py:22-33)."""
    p = np.asarray(p, dtype=np.float32)
    b = p.view(np.uint32)
    e = ((b >> np.uint32(23)) & np.uint32(0xFF)).astype(np.int64) - 127
    e += ((b & np.uint32(0x7FFFFF)) != 0).astype(np.int64)
    scale = ((e + 127).astype(np.uint32) << np.uint32(23)).view(np.float32)
    return scale, e


def e2m1_quant_rne(y):
    """Round-to-nearest-even to E2M1 codes (0..15). |y| <= 6 assumed."""
    y = np.asarray(y, dtype=np.float32)
    sign = np.signbit(y).astype(np.uint8)
    a = np.abs(y)
    # (threshold, lower magnitude index); tie goes to the even index
    bounds = [(0.25, 0), (0.75, 1), (1.25, 2), (1.75, 3),
              (2.5, 4), (3.5, 5), (5.0, 6)]
    mag = np.zeros(a.shape, dtype=np.uint8)
    for t, j in bounds:
        mag += ((a > t) | ((a == t) & (j % 2 == 1))).astype(np.uint8)
    return mag | (sign << np.uint8(3))


def build_e4m3_table():
    """All 256 E4M3 values as float64 (NaN codes 0x7F/0xFF excluded from use)."""
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
E4M3_POS = E4M3_TABLE[:127]          # codes 0x00..0x7E, ascending magnitude


def e4m3_quant_rne(x):
    """Brute-force RNE quantize float32 -> e4m3 codes, clamped to +-448.
    Independent of the C bit-twiddling implementation on purpose."""
    y = np.clip(np.asarray(x, dtype=np.float32), -448.0, 448.0)
    sign = np.signbit(y)
    a = np.abs(y).astype(np.float64)
    idx = np.searchsorted(E4M3_POS, a)          # first table entry >= a
    hi = np.clip(idx, 0, 126)
    lo = np.clip(idx - 1, 0, 126)
    d_lo = a - E4M3_POS[lo]
    d_hi = E4M3_POS[hi] - a
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0))
    code = np.where(pick_hi, hi, lo).astype(np.uint8)
    return code | (sign.astype(np.uint8) << np.uint8(7))


def fp4_quant(x):
    """Port of fp4_quant_kernel. x float32 [O, K] -> packed u8 [O, K/2],
    scales u8 [O, K/32]."""
    O, K = x.shape
    xb = x.reshape(O, K // 32, 32)
    amax = np.abs(xb).max(axis=2)
    amax = np.maximum(amax, AMAX_FLOOR_FP4)
    p = (amax * np.float32(1.0 / 6.0)).astype(np.float32)
    scale, e = ceil_log2_pow2_f32(p)
    y = np.clip(xb / scale[:, :, None], -6.0, 6.0).astype(np.float32)
    codes = e2m1_quant_rne(y)                       # [O, K/32, 32]
    codes = codes.reshape(O, K // 2, 2)
    packed = (codes[:, :, 0] | (codes[:, :, 1] << np.uint8(4))).astype(np.uint8)
    scales = (e + 127).astype(np.uint8)
    return packed, scales


def act_quant_ue8m0(x, block=128):
    """Port of act_quant(round_scale=True). x float32 [M, K] ->
    codes u8 [M, K], scales float32 [M, K/block]."""
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


def dequant_mxfp4(packed, scales):
    """M1-pinned dequant (tests/m1/test_3_dequant.py)."""
    O, half_k = packed.shape
    K = half_k * 2
    low = packed & np.uint8(0x0F)
    high = packed >> np.uint8(4)
    vals = np.empty((O, K), dtype=np.float32)
    vals[:, 0::2] = FP4_LUT[low]
    vals[:, 1::2] = FP4_LUT[high]
    scale_f = np.exp2(scales.astype(np.int32) - 127).astype(np.float32)
    return vals * np.repeat(scale_f, 32, axis=1)


def fp4_gemm_f64(act_codes, act_scales, w_packed, w_scales):
    """fp4_gemm accumulation semantics evaluated in float64."""
    M, K = act_codes.shape
    O = w_packed.shape[0]
    ad = E4M3_TABLE[act_codes]                       # [M, K] f64
    wl = np.empty((O, K), dtype=np.float64)
    lo = w_packed & np.uint8(0x0F)
    hi = w_packed >> np.uint8(4)
    wl[:, 0::2] = FP4_LUT[lo].astype(np.float64)
    wl[:, 1::2] = FP4_LUT[hi].astype(np.float64)
    sb = np.exp2(w_scales.astype(np.int64) - 127)    # [O, K/32] f64
    out = np.zeros((M, O), dtype=np.float64)
    for kb in range(K // 32):
        dot = ad[:, kb * 32:(kb + 1) * 32] @ wl[:, kb * 32:(kb + 1) * 32].T
        sa = act_scales[:, (kb * 32) // 128].astype(np.float64)[:, None]
        out += (dot * sa) * sb[:, kb][None, :]
    return out


def main():
    rng = np.random.default_rng(20260729)
    os.makedirs(OUT, exist_ok=True)

    M, O, K = 3, 64, 256

    # --- weights: random + corner-case blocks -------------------------------
    w_raw = (rng.standard_normal((O, K)) * 2.0).astype(np.float32)
    w_raw[0, 0:32] = 0.0                                   # all-zero block
    w_raw[1, 32:64] = -np.abs(w_raw[1, 32:64])             # all-negative
    w_raw[2, 64:96] = 1000.0                               # beyond +-6 -> clamp
    w_raw[3, 96:128] = -1000.0
    blk = w_raw[4, 128:160].copy()
    blk[np.argmax(np.abs(blk))] = 6.0                      # exact +6 saturation
    w_raw[4, 128:160] = blk
    blk = w_raw[5, 160:192].copy()
    blk[np.argmax(np.abs(blk))] = -6.0                     # exact -6
    w_raw[5, 160:192] = blk
    w_raw[6, 192:224] = 1e-30                              # tiny amax -> floor

    w_packed, w_scales = fp4_quant(w_raw)
    w_deq = dequant_mxfp4(w_packed, w_scales)

    # --- activations ---------------------------------------------------------
    act_x = (rng.standard_normal((M, K)) * 3.0).astype(np.float32)
    act_x[1, :] = 0.0                                      # all-zero row
    act_x[2, 128:] *= 100.0                                # large second group
    act_codes, act_scales = act_quant_ue8m0(act_x)

    out = fp4_gemm_f64(act_codes, act_scales, w_packed, w_scales)

    # --- write fixtures ------------------------------------------------------
    def dump(name, arr):
        arr.tofile(os.path.join(OUT, name))

    dump("w_packed.bin", w_packed)
    dump("w_scales.bin", w_scales)
    dump("w_deq.bin", w_deq.astype(np.float32))
    dump("act_x.bin", act_x)
    dump("act_codes.bin", act_codes)
    dump("act_scales.bin", act_scales.astype(np.float32))
    dump("out.bin", out.astype(np.float64))

    with open(os.path.join(OUT, "manifest.txt"), "w") as f:
        f.write(f"M={M}\nO={O}\nK={K}\n")

    print(f"golden: M={M} O={O} K={K}")
    print(f"  zero-block scale byte (amax floor): {w_scales[0, 0]}")
    print(f"  tiny-block scale byte (1e-30 amax): {w_scales[6, 6]}")
    print(f"  out range: [{out.min():.6g}, {out.max():.6g}]")


if __name__ == "__main__":
    main()
