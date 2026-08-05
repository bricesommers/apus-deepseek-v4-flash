"""M1 test 3 — MXFP4 dequant spot-check (numpy).

Validates the dequantization semantics the C kernel (milestone M3, c/fp4.h)
will implement, against the normative reference in
reference/inference/convert.py and kernel.py:

  * E2M1 nibble -> value via the 16-entry LUT
    [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6]
    (bit3 sign, bits2:1 exponent bias 1, bit0 mantissa; e==0 subnormal .5*m).
  * Packing: 2 nibbles per byte along K; LOW nibble = even K index,
    HIGH nibble = odd K index (reference/inference/convert.py:30-33).
  * UE8M0 scale = 2^(byte - 127), one scale per 32 elements along K,
    stored [O, K/32].
"""

import unittest

import numpy as np

FP4_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def dequant_mxfp4(packed, scales):
    """packed: uint8 [O, K//2]; scales: uint8 [O, K//32]. Returns f32 [O, K]."""
    packed = np.asarray(packed, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.uint8)
    O, half_k = packed.shape
    K = half_k * 2
    assert scales.shape == (O, K // 32), scales.shape
    low = packed & np.uint8(0x0F)
    high = packed >> np.uint8(4)
    vals = np.empty((O, K), dtype=np.float32)
    vals[:, 0::2] = FP4_LUT[low]
    vals[:, 1::2] = FP4_LUT[high]
    scale_f = np.exp2(scales.astype(np.int32) - 127).astype(np.float32)
    return vals * np.repeat(scale_f, 32, axis=1)


class TestDequant(unittest.TestCase):
    def test_lut_structure(self):
        """Derive the LUT from the E2M1 bit definition and compare against
        the reference table."""
        for nibble in range(16):
            sign = -1.0 if nibble & 0x8 else 1.0
            exp = (nibble >> 1) & 0x3
            man = nibble & 0x1
            if exp == 0:
                val = sign * 0.5 * man            # subnormal
            else:
                val = sign * (1.0 + 0.5 * man) * 2.0 ** (exp - 1)
            self.assertEqual(FP4_LUT[nibble], val, f"nibble {nibble:#x}")

    def test_hand_computed_row(self):
        """One row, K=64: 32 packed bytes (16 per 32-elem block), 2 scales.

        Bytes (low nibble = even element):
          0x10 -> elems [0.0, 0.5]      (lo=0, hi=1)
          0x93 -> elems [1.5, -0.5]     (lo=3, hi=9)
          0x66 -> elems [4.0, 4.0]      (lo=6, hi=6)
          0xAA -> elems [-1.0, -1.0]    (lo=10, hi=10)
        Scales: block0 byte 128 -> 2^1, block1 byte 126 -> 2^-1.
        """
        packed = np.array([[0x10, 0x93] + [0x00] * 14 +
                           [0x66, 0xAA] + [0x00] * 14], dtype=np.uint8)
        scales = np.array([[128, 126]], dtype=np.uint8)
        out = dequant_mxfp4(packed, scales)
        self.assertEqual(out.shape, (1, 64))
        # block 0, scale 2^1 = 2
        self.assertEqual(out[0, 0], 0.0 * 2)
        self.assertEqual(out[0, 1], 0.5 * 2)
        self.assertEqual(out[0, 2], 1.5 * 2)
        self.assertEqual(out[0, 3], -0.5 * 2)
        self.assertTrue(np.all(out[0, 4:32] == 0.0))
        # block 1, scale 2^-1 = 0.5
        self.assertEqual(out[0, 32], 4.0 * 0.5)
        self.assertEqual(out[0, 33], 4.0 * 0.5)
        self.assertEqual(out[0, 34], -1.0 * 0.5)
        self.assertEqual(out[0, 35], -1.0 * 0.5)

    def test_all_nibbles_unity_scale(self):
        """Every nibble pattern with scale byte 127 (2^0) reproduces the LUT,
        in both low and high nibble positions."""
        packed = np.arange(256, dtype=np.uint8).reshape(1, 256)
        scales = np.full((1, 16), 127, dtype=np.uint8)
        out = dequant_mxfp4(packed, scales)
        for byte in range(256):
            self.assertEqual(out[0, 2 * byte], FP4_LUT[byte & 0x0F])
            self.assertEqual(out[0, 2 * byte + 1], FP4_LUT[byte >> 4])

    def test_scale_exponent_range(self):
        """Scale byte b multiplies by 2^(b-127), including 0 (2^-127) and
        255 (2^128)."""
        packed = np.zeros((1, 16), dtype=np.uint8)  # K = 32, one block
        packed[0, 0] = 0x06                         # lo=6 -> 4.0, hi=0
        for b, expected in ((0, 2.0 ** -127), (126, 0.5), (127, 1.0),
                            (128, 2.0), (255, 2.0 ** 128)):
            scales = np.array([[b]], dtype=np.uint8)
            out = dequant_mxfp4(packed, scales)
            self.assertEqual(out[0, 0], np.float32(4.0 * expected),
                             f"scale byte {b}")

    def test_block_boundaries(self):
        """Scale index is element_k // 32: elements 31 and 32 use different
        scales."""
        packed = np.zeros((1, 32), dtype=np.uint8)  # K = 64
        packed[0, 15] = 0x02   # elements 30,31: lo=2 -> 1.0, hi=0
        packed[0, 16] = 0x02   # elements 32,33
        scales = np.array([[127, 129]], dtype=np.uint8)  # 2^0, 2^2
        out = dequant_mxfp4(packed, scales)
        self.assertEqual(out[0, 30], 1.0)
        self.assertEqual(out[0, 32], 4.0)


if __name__ == "__main__":
    unittest.main()
