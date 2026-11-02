//! IEEE 754 binary16 ⇄ binary32 conversion helpers.
//!
//! Pure integer bit manipulation — no FPU involved, so the results are
//! identical on every target. Model weights are stored as binary16 bit
//! patterns (`u16`) and expanded to `f32` on load or in-register.

/// Convert one IEEE 754 binary16 bit pattern to `f32`.
///
/// Handles normals, subnormals, ±0, ±inf and NaN (NaN payload bits are
/// shifted into the f32 mantissa, quietness preserved).
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h as u32) >> 10) & 0x1f;
    let mant = (h as u32) & 0x3ff;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // Subnormal: renormalise the mantissa into f32's biased exponent.
            let mut m = mant;
            let mut e: i32 = 127 - 14;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            sign | ((e as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + (127 - 15)) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// Convert an `f32` to the nearest IEEE 754 binary16 bit pattern
/// (round-to-nearest-even). Values beyond the binary16 range become
/// ±inf; NaN becomes a quiet NaN. Mainly a test / tooling helper — the
/// runtime itself only ever reads binary16.
pub fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let man = b & 0x7F_FFFF;
    if exp == 0xFF {
        // inf or NaN
        return if man == 0 {
            sign | 0x7C00
        } else {
            sign | 0x7E00
        };
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let man = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = man >> shift;
        let rem = man & ((1 << shift) - 1);
        let round = (rem > (1 << (shift - 1))) || (rem == (1 << (shift - 1)) && (half & 1) == 1);
        return sign | (half as u16 + round as u16);
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1FFF;
    let round = (rem > 0x1000) || (rem == 0x1000 && (half & 1) == 1);
    sign | (half as u16 + round as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_and_signs() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0_f32).to_bits());
    }

    #[test]
    fn normals() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x4900), 10.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
    }

    #[test]
    fn subnormals_inf_nan() {
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert_eq!(f16_to_f32(0x03ff), 6.097_555e-5);
        assert_eq!(f16_to_f32(0x0400), 6.103_515_6e-5);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn round_trip_every_finite_half() {
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if f.is_nan() {
                continue;
            }
            assert_eq!(f32_to_f16(f), h, "h = {h:#06x}");
        }
    }

    #[test]
    fn encode_rounds_to_nearest_even() {
        // 1 + 2^-11 lies exactly between 1.0 (0x3c00) and 1+2^-10 (0x3c01);
        // ties go to the even mantissa.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        assert_eq!(f32_to_f16(70000.0), 0x7c00);
        assert_eq!(f32_to_f16(-70000.0), 0xfc00);
        assert_eq!(f32_to_f16(1e-9), 0x0000);
    }
}
