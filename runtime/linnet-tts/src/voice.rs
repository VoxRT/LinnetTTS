//! Voice prompt latents: a `[T × latent_dim]` f32 matrix stored as a
//! NumPy `.npy` file (format version 1 or 2, C order, little-endian).

use crate::error::{Error, Result};

/// Codec-space latents of a voice prompt.
#[derive(Debug, Clone)]
pub struct VoiceLatents {
    dim: usize,
    data: Vec<f32>,
}

impl VoiceLatents {
    /// Wrap a flat `[frames × dim]` buffer.
    pub fn from_vec(data: Vec<f32>, dim: usize) -> Result<Self> {
        if dim == 0 || data.is_empty() || data.len() % dim != 0 {
            return Err(Error::Invalid(format!(
                "voice latents: {} values is not a non-empty multiple of {dim}",
                data.len()
            )));
        }
        Ok(VoiceLatents { dim, data })
    }

    /// Parse a `.npy` image.
    pub fn from_npy_bytes(bytes: &[u8]) -> Result<Self> {
        let bad = |m: &str| Error::Format(format!("npy: {m}"));
        if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
            return Err(bad("missing magic"));
        }
        let major = bytes[6];
        let (header_len, header_start) = match major {
            1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
            2 | 3 => {
                if bytes.len() < 12 {
                    return Err(bad("truncated header"));
                }
                (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12)
            }
            v => return Err(bad(&format!("unsupported version {v}"))),
        };
        let data_start = header_start + header_len;
        if data_start > bytes.len() {
            return Err(bad("truncated header"));
        }
        let header = std::str::from_utf8(&bytes[header_start..data_start]).map_err(|_| bad("header not UTF-8"))?;
        let descr = dict_value(header, "descr").ok_or_else(|| bad("no descr"))?;
        let descr = descr.trim().trim_matches(|c| c == '\'' || c == '"');
        let width = match descr {
            "<f4" | "=f4" | "f4" => 4,
            "<f8" | "=f8" | "f8" => 8,
            other => return Err(bad(&format!("dtype {other} not supported (want <f4)"))),
        };
        if dict_value(header, "fortran_order").map(str::trim) != Some("False") {
            return Err(bad("fortran_order must be False"));
        }
        let shape_s = dict_value(header, "shape").ok_or_else(|| bad("no shape"))?;
        let shape_s = shape_s.trim().trim_start_matches('(').trim_end_matches(')');
        let shape: Vec<usize> = shape_s
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<usize>().map_err(|_| bad("shape not integers")))
            .collect::<Result<_>>()?;
        if shape.len() != 2 {
            return Err(bad(&format!("expected a 2-D array, got shape {shape:?}")));
        }
        let numel = shape[0] * shape[1];
        let payload = &bytes[data_start..];
        if payload.len() != numel * width {
            return Err(bad(&format!(
                "payload {} bytes, expected {} for shape {shape:?}",
                payload.len(),
                numel * width
            )));
        }
        let data: Vec<f32> = if width == 4 {
            payload.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
        } else {
            payload
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
                .collect()
        };
        Self::from_vec(data, shape[1])
    }

    /// Load a `.npy` file.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_npy_bytes(&std::fs::read(path)?)
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn frames(&self) -> usize {
        self.data.len() / self.dim
    }

    /// Flat `[frames × dim]` latents.
    pub fn latents(&self) -> &[f32] {
        &self.data
    }
}

/// Value of `'key': value` in a Python-dict-literal header (up to the
/// next top-level comma).
fn dict_value<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("'{key}':");
    let start = header.find(&needle)? + needle.len();
    let rest = &header[start..];
    let mut depth = 0i32;
    for (i, ch) in rest.char_indices() {
        match ch {
            '(' | '[' => depth += 1,
            ')' | ']' => depth -= 1,
            ',' | '}' if depth <= 0 => return Some(&rest[..i]),
            _ => {}
        }
    }
    Some(rest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy(shape: (usize, usize), vals: &[f32]) -> Vec<u8> {
        let header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}", shape.0, shape.1);
        let mut h = header.into_bytes();
        while (10 + h.len() + 1) % 64 != 0 {
            h.push(b' ');
        }
        h.push(b'\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(h.len() as u16).to_le_bytes());
        out.extend_from_slice(&h);
        for v in vals {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn parses_v1_file() {
        let vals: Vec<f32> = (0..96).map(|i| i as f32 * 0.5).collect();
        let v = VoiceLatents::from_npy_bytes(&npy((2, 48), &vals)).unwrap();
        assert_eq!(v.frames(), 2);
        assert_eq!(v.dim(), 48);
        assert_eq!(v.latents(), &vals[..]);
    }

    #[test]
    fn rejects_bad_payload() {
        let mut b = npy((2, 48), &vec![0.0; 96]);
        b.pop();
        assert!(VoiceLatents::from_npy_bytes(&b).is_err());
    }
}
