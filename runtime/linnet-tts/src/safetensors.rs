//! Minimal safetensors reader: 8-byte little-endian header length, a JSON
//! header (`__metadata__` plus one entry per tensor with `dtype`, `shape`
//! and `data_offsets`) and the raw tensor bytes. Only `F32` and `F16`
//! tensors are accepted; both are little-endian in the file.

use std::collections::HashMap;

use serde::Deserialize;

use crate::error::{Error, Result};

/// Element type of a stored tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
}

impl DType {
    fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
        }
    }
}

#[derive(Deserialize)]
struct RawEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

/// One tensor's header entry, resolved against the data section.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: DType,
    pub shape: Vec<usize>,
    start: usize,
    end: usize,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// A parsed safetensors file held in memory.
pub struct SafeTensors {
    metadata: HashMap<String, String>,
    tensors: HashMap<String, TensorInfo>,
    data: Vec<u8>,
    data_start: usize,
}

impl SafeTensors {
    /// Parse a complete file image. The byte vector is kept (tensor data
    /// is sliced out of it on demand).
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        if bytes.len() < 8 {
            return Err(Error::Format("safetensors: file shorter than 8 bytes".into()));
        }
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let data_start = 8usize
            .checked_add(header_len)
            .ok_or_else(|| Error::Format("safetensors: header length overflow".into()))?;
        if data_start > bytes.len() {
            return Err(Error::Format(format!(
                "safetensors: header length {header_len} exceeds file size {}",
                bytes.len()
            )));
        }
        let header: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&bytes[8..data_start])
                .map_err(|e| Error::Format(format!("safetensors: header is not JSON: {e}")))?;
        let data_len = bytes.len() - data_start;

        let mut metadata = HashMap::new();
        let mut tensors = HashMap::new();
        for (name, value) in header {
            if name == "__metadata__" {
                let m: HashMap<String, String> = serde_json::from_value(value).map_err(|e| {
                    Error::Format(format!("safetensors: __metadata__ must map strings: {e}"))
                })?;
                metadata = m;
                continue;
            }
            let raw: RawEntry = serde_json::from_value(value)
                .map_err(|e| Error::Format(format!("safetensors: entry {name}: {e}")))?;
            let dtype = match raw.dtype.as_str() {
                "F32" => DType::F32,
                "F16" => DType::F16,
                other => {
                    return Err(Error::Format(format!(
                        "safetensors: tensor {name} has unsupported dtype {other}"
                    )))
                }
            };
            let numel: usize = raw.shape.iter().product();
            let [start, end] = raw.data_offsets;
            if end < start || end > data_len || end - start != numel * dtype.size() {
                return Err(Error::Format(format!(
                    "safetensors: tensor {name}: offsets [{start}, {end}) inconsistent with shape {:?} ({numel} x {} bytes, data section {data_len} bytes)",
                    raw.shape,
                    dtype.size()
                )));
            }
            tensors.insert(
                name,
                TensorInfo {
                    dtype,
                    shape: raw.shape,
                    start,
                    end,
                },
            );
        }
        Ok(SafeTensors {
            metadata,
            tensors,
            data: bytes,
            data_start,
        })
    }

    /// Read and parse a file from disk.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_bytes(std::fs::read(path)?)
    }

    /// `__metadata__` entries (all strings).
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }

    /// Tensor names in the file (unordered).
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// Header entry of a tensor, or a `Shape` error if it is missing.
    pub fn info(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::Shape(format!("missing tensor {name}")))
    }

    /// Raw little-endian bytes of a tensor.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let t = self.info(name)?;
        Ok(&self.data[self.data_start + t.start..self.data_start + t.end])
    }

    /// Fail unless the tensor exists with exactly `shape`.
    pub fn expect_shape(&self, name: &str, shape: &[usize]) -> Result<&TensorInfo> {
        let t = self.info(name)?;
        if t.shape != shape {
            return Err(Error::Shape(format!(
                "tensor {name}: shape {:?}, expected {:?}",
                t.shape, shape
            )));
        }
        Ok(t)
    }

    /// Tensor as f32 values (F16 is widened). Shape is validated.
    pub fn f32_vec(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let t = self.expect_shape(name, shape)?;
        let b = self.bytes(name)?;
        Ok(match t.dtype {
            DType::F32 => b
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            DType::F16 => b
                .chunks_exact(2)
                .map(|c| linnet_ops::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
        })
    }

    /// Tensor as raw binary16 bit patterns. Fails if the tensor is not F16.
    pub fn f16_vec(&self, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
        let t = self.expect_shape(name, shape)?;
        if t.dtype != DType::F16 {
            return Err(Error::Shape(format!("tensor {name}: expected F16, got {:?}", t.dtype)));
        }
        Ok(self
            .bytes(name)?
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    }
}
