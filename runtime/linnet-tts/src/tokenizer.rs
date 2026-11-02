//! SentencePiece unigram tokenizer read directly from a `.model` file.
//!
//! The file is a protobuf `ModelProto`; only the fields the encoder needs
//! are decoded: the piece table (`piece`, `score`, `type`), the
//! normalizer flags (`add_dummy_prefix`, `remove_extra_whitespaces`,
//! `escape_whitespaces`, a possible precompiled character map) and the
//! trainer flags that change the segmentation (`byte_fallback`,
//! `unk_id`, `treat_whitespace_as_suffix`). Models with a precompiled
//! normalization map are refused: this encoder only implements the
//! identity normalizer, which is what the shipped model uses.
//!
//! Segmentation follows the reference implementation: a Viterbi search
//! over the lattice of all dictionary pieces, with an `unk` node of score
//! `min_score − 10` wherever no single-character piece exists;
//! user-defined symbols score `len · max_score − 0.1`; with
//! `byte_fallback` an `unk` character is emitted as its UTF-8 bytes
//! (`<0xNN>` pieces).

use std::collections::HashMap;

use crate::error::{Error, Result};

const TYPE_NORMAL: u64 = 1;
const TYPE_UNKNOWN: u64 = 2;
const TYPE_USER_DEFINED: u64 = 4;
const TYPE_BYTE: u64 = 6;

const WS: char = '\u{2581}';
const UNK_PENALTY: f32 = 10.0;

/// A loaded unigram model.
pub struct Tokenizer {
    pieces: HashMap<String, PieceEntry>,
    byte_ids: Option<[u32; 256]>,
    unk_id: u32,
    max_piece_chars: usize,
    unk_score: f32,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
    vocab_size: usize,
}

#[derive(Clone, Copy)]
struct PieceEntry {
    id: u32,
    score: f32,
}

// ───────────────────────── protobuf wire reader ─────────────────────────

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

enum Field<'a> {
    Varint(u64),
    Fixed32(u32),
    Fixed64,
    Bytes(&'a [u8]),
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }

    fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        let mut shift = 0;
        loop {
            let b = *self.buf.get(self.pos).ok_or_else(|| bad("truncated varint"))?;
            self.pos += 1;
            if shift < 64 {
                v |= ((b & 0x7f) as u64) << shift;
            }
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
            if shift > 70 {
                return Err(bad("varint too long"));
            }
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or_else(|| bad("length overflow"))?;
        if end > self.buf.len() {
            return Err(bad("truncated field"));
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    /// Next `(field_number, value)`.
    fn field(&mut self) -> Result<(u64, Field<'a>)> {
        let key = self.varint()?;
        let num = key >> 3;
        let value = match key & 7 {
            0 => Field::Varint(self.varint()?),
            1 => {
                self.take(8)?;
                Field::Fixed64
            }
            2 => {
                let n = self.varint()? as usize;
                Field::Bytes(self.take(n)?)
            }
            5 => {
                let b = self.take(4)?;
                Field::Fixed32(u32::from_le_bytes(b.try_into().unwrap()))
            }
            w => return Err(bad(&format!("unsupported wire type {w}"))),
        };
        Ok((num, value))
    }
}

fn bad(m: &str) -> Error {
    Error::Format(format!("sentencepiece model: {m}"))
}

struct RawPiece {
    piece: String,
    score: f32,
    kind: u64,
}

fn parse_piece(buf: &[u8]) -> Result<RawPiece> {
    let mut r = Reader::new(buf);
    let mut piece = None;
    let mut score = 0.0f32;
    let mut kind = TYPE_NORMAL;
    while !r.done() {
        match r.field()? {
            (1, Field::Bytes(b)) => {
                piece = Some(String::from_utf8(b.to_vec()).map_err(|_| bad("piece is not UTF-8"))?)
            }
            (2, Field::Fixed32(v)) => score = f32::from_bits(v),
            (3, Field::Varint(v)) => kind = v,
            _ => {}
        }
    }
    Ok(RawPiece {
        piece: piece.ok_or_else(|| bad("piece without text"))?,
        score,
        kind,
    })
}

#[derive(Default)]
struct NormalizerSpec {
    name: Option<String>,
    charsmap_len: usize,
    add_dummy_prefix: Option<bool>,
    remove_extra_whitespaces: Option<bool>,
    escape_whitespaces: Option<bool>,
}

fn parse_normalizer(buf: &[u8]) -> Result<NormalizerSpec> {
    let mut r = Reader::new(buf);
    let mut n = NormalizerSpec::default();
    while !r.done() {
        match r.field()? {
            (1, Field::Bytes(b)) => n.name = Some(String::from_utf8_lossy(b).into_owned()),
            (2, Field::Bytes(b)) => n.charsmap_len = b.len(),
            (3, Field::Varint(v)) => n.add_dummy_prefix = Some(v != 0),
            (4, Field::Varint(v)) => n.remove_extra_whitespaces = Some(v != 0),
            (5, Field::Varint(v)) => n.escape_whitespaces = Some(v != 0),
            _ => {}
        }
    }
    Ok(n)
}

#[derive(Default)]
struct TrainerSpec {
    model_type: Option<u64>,
    treat_whitespace_as_suffix: bool,
    byte_fallback: bool,
    unk_id: Option<u64>,
}

fn parse_trainer(buf: &[u8]) -> Result<TrainerSpec> {
    let mut r = Reader::new(buf);
    let mut t = TrainerSpec::default();
    while !r.done() {
        match r.field()? {
            (3, Field::Varint(v)) => t.model_type = Some(v),
            (24, Field::Varint(v)) => t.treat_whitespace_as_suffix = v != 0,
            (35, Field::Varint(v)) => t.byte_fallback = v != 0,
            (40, Field::Varint(v)) => t.unk_id = Some(v),
            _ => {}
        }
    }
    Ok(t)
}

impl Tokenizer {
    /// Parse a serialized `ModelProto`.
    pub fn from_model_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        let mut raw: Vec<RawPiece> = Vec::new();
        let mut norm = NormalizerSpec::default();
        let mut trainer = TrainerSpec::default();
        while !r.done() {
            match r.field()? {
                (1, Field::Bytes(b)) => raw.push(parse_piece(b)?),
                (2, Field::Bytes(b)) => trainer = parse_trainer(b)?,
                (3, Field::Bytes(b)) => norm = parse_normalizer(b)?,
                _ => {}
            }
        }
        if raw.is_empty() {
            return Err(bad("no pieces"));
        }
        if let Some(mt) = trainer.model_type {
            if mt != 1 {
                return Err(bad(&format!("model_type {mt} is not unigram")));
            }
        }
        if trainer.treat_whitespace_as_suffix {
            return Err(bad("treat_whitespace_as_suffix is not supported"));
        }
        if norm.charsmap_len > 0 {
            return Err(bad(&format!(
                "normalizer {:?} has a precompiled character map; only the identity normalizer is supported",
                norm.name.as_deref().unwrap_or("")
            )));
        }

        let mut pieces = HashMap::with_capacity(raw.len());
        let mut byte_ids = [u32::MAX; 256];
        let mut n_bytes = 0usize;
        let mut unk_from_type = None;
        let mut max_chars = 1usize;
        let mut min_score = f32::INFINITY;
        let mut max_score = f32::NEG_INFINITY;
        for (i, p) in raw.iter().enumerate() {
            match p.kind {
                TYPE_NORMAL => {
                    min_score = min_score.min(p.score);
                    max_score = max_score.max(p.score);
                }
                TYPE_UNKNOWN => unk_from_type = Some(i as u32),
                TYPE_BYTE => {
                    let b = parse_byte_piece(&p.piece).ok_or_else(|| bad(&format!("byte piece {:?}", p.piece)))?;
                    if byte_ids[b as usize] == u32::MAX {
                        n_bytes += 1;
                    }
                    byte_ids[b as usize] = i as u32;
                }
                _ => {}
            }
        }
        if !min_score.is_finite() {
            return Err(bad("no normal pieces"));
        }
        for (i, p) in raw.iter().enumerate() {
            let score = match p.kind {
                TYPE_NORMAL => p.score,
                TYPE_USER_DEFINED => p.piece.chars().count() as f32 * max_score - 0.1,
                _ => continue,
            };
            max_chars = max_chars.max(p.piece.chars().count());
            pieces.insert(p.piece.clone(), PieceEntry { id: i as u32, score });
        }
        let unk_id = match trainer.unk_id {
            Some(u) if (u as usize) < raw.len() && raw[u as usize].kind == TYPE_UNKNOWN => u as u32,
            _ => unk_from_type.ok_or_else(|| bad("no <unk> piece"))?,
        };
        let byte_ids = if trainer.byte_fallback {
            if n_bytes != 256 {
                return Err(bad("byte_fallback set but the byte pieces are incomplete"));
            }
            Some(byte_ids)
        } else {
            None
        };
        Ok(Tokenizer {
            pieces,
            byte_ids,
            unk_id,
            max_piece_chars: max_chars,
            unk_score: min_score - UNK_PENALTY,
            add_dummy_prefix: norm.add_dummy_prefix.unwrap_or(true),
            remove_extra_whitespaces: norm.remove_extra_whitespaces.unwrap_or(true),
            escape_whitespaces: norm.escape_whitespaces.unwrap_or(true),
            vocab_size: raw.len(),
        })
    }

    /// Load a `.model` file.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_model_bytes(&std::fs::read(path)?)
    }

    /// Number of pieces in the model.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Normalise (identity + whitespace rules) then segment.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let norm = self.normalize(text);
        if norm.is_empty() {
            return Vec::new();
        }
        self.viterbi(&norm)
    }

    fn normalize(&self, text: &str) -> Vec<char> {
        let mut chars: Vec<char> = text.chars().collect();
        if self.remove_extra_whitespaces {
            let mut out = Vec::with_capacity(chars.len());
            let mut prev_space = true; // drops leading spaces
            for &c in &chars {
                if c == ' ' {
                    if !prev_space {
                        out.push(c);
                    }
                    prev_space = true;
                } else {
                    out.push(c);
                    prev_space = false;
                }
            }
            while out.last() == Some(&' ') {
                out.pop();
            }
            chars = out;
        }
        if chars.is_empty() {
            return chars;
        }
        let mut out = Vec::with_capacity(chars.len() + 1);
        if self.add_dummy_prefix {
            out.push(if self.escape_whitespaces { WS } else { ' ' });
        }
        for c in chars {
            out.push(if c == ' ' && self.escape_whitespaces { WS } else { c });
        }
        out
    }

    fn viterbi(&self, chars: &[char]) -> Vec<u32> {
        let n = chars.len();
        let s: String = chars.iter().collect();
        let mut byte_off = Vec::with_capacity(n + 1);
        let mut off = 0usize;
        byte_off.push(0);
        for c in chars {
            off += c.len_utf8();
            byte_off.push(off);
        }
        const UNK: u32 = u32::MAX;
        let mut best = vec![f32::NEG_INFINITY; n + 1];
        let mut back: Vec<(usize, u32)> = vec![(0, UNK); n + 1];
        best[0] = 0.0;
        for i in 0..n {
            if best[i] == f32::NEG_INFINITY {
                continue;
            }
            let mut has_single = false;
            let max_j = (i + self.max_piece_chars).min(n);
            for j in (i + 1)..=max_j {
                if let Some(e) = self.pieces.get(&s[byte_off[i]..byte_off[j]]) {
                    if j == i + 1 {
                        has_single = true;
                    }
                    let cand = best[i] + e.score;
                    if cand > best[j] {
                        best[j] = cand;
                        back[j] = (i, e.id);
                    }
                }
            }
            if !has_single {
                let cand = best[i] + self.unk_score;
                if cand > best[i + 1] {
                    best[i + 1] = cand;
                    back[i + 1] = (i, UNK);
                }
            }
        }
        let mut segs: Vec<(usize, u32)> = Vec::new();
        let mut j = n;
        while j > 0 {
            let (i, id) = back[j];
            segs.push((i, id));
            j = i;
        }
        segs.reverse();
        let mut out = Vec::with_capacity(segs.len());
        for (i, id) in segs {
            if id != UNK {
                out.push(id);
            } else if let Some(bytes) = &self.byte_ids {
                let mut buf = [0u8; 4];
                for &b in chars[i].encode_utf8(&mut buf).as_bytes() {
                    out.push(bytes[b as usize]);
                }
            } else {
                out.push(self.unk_id);
            }
        }
        out
    }
}

fn parse_byte_piece(p: &str) -> Option<u8> {
    let hex = p.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}
