"""Text tokenizer: a SentencePiece unigram model, used as-is (no extra normalisation, no BOS/EOS).

The LM's text embedding table has `vocab_size + 1` rows; the extra row (id == vocab_size) is the
training-time padding id and is never produced here.
"""

from __future__ import annotations

from pathlib import Path

import sentencepiece as spm
import torch


class Tokenizer:
    def __init__(self, model_path: str | Path, expected_vocab: int | None = None):
        self.sp = spm.SentencePieceProcessor(model_file=str(model_path))
        if expected_vocab is not None and self.sp.vocab_size() != expected_vocab:
            raise ValueError(
                f"tokenizer vocab {self.sp.vocab_size()} != model vocab {expected_vocab}"
            )

    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()

    def encode(self, text: str) -> list[int]:
        return list(self.sp.encode(text, out_type=int))

    def encode_tensor(self, text: str) -> torch.Tensor:
        return torch.tensor(self.encode(text), dtype=torch.long)

    def pieces(self, text: str) -> list[str]:
        return list(self.sp.encode(text, out_type=str))
