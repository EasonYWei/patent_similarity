#!/usr/bin/env python3
"""Patent SBERT embeddings + aggregation pipeline.

This script reads a cleaned patent file (default: `data/patents_cleaned.dta`),
computes patent embeddings with SBERT, aggregates by firm-year, and writes:
- output/stkcd_year_embeddings.csv
- output/stkcd_year_citweighted_embeddings.csv

Citation weighting is based on `p_cite` in the cleaned file.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm  # For custom progress bars


os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def get_gpu_info() -> dict:
    """Get GPU information for monitoring."""
    info = {"available": False, "count": 0, "name": "N/A", "memory_total": 0, "memory_free": 0}
    if torch.cuda.is_available():
        info["available"] = True
        info["count"] = torch.cuda.device_count()
        info["name"] = torch.cuda.get_device_name(0)
        info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        info["memory_free"] = (torch.cuda.mem_get_info(0)[0] / (1024**3)) if hasattr(torch.cuda, "mem_get_info") else 0
    return info


def print_gpu_status():
    """Print current GPU memory usage."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def recommend_batch_size(gpu_memory_gb: float, model_name: str) -> int:
    """Recommend batch size based on GPU memory and model."""
    # MiniLM: ~0.5GB per 1000 samples at batch=256
    # For 32GB RTX 5090, can easily do 1024+
    if "minilm" in model_name.lower():
        if gpu_memory_gb >= 24:
            return 1024
        elif gpu_memory_gb >= 16:
            return 512
        elif gpu_memory_gb >= 8:
            return 256
        else:
            return 128
    else:
        # Default for other models
        if gpu_memory_gb >= 24:
            return 512
        elif gpu_memory_gb >= 16:
            return 256
        elif gpu_memory_gb >= 8:
            return 128
        else:
            return 64

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_INPUT_FILE = PROJECT_ROOT / "data" / "patents_cleaned.dta"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

STKCD_COLUMN = "stkcd"
YEAR_COLUMN = "p_year"
KEY_COLUMN = "stkcd_year"
TEXT_COLUMNS = ("p_tt", "p_abs")
CITATION_COLUMN = "p_cite"
REQUIRED_COLUMNS = (STKCD_COLUMN, YEAR_COLUMN, TEXT_COLUMNS[0], TEXT_COLUMNS[1])
ZERO_DIVISION_EPSILON = 1e-12
TOKENIZED_SENTENCE_SPLIT = re.compile(r"(?<=[。；;!?！？。!?.])\s+|\n+")

MODEL_SHORT_NAMES = {
    "paraphrase-multilingual-MiniLM-L12-v2": "minilm",
    "distiluse-base-multilingual-cased-v2": "distiluse",
}


def get_model_short_name(model_name: str) -> str:
    """Get short form of model name for file naming."""
    return MODEL_SHORT_NAMES.get(model_name, model_name.split("-")[0].lower())

EMBEDDING_OUTPUT_COLUMNS = [
    STKCD_COLUMN,
    YEAR_COLUMN,
    KEY_COLUMN,
    "n_patents",
    "n_texts_used",
    "total_citations",
    "mean_citations",
]


@dataclass
class EmbeddingStats:
    fallback_count: int = 0
    total_fallback_chunks: int = 0
    total_fallback_tokens: int = 0

    @property
    def avg_chunks(self) -> float:
        if self.fallback_count <= 0:
            return 0.0
        return self.total_fallback_chunks / self.fallback_count

    @property
    def mean_chunk_tokens(self) -> float:
        if self.total_fallback_chunks <= 0:
            return 0.0
        return self.total_fallback_tokens / self.total_fallback_chunks


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_required_columns(
    df: pd.DataFrame, data_path: Path, required: Iterable[str]
) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input {data_path} missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def build_text_field(
    df: pd.DataFrame, text_cols: Sequence[str] = TEXT_COLUMNS
) -> Tuple[pd.Series, pd.Series]:
    parts: List[pd.Series] = []
    for col in text_cols:
        if col in df.columns:
            parts.append(df[col].astype("string").fillna(""))
        else:
            parts.append(pd.Series([""] * len(df), dtype="string"))

    if len(parts) == 0:
        text = pd.Series([""] * len(df), dtype="string")
    elif len(parts) == 1:
        text = parts[0]
    else:
        text = (parts[0] + " " + parts[1])

    text = text.str.replace(r"\s+", " ", regex=True).str.strip().astype("string")
    text_is_empty = text.str.len().fillna(0).eq(0)
    return text, text_is_empty


def coerce_citations(
    df: pd.DataFrame, citation_col: Optional[str]
) -> Tuple[np.ndarray, int]:
    log = logging.getLogger(__name__)
    if citation_col is None or citation_col not in df.columns:
        return np.zeros(len(df), dtype=np.float64), 0

    raw = pd.to_numeric(df[citation_col], errors="coerce")
    invalid = int(raw.isna().sum())
    if invalid:
        log.warning(
            "Column %s has %d invalid/missing values; treating them as 0.",
            citation_col,
            invalid,
        )
    raw = raw.fillna(0.0).clip(lower=0.0)
    return raw.to_numpy(dtype=np.float64), invalid


def divide_rows(
    sums: np.ndarray, counts: np.ndarray, dtype: type = np.float32
) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if sums.size == 0:
        return np.empty_like(sums, dtype=dtype)

    out = np.empty_like(sums, dtype=dtype)
    out[:] = np.nan
    mask = counts > ZERO_DIVISION_EPSILON
    if np.any(mask):
        out[mask] = (sums[mask] / counts[mask, None]).astype(dtype)
    return out


def load_and_prepare_data(data_path: Path) -> pd.DataFrame:
    """Backwards-compatible alias for loading one cleaned file."""
    return load_single_file(data_path)


def load_single_file(data_path: Path) -> pd.DataFrame:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    try:
        df = pd.read_stata(data_path, convert_categoricals=False)
    except Exception as exc:
        logging.getLogger(__name__).error("Failed to read %s: %s", data_path, exc)
        raise

    # Normalize legacy/alternate field names used in intermediate outputs.
    # (No legacy date fields to normalize currently)

    validate_required_columns(df, data_path, REQUIRED_COLUMNS)

    # Keep memory use reasonable by dropping unused columns early.
    keep = [
        STKCD_COLUMN,
        YEAR_COLUMN,
        "p_tt",
        "p_abs",
        "p_date",
        CITATION_COLUMN,
    ]
    keep = [c for c in keep if c in df.columns]
    if len(keep) < len(df.columns):
        missing = sorted(set(df.columns) - set(keep))
        logging.getLogger(__name__).debug(
            "Dropping non-required columns from %s: %s", data_path.name, missing
        )
        df = df[keep].copy()

    before = len(df)
    df = df[df[STKCD_COLUMN].notna() & df[YEAR_COLUMN].notna()].copy()
    dropped = before - len(df)
    if dropped:
        logging.getLogger(__name__).warning(
            "Dropped %d rows with missing stkcd or year in %s", dropped, data_path.name
        )

    df[YEAR_COLUMN] = pd.to_numeric(df[YEAR_COLUMN], errors="coerce").astype("Int32")
    df = df[df[YEAR_COLUMN].notna()].copy()

    df[STKCD_COLUMN] = df[STKCD_COLUMN].astype("string").str.strip()

    if "p_date" in df.columns:
        df["p_date"] = pd.to_datetime(df["p_date"], errors="coerce")
        df = df.sort_values([STKCD_COLUMN, "p_date"], ascending=True, kind="mergesort")
    else:
        df = df.sort_values([STKCD_COLUMN, YEAR_COLUMN], ascending=True, kind="mergesort")

    df = df.reset_index(drop=True)
    df["text"], df["text_is_empty"] = build_text_field(df)
    df[KEY_COLUMN] = (
        df[STKCD_COLUMN].astype("string") + "_" + df[YEAR_COLUMN].astype("Int32").astype(str)
    )

    if CITATION_COLUMN in df.columns:
        df[CITATION_COLUMN] = (
            pd.to_numeric(df[CITATION_COLUMN], errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
        )

    if df.empty:
        logging.getLogger(__name__).warning("No valid rows in %s after preprocessing", data_path.name)
        return df

    logging.getLogger(__name__).info(
        "Loaded %s -> rows=%d | companies=%d | firm-years=%d | empty_text=%d",
        data_path.name,
        len(df),
        df[STKCD_COLUMN].nunique(),
        df[KEY_COLUMN].nunique(),
        int(df["text_is_empty"].sum()),
    )
    return df


class SBertEmbedder:
    def __init__(
        self,
        model_dir: Path,
        model_name: str,
        device: Optional[str] = None,
        multi_gpu: bool = False,
        fp16: bool = False,
        tf32: bool = False,
        max_seq_length: Optional[int] = None,
        embed_backend: str = "overflow",
    ) -> None:
        """
        Wrapper around SentenceTransformers to embed patent texts efficiently.

        Performance knobs:
          - fp16: cast model weights to float16 (CUDA only). Faster on many GPUs, small numeric drift.
          - tf32: allow TF32 matmul on Ampere+ GPUs (CUDA only). Faster, small numeric drift.
          - multi_gpu: use SentenceTransformers' multi-process pool when >1 GPU is available.
        """
        from sentence_transformers import SentenceTransformer

        log = logging.getLogger(__name__)
        self.model_path = model_dir / model_name
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = str(device)

        if tf32 and self.device.startswith("cuda") and torch.cuda.is_available():
            # TF32 can substantially speed up matmul on Ampere+ GPUs.
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                # Older torch versions may not support this; ignore.
                pass
            log.info("Enabled TF32 matmul (CUDA).")

        log.info("Loading SBERT model from: %s", self.model_path)
        t0 = time.time()
        self.model = SentenceTransformer(str(self.model_path), device=self.device)

        if fp16 and self.device.startswith("cuda") and torch.cuda.is_available():
            try:
                self.model = self.model.half()
                log.info("Using fp16 model weights (CUDA).")
            except Exception as exc:
                log.warning("Failed to cast model to fp16; continuing in fp32. Error: %s", exc)

        self.tokenizer = getattr(self.model, "tokenizer", None)
        if self.tokenizer is None:
            raise RuntimeError("SentenceTransformer tokenizer is not available.")

        tokenizer_max_len = getattr(self.tokenizer, "model_max_length", None)
        if tokenizer_max_len is None or not isinstance(tokenizer_max_len, int) or tokenizer_max_len <= 0:
            tokenizer_max_len = 512
        self.tokenizer_model_max_length = int(tokenizer_max_len)

        # SentenceTransformers sometimes defaults to a small max_seq_length (e.g., 128),
        # which can silently truncate inputs. Prefer 512 when the tokenizer supports it,
        # unless the user explicitly overrides it.
        st_max_len = getattr(self.model, "max_seq_length", None)
        log.info(
            "SentenceTransformer max_seq_length=%s | tokenizer model_max_length=%s",
            st_max_len,
            self.tokenizer_model_max_length,
        )

        if max_seq_length is not None:
            desired_max_len = int(max_seq_length)
        else:
            desired_max_len = min(512, self.tokenizer_model_max_length)

        if hasattr(self.model, "max_seq_length"):
            try:
                self.model.max_seq_length = int(desired_max_len)
            except Exception as exc:
                log.warning("Failed to set model.max_seq_length; continuing. Error: %s", exc)

        self.max_tokens = int(desired_max_len)
        # Reserve room for special tokens when using tokenizer decode/encode boundaries.
        self.chunk_token_limit = max(32, self.max_tokens - 16)
        self.pool = None

        self.multi_gpu = bool(multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1)
        if self.multi_gpu:
            target_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            log.info("Enabling SentenceTransformers multi-process pool: %s", target_devices)
            self.pool = self.model.start_multi_process_pool(target_devices=target_devices)

        log.info("Model loaded in %.2fs | device=%s", time.time() - t0, device)
        log.info("SBERT token budget: max_tokens=%d chunk_limit=%d", self.max_tokens, self.chunk_token_limit)
        if torch.cuda.is_available():
            log.info("CUDA: %s | GPUs: %d", torch.version.cuda, torch.cuda.device_count())
        self.embed_backend = str(embed_backend).lower().strip()
        if self.embed_backend not in ("overflow", "legacy"):
            raise ValueError(f"Unknown embed_backend: {self.embed_backend}")
        self.embed_stats = EmbeddingStats()

    def close(self) -> None:
        if self.pool is not None:
            self.model.stop_multi_process_pool(self.pool)
            self.pool = None

    def _encode_batch(self, texts: list[str], batch_size: int, show_progress: bool) -> np.ndarray:
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        with torch.inference_mode():
            emb = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                device=self.device,
            )
        return np.asarray(emb, dtype=np.float32)

    def _encode_ids(self, text: str) -> list[int]:
        """Tokenize once (no special tokens, no truncation)."""
        if not text:
            return []
        return self.tokenizer.encode(text, add_special_tokens=False, truncation=False)

    def tokenize_and_split(
        self,
        text: str,
        *,
        token_count: Optional[int] = None,
        token_ids: Optional[list[int]] = None,
        long_text_token_chunk_threshold: int = 8,
        max_sentence_splits: int = 256,
    ) -> Tuple[list[str], list[int]]:
        """
        Split a (potentially long) text into chunks that fit within chunk_token_limit.

        Returns:
          - chunks: list[str]
          - chunk_token_counts: list[int] (exact for pure token-chunking, conservative estimate for sentence chunks)
        """
        if not text:
            return [""], [0]

        full_text = str(text).strip()
        if not full_text:
            return [""], [0]

        if token_ids is None and token_count is None:
            token_ids = self._encode_ids(full_text)
            token_count = len(token_ids)
        elif token_ids is not None and token_count is None:
            token_count = len(token_ids)

        token_count = int(token_count or 0)
        if token_count <= self.chunk_token_limit:
            return [full_text], [token_count]

        # If the document is extremely long, sentence-based splitting can be very slow.
        # Fall back to direct token chunking (1 full tokenize + slicing).
        if token_count > self.chunk_token_limit * max(1, int(long_text_token_chunk_threshold)):
            if token_ids is None:
                token_ids = self._encode_ids(full_text)
            return self._split_token_chunks(token_ids)

        # Prefer sentence-like splitting to better preserve semantic boundaries.
        sentences = [seg.strip() for seg in TOKENIZED_SENTENCE_SPLIT.split(full_text) if seg.strip()]
        if len(sentences) <= 1 or len(sentences) > max_sentence_splits:
            if token_ids is None:
                token_ids = self._encode_ids(full_text)
            return self._split_token_chunks(token_ids)

        # Tokenize each sentence once (avoid repeated "candidate" tokenization).
        try:
            encoded = self.tokenizer(
                sentences,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            sent_token_ids = encoded["input_ids"]
        except Exception:
            sent_token_ids = [self._encode_ids(s) for s in sentences]

        # Conservative buffer for the join between sentences. This avoids under-estimation.
        join_buffer = 1

        chunks: list[str] = []
        chunk_token_counts: list[int] = []
        cur_sents: list[str] = []
        cur_tokens = 0

        for sent, ids in zip(sentences, sent_token_ids):
            sent_len = len(ids)

            # Very long single sentence -> split by token ids directly.
            if sent_len > self.chunk_token_limit:
                if cur_sents:
                    chunks.append(" ".join(cur_sents))
                    chunk_token_counts.append(cur_tokens)
                    cur_sents = []
                    cur_tokens = 0
                sub_chunks, sub_counts = self._split_token_chunks(ids)
                chunks.extend(sub_chunks)
                chunk_token_counts.extend(sub_counts)
                continue

            needed = sent_len + (join_buffer if cur_sents else 0)
            if cur_tokens + needed <= self.chunk_token_limit:
                cur_sents.append(sent)
                cur_tokens += needed
                continue

            if cur_sents:
                chunks.append(" ".join(cur_sents))
                chunk_token_counts.append(cur_tokens)

            cur_sents = [sent]
            cur_tokens = sent_len

        if cur_sents:
            chunks.append(" ".join(cur_sents))
            chunk_token_counts.append(cur_tokens)

        if not chunks:
            if token_ids is None:
                token_ids = self._encode_ids(full_text)
            return self._split_token_chunks(token_ids)

        return chunks, chunk_token_counts

    def _split_token_chunks(self, token_ids: list[int]) -> Tuple[list[str], list[int]]:
        chunks: list[str] = []
        counts: list[int] = []
        if not token_ids:
            return [""], [0]
        for start in range(0, len(token_ids), self.chunk_token_limit):
            chunk_ids = token_ids[start : start + self.chunk_token_limit]
            if not chunk_ids:
                continue
            text_piece = self.tokenizer.decode(
                chunk_ids,
                clean_up_tokenization_spaces=True,
                skip_special_tokens=True,
            )
            piece = text_piece.strip()
            if piece:
                chunks.append(piece)
                counts.append(len(chunk_ids))
        if not chunks:
            return [""], [0]
        return chunks, counts

    def embed_texts_with_fallback(self, texts: list[str], batch_size: int, show_progress: bool) -> np.ndarray:
        """
        Embed texts, splitting only the ones that exceed the token budget.

        Major speed optimization vs. the original implementation:
          - All fallback chunks across all long texts are encoded in ONE batched call,
            instead of calling model.encode() once per long text.
        """
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)

        norm_texts = ["" if x is None else str(x) for x in texts]
        n = len(norm_texts)
        dim = self.model.get_sentence_embedding_dimension()
        out = np.empty((n, dim), dtype=np.float32)

        short_texts: list[str] = []
        short_indices: list[int] = []

        flat_chunks: list[str] = []
        flat_owner: list[int] = []

        # Stats (computed from splitting logic; no extra tokenization passes).
        fallback_text_count = 0
        total_chunks = 0
        total_chunk_tokens = 0
        max_chunk_tokens = 0

        for idx, text in enumerate(norm_texts):
            if not text:
                # Empty string is always "short".
                short_indices.append(idx)
                short_texts.append("")
                continue

            token_ids = self._encode_ids(text)
            token_count = len(token_ids)

            if token_count > self.chunk_token_limit:
                chunks, chunk_counts = self.tokenize_and_split(
                    text,
                    token_count=token_count,
                    token_ids=token_ids,
                )
                if not chunks:
                    chunks, chunk_counts = [""], [0]

                fallback_text_count += 1
                total_chunks += len(chunks)
                total_chunk_tokens += int(sum(chunk_counts))
                if chunk_counts:
                    max_chunk_tokens = max(max_chunk_tokens, max(chunk_counts))

                flat_chunks.extend(chunks)
                flat_owner.extend([idx] * len(chunks))
            else:
                short_indices.append(idx)
                short_texts.append(text)

        # If multi-GPU pool is available and no fallback is needed, use the fast multi-process path.
        if self.pool is not None and not flat_chunks:
            emb = self.model.encode_multi_process(
                norm_texts,
                self.pool,
                batch_size=batch_size,
                show_progress_bar=show_progress,
            )
            return np.asarray(emb, dtype=np.float32)

        # To avoid two progress bars, show it only for the larger encoding call.
        show_short = bool(show_progress and (len(short_texts) >= len(flat_chunks)))
        show_chunks = bool(show_progress and (len(flat_chunks) > len(short_texts)))

        if short_texts:
            short_matrix = self._encode_batch(short_texts, batch_size=batch_size, show_progress=show_short)
            out[np.asarray(short_indices, dtype=np.int64), :] = short_matrix

        if flat_chunks:
            # Chunk batches are usually heavier than short texts; keep the batch size conservative.
            chunk_bs = max(8, min(int(batch_size), 1024))
            chunk_matrix = self._encode_batch(flat_chunks, batch_size=chunk_bs, show_progress=show_chunks)

            owner_arr = np.asarray(flat_owner, dtype=np.int64)
            uniq_owner, inv = np.unique(owner_arr, return_inverse=True)

            sums = np.zeros((len(uniq_owner), dim), dtype=np.float32)
            np.add.at(sums, inv, chunk_matrix)
            counts = np.bincount(inv).astype(np.float32)

            means = sums / np.maximum(counts, 1.0)[:, None]
            out[uniq_owner, :] = means.astype(np.float32)

        if fallback_text_count:
            self.embed_stats.fallback_count += fallback_text_count
            self.embed_stats.total_fallback_chunks += total_chunks
            self.embed_stats.total_fallback_tokens += total_chunk_tokens

            log = logging.getLogger(__name__)
            log.debug(
                "Fallback: %d texts, %.2f chunks/text, mean chunk tokens %.1f, max chunk tokens %d",
                fallback_text_count,
                self.embed_stats.avg_chunks,
                self.embed_stats.mean_chunk_tokens,
                max_chunk_tokens,
            )

        return out

    def embed_texts_overflow_windows(
        self,
        texts: list[str],
        *,
        seq_batch_size: int,
        doc_batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Fast embedding path: single tokenization pass using HF tokenizer overflow windows + direct model.forward().

        - Avoids per-row tokenizer.encode() in Python loops
        - Avoids decoding token chunks back to text and re-tokenizing
        - Handles long texts by creating multiple max_length windows per document
        - Aggregates back to one embedding per input document via simple mean over windows

        Notes:
          - Windows are non-overlapping by default (stride=0).
          - This matches the previous "equal weight per chunk" averaging logic.
        """
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)

        norm_texts = ["" if x is None else str(x) for x in texts]
        n = len(norm_texts)
        dim = self.model.get_sentence_embedding_dimension()

        if max_length is None:
            max_length = int(self.max_tokens)

        if doc_batch_size is None:
            # Tokenizer work benefits from larger batches than GPU forward batches.
            doc_batch_size = max(256, int(seq_batch_size) * 2)

        out = np.empty((n, dim), dtype=np.float32)

        device = torch.device(self.device)

        # Progress bar over documents (not windows), to avoid overly noisy output.
        ranges = range(0, n, int(doc_batch_size))
        it = tqdm(ranges, disable=not show_progress, desc="Tokenize+embed", unit="docs")

        for start in it:
            block = norm_texts[start : min(start + int(doc_batch_size), n)]
            bsz = len(block)

            # Tokenize once, creating overflow windows for long texts.
            try:
                enc = self.tokenizer(
                    block,
                    padding=True,
                    truncation=True,
                    max_length=int(max_length),
                    return_overflowing_tokens=True,
                    return_tensors="pt",
                )
            except TypeError:
                # Some older tokenizers require explicit stride arg when returning overflow.
                enc = self.tokenizer(
                    block,
                    padding=True,
                    truncation=True,
                    max_length=int(max_length),
                    stride=0,
                    return_overflowing_tokens=True,
                    return_tensors="pt",
                )

            mapping = enc.pop("overflow_to_sample_mapping")  # [n_windows]

            features = {
                k: v
                for k, v in enc.items()
                if k in ("input_ids", "attention_mask", "token_type_ids")
            }

            # Optional pinning helps H2D transfer when CPU is the bottleneck.
            if device.type == "cuda":
                for k in list(features.keys()):
                    try:
                        features[k] = features[k].pin_memory()
                    except Exception:
                        pass
                try:
                    mapping = mapping.pin_memory()
                except Exception:
                    pass

            # GPU-side accumulators (float32 accumulation even if model is fp16).
            sums = torch.zeros((bsz, dim), device=device, dtype=torch.float32)
            counts = torch.zeros((bsz,), device=device, dtype=torch.float32)

            n_windows = int(features["input_ids"].shape[0])
            mapping = mapping.to(device, non_blocking=True)

            w0 = 0
            while w0 < n_windows:
                w1 = min(w0 + int(seq_batch_size), n_windows)
                mb = {k: v[w0:w1].to(device, non_blocking=True) for k, v in features.items()}
                mb_map = mapping[w0:w1]

                with torch.inference_mode():
                    out_dict = self.model.forward(mb)
                    if isinstance(out_dict, dict) and "sentence_embedding" in out_dict:
                        mb_emb = out_dict["sentence_embedding"]
                    else:
                        # Extremely defensive fallback; SentenceTransformers should return a dict.
                        mb_emb = out_dict

                # Accumulate per-document sums and counts.
                sums.index_add_(0, mb_map, mb_emb.float())
                counts.index_add_(
                    0,
                    mb_map,
                    torch.ones_like(mb_map, dtype=torch.float32),
                )

                w0 = w1

            block_emb = (sums / counts.clamp_min(1.0).unsqueeze(1)).cpu().numpy().astype(
                np.float32, copy=False
            )
            out[start : start + bsz] = block_emb

        return out

    def embed(self, texts: list[str], batch_size: int, show_progress: bool) -> np.ndarray:
        t0 = time.time()
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)

        norm_texts = [
            "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)
            for x in texts
        ]

        if self.embed_backend == "legacy":
            emb = self.embed_texts_with_fallback(
                norm_texts, batch_size=batch_size, show_progress=show_progress
            )
        else:
            emb = self.embed_texts_overflow_windows(
                norm_texts,
                seq_batch_size=int(batch_size),
                doc_batch_size=None,
                max_length=int(self.max_tokens),
                show_progress=show_progress,
            )

        dt = time.time() - t0
        log = logging.getLogger(__name__)
        # Avoid spamming logs when embedding small chunks.
        if len(texts) >= 10_000 or log.isEnabledFor(logging.DEBUG):
            rate = dt / max(len(texts), 1)
            log.info("Embedded %d texts in %.2fs (%.4fs/text)", len(texts), dt, rate)

        if self.embed_stats.fallback_count:
            log.info(
                "Fallback summary: texts=%d, avg_chunks=%.2f, mean_chunk_tokens=%.2f",
                self.embed_stats.fallback_count,
                self.embed_stats.avg_chunks,
                self.embed_stats.mean_chunk_tokens,
            )
        return emb


def aggregate_chunk(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    key_col: str = KEY_COLUMN,
    exclude_empty_text: bool = True,
    citation_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chunk-level aggregation into sufficient statistics.
    Returns:
      - meta (one row per key in this chunk)
      - sum_embedding (sum over rows that contribute to mean)
      - n_text_rows (counts used in mean)
      - sum_citation_weighted_embedding (sum(citation * embedding)
    """
    if len(df) != embeddings.shape[0]:
        raise ValueError(
            f"Row mismatch: df has {len(df)} rows but embeddings has {embeddings.shape[0]} rows"
        )

    if len(df) == 0:
        return (
            pd.DataFrame(columns=EMBEDDING_OUTPUT_COLUMNS),
            np.empty((0, embeddings.shape[1] if embeddings.ndim == 2 else 0), dtype=np.float32),
            np.empty((0,), dtype=np.float64),
            np.empty((0, embeddings.shape[1] if embeddings.ndim == 2 else 0), dtype=np.float32),
        )

    log = logging.getLogger(__name__)
    t0 = time.time()

    keys = df[key_col].astype("string").to_numpy()
    uniq_keys, inv = np.unique(keys, return_inverse=True)
    inv = inv.astype(np.int64)
    dim = embeddings.shape[1]
    n_groups = len(uniq_keys)

    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim != 2:
        raise ValueError("embeddings must be a 2D array")

    if exclude_empty_text and "text_is_empty" in df.columns:
        text_weights = (~df["text_is_empty"].to_numpy()).astype(np.float32)
    else:
        text_weights = np.ones(len(df), dtype=np.float32)

    sum_embedding = np.zeros((n_groups, dim), dtype=np.float32)
    np.add.at(sum_embedding, inv, emb * text_weights[:, None])
    n_text_rows = np.bincount(inv, weights=text_weights).astype(np.float64)

    all_counts = np.bincount(inv).astype(np.float64)
    if np.any(n_text_rows == 0):
        fallback_mask = n_text_rows == 0
        all_sum = np.zeros((n_groups, dim), dtype=np.float32)
        np.add.at(all_sum, inv, emb)
        sum_embedding[fallback_mask] = all_sum[fallback_mask]
        n_text_rows[fallback_mask] = all_counts[fallback_mask]
        if np.any(fallback_mask):
            log.warning(
                "Some groups in current chunk have only empty text; using all rows for those groups."
            )

    citations, invalid_count = coerce_citations(df, citation_col)
    if invalid_count:
        log.warning(
            "Detected %d invalid citation values in %s for chunk aggregation.",
            invalid_count,
            citation_col,
        )
    sum_cit_weight = np.zeros((n_groups, dim), dtype=np.float32)
    np.add.at(sum_cit_weight, inv, emb * citations[:, None].astype(np.float32))

    total_citations = np.bincount(inv, weights=citations).astype(np.float64)

    meta = (
        df.groupby(key_col, sort=True)[[STKCD_COLUMN, YEAR_COLUMN]].first().reindex(uniq_keys).reset_index()
    )
    meta["n_patents"] = np.bincount(inv).astype(np.int64)
    meta["n_texts_used"] = n_text_rows.astype(np.int64)
    meta["total_citations"] = total_citations
    denom = meta["n_patents"].to_numpy(dtype=np.float64)
    meta["mean_citations"] = np.divide(
        meta["total_citations"].to_numpy(dtype=np.float64),
        denom,
        out=np.full(n_groups, np.nan),
        where=denom > 0,
    )

    log.info("Chunk aggregate built for %d groups in %.2fs", n_groups, time.time() - t0)
    return meta, sum_embedding, n_text_rows, sum_cit_weight


def finalize_chunk_aggregates(
    chunk_meta_parts: List[pd.DataFrame],
    chunk_sum_embeddings: List[np.ndarray],
    chunk_text_counts: List[np.ndarray],
    chunk_cit_weight_sums: List[np.ndarray],
) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
    if not chunk_meta_parts:
        return (
            pd.DataFrame(columns=EMBEDDING_OUTPUT_COLUMNS),
            np.empty((0, 0), dtype=np.float32),
            None,
        )

    meta = pd.concat(chunk_meta_parts, ignore_index=True)
    if meta.empty:
        return (
            pd.DataFrame(columns=EMBEDDING_OUTPUT_COLUMNS),
            np.empty((0, 0), dtype=np.float32),
            None,
        )

    if not chunk_sum_embeddings or not chunk_text_counts:
        raise ValueError("Chunk aggregate components are incomplete")

    dim = chunk_sum_embeddings[0].shape[1] if chunk_sum_embeddings[0].size else 0
    global_meta = (
        meta.groupby(KEY_COLUMN, sort=True)
        .agg(
            {
                STKCD_COLUMN: "first",
                YEAR_COLUMN: "first",
                "n_patents": "sum",
                "n_texts_used": "sum",
                "total_citations": "sum",
            }
        )
        .reset_index()
    )
    global_meta["mean_citations"] = np.divide(
        global_meta["total_citations"].to_numpy(dtype=np.float64),
        global_meta["n_patents"].to_numpy(dtype=np.float64),
        out=np.full(len(global_meta), np.nan),
        where=global_meta["n_patents"].to_numpy() > 0,
    )

    key_to_idx = pd.Series(
        np.arange(len(global_meta), dtype=np.int64),
        index=global_meta[KEY_COLUMN].astype("string"),
    )
    n_groups = len(global_meta)
    global_sum_embedding = np.zeros((n_groups, dim), dtype=np.float64)
    global_text_counts = np.zeros(n_groups, dtype=np.float64)
    global_cit_weight_sum = np.zeros((n_groups, dim), dtype=np.float64)

    for part_meta, part_sum, part_counts, part_cit_sum in zip(
        chunk_meta_parts, chunk_sum_embeddings, chunk_text_counts, chunk_cit_weight_sums
    ):
        if len(part_meta) == 0:
            continue
        chunk_key_idx = (
            key_to_idx.reindex(part_meta[KEY_COLUMN].astype("string")).to_numpy(dtype=np.int64)
        )
        if np.any(np.isnan(chunk_key_idx)):
            raise RuntimeError("Failed to map chunk keys to global key space")
        np.add.at(global_sum_embedding, chunk_key_idx, part_sum)
        np.add.at(global_text_counts, chunk_key_idx, part_counts)
        np.add.at(global_cit_weight_sum, chunk_key_idx, part_cit_sum)

    global_sum_embedding = np.asarray(global_sum_embedding, dtype=np.float64)
    global_text_counts = np.asarray(global_text_counts, dtype=np.float64)
    global_meta["n_texts_used"] = global_text_counts.astype(np.int64)

    final_sum_embedding = divide_rows(global_sum_embedding, global_meta["n_texts_used"].to_numpy())

    cit_denom = global_meta["total_citations"].to_numpy(dtype=np.float64)
    final_cit_weight = divide_rows(global_cit_weight_sum, cit_denom)
    fallback = cit_denom <= ZERO_DIVISION_EPSILON
    if np.any(fallback):
        final_cit_weight[fallback] = final_sum_embedding[fallback]

    final_meta = global_meta[EMBEDDING_OUTPUT_COLUMNS].copy()
    return final_meta, final_sum_embedding.astype(np.float32), final_cit_weight.astype(np.float32)


def save_embeddings_bundle(
    output_dir: Path,
    prefix: str,
    meta: pd.DataFrame,
    embeddings: np.ndarray,
    save_npy: bool = False,
    model_short: str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(__name__)

    emb_shape = embeddings.shape
    if len(emb_shape) != 2:
        embeddings = np.asarray(embeddings, dtype=np.float32).reshape((emb_shape[0], -1))

    emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
    full = pd.concat(
        [meta.reset_index(drop=True), pd.DataFrame(embeddings, columns=emb_cols)],
        axis=1,
    )
    suffix = f"_{model_short}" if model_short else ""
    csv_path = output_dir / f"{prefix}{suffix}_embeddings.csv"
    full.to_csv(csv_path, index=False)
    log.info("Saved embeddings CSV: %s", csv_path)

    if save_npy:
        meta_path = output_dir / f"{prefix}{suffix}_meta.csv"
        emb_path = output_dir / f"{prefix}{suffix}_embeddings.npy"
        meta.to_csv(meta_path, index=False)
        np.save(emb_path, embeddings.astype(np.float32, copy=False))
        log.info("Saved meta: %s", meta_path)
        log.info("Saved embeddings NPY: %s", emb_path)


def write_embedding_outputs(
    output_dir: Path,
    meta: pd.DataFrame,
    embeddings: np.ndarray,
    citation_embeddings: Optional[np.ndarray],
    save_npy: bool = False,
    model_short: str = "",
) -> None:
    save_embeddings_bundle(
        output_dir=output_dir,
        prefix="stkcd_year",
        meta=meta,
        embeddings=embeddings,
        save_npy=save_npy,
        model_short=model_short,
    )
    if citation_embeddings is None:
        return
    save_embeddings_bundle(
        output_dir=output_dir,
        prefix="stkcd_year_citweighted",
        meta=meta,
        embeddings=citation_embeddings,
        save_npy=save_npy,
        model_short=model_short,
    )


def save_patent_level_embeddings(
    output_dir: Path,
    prefix: str,
    meta: pd.DataFrame,
    embeddings: np.ndarray,
    model_short: str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(__name__)
    suffix = f"_{model_short}" if model_short else ""
    meta_path = output_dir / f"{prefix}{suffix}_meta.csv"
    emb_path = output_dir / f"{prefix}{suffix}_embeddings.npy"

    meta.to_csv(meta_path, index=False)
    np.save(emb_path, embeddings.astype(np.float32, copy=False))
    log.info("Saved patent-level metadata: %s", meta_path)
    log.info("Saved patent-level embeddings NPY: %s (%s)", emb_path, f"{embeddings.shape}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Patent SBERT embedding + aggregation")
    p.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Input cleaned patent file (default: data/patents_cleaned.dta)",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Deprecated compatibility alias for --input (expects a cleaned .dta file).",
    )
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODELS_DIR)
    p.add_argument("--model-name", type=str, default="paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--batch-size", type=int, default=None, help="Batch size for embedding (auto-detect if not specified)")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--multi-gpu", action="store_true", help="Use multi-GPU encode path")

    # Memory/performance controls
    p.add_argument(
        "--row-chunk-size",
        type=int,
        default=None,
        help=(
            "Optional: process embeddings + aggregation in row chunks of this size "
            "(reduces peak RAM/VRAM; can be faster if you were memory bound). "
            "Example: 50000. Default: None (embed all rows at once)."
        ),
    )
    p.add_argument(
        "--embed-backend",
        choices=["overflow", "legacy"],
        default="overflow",
        help=(
            "Embedding backend. 'overflow' uses tokenizer overflow windows + direct model.forward() "
            "(single tokenization pass; fastest; supports long texts). "
            "'legacy' keeps the previous fallback splitter + model.encode() behavior."
        ),
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help=(
            "Override SentenceTransformer max_seq_length / tokenizer max_length used for embedding windows "
            "(e.g., 512). If not set, the script will bump common defaults (like 128) up to 512 when "
            "the tokenizer supports it."
        ),
    )

    p.add_argument(
        "--fp16",
        action="store_true",
        help="CUDA only: cast model weights to fp16 (often faster; slight numeric drift).",
    )
    p.add_argument(
        "--tf32",
        action="store_true",
        help="CUDA only: allow TF32 matmul on Ampere+ GPUs (often faster; slight numeric drift).",
    )
    p.add_argument(
        "--tokenizers-parallelism",
        choices=["true", "false"],
        default=None,
        help="Override TOKENIZERS_PARALLELISM env var before loading the model (default: keep script default).",
    )

    p.add_argument(
        "--process-by-chunk",
        action="store_true",
        default=False,
        help="Deprecated: chunk mode is no longer used; this flag is accepted for compatibility.",
    )
    p.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Deprecated: ignored when using single-file mode.",
    )
    p.add_argument(
        "--include-empty-in-agg",
        action="store_true",
        help="Do not exclude empty text rows from aggregation",
    )
    p.add_argument("--save-npy", action="store_true", help="Also save .npy outputs")
    p.add_argument(
        "--save-patent-level",
        action="store_true",
        default=False,
        help="Save patent-level embeddings+meta (default: False)",
    )
    p.add_argument(
        "--no-save-patent-level",
        dest="save_patent_level",
        action="store_false",
        help="Disable patent-level embedding save",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def empty_embedding_outputs(output_dir: Path, save_npy: bool) -> None:
    write_embedding_outputs(
        output_dir=output_dir,
        meta=pd.DataFrame(columns=EMBEDDING_OUTPUT_COLUMNS),
        embeddings=np.empty((0, 0), dtype=np.float32),
        citation_embeddings=None,
        save_npy=save_npy,
    )


def process_all_at_once(args: argparse.Namespace) -> int:
    log = logging.getLogger(__name__)
    input_path = args.input

    if args.data_dir is not None:
        # Backward compatible entry point: older scripts pass --data-dir.
        if args.data_dir.is_dir():
            candidate = args.data_dir / "patents_cleaned.dta"
            if candidate.exists():
                input_path = candidate
            else:
                raise ValueError(
                    f"--data-dir is a directory but does not contain patents_cleaned.dta: {args.data_dir}"
                )
        else:
            input_path = args.data_dir
        log.warning(
            "--data-dir / --process-by-chunk / --max-chunks are deprecated. Using single-file mode."
        )

    # Compatibility: allow legacy flag to enable streaming row-chunk processing.
    if getattr(args, "process_by_chunk", False) and (args.row_chunk_size is None):
        args.row_chunk_size = 50_000
        log.warning(
            "Legacy --process-by-chunk detected. Using --row-chunk-size=%d for streaming processing.",
            args.row_chunk_size,
        )
    if getattr(args, "max_chunks", None) is not None:
        log.warning("--max-chunks is deprecated; it will only be used to LIMIT streaming chunks when --row-chunk-size is set.")

    # Allow the user to override TOKENIZERS_PARALLELISM *before* SentenceTransformers is imported.
    if getattr(args, "tokenizers_parallelism", None) is not None:
        os.environ["TOKENIZERS_PARALLELISM"] = str(args.tokenizers_parallelism)
        log.info("TOKENIZERS_PARALLELISM=%s", os.environ["TOKENIZERS_PARALLELISM"])
    elif bool(getattr(args, "multi_gpu", False)):
        # When using SentenceTransformers' multi-process pool, keep tokenizer threading disabled
        # to avoid CPU oversubscription.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        log.info("TOKENIZERS_PARALLELISM=%s (multi-gpu pool)", os.environ["TOKENIZERS_PARALLELISM"])

    df = load_single_file(input_path)
    if df.empty:
        log.warning("No data loaded after preprocessing; writing empty outputs.")
        empty_embedding_outputs(output_dir=args.output_dir, save_npy=args.save_npy)
        return 0

    model_short = get_model_short_name(args.model_name)

    # Auto-recommend batch size if not specified
    batch_size = args.batch_size
    if batch_size is None:
        gpu_info = get_gpu_info()
        if gpu_info["available"]:
            batch_size = recommend_batch_size(gpu_info["memory_total"], args.model_name)
            log.info("Auto-selected batch size: %d (GPU: %s, %.1fGB)", batch_size, gpu_info["name"], gpu_info["memory_total"])
        else:
            batch_size = 64  # Conservative default for CPU
            log.info("Auto-selected batch size: %d (CPU mode)", batch_size)
    else:
        log.info("Using specified batch size: %d", batch_size)

    embedder = SBertEmbedder(
        model_dir=args.model_dir,
        model_name=args.model_name,
        device=args.device,
        multi_gpu=args.multi_gpu,
        fp16=bool(getattr(args, "fp16", False)),
        tf32=bool(getattr(args, "tf32", False)),
        max_seq_length=getattr(args, "max_seq_length", None),
        embed_backend=getattr(args, "embed_backend", "overflow"),
    )

    try:
        row_chunk_size = getattr(args, "row_chunk_size", None)
        if row_chunk_size is not None:
            row_chunk_size = int(row_chunk_size)
            if row_chunk_size <= 0:
                row_chunk_size = None

        # ---- Streaming row-chunk path (reduces peak memory, can be faster if memory bound) ----
        if row_chunk_size is not None and len(df) > row_chunk_size:
            log.info("Processing in row chunks: chunk_size=%d | total_rows=%d", row_chunk_size, len(df))

            # Optional: patent-level save using an on-disk .npy memmap to avoid holding all embeddings in RAM.
            patent_meta = None
            emb_memmap = None
            if args.save_patent_level:
                patent_cols = [
                    STKCD_COLUMN,
                    YEAR_COLUMN,
                    KEY_COLUMN,
                    "p_date",
                    "text_is_empty",
                    CITATION_COLUMN,
                ]
                patent_meta = df[[c for c in patent_cols if c in df.columns]].copy()

                args.output_dir.mkdir(parents=True, exist_ok=True)
                suffix = f"_{model_short}" if model_short else ""
                meta_path = args.output_dir / f"patent_level{suffix}_meta.csv"
                emb_path = args.output_dir / f"patent_level{suffix}_embeddings.npy"

                patent_meta.to_csv(meta_path, index=False)
                dim = embedder.model.get_sentence_embedding_dimension()
                emb_memmap = np.lib.format.open_memmap(
                    emb_path, mode="w+", dtype=np.float32, shape=(len(df), dim)
                )
                log.info("Streaming patent-level embeddings to: %s", emb_path)
                log.info("Saved patent-level metadata to: %s", meta_path)

            chunk_meta_parts: List[pd.DataFrame] = []
            chunk_sum_embeddings: List[np.ndarray] = []
            chunk_text_counts: List[np.ndarray] = []
            chunk_cit_weight_sums: List[np.ndarray] = []

            max_chunks = getattr(args, "max_chunks", None)
            n_rows = len(df)
            n_chunks = (n_rows + row_chunk_size - 1) // row_chunk_size
            if max_chunks is not None:
                n_chunks = min(n_chunks, int(max_chunks))

            for chunk_id in range(n_chunks):
                start = chunk_id * row_chunk_size
                end = min((chunk_id + 1) * row_chunk_size, n_rows)
                df_chunk = df.iloc[start:end]

                log.info("Embedding chunk %d/%d: rows [%d:%d)", chunk_id + 1, n_chunks, start, end)
                texts = df_chunk["text"].tolist()
                emb_chunk = embedder.embed(texts, batch_size=batch_size, show_progress=False)

                if emb_memmap is not None:
                    emb_memmap[start:end, :] = emb_chunk

                meta, simple_sum, text_counts, cit_weight_sum = aggregate_chunk(
                    df_chunk,
                    emb_chunk,
                    key_col=KEY_COLUMN,
                    exclude_empty_text=not args.include_empty_in_agg,
                    citation_col=CITATION_COLUMN,
                )
                chunk_meta_parts.append(meta)
                chunk_sum_embeddings.append(simple_sum)
                chunk_text_counts.append(text_counts)
                chunk_cit_weight_sums.append(cit_weight_sum)

            # Ensure memmap is flushed to disk.
            if emb_memmap is not None:
                emb_memmap.flush()
                del emb_memmap

            final_meta, stkcd_year_embeddings, cit_weighted = finalize_chunk_aggregates(
                chunk_meta_parts,
                chunk_sum_embeddings,
                chunk_text_counts,
                chunk_cit_weight_sums,
            )

            write_embedding_outputs(
                output_dir=args.output_dir,
                meta=final_meta,
                embeddings=stkcd_year_embeddings,
                citation_embeddings=cit_weighted,
                save_npy=args.save_npy,
                model_short=model_short,
            )

            log.info("Done (streaming). Outputs in: %s", args.output_dir)
            return 0

        # ---- Original full-file path ----
        texts = df["text"].tolist()
        patent_embeddings = embedder.embed(texts, batch_size=batch_size, show_progress=True)

        if args.save_patent_level:
            patent_cols = [
                STKCD_COLUMN,
                YEAR_COLUMN,
                KEY_COLUMN,
                "p_date",
                "text_is_empty",
                CITATION_COLUMN,
            ]
            patent_meta = df[[c for c in patent_cols if c in df.columns]].copy()
            save_patent_level_embeddings(
                output_dir=args.output_dir,
                prefix="patent_level",
                meta=patent_meta,
                embeddings=patent_embeddings,
                model_short=model_short,
            )

        meta, simple_sum, text_counts, cit_weight_sum = aggregate_chunk(
            df,
            patent_embeddings,
            key_col=KEY_COLUMN,
            exclude_empty_text=not args.include_empty_in_agg,
            citation_col=CITATION_COLUMN,
        )

        final_meta, stkcd_year_embeddings, cit_weighted = finalize_chunk_aggregates(
            [meta],
            [simple_sum],
            [text_counts],
            [cit_weight_sum],
        )

        write_embedding_outputs(
            output_dir=args.output_dir,
            meta=final_meta,
            embeddings=stkcd_year_embeddings,
            citation_embeddings=cit_weighted,
            save_npy=args.save_npy,
            model_short=model_short,
        )

        log.info("Done. Outputs in: %s", args.output_dir)
        return 0
    finally:
        embedder.close()


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    log = logging.getLogger(__name__)

    log.info(
        "PyTorch: %s | CUDA available: %s",
        torch.__version__,
        torch.cuda.is_available(),
    )

    return process_all_at_once(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
