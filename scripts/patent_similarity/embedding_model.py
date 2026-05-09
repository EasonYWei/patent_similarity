"""SBERT model loading and embedding helpers."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

TOKENIZED_SENTENCE_SPLIT = re.compile(r"(?<=[。；;!?！？。!?.])\s+|\n+")


def get_gpu_info() -> dict:
    """Get GPU information for batch-size recommendations."""
    info = {"available": False, "count": 0, "name": "N/A", "memory_total": 0, "memory_free": 0}
    if torch.cuda.is_available():
        info["available"] = True
        info["count"] = torch.cuda.device_count()
        info["name"] = torch.cuda.get_device_name(0)
        info["memory_total"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        info["memory_free"] = (torch.cuda.mem_get_info(0)[0] / (1024**3)) if hasattr(torch.cuda, "mem_get_info") else 0
    return info


def print_gpu_status() -> None:
    """Print current GPU memory usage."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    print(f"  GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def recommend_batch_size(gpu_memory_gb: float, model_name: str) -> int:
    """Recommend a conservative embedding batch size based on GPU memory."""
    if "minilm" in model_name.lower():
        if gpu_memory_gb >= 24:
            return 1024
        if gpu_memory_gb >= 16:
            return 512
        if gpu_memory_gb >= 8:
            return 256
        return 128
    if gpu_memory_gb >= 24:
        return 512
    if gpu_memory_gb >= 16:
        return 256
    if gpu_memory_gb >= 8:
        return 128
    return 64


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


