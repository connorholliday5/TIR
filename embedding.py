#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Module: src.embedding
# Implements:
#   - REQ-013 (Deterministic dense text embeddings, no external model files)
#
# Drop-in replacement for the previous FastText sentence vectors.
#
# The embedder needs no training corpus, no downloaded .bin model and no deep
# learning framework: every subword is mapped to a fixed pseudo-random vector
# that is derived from a SHA-256 digest of the subword itself, and a text is
# embedded as the mean of its subword vectors.  Because the digest is stable
# across processes, machines and Python versions, the same text always yields
# exactly the same vector - which is what the FastText model was relied upon
# for, minus the 1 GB binary.
#
# Dependencies: numpy only.

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


# Bumped whenever a change would alter the vectors produced for a given
# configuration.  Stored in the metadata file so that inference can refuse a
# model that was trained with an incompatible embedder.
EMBEDDING_VERSION = 1

DEFAULT_DIM = 192
DEFAULT_MIN_N = 3
DEFAULT_MAX_N = 5

# Upper bound on the number of cached subword vectors.  Character n-grams over
# a large corpus can run into the millions of distinct subwords, so the cache
# is capped to keep memory bounded (100k * 192 * 4 B is roughly 77 MB).  Once
# full, further subwords are recomputed on demand instead of stored.
#
# Overridable through the "embedding" block of config.json.  Unlike dim, min_n,
# max_n and seed, this is purely a memory/speed trade-off: it never changes the
# vectors produced, so it is not part of the embedding version.
DEFAULT_MAX_CACHE = 100_000


# Returns a deterministic 32-bit seed for a subword.
def stable_hash(subword: str, seed: int = 0) -> int:
    """Derive a reproducible ``np.random.RandomState`` seed from a subword.

    Uses SHA-256 rather than :func:`hash` because the built-in hash is salted
    per process (``PYTHONHASHSEED``) and would break reproducibility.

    Args:
        subword: The subword to hash.
        seed: Optional namespace.  ``0`` hashes the bare subword, which keeps
            the vectors identical to earlier revisions of this pipeline; any
            other value produces a completely independent embedding space.
    """
    payload = subword if seed == 0 else f"{seed}:{subword}"
    return int(hashlib.sha256(payload.encode("utf8")).hexdigest()[:8], 16)


# Turns text into a fixed-length vector without any trained model file.
class SubwordHashEmbedder:
    """Deterministic dense sentence embeddings built from hashed subwords.

    Example:
        >>> embedder = SubwordHashEmbedder(dim=192)
        >>> vec = embedder.embed("cracked weld on pipe hanger")
        >>> vec.shape
        (192,)

    Args:
        dim: Length of the produced vectors.
        min_n: Shortest character n-gram taken from the text.
        max_n: Longest character n-gram taken from the text.
        seed: Namespace for the subword hashes; change it to obtain a fresh,
            uncorrelated embedding space (models must then be retrained).
        lowercase: Lower-case the text before extracting subwords.
        l2_normalize: Scale each sentence vector to unit length.  Helpful when
            the downstream model is sensitive to text length; off by default
            so that existing artifacts keep their exact behaviour.
        max_cache: Maximum number of subword vectors held in memory.
    """

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        min_n: int = DEFAULT_MIN_N,
        max_n: int = DEFAULT_MAX_N,
        seed: int = 0,
        lowercase: bool = True,
        l2_normalize: bool = False,
        max_cache: int = DEFAULT_MAX_CACHE,
    ) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        if min_n < 1:
            raise ValueError(f"min_n must be >= 1, got {min_n}")
        if max_n < min_n:
            raise ValueError(f"max_n ({max_n}) must be >= min_n ({min_n})")

        self.dim = int(dim)
        self.min_n = int(min_n)
        self.max_n = int(max_n)
        self.seed = int(seed)
        self.lowercase = bool(lowercase)
        self.l2_normalize = bool(l2_normalize)
        self.max_cache = int(max_cache)

        self._cache: Dict[str, np.ndarray] = {}

    # -- subword handling ------------------------------------------------

    # Returns the fixed pseudo-random vector for a single subword.
    def subword_vector(self, subword: str) -> np.ndarray:
        """Return the deterministic vector assigned to ``subword``."""
        cached = self._cache.get(subword)
        if cached is not None:
            return cached

        rng = np.random.RandomState(stable_hash(subword, self.seed))
        vec = rng.normal(0, 1, self.dim).astype(np.float32)

        if len(self._cache) < self.max_cache:
            self._cache[subword] = vec
        return vec

    # Extracts character n-grams plus whole-word and word-affix units.
    def extract_subwords(self, text: str) -> List[str]:
        """Split ``text`` into the subword units that make up its embedding.

        Leading and trailing whitespace is stripped first, so padding cannot
        change a text's vector; interior spaces are kept and n-grams may span
        them, exactly as they did before.

        The result is sorted, not merely de-duplicated: averaging float32
        vectors is not associative, so an unordered set would make the final
        embedding depend on the interpreter's string hash randomisation
        (``PYTHONHASHSEED``) and drift by ~1e-8 between processes.
        """
        t = text.strip()
        if self.lowercase:
            t = t.lower()
        subs = set()

        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(t) - n + 1):
                subs.add(t[i:i + n])

        # Whole words and their affixes; short words are kept even when they
        # are below min_n, so "ab" still contributes a unit of its own.
        for w in t.split():
            subs.add(w)
            if len(w) > 4:
                subs.add(w[:4])
                subs.add(w[-4:])

        return sorted(subs)

    # -- embedding -------------------------------------------------------

    # Embeds a single string.
    def embed(self, text: str) -> np.ndarray:
        """Return the ``(dim,)`` float32 embedding of ``text``.

        Texts with no extractable subword (empty or whitespace-only, or
        shorter than ``min_n``) embed to the zero vector.
        """
        subs = self.extract_subwords(str(text))
        if not subs:
            return np.zeros(self.dim, dtype=np.float32)

        mat = np.empty((len(subs), self.dim), dtype=np.float32)
        for i, sub in enumerate(subs):
            mat[i] = self.subword_vector(sub)

        vec = mat.mean(axis=0)
        if self.l2_normalize:
            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                vec = vec / norm
        return vec.astype(np.float32)

    # Embeds a sequence of strings.
    def embed_many(self, texts: Iterable[str]) -> np.ndarray:
        """Return an ``(n_texts, dim)`` float32 matrix of embeddings."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed(t) for t in texts])

    # Drops every cached subword vector.
    def clear_cache(self) -> None:
        """Release the cached subword vectors."""
        self._cache.clear()

    # -- persistence -----------------------------------------------------

    # Serialises the configuration needed to rebuild an identical embedder.
    def to_meta(self) -> dict:
        """Return the JSON-serialisable configuration of this embedder."""
        return {
            "version": EMBEDDING_VERSION,
            "kind": "subword_hash",
            "dim": self.dim,
            "min_n": self.min_n,
            "max_n": self.max_n,
            "seed": self.seed,
            "lowercase": self.lowercase,
            "l2_normalize": self.l2_normalize,
            # Recorded so inference reuses the size training ran with; it does
            # not affect the vectors, only how many are held in memory.
            "max_cache": self.max_cache,
        }

    # Rebuilds an embedder from a metadata dictionary.
    @classmethod
    def from_meta(cls, meta: dict) -> "SubwordHashEmbedder":
        """Rebuild an embedder from :meth:`to_meta` output.

        Raises:
            ValueError: If the metadata was written by a newer, incompatible
                version of this module.
        """
        version = int(meta.get("version", EMBEDDING_VERSION))
        if version > EMBEDDING_VERSION:
            raise ValueError(
                f"dense_meta was written by embedding version {version}, but this "
                f"install only understands up to {EMBEDDING_VERSION}. Retrain the "
                f"models or upgrade the code."
            )

        return cls(
            dim=meta.get("dim", DEFAULT_DIM),
            min_n=meta.get("min_n", DEFAULT_MIN_N),
            max_n=meta.get("max_n", DEFAULT_MAX_N),
            seed=meta.get("seed", 0),
            lowercase=meta.get("lowercase", True),
            l2_normalize=meta.get("l2_normalize", False),
            max_cache=meta.get("max_cache", DEFAULT_MAX_CACHE),
        )

    # Writes the metadata file that inference reads back.
    def save_meta(self, path: Path) -> None:
        """Write :meth:`to_meta` to ``path`` as JSON."""
        Path(path).write_text(json.dumps(self.to_meta(), indent=4))

    # Loads an embedder from a metadata file on disk.
    @classmethod
    def load(cls, path: Path) -> "SubwordHashEmbedder":
        """Rebuild the embedder described by the JSON file at ``path``."""
        return cls.from_meta(json.loads(Path(path).read_text()))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SubwordHashEmbedder(dim={self.dim}, min_n={self.min_n}, "
            f"max_n={self.max_n}, seed={self.seed}, "
            f"l2_normalize={self.l2_normalize})"
        )


# Embeds a sequence of strings with a throwaway embedder.
def embed_texts(texts: Sequence[str], **kwargs) -> np.ndarray:
    """Convenience wrapper: embed ``texts`` with a one-off embedder."""
    return SubwordHashEmbedder(**kwargs).embed_many(texts)