"""
Axiom Embedding Kernel (Phase 4 Preview)
═════════════════════════════════════════
Fast local text → vector embedding without network calls.

Current implementation: Character-level hash embedding.
This is a lightweight fallback that produces deterministic embeddings
based on character n-gram hashing. It's not as good as a neural model
(nomic-embed-text) but it's:
  - Zero latency (no network)
  - Zero dependencies (no Python, no model weights)
  - Deterministic (same input = same output, always)

Phase 4 roadmap: Load quantized MiniLM weights directly in Mojo
for neural-quality embeddings at SIMD speed.

Usage:
    mojo run embedding.mojo embed "your text here" 768 output.bin
    mojo run embedding.mojo embed-file input.txt 768 output.bin
    mojo run embedding.mojo batch-embed filelist.txt 768 output.bin
"""

from math import sqrt, sin, cos
from memory import alloc
from sys import argv
from algorithm.functional import vectorize
from sys import simd_width_of
from pathlib import Path


alias SIMD_WIDTH = simd_width_of[DType.float32]()


# ─── Hash Embedding ─────────────────────────────────────────────────────────
# Produces a fixed-dimensional vector from arbitrary text using a combination
# of character n-gram hashing and positional encoding.
#
# The key insight: for similarity search, we don't need PERFECT embeddings.
# We need embeddings where SIMILAR texts produce SIMILAR vectors. Hash-based
# approaches achieve this because texts sharing n-grams will activate the
# same hash buckets, producing correlated vectors.

fn hash_u64(value: UInt64) -> UInt64:
    """Fast non-cryptographic hash (splitmix64 finalizer)."""
    var x = value
    x ^= x >> 30
    x *= 0xBF58476D1CE4E5B9
    x ^= x >> 27
    x *= 0x94D049BB133111EB
    x ^= x >> 31
    return x


fn text_to_embedding(
    text: String,
    dim: Int,
    output: UnsafePointer[Scalar[DType.float32]],
):
    """Convert text to a float32 embedding vector.

    Algorithm:
    1. Extract character trigrams from the text
    2. Hash each trigram to a bucket index (mod dim)
    3. Accumulate weighted values at each bucket
    4. Apply positional encoding for word order sensitivity
    5. L2-normalize the final vector
    """
    # Zero-initialize
    for i in range(dim):
        output.store(i, Float32(0.0))

    var text_len = len(text)
    if text_len == 0:
        return

    # ── Pass 1: Character unigram features ───────────────────────────────
    for i in range(text_len):
        var ch = UInt64(ord(text[i]))
        var h = hash_u64(ch * 2654435761 + UInt64(i))
        var bucket = Int(h % UInt64(dim))
        var sign: Float32 = 1.0
        if h & 1 == 0:
            sign = -1.0
        var weight = 1.0 / sqrt(Float32(text_len))
        var current = output.load(bucket)[0]
        output.store(bucket, current + sign * weight)

    # ── Pass 2: Character bigram features ────────────────────────────────
    if text_len >= 2:
        for i in range(text_len - 1):
            var c0 = UInt64(ord(text[i]))
            var c1 = UInt64(ord(text[i + 1]))
            var h = hash_u64(c0 * 31 + c1 + UInt64(i) * 7)
            var bucket = Int(h % UInt64(dim))
            var sign: Float32 = 1.0
            if (h >> 1) & 1 == 0:
                sign = -1.0
            var weight = 1.5 / sqrt(Float32(text_len))
            var current = output.load(bucket)[0]
            output.store(bucket, current + sign * weight)

    # ── Pass 3: Character trigram features ───────────────────────────────
    if text_len >= 3:
        for i in range(text_len - 2):
            var c0 = UInt64(ord(text[i]))
            var c1 = UInt64(ord(text[i + 1]))
            var c2 = UInt64(ord(text[i + 2]))
            var h = hash_u64(c0 * 961 + c1 * 31 + c2 + UInt64(i) * 13)
            var bucket = Int(h % UInt64(dim))
            var sign: Float32 = 1.0
            if (h >> 2) & 1 == 0:
                sign = -1.0
            var weight = 2.0 / sqrt(Float32(text_len))
            var current = output.load(bucket)[0]
            output.store(bucket, current + sign * weight)

    # ── Pass 4: Positional encoding ──────────────────────────────────────
    # Adds sinusoidal position signal so word order matters
    var pos_weight = 0.3 / sqrt(Float32(text_len))
    for i in range(min(text_len, 128)):  # Cap at first 128 chars
        var ch = UInt64(ord(text[i]))
        var pos = Float32(i)
        for d in range(0, min(dim, 64), 2):
            var freq = 1.0 / (10000.0 ** (Float32(d) / 64.0))
            var idx_sin = Int(hash_u64(ch + UInt64(d)) % UInt64(dim))
            var idx_cos = Int(hash_u64(ch + UInt64(d + 1)) % UInt64(dim))
            var current_sin = output.load(idx_sin)[0]
            var current_cos = output.load(idx_cos)[0]
            output.store(idx_sin, current_sin + sin(pos * freq) * pos_weight)
            output.store(idx_cos, current_cos + cos(pos * freq) * pos_weight)

    # ── L2 Normalize ─────────────────────────────────────────────────────
    var norm_sq: Float32 = 0.0

    fn _sum_sq[width: Int](idx: Int) unified {mut}:
        var val = output.load[width=width](idx)
        norm_sq += (val * val).reduce_add()

    vectorize[SIMD_WIDTH](dim, _sum_sq)

    var norm = sqrt(norm_sq)
    if norm > 0.0:
        var inv_norm = 1.0 / norm

        fn _normalize[width: Int](idx: Int) unified {mut}:
            var val = output.load[width=width](idx)
            output.store[width=width](idx, val * inv_norm)

        vectorize[SIMD_WIDTH](dim, _normalize)


# ─── File I/O ────────────────────────────────────────────────────────────────

fn write_float32_binary(
    path: String,
    data: UnsafePointer[Scalar[DType.float32]],
    count: Int,
) raises:
    var byte_ptr = data.bitcast[UInt8]()
    var bytes_list = List[UInt8]()
    for i in range(count * 4):
        bytes_list.append(byte_ptr.load(i)[0])
    Path(path).write_bytes(bytes_list)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    var args = argv()

    if len(args) < 2:
        print("Usage:")
        print("  mojo run embedding.mojo embed <text> <dim> <output.bin>")
        print("  mojo run embedding.mojo embed-file <input.txt> <dim> <output.bin>")
        print("  mojo run embedding.mojo info")
        return

    var op = String(args[1])

    if op == "info":
        print("Axiom Hash Embedder v1.0")
        print("SIMD width (float32):", SIMD_WIDTH)
        print("Method: character n-gram hashing + positional encoding")
        print("Use for: fast local fallback when Ollama is unavailable")
        return

    if op == "embed":
        if len(args) < 5:
            print("Usage: embed <text> <dim> <output.bin>")
            return

        var text = String(args[2])
        var dim = Int(String(args[3]))
        var output_path = String(args[4])

        var embedding = alloc[Scalar[DType.float32]](dim)
        text_to_embedding(text, dim, embedding)
        write_float32_binary(output_path, embedding, dim)

        # Print first 8 values for debugging
        print("Embedding (first 8 of", dim, "dims):")
        for i in range(min(8, dim)):
            print("  [" + String(i) + "]:", embedding.load(i)[0])

        embedding.free()

    elif op == "embed-file":
        if len(args) < 5:
            print("Usage: embed-file <input.txt> <dim> <output.bin>")
            return

        var input_path = String(args[2])
        var dim = Int(String(args[3]))
        var output_path = String(args[4])

        var text = Path(input_path).read_text()
        var embedding = alloc[Scalar[DType.float32]](dim)
        text_to_embedding(text, dim, embedding)
        write_float32_binary(output_path, embedding, dim)

        print("Embedded", len(text), "chars →", dim, "dims →", output_path)
        embedding.free()

    else:
        print("Unknown operation:", op)
