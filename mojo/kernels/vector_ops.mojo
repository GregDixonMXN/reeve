"""
Herald Vector Operations Kernel
═══════════════════════════════
SIMD-accelerated vector math for the RAG memory pipeline.
Replaces Python/numpy for all vector similarity operations.

Protocol:
    Go writes query + doc vectors as raw float32 binary files.
    Mojo reads them, computes SIMD similarity, writes results.

Usage:
    mojo run vector_ops.mojo cosine query.bin docs.bin 384 10 results.bin
    mojo run vector_ops.mojo dot query.bin docs.bin 384 10 results.bin
    mojo run vector_ops.mojo normalize vectors.bin 384 10 out.bin

    Args: <op> <query_path> <docs_path> <dim> <num_docs> <output_path>
"""

from math import sqrt
from memory import alloc
from sys import argv, simd_width_of
from algorithm.functional import vectorize
from pathlib import Path


# ─── SIMD width for float32 on this hardware ────────────────────────────────
alias FLOAT_SIMD_WIDTH = simd_width_of[DType.float32]()


# ─── Core SIMD Operations ───────────────────────────────────────────────────

fn cosine_similarity_simd(
    a: UnsafePointer[Scalar[DType.float32]],
    b: UnsafePointer[Scalar[DType.float32]],
    dim: Int,
) -> Float32:
    """Compute cosine similarity between two float32 vectors using SIMD.

    Processes FLOAT_SIMD_WIDTH elements per cycle instead of one at a time.
    This is the core hot-path primitive for semantic memory search.
    """
    var dot: Float32 = 0.0
    var norm_a: Float32 = 0.0
    var norm_b: Float32 = 0.0

    fn _compute[width: Int](idx: Int) unified {mut}:
        var va = a.load[width=width](idx)
        var vb = b.load[width=width](idx)
        dot += (va * vb).reduce_add()
        norm_a += (va * va).reduce_add()
        norm_b += (vb * vb).reduce_add()

    vectorize[FLOAT_SIMD_WIDTH](dim, _compute)

    var denom = sqrt(norm_a) * sqrt(norm_b)
    if denom == 0.0:
        return 0.0
    return dot / denom


fn dot_product_simd(
    a: UnsafePointer[Scalar[DType.float32]],
    b: UnsafePointer[Scalar[DType.float32]],
    dim: Int,
) -> Float32:
    """Compute dot product of two float32 vectors using SIMD."""
    var result: Float32 = 0.0

    fn _compute[width: Int](idx: Int) unified {mut}:
        var va = a.load[width=width](idx)
        var vb = b.load[width=width](idx)
        result += (va * vb).reduce_add()

    vectorize[FLOAT_SIMD_WIDTH](dim, _compute)
    return result


fn l2_norm_simd(
    v: UnsafePointer[Scalar[DType.float32]],
    dim: Int,
) -> Float32:
    """Compute L2 (Euclidean) norm of a float32 vector using SIMD."""
    var sum_sq: Float32 = 0.0

    fn _compute[width: Int](idx: Int) unified {mut}:
        var val = v.load[width=width](idx)
        sum_sq += (val * val).reduce_add()

    vectorize[FLOAT_SIMD_WIDTH](dim, _compute)
    return sqrt(sum_sq)


fn normalize_inplace(
    v: UnsafePointer[Scalar[DType.float32]],
    dim: Int,
):
    """L2-normalize a vector in place using SIMD."""
    var norm = l2_norm_simd(v, dim)
    if norm == 0.0:
        return

    var inv_norm = 1.0 / norm

    fn _normalize[width: Int](idx: Int) unified {mut}:
        var val = v.load[width=width](idx)
        v.store[width=width](idx, val * inv_norm)

    vectorize[FLOAT_SIMD_WIDTH](dim, _normalize)


fn euclidean_distance_simd(
    a: UnsafePointer[Scalar[DType.float32]],
    b: UnsafePointer[Scalar[DType.float32]],
    dim: Int,
) -> Float32:
    """Compute Euclidean distance between two vectors using SIMD."""
    var sum_sq: Float32 = 0.0

    fn _compute[width: Int](idx: Int) unified {mut}:
        var va = a.load[width=width](idx)
        var vb = b.load[width=width](idx)
        var diff = va - vb
        sum_sq += (diff * diff).reduce_add()

    vectorize[FLOAT_SIMD_WIDTH](dim, _compute)
    return sqrt(sum_sq)


# ─── Batch Operations ───────────────────────────────────────────────────────

fn batch_cosine_similarity(
    query: UnsafePointer[Scalar[DType.float32]],
    docs: UnsafePointer[Scalar[DType.float32]],
    num_docs: Int,
    dim: Int,
    results: UnsafePointer[Scalar[DType.float32]],
):
    """Compute cosine similarity between a query and a batch of doc vectors.

    Memory layout for docs: [doc0_d0, doc0_d1, ..., doc1_d0, doc1_d1, ...]
    Results: [score0, score1, ..., scoreN]
    """
    for i in range(num_docs):
        var doc_ptr = docs.offset(i * dim)
        var score = cosine_similarity_simd(query, doc_ptr, dim)
        results.store(i, score)


fn batch_dot_product(
    query: UnsafePointer[Scalar[DType.float32]],
    docs: UnsafePointer[Scalar[DType.float32]],
    num_docs: Int,
    dim: Int,
    results: UnsafePointer[Scalar[DType.float32]],
):
    """Compute dot product between a query and a batch of doc vectors."""
    for i in range(num_docs):
        var doc_ptr = docs.offset(i * dim)
        var score = dot_product_simd(query, doc_ptr, dim)
        results.store(i, score)


fn batch_normalize(
    vectors: UnsafePointer[Scalar[DType.float32]],
    num_vectors: Int,
    dim: Int,
):
    """L2-normalize a batch of vectors in place."""
    for i in range(num_vectors):
        normalize_inplace(vectors.offset(i * dim), dim)


# ─── Top-K Selection ────────────────────────────────────────────────────────

fn top_k_indices(
    scores: UnsafePointer[Scalar[DType.float32]],
    num_scores: Int,
    k: Int,
    out_indices: UnsafePointer[Scalar[DType.int32]],
    out_scores: UnsafePointer[Scalar[DType.float32]],
):
    """Find the top-k highest scores and their indices.

    Uses a simple selection approach (optimal for small k, which is
    typical in RAG retrieval where k=5-20).
    """
    # Initialize with -inf
    for i in range(k):
        out_scores.store(i, Float32(-1e30))
        out_indices.store(i, Int32(-1))

    for i in range(num_scores):
        var score = scores.load(i)[0]

        # Find the minimum in our top-k buffer
        var min_idx = 0
        var min_val = out_scores.load(0)[0]
        for j in range(1, k):
            var val = out_scores.load(j)[0]
            if val < min_val:
                min_val = val
                min_idx = j

        # Replace if current score is higher than the minimum
        if score > min_val:
            out_scores.store(min_idx, score)
            out_indices.store(min_idx, Int32(i))


# ─── Binary File I/O ────────────────────────────────────────────────────────
# Go writes raw float32 arrays as binary files.
# Mojo reads/writes them directly — zero parsing overhead.

fn read_float32_binary(
    path: String,
    count: Int,
) raises -> UnsafePointer[Scalar[DType.float32]]:
    """Read `count` float32 values from a raw binary file."""
    var data = Path(path).read_bytes()
    var expected_bytes = count * 4
    if len(data) < expected_bytes:
        raise Error("File too small: expected " + String(expected_bytes) + " bytes, got " + String(len(data)))

    var ptr = alloc[Scalar[DType.float32]](count)
    var src = data.unsafe_ptr().bitcast[Scalar[DType.float32]]()
    for i in range(count):
        ptr.store(i, src.load(i))
    return ptr


fn write_float32_binary(
    path: String,
    data: UnsafePointer[Scalar[DType.float32]],
    count: Int,
) raises:
    """Write `count` float32 values to a raw binary file."""
    var byte_ptr = data.bitcast[UInt8]()
    var byte_count = count * 4
    var bytes_list = List[UInt8]()
    for i in range(byte_count):
        bytes_list.append(byte_ptr.load(i)[0])
    Path(path).write_bytes(bytes_list)


# ─── CLI Entry Point ────────────────────────────────────────────────────────

def main():
    var args = argv()

    if len(args) < 2:
        print("Usage: mojo run vector_ops.mojo <op> [args...]")
        print("  cosine    <query.bin> <docs.bin> <dim> <num_docs> <results.bin>")
        print("  dot       <query.bin> <docs.bin> <dim> <num_docs> <results.bin>")
        print("  normalize <vectors.bin> <dim> <num_vectors> <output.bin>")
        print("  topk      <scores.bin> <num_scores> <k> <indices.bin> <topscores.bin>")
        print("  info      — print SIMD capabilities")
        return

    var op = String(args[1])

    if op == "info":
        print("SIMD width (float32):", FLOAT_SIMD_WIDTH)
        print("Elements per cycle:  ", FLOAT_SIMD_WIDTH)
        return

    if op == "cosine" or op == "dot":
        if len(args) < 7:
            print("Usage: " + op + " <query.bin> <docs.bin> <dim> <num_docs> <results.bin>")
            return

        var query_path = String(args[2])
        var docs_path = String(args[3])
        var dim = Int(String(args[4]))
        var num_docs = Int(String(args[5]))
        var results_path = String(args[6])

        var query = read_float32_binary(query_path, dim)
        var docs = read_float32_binary(docs_path, dim * num_docs)
        var results = alloc[Scalar[DType.float32]](num_docs)

        if op == "cosine":
            batch_cosine_similarity(query, docs, num_docs, dim, results)
        else:
            batch_dot_product(query, docs, num_docs, dim, results)

        write_float32_binary(results_path, results, num_docs)

        # Print results to stdout as well for debugging
        for i in range(num_docs):
            print("doc[" + String(i) + "]: " + String(results.load(i)[0]))

        query.free()
        docs.free()
        results.free()

    elif op == "normalize":
        if len(args) < 6:
            print("Usage: normalize <vectors.bin> <dim> <num_vectors> <output.bin>")
            return

        var vectors_path = String(args[2])
        var dim = Int(String(args[3]))
        var num_vectors = Int(String(args[4]))
        var output_path = String(args[5])

        var vectors = read_float32_binary(vectors_path, dim * num_vectors)
        batch_normalize(vectors, num_vectors, dim)
        write_float32_binary(output_path, vectors, dim * num_vectors)

        print("Normalized", num_vectors, "vectors of dim", dim)
        vectors.free()

    elif op == "topk":
        if len(args) < 7:
            print("Usage: topk <scores.bin> <num_scores> <k> <indices.bin> <topscores.bin>")
            return

        var scores_path = String(args[2])
        var num_scores = Int(String(args[3]))
        var k = Int(String(args[4]))
        var indices_path = String(args[5])
        var topscores_path = String(args[6])

        var scores = read_float32_binary(scores_path, num_scores)
        var out_indices = alloc[Scalar[DType.int32]](k)
        var out_scores = alloc[Scalar[DType.float32]](k)

        top_k_indices(scores, num_scores, k, out_indices, out_scores)

        write_float32_binary(topscores_path, out_scores, k)
        # Write indices as int32 binary
        var idx_byte_ptr = out_indices.bitcast[UInt8]()
        var idx_bytes = List[UInt8]()
        for i in range(k * 4):
            idx_bytes.append(idx_byte_ptr.load(i)[0])
        Path(indices_path).write_bytes(idx_bytes)

        for i in range(k):
            print("top[" + String(i) + "]: idx=" + String(out_indices.load(i)[0]) + " score=" + String(out_scores.load(i)[0]))

        scores.free()
        out_indices.free()
        out_scores.free()

    else:
        print("Unknown operation:", op)
        print("Available: cosine, dot, normalize, topk, info")
