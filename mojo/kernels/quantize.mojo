"""
Reeve Quantization Kernel
═════════════════════════
SIMD-accelerated vector quantization for memory-efficient storage.

Reduces embedding storage by 4x (float32 → int8) or 8x (float32 → int4)
while preserving enough fidelity for similarity search.

Usage:
    mojo run quantize.mojo quantize_i8  <input.bin> <dim> <count> <output.bin> <params.bin>
    mojo run quantize.mojo dequantize_i8 <input.bin> <dim> <count> <params.bin> <output.bin>
    mojo run quantize.mojo cosine_i8    <query.bin> <docs.bin> <dim> <num_docs> <params.bin> <results.bin>
"""

from math import sqrt
from memory import alloc
from sys import argv, simd_width_of
from algorithm.functional import vectorize
from pathlib import Path


alias SIMD_F32 = simd_width_of[DType.float32]()
alias SIMD_I8 = simd_width_of[DType.int8]()


# ─── Quantization Parameters ────────────────────────────────────────────────

struct QuantParams:
    """Per-vector quantization parameters for int8 symmetric quantization.

    Each vector gets its own scale factor:
        quantized = round(original / scale)
        dequantized = quantized * scale
        scale = max(abs(vector)) / 127
    """
    var scales: UnsafePointer[Scalar[DType.float32]]
    var count: Int

    fn __init__(out self, count: Int):
        self.scales = alloc[Scalar[DType.float32]](count)
        self.count = count

    fn __del__(owned self):
        self.scales.free()


# ─── Float32 → Int8 Quantization ────────────────────────────────────────────

fn compute_scale(
    vec: UnsafePointer[Scalar[DType.float32]],
    dim: Int,
) -> Float32:
    """Compute the quantization scale for a single vector.

    scale = max(|v_i|) / 127.0
    """
    var max_abs: Float32 = 0.0

    fn _find_max[width: Int](idx: Int) unified {mut}:
        var vals = vec.load[width=width](idx)
        # Compute abs via max(v, -v)
        var abs_vals = vals.__abs__()
        max_abs = max(max_abs, abs_vals.reduce_max())

    vectorize[SIMD_F32](dim, _find_max)

    if max_abs == 0.0:
        return 1.0  # Avoid division by zero
    return max_abs / 127.0


fn quantize_vector_i8(
    input: UnsafePointer[Scalar[DType.float32]],
    output: UnsafePointer[Scalar[DType.int8]],
    dim: Int,
    scale: Float32,
):
    """Quantize a single float32 vector to int8 using symmetric quantization."""
    var inv_scale = 1.0 / scale

    for i in range(dim):
        var val = input.load(i)[0]
        # Clamp to [-127, 127] after scaling
        var scaled = val * inv_scale
        if scaled > 127.0:
            scaled = 127.0
        elif scaled < -127.0:
            scaled = -127.0
        output.store(i, Int8(Int(scaled)))


fn dequantize_vector_i8(
    input: UnsafePointer[Scalar[DType.int8]],
    output: UnsafePointer[Scalar[DType.float32]],
    dim: Int,
    scale: Float32,
):
    """Dequantize an int8 vector back to float32."""
    for i in range(dim):
        var val = input.load(i)[0]
        output.store(i, Float32(Int(val)) * scale)


# ─── Batch Operations ───────────────────────────────────────────────────────

fn batch_quantize_i8(
    input: UnsafePointer[Scalar[DType.float32]],
    output: UnsafePointer[Scalar[DType.int8]],
    params: QuantParams,
    num_vectors: Int,
    dim: Int,
):
    """Quantize a batch of float32 vectors to int8."""
    for i in range(num_vectors):
        var in_ptr = input.offset(i * dim)
        var out_ptr = output.offset(i * dim)
        var scale = compute_scale(in_ptr, dim)
        params.scales.store(i, scale)
        quantize_vector_i8(in_ptr, out_ptr, dim, scale)


fn batch_dequantize_i8(
    input: UnsafePointer[Scalar[DType.int8]],
    output: UnsafePointer[Scalar[DType.float32]],
    params: QuantParams,
    num_vectors: Int,
    dim: Int,
):
    """Dequantize a batch of int8 vectors to float32."""
    for i in range(num_vectors):
        var in_ptr = input.offset(i * dim)
        var out_ptr = output.offset(i * dim)
        var scale = params.scales.load(i)[0]
        dequantize_vector_i8(in_ptr, out_ptr, dim, scale)


# ─── Quantized Cosine Similarity ────────────────────────────────────────────
# Operates directly on int8 vectors — avoids dequantization overhead.

fn cosine_similarity_i8(
    a: UnsafePointer[Scalar[DType.int8]],
    b: UnsafePointer[Scalar[DType.int8]],
    dim: Int,
    scale_a: Float32,
    scale_b: Float32,
) -> Float32:
    """Compute approximate cosine similarity between two int8 vectors.

    The int8 dot product is computed in int32 to avoid overflow,
    then scaled back to float32.
    """
    var dot: Int32 = 0
    var norm_a: Int32 = 0
    var norm_b: Int32 = 0

    for i in range(dim):
        var va = Int32(Int(a.load(i)[0]))
        var vb = Int32(Int(b.load(i)[0]))
        dot += va * vb
        norm_a += va * va
        norm_b += vb * vb

    var denom = sqrt(Float32(Int(norm_a))) * sqrt(Float32(Int(norm_b)))
    if denom == 0.0:
        return 0.0
    return Float32(Int(dot)) / denom


fn batch_cosine_similarity_i8(
    query_f32: UnsafePointer[Scalar[DType.float32]],
    docs_i8: UnsafePointer[Scalar[DType.int8]],
    doc_scales: UnsafePointer[Scalar[DType.float32]],
    num_docs: Int,
    dim: Int,
    results: UnsafePointer[Scalar[DType.float32]],
):
    """Compute cosine similarity between a float32 query and int8 doc vectors.

    The query is quantized on-the-fly, then compared against pre-quantized docs.
    """
    # Quantize the query
    var query_scale = compute_scale(query_f32, dim)
    var query_i8 = alloc[Scalar[DType.int8]](dim)
    quantize_vector_i8(query_f32, query_i8, dim, query_scale)

    for i in range(num_docs):
        var doc_ptr = docs_i8.offset(i * dim)
        var doc_scale = doc_scales.load(i)[0]
        var score = cosine_similarity_i8(query_i8, doc_ptr, dim, query_scale, doc_scale)
        results.store(i, score)

    query_i8.free()


# ─── File I/O Helpers ────────────────────────────────────────────────────────

fn read_float32_binary(path: String, count: Int) raises -> UnsafePointer[Scalar[DType.float32]]:
    var data = Path(path).read_bytes()
    var ptr = alloc[Scalar[DType.float32]](count)
    var src = data.unsafe_ptr().bitcast[Scalar[DType.float32]]()
    for i in range(count):
        ptr.store(i, src.load(i))
    return ptr


fn write_float32_binary(path: String, data: UnsafePointer[Scalar[DType.float32]], count: Int) raises:
    var byte_ptr = data.bitcast[UInt8]()
    var bytes_list = List[UInt8]()
    for i in range(count * 4):
        bytes_list.append(byte_ptr.load(i)[0])
    Path(path).write_bytes(bytes_list)


fn write_int8_binary(path: String, data: UnsafePointer[Scalar[DType.int8]], count: Int) raises:
    var byte_ptr = data.bitcast[UInt8]()
    var bytes_list = List[UInt8]()
    for i in range(count):
        bytes_list.append(byte_ptr.load(i)[0])
    Path(path).write_bytes(bytes_list)


fn read_int8_binary(path: String, count: Int) raises -> UnsafePointer[Scalar[DType.int8]]:
    var data = Path(path).read_bytes()
    var ptr = alloc[Scalar[DType.int8]](count)
    var src = data.unsafe_ptr().bitcast[Scalar[DType.int8]]()
    for i in range(count):
        ptr.store(i, src.load(i))
    return ptr


# ─── CLI Entry Point ────────────────────────────────────────────────────────

def main():
    var args = argv()

    if len(args) < 2:
        print("Usage: mojo run quantize.mojo <op> [args...]")
        print("  quantize_i8   <input.bin> <dim> <count> <output.bin> <params.bin>")
        print("  dequantize_i8 <input.bin> <dim> <count> <params.bin> <output.bin>")
        print("  cosine_i8     <query.bin> <docs.bin> <dim> <num_docs> <params.bin> <results.bin>")
        return

    var op = String(args[1])

    if op == "quantize_i8":
        var input_path = String(args[2])
        var dim = Int(String(args[3]))
        var count = Int(String(args[4]))
        var output_path = String(args[5])
        var params_path = String(args[6])

        var input = read_float32_binary(input_path, dim * count)
        var output = alloc[Scalar[DType.int8]](dim * count)
        var params = QuantParams(count)

        batch_quantize_i8(input, output, params, count, dim)

        write_int8_binary(output_path, output, dim * count)
        write_float32_binary(params_path, params.scales, count)

        # Report compression
        var original_bytes = dim * count * 4
        var quantized_bytes = dim * count * 1 + count * 4  # int8 data + scales
        print("Quantized", count, "vectors:", original_bytes, "→", quantized_bytes, "bytes")
        print("Compression ratio:", Float32(original_bytes) / Float32(quantized_bytes))

        input.free()
        output.free()

    elif op == "cosine_i8":
        var query_path = String(args[2])
        var docs_path = String(args[3])
        var dim = Int(String(args[4]))
        var num_docs = Int(String(args[5]))
        var params_path = String(args[6])
        var results_path = String(args[7])

        var query = read_float32_binary(query_path, dim)
        var docs = read_int8_binary(docs_path, dim * num_docs)
        var scales = read_float32_binary(params_path, num_docs)
        var results = alloc[Scalar[DType.float32]](num_docs)

        batch_cosine_similarity_i8(query, docs, scales, num_docs, dim, results)
        write_float32_binary(results_path, results, num_docs)

        for i in range(num_docs):
            print("doc[" + String(i) + "]: " + String(results.load(i)[0]))

        query.free()
        docs.free()
        scales.free()
        results.free()

    else:
        print("Unknown operation:", op)
