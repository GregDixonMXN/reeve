#!/usr/bin/env python3
"""
Export MiniLM-L6-v2 weights to quantized int8 binary format for the Mojo kernel.

This is a ONE-TIME operation. Run once to generate weight files,
then the Mojo kernel loads them directly without Python.

Usage:
    pip install torch transformers numpy
    python3 scripts/export_minilm_weights.py --output mojo/kernels/weights/

Output files:
    token_embeddings.qbin    — [30522 x 384] int8 + float32 scale
    layer_N_attn_qkv.qbin   — [384 x 1152] int8 (Q, K, V concatenated)
    layer_N_attn_out.qbin    — [384 x 384] int8
    layer_N_ffn_up.qbin      — [384 x 1536] int8
    layer_N_ffn_down.qbin    — [1536 x 384] int8
    layer_N_ln1_gamma.bin    — [384] float32
    layer_N_ln1_beta.bin     — [384] float32
    layer_N_ln2_gamma.bin    — [384] float32
    layer_N_ln2_beta.bin     — [384] float32

Total size: ~23MB (vs ~90MB float32)
"""

import argparse
import os
import struct

import numpy as np


def quantize_to_int8(weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric int8 quantization: scale = max(abs(w)) / 127."""
    scale = float(np.max(np.abs(weights))) / 127.0
    if scale == 0:
        scale = 1e-10
    quantized = np.clip(np.round(weights / scale), -127, 127).astype(np.int8)
    return quantized, scale


def save_quantized(path: str, weights: np.ndarray):
    """Save as: [4 bytes float32 scale] + [N bytes int8 weights]."""
    quantized, scale = quantize_to_int8(weights.flatten())
    with open(path, "wb") as f:
        f.write(struct.pack("<f", scale))
        f.write(quantized.tobytes())
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  {path}: {weights.shape} → {size_mb:.2f} MB (scale={scale:.6f})")


def save_float32(path: str, weights: np.ndarray):
    """Save raw float32 array."""
    with open(path, "wb") as f:
        f.write(weights.astype(np.float32).tobytes())
    print(f"  {path}: {weights.shape} float32")


def export_from_huggingface(output_dir: str):
    """Download and export all-MiniLM-L6-v2 weights."""
    try:
        import torch
        from transformers import AutoModel
    except ImportError:
        print("ERROR: pip install torch transformers")
        return

    print("Loading sentence-transformers/all-MiniLM-L6-v2...")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    state = model.state_dict()

    os.makedirs(output_dir, exist_ok=True)

    # Token embeddings
    word_emb = state["embeddings.word_embeddings.weight"].numpy()
    save_quantized(os.path.join(output_dir, "token_embeddings.qbin"), word_emb)

    # Position embeddings
    pos_emb = state["embeddings.position_embeddings.weight"].numpy()
    save_quantized(os.path.join(output_dir, "position_embeddings.qbin"), pos_emb)

    # Transformer layers
    for layer_idx in range(6):
        prefix = f"encoder.layer.{layer_idx}"
        lp = f"layer_{layer_idx}"

        # Self-attention Q, K, V weights (concatenated)
        q_w = state[f"{prefix}.attention.self.query.weight"].numpy()
        k_w = state[f"{prefix}.attention.self.key.weight"].numpy()
        v_w = state[f"{prefix}.attention.self.value.weight"].numpy()
        qkv = np.concatenate([q_w, k_w, v_w], axis=0)
        save_quantized(os.path.join(output_dir, f"{lp}_attn_qkv.qbin"), qkv)

        # Attention output projection
        out_w = state[f"{prefix}.attention.output.dense.weight"].numpy()
        save_quantized(os.path.join(output_dir, f"{lp}_attn_out.qbin"), out_w)

        # FFN up projection (intermediate)
        ffn_up = state[f"{prefix}.intermediate.dense.weight"].numpy()
        save_quantized(os.path.join(output_dir, f"{lp}_ffn_up.qbin"), ffn_up)

        # FFN down projection (output)
        ffn_down = state[f"{prefix}.output.dense.weight"].numpy()
        save_quantized(os.path.join(output_dir, f"{lp}_ffn_down.qbin"), ffn_down)

        # LayerNorm parameters (kept as float32 — tiny)
        for ln_name, ln_key in [
            ("ln1", f"{prefix}.attention.output.LayerNorm"),
            ("ln2", f"{prefix}.output.LayerNorm"),
        ]:
            gamma = state[f"{ln_key}.weight"].numpy()
            beta = state[f"{ln_key}.bias"].numpy()
            save_float32(os.path.join(output_dir, f"{lp}_{ln_name}_gamma.bin"), gamma)
            save_float32(os.path.join(output_dir, f"{lp}_{ln_name}_beta.bin"), beta)

    # Pooler (optional — we use mean pooling instead)
    print(f"\nExport complete: {output_dir}/")
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir)
    )
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"Files: {len(os.listdir(output_dir))}")


def export_random_weights(output_dir: str, dim: int = 384):
    """Generate random weights for testing without downloading the model."""
    print(f"Generating random test weights (dim={dim})...")
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(42)

    # Token embeddings
    save_quantized(
        os.path.join(output_dir, "token_embeddings.qbin"),
        np.random.randn(30522, dim).astype(np.float32) * 0.02,
    )

    for layer_idx in range(6):
        lp = f"layer_{layer_idx}"
        save_quantized(
            os.path.join(output_dir, f"{lp}_attn_qkv.qbin"),
            np.random.randn(dim * 3, dim).astype(np.float32) * 0.02,
        )
        save_quantized(
            os.path.join(output_dir, f"{lp}_attn_out.qbin"),
            np.random.randn(dim, dim).astype(np.float32) * 0.02,
        )
        save_quantized(
            os.path.join(output_dir, f"{lp}_ffn_up.qbin"),
            np.random.randn(dim * 4, dim).astype(np.float32) * 0.02,
        )
        save_quantized(
            os.path.join(output_dir, f"{lp}_ffn_down.qbin"),
            np.random.randn(dim, dim * 4).astype(np.float32) * 0.02,
        )

        for ln in ["ln1", "ln2"]:
            save_float32(
                os.path.join(output_dir, f"{lp}_{ln}_gamma.bin"),
                np.ones(dim, dtype=np.float32),
            )
            save_float32(
                os.path.join(output_dir, f"{lp}_{ln}_beta.bin"),
                np.zeros(dim, dtype=np.float32),
            )

    print(f"\nRandom weights exported to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MiniLM weights for Mojo")
    parser.add_argument(
        "--output",
        default="mojo/kernels/weights/",
        help="Output directory for weight files",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Generate random weights (no model download)",
    )
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension")
    args = parser.parse_args()

    if args.random:
        export_random_weights(args.output, args.dim)
    else:
        export_from_huggingface(args.output)
