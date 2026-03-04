package bridge

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// MojoBridge executes Mojo kernels via binary file I/O.
// Automatically dispatches to GPU kernels when available.
type MojoBridge struct {
	binary     string
	kernelsDir string
	tempDir    string
	hasGPU     bool // cached GPU detection result
}

func NewMojoBridge(mojoPath, kernelsDir string) (*MojoBridge, error) {
	if mojoPath == "" {
		mojoPath = "mojo"
	}
	if kernelsDir == "" {
		kernelsDir = "mojo/kernels"
	}

	tempDir, err := os.MkdirTemp("", "axiom-mojo-*")
	if err != nil {
		return nil, fmt.Errorf("create temp dir: %w", err)
	}

	b := &MojoBridge{
		binary:     mojoPath,
		kernelsDir: kernelsDir,
		tempDir:    tempDir,
	}

	// Probe GPU availability at startup
	b.hasGPU = b.probeGPU()

	return b, nil
}

func (b *MojoBridge) Close() error {
	return os.RemoveAll(b.tempDir)
}

// HasGPU returns whether GPU acceleration is available.
func (b *MojoBridge) HasGPU() bool {
	return b.hasGPU
}

// probeGPU runs the GPU kernel info command to check for hardware.
func (b *MojoBridge) probeGPU() bool {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	kernel := filepath.Join(b.kernelsDir, "gpu_kernels.mojo")
	if _, err := os.Stat(kernel); err != nil {
		return false
	}

	cmd := exec.CommandContext(ctx, b.binary, "run", kernel, "info")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return false
	}

	return strings.Contains(string(output), "DETECTED")
}

// ─── Neural Embedding (Upgrade 3) ──────────────────────────────────────────

// NeuralEmbed generates a neural embedding using Mojo.
// Uses GPU kernels if available, falls back to CPU SIMD.
func (b *MojoBridge) NeuralEmbed(ctx context.Context, text string, dim int) ([]float32, error) {
	outputFile := filepath.Join(b.tempDir, fmt.Sprintf("embed_%d.bin", time.Now().UnixNano()))
	defer os.Remove(outputFile)

	// gpu_kernels.mojo uses @parameter if has_accelerator() internally,
	// so it dispatches to GPU when available and CPU SIMD otherwise.
	kernel := filepath.Join(b.kernelsDir, "gpu_kernels.mojo")
	if _, err := os.Stat(kernel); err != nil {
		kernel = filepath.Join(b.kernelsDir, "neural_embed.mojo")
	}

	args := []string{"run", kernel, "embed", text, fmt.Sprintf("%d", dim), outputFile}
	return b.runAndReadFloats(ctx, args, outputFile, dim)
}

// NeuralEmbedFile generates a neural embedding from a text file.
func (b *MojoBridge) NeuralEmbedFile(ctx context.Context, inputFile string, dim int) ([]float32, error) {
	outputFile := filepath.Join(b.tempDir, fmt.Sprintf("embed_%d.bin", time.Now().UnixNano()))
	defer os.Remove(outputFile)

	kernel := filepath.Join(b.kernelsDir, "gpu_kernels.mojo")
	if _, err := os.Stat(kernel); err != nil {
		kernel = filepath.Join(b.kernelsDir, "neural_embed.mojo")
	}

	args := []string{"run", kernel, "embed-file", inputFile, fmt.Sprintf("%d", dim), outputFile}
	return b.runAndReadFloats(ctx, args, outputFile, dim)
}

// Benchmark runs the embedding benchmark and returns the output.
func (b *MojoBridge) Benchmark(ctx context.Context, dim int) (string, error) {
	kernel := filepath.Join(b.kernelsDir, "gpu_kernels.mojo")
	if _, err := os.Stat(kernel); err != nil {
		kernel = filepath.Join(b.kernelsDir, "neural_embed.mojo")
	}

	cmd := exec.CommandContext(ctx, b.binary, "run", kernel, "bench", fmt.Sprintf("%d", dim))
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("benchmark: %w\n%s", err, string(output))
	}
	return strings.TrimSpace(string(output)), nil
}

// SIMDInfo returns the Mojo kernel's hardware capabilities.
func (b *MojoBridge) SIMDInfo(ctx context.Context) (string, error) {
	kernel := filepath.Join(b.kernelsDir, "gpu_kernels.mojo")
	if _, err := os.Stat(kernel); err != nil {
		kernel = filepath.Join(b.kernelsDir, "neural_embed.mojo")
	}

	cmd := exec.CommandContext(ctx, b.binary, "run", kernel, "info")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("mojo info: %w\n%s", err, string(output))
	}
	return strings.TrimSpace(string(output)), nil
}

// ─── Vector Operations ──────────────────────────────────────────────────────

// CosineSimilarity computes similarity between query and each document vector.
func (b *MojoBridge) CosineSimilarity(ctx context.Context, query []float32, docs [][]float32) ([]float32, error) {
	if len(docs) == 0 {
		return nil, nil
	}
	dim := len(query)

	queryFile := filepath.Join(b.tempDir, "query.bin")
	docsFile := filepath.Join(b.tempDir, "docs.bin")
	resultFile := filepath.Join(b.tempDir, "result.bin")
	defer func() {
		os.Remove(queryFile)
		os.Remove(docsFile)
		os.Remove(resultFile)
	}()

	if err := writeFloat32Binary(queryFile, query); err != nil {
		return nil, err
	}

	// Flatten docs
	flat := make([]float32, 0, len(docs)*dim)
	for _, d := range docs {
		flat = append(flat, d...)
	}
	if err := writeFloat32Binary(docsFile, flat); err != nil {
		return nil, err
	}

	kernel := filepath.Join(b.kernelsDir, "vector_ops.mojo")
	args := []string{
		"run", kernel, "cosine",
		queryFile, docsFile,
		fmt.Sprintf("%d", dim), fmt.Sprintf("%d", len(docs)),
		resultFile,
	}

	return b.runAndReadFloats(ctx, args, resultFile, len(docs))
}

// TopK returns the indices and scores of the top-k highest values.
func (b *MojoBridge) TopK(ctx context.Context, scores []float32, k int) ([]int32, []float32, error) {
	if k > len(scores) {
		k = len(scores)
	}

	scoresFile := filepath.Join(b.tempDir, "scores.bin")
	indicesFile := filepath.Join(b.tempDir, "indices.bin")
	topscoresFile := filepath.Join(b.tempDir, "topscores.bin")
	defer func() {
		os.Remove(scoresFile)
		os.Remove(indicesFile)
		os.Remove(topscoresFile)
	}()

	if err := writeFloat32Binary(scoresFile, scores); err != nil {
		return nil, nil, err
	}

	kernel := filepath.Join(b.kernelsDir, "vector_ops.mojo")
	args := []string{
		"run", kernel, "topk",
		scoresFile,
		fmt.Sprintf("%d", len(scores)), fmt.Sprintf("%d", k),
		indicesFile, topscoresFile,
	}

	indices, err := b.runAndReadInt32s(ctx, args, indicesFile, k)
	if err != nil {
		return nil, nil, err
	}

	topscores, err := readFloat32Binary(topscoresFile, k)
	if err != nil {
		return nil, nil, err
	}

	return indices, topscores, nil
}

// ─── Internal Helpers ───────────────────────────────────────────────────────

func (b *MojoBridge) runAndReadFloats(ctx context.Context, args []string, outputFile string, count int) ([]float32, error) {
	if err := b.runKernel(ctx, args); err != nil {
		return nil, err
	}
	return readFloat32Binary(outputFile, count)
}

func (b *MojoBridge) runAndReadInt32s(ctx context.Context, args []string, outputFile string, count int) ([]int32, error) {
	if err := b.runKernel(ctx, args); err != nil {
		return nil, err
	}
	return readInt32Binary(outputFile, count)
}

func (b *MojoBridge) runKernel(ctx context.Context, args []string) error {
	ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, b.binary, args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("mojo kernel failed: %w\n%s", err, string(output))
	}
	return nil
}

// ─── Binary I/O ─────────────────────────────────────────────────────────────

func writeFloat32Binary(path string, data []float32) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	buf := make([]byte, 4*len(data))
	for i, v := range data {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	_, err = f.Write(buf)
	return err
}

func readFloat32Binary(path string, count int) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	if len(data) < count*4 {
		return nil, fmt.Errorf("short read: got %d bytes, need %d", len(data), count*4)
	}

	result := make([]float32, count)
	for i := range result {
		result[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return result, nil
}

func readInt32Binary(path string, count int) ([]int32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	if len(data) < count*4 {
		return nil, fmt.Errorf("short read: got %d bytes, need %d", len(data), count*4)
	}

	result := make([]int32, count)
	for i := range result {
		result[i] = int32(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return result, nil
}
