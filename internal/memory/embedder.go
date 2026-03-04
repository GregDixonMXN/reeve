package memory

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Embedder generates vector embeddings from text.
// Phase 3 uses Ollama's /api/embed endpoint.
// Phase 4+ can swap to the Mojo embedding kernel for zero-network operation.
type Embedder struct {
	baseURL string
	model   string
	dim     int
	client  *http.Client
}

type EmbedderConfig struct {
	BaseURL string // e.g. "http://localhost:11434"
	Model   string // embedding model name, e.g. "nomic-embed-text" or "all-minilm"
	Dim     int    // expected embedding dimension (384 for MiniLM, 768 for nomic)
}

func NewEmbedder(cfg EmbedderConfig) *Embedder {
	if cfg.BaseURL == "" {
		cfg.BaseURL = "http://localhost:11434"
	}
	if cfg.Model == "" {
		cfg.Model = "nomic-embed-text"
	}
	if cfg.Dim == 0 {
		cfg.Dim = 768
	}
	return &Embedder{
		baseURL: cfg.BaseURL,
		model:   cfg.Model,
		dim:     cfg.Dim,
		client:  &http.Client{Timeout: 30 * time.Second},
	}
}

// Embed generates a vector embedding for a single text string.
func (e *Embedder) Embed(ctx context.Context, text string) ([]float32, error) {
	results, err := e.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(results) == 0 {
		return nil, fmt.Errorf("embedder: empty response")
	}
	return results[0], nil
}

// EmbedBatch generates embeddings for multiple texts in one call.
func (e *Embedder) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	// Ollama /api/embed accepts "input" as string or []string
	reqBody, err := json.Marshal(map[string]interface{}{
		"model": e.model,
		"input": texts,
	})
	if err != nil {
		return nil, fmt.Errorf("embedder marshal: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", e.baseURL+"/api/embed", bytes.NewReader(reqBody))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embedder: Ollama unreachable at %s — is it running? %w", e.baseURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embedder: HTTP %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Embeddings [][]float64 `json:"embeddings"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("embedder decode: %w", err)
	}

	// Convert float64 (JSON default) to float32
	out := make([][]float32, len(result.Embeddings))
	for i, emb := range result.Embeddings {
		f32 := make([]float32, len(emb))
		for j, v := range emb {
			f32[j] = float32(v)
		}
		out[i] = f32
	}

	return out, nil
}

// Dim returns the expected embedding dimension.
func (e *Embedder) Dim() int {
	return e.dim
}

// Model returns the embedding model name.
func (e *Embedder) Model() string {
	return e.model
}
