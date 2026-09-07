package adapters

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const ollamaHealthErrorLimit = int64(8 * 1024)

// CheckOllamaModel verifies both that an Ollama server is reachable and that
// the exact configured model tag is installed. Reeve performs this preflight
// before advertising local mode as ready.
func CheckOllamaModel(ctx context.Context, baseURL, model string, timeout time.Duration) error {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	model = strings.TrimSpace(model)
	if baseURL == "" {
		return fmt.Errorf("ollama base URL is empty")
	}
	if model == "" {
		return fmt.Errorf("ollama model is empty")
	}
	if timeout <= 0 {
		timeout = 5 * time.Second
	}

	client := &http.Client{Timeout: timeout}
	versionReq, err := http.NewRequestWithContext(ctx, http.MethodGet, baseURL+"/api/version", nil)
	if err != nil {
		return fmt.Errorf("build Ollama version request: %w", err)
	}
	versionResp, err := client.Do(versionReq)
	if err != nil {
		return fmt.Errorf("Ollama is unreachable at %s: %w", baseURL, err)
	}
	if err := requireOllamaSuccess(versionResp, "version"); err != nil {
		return err
	}

	body, err := json.Marshal(map[string]string{"model": model})
	if err != nil {
		return fmt.Errorf("encode Ollama model request: %w", err)
	}
	showReq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/api/show", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("build Ollama model request: %w", err)
	}
	showReq.Header.Set("Content-Type", "application/json")
	showResp, err := client.Do(showReq)
	if err != nil {
		return fmt.Errorf("check Ollama model %q: %w", model, err)
	}
	if err := requireOllamaSuccess(showResp, "model "+model); err != nil {
		return err
	}
	return nil
}

func requireOllamaSuccess(resp *http.Response, operation string) error {
	defer resp.Body.Close()
	if resp.StatusCode >= http.StatusOK && resp.StatusCode < http.StatusMultipleChoices {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, ollamaHealthErrorLimit))
		return nil
	}
	detail, _ := io.ReadAll(io.LimitReader(resp.Body, ollamaHealthErrorLimit))
	message := strings.TrimSpace(string(detail))
	if message == "" {
		message = http.StatusText(resp.StatusCode)
	}
	return fmt.Errorf("Ollama %s check failed (HTTP %d): %s", operation, resp.StatusCode, message)
}
