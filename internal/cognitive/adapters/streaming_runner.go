package adapters

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"reeve/pkg/models"
)

// TokenCallback is called for each streamed token.
type TokenCallback func(token string)

// StreamingRunner wraps RemoteRunner with streaming support.
// It implements cognitive.LLMRunner for the non-streaming interface,
// and adds StreamComplete for token-by-token output.
type StreamingRunner struct {
	baseURL     string
	model       string
	protocol    Protocol
	client      *http.Client
	onToken     TokenCallback
	contextSize int
}

type StreamingRunnerConfig struct {
	BaseURL     string
	Model       string
	Protocol    Protocol
	TimeoutS    int
	OnToken     TokenCallback // Called for each streamed token
	ContextSize int
}

type safeStreamEmitter struct {
	callback TokenCallback
	buffer   strings.Builder
	mode     uint8 // 0 undecided, 1 plain text, 2 structured/protocol output
}

func (e *safeStreamEmitter) Write(chunk string) {
	if chunk == "" || e.callback == nil {
		return
	}
	if e.mode == 1 {
		e.callback(chunk)
		return
	}
	e.buffer.WriteString(chunk)
	if e.mode == 2 {
		return
	}
	trimmed := strings.TrimSpace(e.buffer.String())
	if trimmed == "" {
		return
	}
	switch trimmed[0] {
	case '{', '[', '`', '<':
		e.mode = 2
	default:
		e.mode = 1
		e.callback(e.buffer.String())
		e.buffer.Reset()
	}
}

func (e *safeStreamEmitter) Finish(allowOutput bool) {
	if e.callback == nil || !allowOutput || e.mode == 1 {
		return
	}
	raw := strings.TrimSpace(e.buffer.String())
	if raw == "" {
		return
	}
	if strings.HasPrefix(raw, "```") {
		lines := strings.Split(raw, "\n")
		if len(lines) >= 3 {
			raw = strings.Join(lines[1:len(lines)-1], "\n")
		}
	}
	var payload struct {
		Content string `json:"content"`
	}
	if json.Unmarshal([]byte(raw), &payload) == nil && payload.Content != "" {
		e.callback(payload.Content)
		return
	}
	if e.mode == 0 {
		e.callback(e.buffer.String())
	}
}

func NewStreamingRunner(cfg StreamingRunnerConfig) *StreamingRunner {
	timeout := cfg.TimeoutS
	if timeout <= 0 {
		timeout = 300 // Longer timeout for streaming
	}
	return &StreamingRunner{
		baseURL:     cfg.BaseURL,
		model:       cfg.Model,
		protocol:    cfg.Protocol,
		client:      &http.Client{Timeout: time.Duration(timeout) * time.Second},
		onToken:     cfg.OnToken,
		contextSize: cfg.ContextSize,
	}
}

// Complete satisfies cognitive.LLMRunner — collects all streamed tokens into one string.
func (s *StreamingRunner) Complete(ctx context.Context, prompt string, maxTokens int) (string, error) {
	switch s.protocol {
	case ProtocolOllama:
		return s.streamOllama(ctx, prompt, maxTokens)
	default:
		return "", fmt.Errorf("streaming not implemented for protocol %d", s.protocol)
	}
}

// CompleteWithTools sends a streaming chat request to Ollama with native tool definitions.
// systemContext is the system prompt plus repo and memory context. Native tool
// names, descriptions, and schemas arrive separately in tools.
// messages is the structured conversation history for proper multi-turn Ollama chat.
func (s *StreamingRunner) CompleteWithTools(ctx context.Context, systemContext string, messages []models.Message, maxTokens int, tools []models.ToolDefinition) (string, error) {
	if s.protocol != ProtocolOllama || len(tools) == 0 {
		// Fallback: build a flat prompt and use /api/generate
		prompt := systemContext + "\n"
		for _, msg := range messages {
			prompt += fmt.Sprintf("[%s]: %s\n", msg.Role, msg.Content)
		}
		return s.Complete(ctx, prompt, maxTokens)
	}
	return s.streamOllamaWithTools(ctx, systemContext, messages, maxTokens, tools)
}

// Unload signals the server to release the model.
func (s *StreamingRunner) Unload() error {
	body, _ := json.Marshal(map[string]interface{}{
		"model": s.model, "keep_alive": 0,
	})
	_, err := s.client.Post(s.baseURL+"/api/generate", "application/json", bytes.NewReader(body))
	return err
}

// ── Ollama /api/chat streaming with tools ───────────────────────────────────

func (s *StreamingRunner) streamOllamaWithTools(ctx context.Context, systemContext string, messages []models.Message, maxTokens int, tools []models.ToolDefinition) (string, error) {
	ollamaTools := convertToOllamaTools(tools)

	// Build proper multi-turn message array with system message first.
	chatMessages := []ollamaChatMsg{
		{Role: "system", Content: systemContext},
	}
	for _, msg := range messages {
		chatMessages = append(chatMessages, ollamaChatMsg{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	options := map[string]interface{}{
		"num_predict": maxTokens,
		"temperature": 0.1, // Low temp for deterministic tool calling
		"top_p":       0.9,
	}
	if s.contextSize > 0 {
		options["num_ctx"] = s.contextSize
	}

	body, err := json.Marshal(ollamaChatReq{
		Model:    s.model,
		Messages: chatMessages,
		Stream:   true,
		Tools:    ollamaTools,
		Options:  options,
	})
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", s.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("ollama unreachable at %s: %w", s.baseURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("ollama HTTP %d", resp.StatusCode)
	}

	var full strings.Builder
	emitter := safeStreamEmitter{callback: s.onToken}
	var lastToolCalls []struct {
		Function struct {
			Name      string                 `json:"name"`
			Arguments map[string]interface{} `json:"arguments"`
		} `json:"function"`
	}

	scanner := bufio.NewScanner(resp.Body)
	scanBuf := make([]byte, 0, 64*1024)
	scanner.Buffer(scanBuf, 1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var chunk ollamaChatResp
		if err := json.Unmarshal(line, &chunk); err != nil {
			continue
		}

		if chunk.Message.Content != "" {
			full.WriteString(chunk.Message.Content)
			emitter.Write(chunk.Message.Content)
		}

		if len(chunk.Message.ToolCalls) > 0 {
			lastToolCalls = chunk.Message.ToolCalls
		}

		if chunk.Done {
			break
		}
	}

	if err := scanner.Err(); err != nil {
		return full.String(), fmt.Errorf("stream read error: %w", err)
	}

	// If tool call was returned, format as structured JSON for the engine
	if len(lastToolCalls) > 0 {
		emitter.Finish(false)
		tc := lastToolCalls[0]
		structured := map[string]interface{}{
			"reasoning": full.String(),
			"tool_call": map[string]interface{}{
				"name": tc.Function.Name,
				"args": tc.Function.Arguments,
			},
			"content": "",
		}
		out, _ := json.Marshal(structured)
		return string(out), nil
	}

	emitter.Finish(true)
	return normalizeNativeContent(full.String()), nil
}

// ── Ollama /api/generate streaming (legacy, no tools) ───────────────────────

type ollamaStreamChunk struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

func (s *StreamingRunner) streamOllama(ctx context.Context, prompt string, maxTokens int) (string, error) {
	options := map[string]interface{}{
		"num_predict": maxTokens,
		"temperature": 0.1, // Low temp for deterministic JSON output
		"top_p":       0.9,
	}
	if s.contextSize > 0 {
		options["num_ctx"] = s.contextSize
	}

	body, err := json.Marshal(map[string]interface{}{
		"model":   s.model,
		"prompt":  prompt,
		"stream":  true,
		"format":  "json", // Enforce valid JSON output at the API level
		"options": options,
	})
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", s.baseURL+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("ollama unreachable at %s: %w", s.baseURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("ollama HTTP %d", resp.StatusCode)
	}

	var full strings.Builder
	emitter := safeStreamEmitter{callback: s.onToken}
	scanner := bufio.NewScanner(resp.Body)
	scanBuf := make([]byte, 0, 64*1024)
	scanner.Buffer(scanBuf, 1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		var chunk ollamaStreamChunk
		if err := json.Unmarshal(line, &chunk); err != nil {
			continue
		}

		if chunk.Response != "" {
			full.WriteString(chunk.Response)
			emitter.Write(chunk.Response)
		}

		if chunk.Done {
			break
		}
	}

	if err := scanner.Err(); err != nil {
		return full.String(), fmt.Errorf("stream read error: %w", err)
	}

	emitter.Finish(true)
	return full.String(), nil
}
