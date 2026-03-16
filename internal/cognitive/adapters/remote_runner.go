package adapters

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"axiom/pkg/models"
)

// Protocol selects which API format to use.
type Protocol int

const (
	ProtocolOllama Protocol = iota
	ProtocolOpenAI
)

type RemoteRunnerConfig struct {
	BaseURL  string
	Model    string
	Protocol Protocol
	TimeoutS int
}

// RemoteRunner talks to a local LLM server over HTTP.
// Supports Ollama (/api/chat with tools) and OpenAI-compatible (/v1/completions).
type RemoteRunner struct {
	baseURL  string
	model    string
	protocol Protocol
	client   *http.Client
}

func NewRemoteRunner(cfg RemoteRunnerConfig) *RemoteRunner {
	timeout := cfg.TimeoutS
	if timeout <= 0 {
		timeout = 120
	}
	return &RemoteRunner{
		baseURL:  cfg.BaseURL,
		model:    cfg.Model,
		protocol: cfg.Protocol,
		client:   &http.Client{Timeout: time.Duration(timeout) * time.Second},
	}
}

func (r *RemoteRunner) Complete(ctx context.Context, prompt string, maxTokens int) (string, error) {
	switch r.protocol {
	case ProtocolOllama:
		return r.completeOllama(ctx, prompt, maxTokens)
	case ProtocolOpenAI:
		return r.completeOpenAI(ctx, prompt, maxTokens)
	default:
		return "", fmt.Errorf("unknown protocol: %d", r.protocol)
	}
}

// CompleteWithTools sends a chat request to Ollama with native tool definitions.
// systemContext is the system prompt + repo tree + tools schema block.
// messages is the structured conversation history passed as proper multi-turn turns.
func (r *RemoteRunner) CompleteWithTools(ctx context.Context, systemContext string, messages []models.Message, maxTokens int, tools []models.ToolDefinition) (string, error) {
	if r.protocol != ProtocolOllama || len(tools) == 0 {
		// Fallback: build a flat prompt from system + history and use /api/generate
		prompt := systemContext + "\n"
		for _, msg := range messages {
			prompt += fmt.Sprintf("[%s]: %s\n", msg.Role, msg.Content)
		}
		return r.Complete(ctx, prompt, maxTokens)
	}
	return r.chatOllamaWithTools(ctx, systemContext, messages, maxTokens, tools)
}

func (r *RemoteRunner) Unload() error {
	if r.protocol == ProtocolOllama {
		body, _ := json.Marshal(map[string]interface{}{
			"model":      r.model,
			"keep_alive": 0,
		})
		_, err := r.client.Post(r.baseURL+"/api/generate", "application/json", bytes.NewReader(body))
		return err
	}
	return nil
}

// ── Ollama /api/chat with native tools ──────────────────────────────────────

// ollamaTool follows the Ollama tool calling spec (OpenAI-compatible format).
type ollamaTool struct {
	Type     string             `json:"type"`
	Function ollamaToolFunction `json:"function"`
}

type ollamaToolFunction struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

type ollamaChatReq struct {
	Model    string                 `json:"model"`
	Messages []ollamaChatMsg        `json:"messages"`
	Stream   bool                   `json:"stream"`
	Tools    []ollamaTool           `json:"tools,omitempty"`
	Options  map[string]interface{} `json:"options,omitempty"`
}

type ollamaChatMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ollamaChatResp struct {
	Message struct {
		Role      string `json:"role"`
		Content   string `json:"content"`
		ToolCalls []struct {
			Function struct {
				Name      string                 `json:"name"`
				Arguments map[string]interface{} `json:"arguments"`
			} `json:"function"`
		} `json:"tool_calls,omitempty"`
	} `json:"message"`
	Done bool `json:"done"`
}

func (r *RemoteRunner) chatOllamaWithTools(ctx context.Context, systemContext string, messages []models.Message, maxTokens int, tools []models.ToolDefinition) (string, error) {
	ollamaTools := convertToOllamaTools(tools)

	// Build a proper multi-turn message array so Ollama's chat API sees real
	// conversation structure rather than one giant user string.
	chatMessages := []ollamaChatMsg{
		{Role: "system", Content: systemContext},
	}
	for _, msg := range messages {
		chatMessages = append(chatMessages, ollamaChatMsg{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	body, err := json.Marshal(ollamaChatReq{
		Model:    r.model,
		Messages: chatMessages,
		Stream:   false,
		Tools:    ollamaTools,
		Options: map[string]interface{}{
			"num_predict": maxTokens,
			"temperature": 0.1, // Low temp for deterministic tool calling
			"top_p":       0.9,
		},
	})
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", r.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := r.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("ollama unreachable at %s — is it running? %w", r.baseURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("ollama HTTP %d: %s", resp.StatusCode, string(b))
	}

	var result ollamaChatResp
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}

	// If Ollama returned a tool call, format it as the JSON Axiom's engine expects
	if len(result.Message.ToolCalls) > 0 {
		tc := result.Message.ToolCalls[0]
		structured := map[string]interface{}{
			"reasoning": result.Message.Content,
			"tool_call": map[string]interface{}{
				"name": tc.Function.Name,
				"args": tc.Function.Arguments,
			},
			"content": "",
		}
		out, _ := json.Marshal(structured)
		return string(out), nil
	}

	return result.Message.Content, nil
}

// convertToOllamaTools translates Axiom's ToolDefinition slice into the
// OpenAI-style JSON format that Ollama's /api/chat endpoint expects.
func convertToOllamaTools(defs []models.ToolDefinition) []ollamaTool {
	tools := make([]ollamaTool, 0, len(defs))
	for _, def := range defs {
		params := parseArgsSchemaToJSONSchema(def.ArgsSchema)
		tools = append(tools, ollamaTool{
			Type: "function",
			Function: ollamaToolFunction{
				Name:        def.Name,
				Description: def.Description,
				Parameters:  params,
			},
		})
	}
	return tools
}

// parseArgsSchemaToJSONSchema converts Axiom's simple schema format into
// a proper JSON Schema object for the Ollama tool calling API.
func parseArgsSchemaToJSONSchema(schema string) map[string]interface{} {
	result := map[string]interface{}{
		"type":       "object",
		"properties": map[string]interface{}{},
	}

	if schema == "" || schema == "{}" {
		return result
	}

	var raw map[string]interface{}
	if err := json.Unmarshal([]byte(schema), &raw); err != nil {
		return result
	}

	properties := map[string]interface{}{}
	required := []string{}

	for key, val := range raw {
		desc := ""
		propType := "string"

		switch v := val.(type) {
		case string:
			desc = v
			if !isOptional(v) {
				required = append(required, key)
			}
			// Infer type from description
			if containsWord(v, "int") || containsWord(v, "number") {
				propType = "integer"
			} else if containsWord(v, "bool") {
				propType = "boolean"
			}
		case map[string]interface{}:
			if t, ok := v["type"].(string); ok {
				propType = t
			}
			if d, ok := v["description"].(string); ok {
				desc = d
			}
		}

		properties[key] = map[string]interface{}{
			"type":        propType,
			"description": desc,
		}
	}

	result["properties"] = properties
	if len(required) > 0 {
		result["required"] = required
	}
	return result
}

func isOptional(s string) bool {
	for i := 0; i <= len(s)-8; i++ {
		if s[i:i+8] == "optional" {
			return true
		}
	}
	return false
}

func containsWord(s, word string) bool {
	for i := 0; i <= len(s)-len(word); i++ {
		if s[i:i+len(word)] == word {
			return true
		}
	}
	return false
}

// ── Ollama /api/generate (legacy, no tools) ─────────────────────────────────

type ollamaResp struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

func (r *RemoteRunner) completeOllama(ctx context.Context, prompt string, maxTokens int) (string, error) {
	body, err := json.Marshal(map[string]interface{}{
		"model":  r.model,
		"prompt": prompt,
		"stream": false,
		"format": "json", // Enforce valid JSON output at the API level
		"options": map[string]interface{}{
			"num_predict": maxTokens,
			"temperature": 0.1, // Low temp for deterministic tool/JSON calling
			"top_p":       0.9,
		},
	})
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", r.baseURL+"/api/generate", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := r.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("ollama unreachable at %s — is it running? %w", r.baseURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("ollama HTTP %d: %s", resp.StatusCode, string(b))
	}

	var result ollamaResp
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	return result.Response, nil
}

// ── OpenAI-compatible ───────────────────────────────────────────────────────

type openAIReq struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature"`
	Stream      bool    `json:"stream"`
}

type openAIResp struct {
	Choices []struct {
		Text string `json:"text"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

func (r *RemoteRunner) completeOpenAI(ctx context.Context, prompt string, maxTokens int) (string, error) {
	body, err := json.Marshal(openAIReq{
		Model:       r.model,
		Prompt:      prompt,
		MaxTokens:   maxTokens,
		Temperature: 0.7,
		Stream:      false,
	})
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", r.baseURL+"/v1/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := r.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("server unreachable at %s: %w", r.baseURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(b))
	}

	var result openAIResp
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}

	if result.Error != nil {
		return "", fmt.Errorf("server error: %s", result.Error.Message)
	}
	if len(result.Choices) == 0 {
		return "", fmt.Errorf("empty response (no choices)")
	}
	return result.Choices[0].Text, nil
}
