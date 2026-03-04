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

// AnthropicRunner implements cognitive.LLMRunner using the Anthropic Messages API.
// Used in "Cloud" mode where Claude drives the entire Think-Verify-Act loop.
// Claude receives Axiom's tools as native tool definitions with input_schema,
// and returns tool_use blocks when it wants to call a tool.
type AnthropicRunner struct {
	apiKey    string
	model     string
	maxTokens int
	client    *http.Client
}

type AnthropicRunnerConfig struct {
	APIKey    string
	Model     string
	MaxTokens int
	TimeoutS  int
}

func NewAnthropicRunner(cfg AnthropicRunnerConfig) *AnthropicRunner {
	if cfg.Model == "" {
		cfg.Model = "claude-sonnet-4-5-20250929"
	}
	if cfg.MaxTokens == 0 {
		cfg.MaxTokens = 4096
	}
	timeout := cfg.TimeoutS
	if timeout == 0 {
		timeout = 120
	}

	return &AnthropicRunner{
		apiKey:    cfg.APIKey,
		model:     cfg.Model,
		maxTokens: cfg.MaxTokens,
		client:    &http.Client{Timeout: time.Duration(timeout) * time.Second},
	}
}

// ─── Request Types (Anthropic Messages API) ─────────────────────────────────

type anthropicRequest struct {
	Model     string             `json:"model"`
	MaxTokens int                `json:"max_tokens"`
	System    string             `json:"system,omitempty"`
	Messages  []anthropicMessage `json:"messages"`
	Tools     []anthropicTool    `json:"tools,omitempty"`
}

type anthropicMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // string or []contentBlock
}

type contentBlock struct {
	Type      string      `json:"type"`
	Text      string      `json:"text,omitempty"`
	ID        string      `json:"id,omitempty"`
	Name      string      `json:"name,omitempty"`
	Input     interface{} `json:"input,omitempty"`
	ToolUseID string      `json:"tool_use_id,omitempty"`
	Content   string      `json:"content,omitempty"`
}

type anthropicTool struct {
	Name        string              `json:"name"`
	Description string              `json:"description"`
	InputSchema anthropicToolSchema `json:"input_schema"`
}

type anthropicToolSchema struct {
	Type       string                       `json:"type"`
	Properties map[string]anthropicToolProp `json:"properties"`
	Required   []string                     `json:"required,omitempty"`
}

type anthropicToolProp struct {
	Type        string `json:"type"`
	Description string `json:"description,omitempty"`
}

// ─── Response Types ─────────────────────────────────────────────────────────

type anthropicResponse struct {
	ID         string `json:"id"`
	Role       string `json:"role"`
	StopReason string `json:"stop_reason"`
	Content    []struct {
		Type  string          `json:"type"`
		Text  string          `json:"text,omitempty"`
		ID    string          `json:"id,omitempty"`
		Name  string          `json:"name,omitempty"`
		Input json.RawMessage `json:"input,omitempty"`
	} `json:"content"`
	Error *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// ─── LLMRunner Interface ────────────────────────────────────────────────────

// Generate sends a prompt to Claude and parses the response into Axiom's format.
// Claude returns either text (for direct answers) or tool_use blocks (for tool calls).
func (r *AnthropicRunner) Generate(ctx context.Context, prompt string, toolDefs []models.ToolDefinition) (*models.LLMResponse, error) {
	// Build native Claude tools from Axiom's tool definitions
	claudeTools := r.convertTools(toolDefs)

	// Build messages — we receive the full formatted prompt from the cognitive engine
	reqBody := anthropicRequest{
		Model:     r.model,
		MaxTokens: r.maxTokens,
		System: `You are Axiom, a local-first AI agent. You have access to tools for file operations, code execution, web search, and cloud delegation.

When asked to perform a task:
1. Think step by step about what tools you need
2. Use tools one at a time, waiting for results
3. If a tool fails, analyze the error and try a different approach
4. Report success when the task is complete

Always respond with valid JSON matching this schema:
{"reasoning": "your chain of thought", "tool_call": {"name": "tool_name", "args": {}} or null, "content": "your response to the user"}`,
		Messages: []anthropicMessage{
			{Role: "user", Content: prompt},
		},
		Tools: claudeTools,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", r.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := r.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("anthropic unreachable: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("anthropic HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	var result anthropicResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("anthropic decode: %w", err)
	}

	if result.Error != nil {
		return nil, fmt.Errorf("anthropic: %s", result.Error.Message)
	}

	return r.parseResponse(&result)
}

// parseResponse converts Claude's response into Axiom's LLMResponse format.
func (r *AnthropicRunner) parseResponse(resp *anthropicResponse) (*models.LLMResponse, error) {
	llmResp := &models.LLMResponse{}

	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			// Try to parse as Axiom's structured JSON format
			var structured struct {
				Reasoning string `json:"reasoning"`
				ToolCall  *struct {
					Name string                 `json:"name"`
					Args map[string]interface{} `json:"args"`
				} `json:"tool_call"`
				Content string `json:"content"`
			}

			if err := json.Unmarshal([]byte(block.Text), &structured); err == nil {
				llmResp.Reasoning = structured.Reasoning
				llmResp.Content = structured.Content
				if structured.ToolCall != nil {
					llmResp.ToolCall = &models.ToolCall{
						Name: structured.ToolCall.Name,
						Args: structured.ToolCall.Args,
					}
				}
			} else {
				// Not structured JSON — use raw text as content
				llmResp.Content += block.Text
			}

		case "tool_use":
			// Claude wants to call one of our tools natively
			var args map[string]interface{}
			if err := json.Unmarshal(block.Input, &args); err != nil {
				args = map[string]interface{}{"raw": string(block.Input)}
			}

			llmResp.ToolCall = &models.ToolCall{
				Name: block.Name,
				Args: args,
			}
			llmResp.ToolUseID = block.ID
		}
	}

	// If stop_reason is "tool_use", Claude is waiting for tool results
	if resp.StopReason == "tool_use" && llmResp.ToolCall == nil {
		return nil, fmt.Errorf("anthropic stop_reason=tool_use but no tool_use block found")
	}

	return llmResp, nil
}

// convertTools translates Axiom tool definitions into Claude's native format.
func (r *AnthropicRunner) convertTools(defs []models.ToolDefinition) []anthropicTool {
	if len(defs) == 0 {
		return nil
	}

	tools := make([]anthropicTool, 0, len(defs))
	for _, def := range defs {
		tool := anthropicTool{
			Name:        def.Name,
			Description: def.Description,
			InputSchema: r.parseSchema(def.ArgsSchema),
		}
		tools = append(tools, tool)
	}
	return tools
}

// parseSchema converts Axiom's simple schema strings into Claude's input_schema format.
func (r *AnthropicRunner) parseSchema(schema string) anthropicToolSchema {
	result := anthropicToolSchema{
		Type:       "object",
		Properties: make(map[string]anthropicToolProp),
	}

	if schema == "" || schema == "{}" {
		return result
	}

	// Try parsing as JSON
	var raw map[string]interface{}
	if err := json.Unmarshal([]byte(schema), &raw); err != nil {
		return result
	}

	for key, val := range raw {
		desc := ""
		propType := "string"

		switch v := val.(type) {
		case string:
			desc = v
			// Infer type from description
			if containsAny(v, "int", "number", "count") {
				propType = "integer"
			} else if containsAny(v, "bool", "true", "false") {
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

		result.Properties[key] = anthropicToolProp{
			Type:        propType,
			Description: desc,
		}
	}

	return result
}

func containsAny(s string, substrs ...string) bool {
	lower := fmt.Sprintf("%s", s)
	for _, sub := range substrs {
		if len(lower) > 0 && len(sub) > 0 {
			for i := 0; i <= len(lower)-len(sub); i++ {
				if lower[i:i+len(sub)] == sub {
					return true
				}
			}
		}
	}
	return false
}

// ─── LLMRunner Interface Implementation ─────────────────────────────────────

// Complete satisfies the cognitive.LLMRunner interface so Claude can act as a
// drop-in replacement for the local model, using the exact same JSON prompting.
func (r *AnthropicRunner) Complete(ctx context.Context, prompt string, maxTokens int) (string, error) {
	reqBody := anthropicRequest{
		Model:     r.model,
		MaxTokens: maxTokens,
		// We pass the raw prompt (which includes Axiom's system instructions and tools)
		// directly to Claude as a user message.
		Messages: []anthropicMessage{
			{Role: "user", Content: prompt},
		},
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", r.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := r.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("anthropic API unreachable: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("anthropic HTTP %d: %s", resp.StatusCode, string(respBody))
	}

	var result anthropicResponse
	if err := json.Unmarshal(respBody, &result); err != nil {
		return "", fmt.Errorf("anthropic decode: %w", err)
	}

	if result.Error != nil {
		return "", fmt.Errorf("anthropic error: %s", result.Error.Message)
	}

	if len(result.Content) > 0 && result.Content[0].Type == "text" {
		return result.Content[0].Text, nil
	}

	return "", fmt.Errorf("empty or unexpected response format from Claude")
}

// Unload satisfies the cognitive.LLMRunner interface.
// Cloud models don't use local VRAM, so this is just a safe no-op.
func (r *AnthropicRunner) Unload() error {
	return nil
}
