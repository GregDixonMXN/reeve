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
	if cfg.MaxTokens <= 0 {
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

type anthropicToolSchema = models.JSONSchema

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
		System: `You are Axiom, an AI agent with access to tools for file operations, code execution, web search, and cloud delegation.

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
	if err := validateSingleToolUse(resp); err != nil {
		return nil, err
	}

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
			InputSchema: def.CanonicalSchema(),
		}
		tools = append(tools, tool)
	}
	return tools
}

// parseSchema is the compatibility adapter retained for legacy callers and
// tests. Provider requests consume ToolDefinition.CanonicalSchema directly.
func (r *AnthropicRunner) parseSchema(schema string) anthropicToolSchema {
	return models.ParseLegacyArgsSchema(schema)
}

// ─── ToolAwareRunner Interface (CompleteWithTools) ──────────────────────────

// CompleteWithTools implements cognitive.ToolAwareRunner so that in cloud mode,
// Claude receives the full structured multi-turn conversation history and native
// tool definitions — instead of the monolithic Phi-4-style JSON prompt.
//
// This is the critical fix: previously AnthropicRunner only implemented Complete(),
// so the engine fell back to flattening everything into one giant user message,
// bypassing Claude's native tool-use capability entirely.
func (r *AnthropicRunner) CompleteWithTools(
	ctx context.Context,
	systemContext string,
	messages []models.Message,
	_ int,
	tools []models.ToolDefinition,
) (string, error) {
	claudeTools := r.convertTools(tools)
	claudeMessages, err := r.convertMessages(messages)
	if err != nil {
		return "", fmt.Errorf("message conversion: %w", err)
	}

	reqBody := anthropicRequest{
		Model:     r.model,
		MaxTokens: r.maxTokens,
		System:    systemContext,
		Messages:  claudeMessages,
		Tools:     claudeTools,
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
		return "", fmt.Errorf("anthropic unreachable: %w", err)
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
		return "", fmt.Errorf("anthropic: %s", result.Error.Message)
	}

	// Serialize Claude's native response back to Axiom's JSON format so that
	// the cognitive engine's parseResponse() can handle it uniformly.
	return r.serializeToAxiomJSON(&result)
}

// convertMessages translates Axiom's flat []models.Message into the
// alternating user/assistant format Anthropic's API requires.
//
// Critical invariant: a tool_result block MUST be immediately preceded by an
// assistant message containing a tool_use block with a matching ID. If we can't
// reconstruct that pairing (e.g. the assistant message isn't parseable JSON or
// has no tool_call field), we fall back to plain user text so we never send an
// orphaned tool_result — which causes Anthropic to return HTTP 400.
func (r *AnthropicRunner) convertMessages(messages []models.Message) ([]anthropicMessage, error) {
	var out []anthropicMessage
	toolUseCounter := 0

	i := 0
	for i < len(messages) {
		msg := messages[i]

		switch msg.Role {
		case models.RoleUser:
			out = append(out, anthropicMessage{Role: "user", Content: msg.Content})
			i++

		case models.RoleAssistant:
			// Try to parse Axiom's stored JSON format to detect a tool call.
			var assistantJSON struct {
				Reasoning string `json:"reasoning"`
				ToolCall  *struct {
					Name string                 `json:"name"`
					Args map[string]interface{} `json:"args"`
				} `json:"tool_call"`
				Content string `json:"content"`
			}
			hasToolCall := json.Unmarshal([]byte(msg.Content), &assistantJSON) == nil &&
				assistantJSON.ToolCall != nil

			// Peek ahead: does a tool result immediately follow?
			nextIsToolResult := i+1 < len(messages) && messages[i+1].Role == models.RoleTool

			if hasToolCall && nextIsToolResult {
				// Emit assistant message with a proper tool_use block, then pair
				// the following tool result as a user tool_result block.
				toolUseCounter++
				toolUseID := fmt.Sprintf("axiom_tool_%d", toolUseCounter)

				var assistantBlocks []contentBlock
				// Prepend a text block for any reasoning/content
				textContent := assistantJSON.Reasoning
				if assistantJSON.Content != "" {
					textContent = assistantJSON.Content
				}
				if textContent != "" {
					assistantBlocks = append(assistantBlocks, contentBlock{Type: "text", Text: textContent})
				}
				assistantBlocks = append(assistantBlocks, contentBlock{
					Type:  "tool_use",
					ID:    toolUseID,
					Name:  assistantJSON.ToolCall.Name,
					Input: assistantJSON.ToolCall.Args,
				})
				out = append(out, anthropicMessage{Role: "assistant", Content: assistantBlocks})
				i++

				// Paired tool_result user message — ID matches the tool_use above
				out = append(out, anthropicMessage{
					Role: "user",
					Content: []contentBlock{{
						Type:      "tool_result",
						ToolUseID: toolUseID,
						Content:   messages[i].Content,
					}},
				})
				i++

			} else {
				// No parseable tool call, OR tool call with no following result.
				// Emit as plain assistant text so we never create an unmatched tool_use.
				textContent := msg.Content
				if hasToolCall {
					// Extract human-readable part from the JSON
					if assistantJSON.Content != "" {
						textContent = assistantJSON.Content
					} else if assistantJSON.Reasoning != "" {
						textContent = assistantJSON.Reasoning
					}
				}
				if textContent == "" {
					textContent = "..."
				}
				out = append(out, anthropicMessage{Role: "assistant", Content: textContent})
				i++

				// If a tool result follows a non-tool-call assistant message,
				// emit it as plain user text — never as a tool_result block.
				if i < len(messages) && messages[i].Role == models.RoleTool {
					out = append(out, anthropicMessage{
						Role:    "user",
						Content: "[Tool Result]\n" + messages[i].Content,
					})
					i++
				}
			}

		case models.RoleTool:
			// Orphaned tool result — no preceding assistant message at all.
			// Convert to plain user text; never emit a tool_result block without
			// a matching tool_use.
			out = append(out, anthropicMessage{
				Role:    "user",
				Content: "[Tool Result]\n" + msg.Content,
			})
			i++

		default:
			i++
		}
	}

	// Anthropic requires the first message to be a user turn
	if len(out) == 0 || out[0].Role != "user" {
		out = append([]anthropicMessage{{Role: "user", Content: "Hello"}}, out...)
	}

	// Merge consecutive same-role messages to ensure strict alternation
	out = mergeConsecutiveRoles(out)
	return out, nil
}

// mergeConsecutiveRoles ensures the message list alternates user/assistant.
// Consecutive messages with the same role are merged into one.
func mergeConsecutiveRoles(messages []anthropicMessage) []anthropicMessage {
	if len(messages) == 0 {
		return messages
	}
	var merged []anthropicMessage
	merged = append(merged, messages[0])

	for i := 1; i < len(messages); i++ {
		last := &merged[len(merged)-1]
		curr := messages[i]

		if last.Role == curr.Role {
			// Merge: append current content to last
			switch v := last.Content.(type) {
			case string:
				switch c := curr.Content.(type) {
				case string:
					last.Content = v + "\n" + c
				case []contentBlock:
					blocks := append([]contentBlock{{Type: "text", Text: v}}, c...)
					last.Content = blocks
				}
			case []contentBlock:
				switch c := curr.Content.(type) {
				case string:
					last.Content = append(v, contentBlock{Type: "text", Text: c})
				case []contentBlock:
					last.Content = append(v, c...)
				}
			}
		} else {
			merged = append(merged, curr)
		}
	}
	return merged
}

// serializeToAxiomJSON converts Claude's native API response back to Axiom's
// internal JSON format: {"reasoning":"...","tool_call":{...}|null,"content":"..."}.
// This lets the cognitive engine's parseResponse() handle cloud and local responses uniformly.
func (r *AnthropicRunner) serializeToAxiomJSON(resp *anthropicResponse) (string, error) {
	if err := validateSingleToolUse(resp); err != nil {
		return "", err
	}

	type axiomOutput struct {
		Reasoning string      `json:"reasoning"`
		ToolCall  interface{} `json:"tool_call"`
		Content   string      `json:"content"`
	}

	out := axiomOutput{ToolCall: nil}

	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			if out.Content != "" {
				out.Content += "\n"
			}
			out.Content += block.Text

		case "tool_use":
			var args map[string]interface{}
			if err := json.Unmarshal(block.Input, &args); err != nil {
				args = map[string]interface{}{}
			}
			out.ToolCall = map[string]interface{}{
				"name": block.Name,
				"args": args,
			}
		}
	}

	result, err := json.Marshal(out)
	if err != nil {
		return "", fmt.Errorf("serialize axiom JSON: %w", err)
	}
	return string(result), nil
}

// validateSingleToolUse enforces Axiom's sequential execution contract. Claude
// may return parallel tool_use blocks, but the orchestrator can execute and
// correlate only one tool call per model turn. Rejecting the response avoids
// silently discarding all but the last requested action.
func validateSingleToolUse(resp *anthropicResponse) error {
	toolUses := 0
	for _, block := range resp.Content {
		if block.Type == "tool_use" {
			toolUses++
		}
	}

	if toolUses > 1 {
		return fmt.Errorf("anthropic returned %d tool_use blocks; Axiom supports one tool call per turn", toolUses)
	}
	return nil
}

// ─── LLMRunner Interface Implementation ─────────────────────────────────────

// Complete satisfies the cognitive.LLMRunner interface so Claude can act as a
// drop-in replacement for the local model, using the exact same JSON prompting.
func (r *AnthropicRunner) Complete(ctx context.Context, prompt string, _ int) (string, error) {
	reqBody := anthropicRequest{
		Model:     r.model,
		MaxTokens: r.maxTokens,
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
