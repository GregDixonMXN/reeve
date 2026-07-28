package adapters

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"axiom/pkg/models"
)

const (
	defaultOpenAIBaseURL        = "https://api.openai.com"
	defaultOpenAITimeout        = 120 * time.Second
	defaultOpenAIMaxTokens      = 4096
	defaultOpenAIReasoning      = "medium"
	defaultOpenAIErrorBodyBytes = int64(64 * 1024)
	openAIPendingStateTTL       = 10 * time.Minute
)

// OpenAIRunner implements Axiom's completion and native-tool interfaces using
// the OpenAI Responses API. Responses are kept stateless at the service
// boundary (store=false). During one active tool loop, opaque response output
// items are retained in memory and replayed with the matching tool result.
type OpenAIRunner struct {
	apiKey            string
	model             string
	baseURL           string
	reasoningEffort   string
	maxTokens         int
	maxErrorBodyBytes int64
	client            *http.Client

	mu      sync.Mutex
	pending map[string]openAIPendingToolLoop
}

type OpenAIRunnerConfig struct {
	APIKey          string
	Model           string
	BaseURL         string
	MaxTokens       int
	ReasoningEffort string
	TimeoutS        int

	// MaxErrorBodyBytes bounds text included in HTTP error messages. It is
	// primarily exposed for deterministic tests and custom OpenAI-compatible
	// gateways.
	MaxErrorBodyBytes int64
}

type openAIPendingToolLoop struct {
	input    []json.RawMessage
	output   []json.RawMessage
	callID   string
	name     string
	argsJSON string
	updated  time.Time
}

type openAIResponsesRequest struct {
	Model             string                       `json:"model"`
	Instructions      string                       `json:"instructions,omitempty"`
	Input             []json.RawMessage            `json:"input"`
	MaxOutputTokens   int                          `json:"max_output_tokens"`
	Reasoning         openAIReasoningConfig        `json:"reasoning"`
	Store             bool                         `json:"store"`
	Tools             []openAIResponseFunctionTool `json:"tools,omitempty"`
	ParallelToolCalls bool                         `json:"parallel_tool_calls"`
	Include           []string                     `json:"include,omitempty"`
}

type openAIReasoningConfig struct {
	Effort string `json:"effort"`
}

type openAIResponseFunctionTool struct {
	Type        string            `json:"type"`
	Name        string            `json:"name"`
	Description string            `json:"description,omitempty"`
	Parameters  models.JSONSchema `json:"parameters"`
	Strict      bool              `json:"strict"`
}

type openAIResponsesResponse struct {
	ID                string            `json:"id"`
	Status            string            `json:"status"`
	Output            []json.RawMessage `json:"output"`
	Error             *openAIAPIError   `json:"error,omitempty"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details,omitempty"`
}

type openAIAPIError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Type    string `json:"type"`
}

type openAIInputMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIFunctionCallOutput struct {
	Type   string `json:"type"`
	CallID string `json:"call_id"`
	Output string `json:"output"`
}

type openAIOutputHeader struct {
	Type string `json:"type"`
}

type openAIReasoningItem struct {
	Type    string `json:"type"`
	Summary []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"summary"`
}

type openAIMessageItem struct {
	Type    string `json:"type"`
	Role    string `json:"role"`
	Content []struct {
		Type    string `json:"type"`
		Text    string `json:"text,omitempty"`
		Refusal string `json:"refusal,omitempty"`
	} `json:"content"`
}

type openAIFunctionCallItem struct {
	Type      string `json:"type"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openAIParsedResponse struct {
	serialized   string
	functionCall *openAIFunctionCallItem
	argsJSON     string
}

type openAIAxiomOutput struct {
	Reasoning string          `json:"reasoning"`
	ToolCall  *openAIToolCall `json:"tool_call"`
	Content   string          `json:"content"`
}

type openAIToolCall struct {
	Name string                 `json:"name"`
	Args map[string]interface{} `json:"args"`
}

func NewOpenAIRunner(cfg OpenAIRunnerConfig) *OpenAIRunner {
	baseURL := strings.TrimSpace(cfg.BaseURL)
	if baseURL == "" {
		baseURL = defaultOpenAIBaseURL
	}
	reasoningEffort := strings.ToLower(strings.TrimSpace(cfg.ReasoningEffort))
	if reasoningEffort == "" {
		reasoningEffort = defaultOpenAIReasoning
	}
	maxTokens := cfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultOpenAIMaxTokens
	}
	timeout := time.Duration(cfg.TimeoutS) * time.Second
	if timeout <= 0 {
		timeout = defaultOpenAITimeout
	}
	maxErrorBodyBytes := cfg.MaxErrorBodyBytes
	if maxErrorBodyBytes <= 0 {
		maxErrorBodyBytes = defaultOpenAIErrorBodyBytes
	}

	return &OpenAIRunner{
		apiKey:            strings.TrimSpace(cfg.APIKey),
		model:             strings.TrimSpace(cfg.Model),
		baseURL:           strings.TrimRight(baseURL, "/"),
		reasoningEffort:   reasoningEffort,
		maxTokens:         maxTokens,
		maxErrorBodyBytes: maxErrorBodyBytes,
		client:            &http.Client{Timeout: timeout},
		pending:           make(map[string]openAIPendingToolLoop),
	}
}

// Complete satisfies cognitive.LLMRunner for compatibility callers that only
// have a flattened prompt.
func (r *OpenAIRunner) Complete(ctx context.Context, prompt string, maxTokens int) (string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	input, err := marshalOpenAIInputMessages([]models.Message{{
		Role:    models.RoleUser,
		Content: prompt,
	}})
	if err != nil {
		return "", err
	}
	return r.completeLocked(ctx, "", input, maxTokens, nil, "", false)
}

// CompleteWithTools satisfies cognitive.ToolAwareRunner. Without a stable
// conversation ID it remains stateless and does not retain opaque output items.
func (r *OpenAIRunner) CompleteWithTools(
	ctx context.Context,
	systemContext string,
	messages []models.Message,
	maxTokens int,
	tools []models.ToolDefinition,
) (string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	input, err := marshalOpenAIInputMessages(messages)
	if err != nil {
		return "", err
	}
	return r.completeLocked(ctx, systemContext, input, maxTokens, tools, "", false)
}

// CompleteWithToolsForConversation satisfies
// cognitive.ConversationToolAwareRunner. Raw reasoning and function-call output
// items are replayed only when the same conversation immediately supplies the
// matching tool result.
func (r *OpenAIRunner) CompleteWithToolsForConversation(
	ctx context.Context,
	conversationID string,
	systemContext string,
	messages []models.Message,
	maxTokens int,
	tools []models.ToolDefinition,
) (string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	conversationID = strings.TrimSpace(conversationID)
	r.prunePendingLocked(time.Now())

	input, err := r.inputForConversationLocked(conversationID, messages)
	if err != nil {
		return "", err
	}
	return r.completeLocked(ctx, systemContext, input, maxTokens, tools, conversationID, conversationID != "")
}

func (r *OpenAIRunner) completeLocked(
	ctx context.Context,
	systemContext string,
	input []json.RawMessage,
	maxTokens int,
	tools []models.ToolDefinition,
	conversationID string,
	trackPending bool,
) (string, error) {
	if r.apiKey == "" {
		return "", fmt.Errorf("openai responses: API key is required")
	}
	if r.model == "" {
		return "", fmt.Errorf("openai responses: model is required")
	}

	effectiveMaxTokens := r.maxTokens
	if effectiveMaxTokens <= 0 {
		effectiveMaxTokens = maxTokens
	}
	if effectiveMaxTokens <= 0 {
		effectiveMaxTokens = defaultOpenAIMaxTokens
	}

	payload := openAIResponsesRequest{
		Model:             r.model,
		Instructions:      systemContext,
		Input:             input,
		MaxOutputTokens:   effectiveMaxTokens,
		Reasoning:         openAIReasoningConfig{Effort: r.reasoningEffort},
		Store:             false,
		Tools:             convertToOpenAIResponseTools(tools),
		ParallelToolCalls: false,
		// Stateless Responses include encrypted reasoning by default. Explicitly
		// requesting it remains supported and keeps compatibility with older
		// Responses deployments.
		Include: []string{"reasoning.encrypted_content"},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("openai responses request encode: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, r.responsesEndpoint(), bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("openai responses request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+r.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := r.client.Do(req)
	if err != nil {
		return "", fmt.Errorf("openai responses request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		detail := r.readBoundedError(resp.Body)
		if detail == "" {
			detail = http.StatusText(resp.StatusCode)
		}
		return "", fmt.Errorf("openai responses HTTP %d: %s", resp.StatusCode, detail)
	}

	var result openAIResponsesResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("openai responses decode: %w", err)
	}
	if result.Error != nil {
		detail := result.Error.Message
		if detail == "" {
			detail = result.Error.Code
		}
		return "", fmt.Errorf("openai responses error: %s", r.boundAndRedact(detail))
	}
	if result.Status != "" && result.Status != "completed" {
		reason := result.Status
		if result.IncompleteDetails != nil && result.IncompleteDetails.Reason != "" {
			reason += ": " + result.IncompleteDetails.Reason
		}
		return "", fmt.Errorf("openai responses did not complete: %s", r.boundAndRedact(reason))
	}

	parsed, err := parseOpenAIResponsesOutput(result.Output)
	if err != nil {
		return "", err
	}

	if trackPending {
		if parsed.functionCall == nil {
			delete(r.pending, conversationID)
		} else {
			r.pending[conversationID] = openAIPendingToolLoop{
				input:    cloneRawMessages(input),
				output:   cloneRawMessages(result.Output),
				callID:   parsed.functionCall.CallID,
				name:     parsed.functionCall.Name,
				argsJSON: parsed.argsJSON,
				updated:  time.Now(),
			}
		}
	}

	return parsed.serialized, nil
}

func (r *OpenAIRunner) inputForConversationLocked(conversationID string, messages []models.Message) ([]json.RawMessage, error) {
	if conversationID != "" {
		if pending, ok := r.pending[conversationID]; ok {
			if output, matches := matchingOpenAIToolResult(messages, pending); matches {
				input := cloneRawMessages(pending.input)
				input = append(input, cloneRawMessages(pending.output)...)
				toolOutput, err := marshalRawJSON(openAIFunctionCallOutput{
					Type:   "function_call_output",
					CallID: pending.callID,
					Output: output,
				})
				if err != nil {
					return nil, err
				}
				return append(input, toolOutput), nil
			}
			// A new, non-matching turn supersedes stale provider state.
			delete(r.pending, conversationID)
		}
	}
	return marshalOpenAIInputMessages(messages)
}

func (r *OpenAIRunner) responsesEndpoint() string {
	switch {
	case strings.HasSuffix(r.baseURL, "/v1/responses"):
		return r.baseURL
	case strings.HasSuffix(r.baseURL, "/v1"):
		return r.baseURL + "/responses"
	default:
		return r.baseURL + "/v1/responses"
	}
}

func (r *OpenAIRunner) readBoundedError(body io.Reader) string {
	// Read far enough past the display boundary to recognize a credential that
	// starts just before it. Truncating first could otherwise leak a key prefix
	// that no longer matches the complete value during redaction.
	readLimit := r.maxErrorBodyBytes + int64(len(r.apiKey)) + 1
	data, _ := io.ReadAll(io.LimitReader(body, readLimit))
	truncated := int64(len(data)) > r.maxErrorBodyBytes
	// Do not trim leading whitespace: doing so after a bounded read could shift
	// a partial credential from beyond the display window into visible output.
	text := strings.TrimRight(r.redact(string(data)), " \t\r\n")
	text = truncateOpenAIErrorText(text, r.maxErrorBodyBytes)
	if truncated {
		text += " ... [truncated]"
	}
	return text
}

func (r *OpenAIRunner) boundAndRedact(text string) string {
	text = r.redact(text)
	if int64(len(text)) > r.maxErrorBodyBytes {
		text = truncateOpenAIErrorText(text, r.maxErrorBodyBytes) + " ... [truncated]"
	}
	return text
}

func (r *OpenAIRunner) redact(text string) string {
	if r.apiKey != "" {
		text = strings.ReplaceAll(text, r.apiKey, "[REDACTED]")
	}
	return text
}

func truncateOpenAIErrorText(text string, limit int64) string {
	if limit <= 0 || int64(len(text)) <= limit {
		return text
	}
	cut := int(limit)
	const marker = "[REDACTED]"
	for searchAt := 0; searchAt < len(text); {
		relative := strings.Index(text[searchAt:], marker)
		if relative < 0 {
			break
		}
		start := searchAt + relative
		end := start + len(marker)
		if start < cut && end > cut {
			// Preserve the whole marker even if it crosses the byte boundary.
			// This exceeds the configured display limit by at most one short,
			// constant marker and never exposes credential material.
			cut = end
			break
		}
		searchAt = end
	}
	if cut > len(text) {
		cut = len(text)
	}
	return text[:cut]
}

func (r *OpenAIRunner) prunePendingLocked(now time.Time) {
	for conversationID, pending := range r.pending {
		if now.Sub(pending.updated) > openAIPendingStateTTL {
			delete(r.pending, conversationID)
		}
	}
}

func (r *OpenAIRunner) Unload() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pending = make(map[string]openAIPendingToolLoop)
	return nil
}

func convertToOpenAIResponseTools(defs []models.ToolDefinition) []openAIResponseFunctionTool {
	if len(defs) == 0 {
		return nil
	}
	tools := make([]openAIResponseFunctionTool, 0, len(defs))
	for _, def := range defs {
		tools = append(tools, openAIResponseFunctionTool{
			Type:        "function",
			Name:        def.Name,
			Description: def.Description,
			Parameters:  def.CanonicalSchema(),
			// Existing Axiom schemas contain genuinely optional properties. They
			// are not compatible with OpenAI strict mode's all-fields-required
			// invariant, so request best-effort schema adherence explicitly.
			Strict: false,
		})
	}
	return tools
}

func marshalOpenAIInputMessages(messages []models.Message) ([]json.RawMessage, error) {
	input := make([]json.RawMessage, 0, len(messages))
	for _, message := range messages {
		role := message.Role
		content := message.Content
		switch role {
		case models.RoleUser, models.RoleAssistant, models.RoleSystem:
		case models.RoleTool:
			// Without live provider state there is no trustworthy call_id.
			// Preserve the result as user-provided context instead of inventing
			// protocol linkage that the Responses API may reject.
			role = models.RoleUser
			content = "[Tool Result]\n" + content
		default:
			continue
		}
		raw, err := marshalRawJSON(openAIInputMessage{Role: role, Content: content})
		if err != nil {
			return nil, err
		}
		input = append(input, raw)
	}
	return input, nil
}

func matchingOpenAIToolResult(messages []models.Message, pending openAIPendingToolLoop) (string, bool) {
	if len(messages) < 2 {
		return "", false
	}
	assistant := messages[len(messages)-2]
	tool := messages[len(messages)-1]
	if assistant.Role != models.RoleAssistant || tool.Role != models.RoleTool {
		return "", false
	}

	var stored openAIAxiomOutput
	if err := json.Unmarshal([]byte(assistant.Content), &stored); err != nil || stored.ToolCall == nil {
		return "", false
	}
	if stored.ToolCall.Name != pending.name {
		return "", false
	}
	argsJSON, err := json.Marshal(stored.ToolCall.Args)
	if err != nil || string(argsJSON) != pending.argsJSON {
		return "", false
	}
	return tool.Content, true
}

func parseOpenAIResponsesOutput(output []json.RawMessage) (*openAIParsedResponse, error) {
	var (
		contentParts   []string
		reasoningParts []string
		functionCalls  []openAIFunctionCallItem
	)

	for _, raw := range output {
		var header openAIOutputHeader
		if err := json.Unmarshal(raw, &header); err != nil {
			return nil, fmt.Errorf("openai responses output decode: %w", err)
		}
		switch header.Type {
		case "reasoning":
			var item openAIReasoningItem
			if err := json.Unmarshal(raw, &item); err != nil {
				return nil, fmt.Errorf("openai reasoning output decode: %w", err)
			}
			for _, summary := range item.Summary {
				if text := strings.TrimSpace(summary.Text); text != "" {
					reasoningParts = append(reasoningParts, text)
				}
			}

		case "message":
			var item openAIMessageItem
			if err := json.Unmarshal(raw, &item); err != nil {
				return nil, fmt.Errorf("openai message output decode: %w", err)
			}
			for _, part := range item.Content {
				switch part.Type {
				case "output_text":
					if part.Text != "" {
						contentParts = append(contentParts, part.Text)
					}
				case "refusal":
					if part.Refusal != "" {
						contentParts = append(contentParts, part.Refusal)
					}
				}
			}

		case "function_call":
			var item openAIFunctionCallItem
			if err := json.Unmarshal(raw, &item); err != nil {
				return nil, fmt.Errorf("openai function call decode: %w", err)
			}
			functionCalls = append(functionCalls, item)
		}
	}

	if len(functionCalls) > 1 {
		return nil, fmt.Errorf("openai returned %d function calls; Axiom supports one tool call per turn", len(functionCalls))
	}

	reasoning := strings.Join(reasoningParts, "\n\n")
	content := strings.Join(contentParts, "\n")
	out := openAIAxiomOutput{
		Reasoning: reasoning,
		Content:   content,
	}
	parsed := &openAIParsedResponse{}

	if len(functionCalls) == 1 {
		call := functionCalls[0]
		if strings.TrimSpace(call.CallID) == "" || strings.TrimSpace(call.Name) == "" {
			return nil, fmt.Errorf("openai returned an incomplete function call")
		}
		var args map[string]interface{}
		arguments := strings.TrimSpace(call.Arguments)
		if arguments == "" {
			arguments = "{}"
		}
		if err := json.Unmarshal([]byte(arguments), &args); err != nil {
			return nil, fmt.Errorf("openai function %q arguments decode: %w", call.Name, err)
		}
		if args == nil {
			return nil, fmt.Errorf("openai function %q arguments must be a JSON object", call.Name)
		}
		argsJSON, err := json.Marshal(args)
		if err != nil {
			return nil, fmt.Errorf("openai function %q arguments encode: %w", call.Name, err)
		}

		// Do not expose a tool-turn preamble as final user-visible content. Keep
		// it with the safe reasoning summary until the tool loop completes.
		if content != "" {
			if out.Reasoning != "" {
				out.Reasoning += "\n\n"
			}
			out.Reasoning += content
		}
		out.Content = ""
		out.ToolCall = &openAIToolCall{Name: call.Name, Args: args}
		parsed.functionCall = &call
		parsed.argsJSON = string(argsJSON)
	} else if content == "" {
		return nil, fmt.Errorf("openai response contained no assistant message or function call")
	}

	serialized, err := json.Marshal(out)
	if err != nil {
		return nil, fmt.Errorf("openai response serialize: %w", err)
	}
	parsed.serialized = string(serialized)
	return parsed, nil
}

func marshalRawJSON(value interface{}) (json.RawMessage, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("openai input encode: %w", err)
	}
	return json.RawMessage(encoded), nil
}

func cloneRawMessages(messages []json.RawMessage) []json.RawMessage {
	cloned := make([]json.RawMessage, len(messages))
	for i, message := range messages {
		cloned[i] = append(json.RawMessage(nil), message...)
	}
	return cloned
}
