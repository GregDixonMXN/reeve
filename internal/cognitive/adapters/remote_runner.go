package adapters

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"

	"axiom/pkg/models"
)

// Protocol selects which API format to use.
type Protocol int

const (
	ProtocolOllama Protocol = iota
	ProtocolOpenAI

	defaultRemoteTemperature = 0.1
	defaultRemoteTopP        = 0.9
	defaultRemoteRepeat      = 1.0
	ollamaPendingStateTTL    = 10 * time.Minute
)

type RemoteRunnerConfig struct {
	BaseURL         string
	Model           string
	Protocol        Protocol
	TimeoutS        int
	ContextSize     int
	Temperature     float64
	TopP            float64
	TopK            int
	MinP            float64
	PresencePenalty float64
	RepeatPenalty   float64
	Threads         int
	Think           bool
}

// RemoteRunner talks to a local LLM server over HTTP.
// Supports Ollama (/api/chat with tools) and OpenAI-compatible (/v1/completions).
type RemoteRunner struct {
	baseURL         string
	model           string
	protocol        Protocol
	client          *http.Client
	contextSize     int
	temperature     float64
	topP            float64
	topK            int
	minP            float64
	presencePenalty float64
	repeatPenalty   float64
	threads         int
	think           bool

	mu      sync.Mutex
	pending map[string]ollamaPendingToolLoop
	locks   map[string]*ollamaConversationLock
	epoch   uint64
}

var nativeThinkBlockPattern = regexp.MustCompile(`(?s)<think>(.*?)</think>`)

type ollamaPendingToolLoop struct {
	input     []ollamaChatMsg
	assistant ollamaChatMsg
	name      string
	argsJSON  string
	updated   time.Time
}

type ollamaConversationLock struct {
	mu   sync.Mutex
	refs int
}

func NewRemoteRunner(cfg RemoteRunnerConfig) *RemoteRunner {
	timeout := cfg.TimeoutS
	if timeout <= 0 {
		timeout = 120
	}
	temperature := cfg.Temperature
	if temperature <= 0 {
		temperature = defaultRemoteTemperature
	}
	topP := cfg.TopP
	if topP <= 0 {
		topP = defaultRemoteTopP
	}
	repeatPenalty := cfg.RepeatPenalty
	if repeatPenalty <= 0 {
		repeatPenalty = defaultRemoteRepeat
	}
	return &RemoteRunner{
		baseURL:         cfg.BaseURL,
		model:           cfg.Model,
		protocol:        cfg.Protocol,
		client:          &http.Client{Timeout: time.Duration(timeout) * time.Second},
		contextSize:     cfg.ContextSize,
		temperature:     temperature,
		topP:            topP,
		topK:            cfg.TopK,
		minP:            cfg.MinP,
		presencePenalty: cfg.PresencePenalty,
		repeatPenalty:   repeatPenalty,
		threads:         cfg.Threads,
		think:           cfg.Think,
		pending:         make(map[string]ollamaPendingToolLoop),
		locks:           make(map[string]*ollamaConversationLock),
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
// systemContext is the system prompt plus repo and memory context. Native tool
// names, descriptions, and schemas arrive separately in tools.
// messages is the structured conversation history passed as proper multi-turn turns.
func (r *RemoteRunner) CompleteWithTools(ctx context.Context, systemContext string, messages []models.Message, maxTokens int, tools []models.ToolDefinition) (string, error) {
	if r.protocol != ProtocolOllama {
		// OpenAI-compatible completion servers do not expose Ollama's chat API.
		systemContext = appendFallbackToolListing(systemContext, tools)
		prompt := systemContext + "\n"
		for _, msg := range messages {
			prompt += fmt.Sprintf("[%s]: %s\n", msg.Role, msg.Content)
		}
		return r.Complete(ctx, prompt, maxTokens)
	}
	return r.chatOllama(ctx, buildOllamaChatMessages(systemContext, messages), maxTokens, tools, "", false, 0)
}

func appendFallbackToolListing(systemContext string, tools []models.ToolDefinition) string {
	if len(tools) == 0 {
		return systemContext
	}

	var listing strings.Builder
	listing.WriteString("\nAVAILABLE TOOLS:\n")
	for _, tool := range tools {
		fmt.Fprintf(&listing, "- %s: %s\n  Args: %s\n", tool.Name, tool.Description, tool.SchemaJSON())
	}
	return systemContext + listing.String()
}

// CompleteWithToolsForConversation preserves Ollama's native assistant
// thinking/tool-call message across the immediately following tool result.
// Axiom persists a provider-neutral JSON assistant turn, so the native message
// is retained only in memory and keyed by the stable conversation ID.
func (r *RemoteRunner) CompleteWithToolsForConversation(
	ctx context.Context,
	conversationID string,
	systemContext string,
	messages []models.Message,
	maxTokens int,
	tools []models.ToolDefinition,
) (string, error) {
	if r.protocol != ProtocolOllama {
		return r.CompleteWithTools(ctx, systemContext, messages, maxTokens, tools)
	}

	conversationID = strings.TrimSpace(conversationID)
	release := r.acquireConversation(conversationID)
	defer release()

	r.mu.Lock()
	r.prunePendingLocked(time.Now())
	chatMessages := r.inputForConversationLocked(conversationID, systemContext, messages)
	epoch := r.epoch
	r.mu.Unlock()

	return r.chatOllama(
		ctx,
		chatMessages,
		maxTokens,
		tools,
		conversationID,
		conversationID != "",
		epoch,
	)
}

func (r *RemoteRunner) Unload() error {
	r.mu.Lock()
	r.epoch++
	r.pending = make(map[string]ollamaPendingToolLoop)
	r.mu.Unlock()

	if r.protocol == ProtocolOllama {
		body, _ := json.Marshal(map[string]interface{}{
			"model":      r.model,
			"keep_alive": 0,
		})
		resp, err := r.client.Post(r.baseURL+"/api/generate", "application/json", bytes.NewReader(body))
		if err != nil {
			return err
		}
		resp.Body.Close()
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
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Parameters  models.JSONSchema `json:"parameters"`
}

type ollamaChatReq struct {
	Model    string                 `json:"model"`
	Messages []ollamaChatMsg        `json:"messages"`
	Stream   bool                   `json:"stream"`
	Think    *bool                  `json:"think,omitempty"`
	Tools    []ollamaTool           `json:"tools,omitempty"`
	Options  map[string]interface{} `json:"options,omitempty"`
}

type ollamaToolCall struct {
	Type     string `json:"type,omitempty"`
	Function struct {
		Index     *int                   `json:"index,omitempty"`
		Name      string                 `json:"name"`
		Arguments map[string]interface{} `json:"arguments"`
	} `json:"function"`
}

// StreamingRunner predates Ollama's type/function.index fields. Keep its
// response-facing shape assignment-compatible while the non-streaming runner
// uses NativeToolCalls for lossless replay.
type legacyOllamaToolCall = struct {
	Function struct {
		Name      string                 `json:"name"`
		Arguments map[string]interface{} `json:"arguments"`
	} `json:"function"`
}

type ollamaChatMsg struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	Thinking  string           `json:"thinking,omitempty"`
	ToolCalls []ollamaToolCall `json:"tool_calls,omitempty"`
	ToolName  string           `json:"tool_name,omitempty"`
}

type ollamaChatResponseMessage struct {
	Role            string                 `json:"role"`
	Content         string                 `json:"content"`
	Thinking        string                 `json:"thinking,omitempty"`
	ToolCalls       []legacyOllamaToolCall `json:"-"`
	NativeToolCalls []ollamaToolCall       `json:"-"`
}

func (m *ollamaChatResponseMessage) UnmarshalJSON(data []byte) error {
	var wire struct {
		Role      string           `json:"role"`
		Content   string           `json:"content"`
		Thinking  string           `json:"thinking,omitempty"`
		ToolCalls []ollamaToolCall `json:"tool_calls,omitempty"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	m.Role = wire.Role
	m.Content = wire.Content
	m.Thinking = wire.Thinking
	m.NativeToolCalls = wire.ToolCalls
	m.ToolCalls = make([]legacyOllamaToolCall, len(wire.ToolCalls))
	for i, call := range wire.ToolCalls {
		m.ToolCalls[i].Function.Name = call.Function.Name
		m.ToolCalls[i].Function.Arguments = call.Function.Arguments
	}
	return nil
}

func (m ollamaChatResponseMessage) nativeMessage() ollamaChatMsg {
	return ollamaChatMsg{
		Role:      m.Role,
		Content:   m.Content,
		Thinking:  m.Thinking,
		ToolCalls: m.NativeToolCalls,
	}
}

type ollamaChatResp struct {
	Message ollamaChatResponseMessage `json:"message"`
	Done    bool                      `json:"done"`
}

func (r *RemoteRunner) chatOllama(
	ctx context.Context,
	chatMessages []ollamaChatMsg,
	maxTokens int,
	tools []models.ToolDefinition,
	conversationID string,
	trackPending bool,
	epoch uint64,
) (string, error) {
	ollamaTools := convertToOllamaTools(tools)
	think := r.think

	body, err := json.Marshal(ollamaChatReq{
		Model:    r.model,
		Messages: chatMessages,
		Stream:   false,
		Think:    &think,
		Tools:    ollamaTools,
		Options:  r.ollamaOptions(maxTokens),
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
	nativeMessage := result.Message.nativeMessage()

	if len(nativeMessage.ToolCalls) > 1 {
		if trackPending {
			r.clearPending(conversationID, epoch)
		}
		return "", fmt.Errorf(
			"ollama returned %d function calls; Axiom supports one tool call per turn",
			len(nativeMessage.ToolCalls),
		)
	}

	// If Ollama returned a tool call, format it as the JSON Axiom's engine
	// expects while retaining the exact native assistant turn for replay.
	if len(nativeMessage.ToolCalls) == 1 {
		tc := nativeMessage.ToolCalls[0]
		name := strings.TrimSpace(tc.Function.Name)
		if name == "" {
			if trackPending {
				r.clearPending(conversationID, epoch)
			}
			return "", fmt.Errorf("ollama returned an incomplete function call with no name")
		}
		args := tc.Function.Arguments
		if args == nil {
			args = map[string]interface{}{}
			nativeMessage.ToolCalls[0].Function.Arguments = args
		}
		argsJSON, err := json.Marshal(args)
		if err != nil {
			if trackPending {
				r.clearPending(conversationID, epoch)
			}
			return "", fmt.Errorf("ollama function %q arguments encode: %w", name, err)
		}

		structured := models.LLMResponse{
			Reasoning: nativeToolReasoning(nativeMessage.Thinking, nativeMessage.Content),
			ToolCall: &models.ToolCall{
				Name: name,
				Args: args,
			},
			Content: "",
		}
		out, err := json.Marshal(structured)
		if err != nil {
			if trackPending {
				r.clearPending(conversationID, epoch)
			}
			return "", fmt.Errorf("ollama response serialize: %w", err)
		}
		if trackPending {
			r.storePending(conversationID, ollamaPendingToolLoop{
				input:     cloneOllamaChatMessages(chatMessages),
				assistant: cloneOllamaChatMessage(nativeMessage),
				name:      name,
				argsJSON:  string(argsJSON),
				updated:   time.Now(),
			}, epoch)
		}
		return string(out), nil
	}

	if trackPending {
		r.clearPending(conversationID, epoch)
	}
	if strings.TrimSpace(nativeMessage.Content) == "" {
		return "", fmt.Errorf("ollama response contained no assistant content or function call")
	}
	return normalizeNativeMessage(nativeMessage.Content, nativeMessage.Thinking), nil
}

func (r *RemoteRunner) ollamaOptions(maxTokens int) map[string]interface{} {
	options := map[string]interface{}{
		"num_predict":      maxTokens,
		"temperature":      r.temperature,
		"top_p":            r.topP,
		"min_p":            r.minP,
		"presence_penalty": r.presencePenalty,
		"repeat_penalty":   r.repeatPenalty,
	}
	if r.contextSize > 0 {
		options["num_ctx"] = r.contextSize
	}
	if r.topK > 0 {
		options["top_k"] = r.topK
	}
	if r.threads > 0 {
		options["num_thread"] = r.threads
	}
	return options
}

func buildOllamaChatMessages(systemContext string, messages []models.Message) []ollamaChatMsg {
	chatMessages := make([]ollamaChatMsg, 0, len(messages)+1)
	if systemContext != "" {
		chatMessages = append(chatMessages, ollamaChatMsg{
			Role:    models.RoleSystem,
			Content: systemContext,
		})
	}
	for _, msg := range messages {
		switch msg.Role {
		case models.RoleUser, models.RoleAssistant, models.RoleSystem:
			chatMessages = append(chatMessages, ollamaChatMsg{
				Role:    msg.Role,
				Content: msg.Content,
			})
		case models.RoleTool:
			// Without live provider state there is no trustworthy tool name to
			// attach. Preserve the output as user context instead of emitting an
			// invalid, unlinked Ollama tool-result message.
			chatMessages = append(chatMessages, ollamaChatMsg{
				Role:    models.RoleUser,
				Content: "[Tool Result]\n" + msg.Content,
			})
		}
	}
	return chatMessages
}

func (r *RemoteRunner) inputForConversationLocked(
	conversationID string,
	systemContext string,
	messages []models.Message,
) []ollamaChatMsg {
	if conversationID != "" {
		if pending, ok := r.pending[conversationID]; ok {
			if output, matches := matchingOllamaToolResult(messages, pending); matches {
				input := cloneOllamaChatMessages(pending.input)
				input = append(input, cloneOllamaChatMessage(pending.assistant))
				input = append(input, ollamaChatMsg{
					Role:     models.RoleTool,
					Content:  output,
					ToolName: pending.name,
				})
				return input
			}
			// A new or mismatched turn supersedes stale native provider state.
			delete(r.pending, conversationID)
		}
	}
	return buildOllamaChatMessages(systemContext, messages)
}

func matchingOllamaToolResult(messages []models.Message, pending ollamaPendingToolLoop) (string, bool) {
	if len(messages) < 2 {
		return "", false
	}
	assistant := messages[len(messages)-2]
	tool := messages[len(messages)-1]
	if assistant.Role != models.RoleAssistant || tool.Role != models.RoleTool {
		return "", false
	}

	var stored models.LLMResponse
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

func (r *RemoteRunner) acquireConversation(conversationID string) func() {
	if conversationID == "" {
		return func() {}
	}

	r.mu.Lock()
	entry := r.locks[conversationID]
	if entry == nil {
		entry = &ollamaConversationLock{}
		r.locks[conversationID] = entry
	}
	entry.refs++
	r.mu.Unlock()

	entry.mu.Lock()
	return func() {
		entry.mu.Unlock()
		r.mu.Lock()
		entry.refs--
		if entry.refs == 0 {
			delete(r.locks, conversationID)
		}
		r.mu.Unlock()
	}
}

func (r *RemoteRunner) storePending(conversationID string, pending ollamaPendingToolLoop, epoch uint64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.epoch == epoch {
		r.pending[conversationID] = pending
	}
}

func (r *RemoteRunner) clearPending(conversationID string, epoch uint64) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.epoch == epoch {
		delete(r.pending, conversationID)
	}
}

func (r *RemoteRunner) prunePendingLocked(now time.Time) {
	for conversationID, pending := range r.pending {
		if now.Sub(pending.updated) > ollamaPendingStateTTL {
			delete(r.pending, conversationID)
		}
	}
}

func cloneOllamaChatMessages(messages []ollamaChatMsg) []ollamaChatMsg {
	cloned := make([]ollamaChatMsg, len(messages))
	for i, message := range messages {
		cloned[i] = cloneOllamaChatMessage(message)
	}
	return cloned
}

func cloneOllamaChatMessage(message ollamaChatMsg) ollamaChatMsg {
	message.ToolCalls = append([]ollamaToolCall(nil), message.ToolCalls...)
	return message
}

func nativeToolReasoning(thinking, content string) string {
	parts := make([]string, 0, 2)
	if thinking = strings.TrimSpace(thinking); thinking != "" {
		parts = append(parts, thinking)
	}
	if content = strings.TrimSpace(content); content != "" {
		parts = append(parts, content)
	}
	return strings.Join(parts, "\n\n")
}

// normalizeNativeContent makes native chat responses obey the same internal
// JSON contract as native tool calls. This lets Engine distinguish a genuine
// native answer from prompt-based compatibility fallbacks that may need its
// corrective JSON retry.
func normalizeNativeContent(content string) string {
	return normalizeNativeMessage(content, "")
}

// normalizeNativeMessage adds Ollama's native thinking field to Axiom's
// provider-neutral reasoning field while preserving already structured output.
func normalizeNativeMessage(content, thinking string) string {
	visibleContent := content
	reasoningParts := make([]string, 0, 2)
	if thinking = strings.TrimSpace(thinking); thinking != "" {
		reasoningParts = append(reasoningParts, thinking)
	}
	thinkMatches := nativeThinkBlockPattern.FindAllStringSubmatch(content, -1)
	if len(thinkMatches) > 0 {
		for _, match := range thinkMatches {
			if part := strings.TrimSpace(match[1]); part != "" {
				reasoningParts = append(reasoningParts, part)
			}
		}
		visibleContent = strings.TrimSpace(nativeThinkBlockPattern.ReplaceAllString(content, ""))
	}
	reasoning := strings.Join(reasoningParts, "\n\n")

	candidate := strings.TrimSpace(visibleContent)
	if strings.HasPrefix(candidate, "```") {
		lines := strings.Split(candidate, "\n")
		if len(lines) >= 3 && strings.HasPrefix(strings.TrimSpace(lines[len(lines)-1]), "```") {
			candidate = strings.Join(lines[1:len(lines)-1], "\n")
		}
	}
	if start, end := strings.Index(candidate, "{"), strings.LastIndex(candidate, "}"); start >= 0 && end > start {
		candidate = candidate[start : end+1]
	}
	var object map[string]json.RawMessage
	if json.Unmarshal([]byte(candidate), &object) == nil {
		_, hasContent := object["content"]
		_, hasToolCall := object["tool_call"]
		if hasContent && hasToolCall {
			if reasoning == "" {
				return content
			}

			var existingReasoning string
			reasoningJSON, hasReasoning := object["reasoning"]
			if hasReasoning {
				_ = json.Unmarshal(reasoningJSON, &existingReasoning)
			}
			if existingReasoning = strings.TrimSpace(existingReasoning); existingReasoning != "" {
				reasoning += "\n\n" + existingReasoning
			}
			object["reasoning"], _ = json.Marshal(reasoning)
			encoded, _ := json.Marshal(object)
			return string(encoded)
		}
	}
	encoded, _ := json.Marshal(map[string]interface{}{
		"reasoning": reasoning,
		"tool_call": nil,
		"content":   visibleContent,
	})
	return string(encoded)
}

// convertToOllamaTools translates Axiom's ToolDefinition slice into the
// OpenAI-style JSON format that Ollama's /api/chat endpoint expects.
func convertToOllamaTools(defs []models.ToolDefinition) []ollamaTool {
	tools := make([]ollamaTool, 0, len(defs))
	for _, def := range defs {
		tools = append(tools, ollamaTool{
			Type: "function",
			Function: ollamaToolFunction{
				Name:        def.Name,
				Description: def.Description,
				Parameters:  def.CanonicalSchema(),
			},
		})
	}
	return tools
}

// parseArgsSchemaToJSONSchema remains as a compatibility shim for legacy
// package callers. All providers consume ToolDefinition.CanonicalSchema.
func parseArgsSchemaToJSONSchema(schema string) map[string]interface{} {
	encoded, _ := json.Marshal(models.ParseLegacyArgsSchema(schema))
	var result map[string]interface{}
	_ = json.Unmarshal(encoded, &result)
	return result
}

// ── Ollama /api/generate (legacy, no tools) ─────────────────────────────────

type ollamaResp struct {
	Response string `json:"response"`
	Thinking string `json:"thinking,omitempty"`
	Done     bool   `json:"done"`
}

func (r *RemoteRunner) completeOllama(ctx context.Context, prompt string, maxTokens int) (string, error) {
	body, err := json.Marshal(map[string]interface{}{
		"model":   r.model,
		"prompt":  prompt,
		"stream":  false,
		"think":   r.think,
		"format":  "json", // Enforce valid JSON output at the API level
		"options": r.ollamaOptions(maxTokens),
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
	if strings.TrimSpace(result.Thinking) == "" {
		return result.Response, nil
	}
	// Preserve the legacy runner's raw response semantics so Engine can still
	// detect malformed JSON and issue its corrective retry. Engine already
	// understands <think> blocks and maps them into Axiom reasoning.
	return "<think>" + result.Thinking + "</think>" + result.Response, nil
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
