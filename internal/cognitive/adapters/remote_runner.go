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
	// Tokenization varies by model and content. Two bytes per token deliberately
	// overestimates English source code and JSON-heavy native tool schemas.
	ollamaEstimatedBytesPerToken = 2
	// Leave room for chat-template variance and tokenizer estimation error in
	// addition to the explicit generation reserve.
	ollamaPromptSafetyTokens = 768
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
	Role                string           `json:"role"`
	Content             string           `json:"content"`
	Thinking            string           `json:"thinking,omitempty"`
	ToolCalls           []ollamaToolCall `json:"tool_calls,omitempty"`
	ToolName            string           `json:"tool_name,omitempty"`
	SyntheticToolResult bool             `json:"-"`
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
	Message         ollamaChatResponseMessage `json:"message"`
	Done            bool                      `json:"done"`
	DoneReason      string                    `json:"done_reason,omitempty"`
	PromptEvalCount int                       `json:"prompt_eval_count,omitempty"`
	EvalCount       int                       `json:"eval_count,omitempty"`
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
	compactedMessages, err := compactOllamaChatMessages(
		chatMessages,
		ollamaTools,
		r.contextSize,
		maxTokens,
	)
	if err != nil {
		return "", err
	}
	chatMessages = compactedMessages

	result, err := r.requestOllamaChat(ctx, chatMessages, maxTokens, ollamaTools, r.think)
	if err != nil {
		return "", err
	}
	nativeMessage := result.Message.nativeMessage()

	// Thinking-capable models can consume the entire generation allowance
	// without ever reaching assistant content or a tool call. Retry once with
	// thinking disabled and an explicit recovery instruction. The request is
	// side-effect free until a tool call is returned, so this cannot duplicate
	// tool execution.
	if len(nativeMessage.ToolCalls) == 0 &&
		strings.TrimSpace(nativeMessage.Content) == "" &&
		strings.EqualFold(strings.TrimSpace(result.DoneReason), "length") {
		chatMessages = appendOllamaBudgetRecovery(chatMessages)
		chatMessages, err = compactOllamaChatMessages(
			chatMessages,
			ollamaTools,
			r.contextSize,
			maxTokens,
		)
		if err != nil {
			return "", fmt.Errorf("ollama output-budget recovery could not fit its prompt: %w", err)
		}
		result, err = r.requestOllamaChat(ctx, chatMessages, maxTokens, ollamaTools, false)
		if err != nil {
			return "", fmt.Errorf("ollama output-budget recovery failed: %w", err)
		}
		nativeMessage = result.Message.nativeMessage()
	}

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
		if strings.EqualFold(strings.TrimSpace(result.DoneReason), "length") {
			return "", fmt.Errorf(
				"ollama exhausted its output budget after %d generated tokens without assistant content or a function call; increase model.max_output_tokens or disable model.enable_thinking",
				result.EvalCount,
			)
		}
		if strings.TrimSpace(nativeMessage.Thinking) != "" {
			return "", fmt.Errorf(
				"ollama returned reasoning but no assistant content or function call (done_reason=%q)",
				result.DoneReason,
			)
		}
		return "", fmt.Errorf(
			"ollama response contained no assistant content or function call (done_reason=%q)",
			result.DoneReason,
		)
	}
	return normalizeNativeMessage(nativeMessage.Content, nativeMessage.Thinking), nil
}

func (r *RemoteRunner) requestOllamaChat(
	ctx context.Context,
	chatMessages []ollamaChatMsg,
	maxTokens int,
	ollamaTools []ollamaTool,
	think bool,
) (ollamaChatResp, error) {
	body, err := json.Marshal(ollamaChatReq{
		Model:    r.model,
		Messages: chatMessages,
		Stream:   false,
		Think:    &think,
		Tools:    ollamaTools,
		Options:  r.ollamaOptions(maxTokens),
	})
	if err != nil {
		return ollamaChatResp{}, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", r.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return ollamaChatResp{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := r.client.Do(req)
	if err != nil {
		return ollamaChatResp{}, fmt.Errorf("ollama unreachable at %s — is it running? %w", r.baseURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return ollamaChatResp{}, fmt.Errorf("ollama HTTP %d: %s", resp.StatusCode, string(b))
	}

	var result ollamaChatResp
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return ollamaChatResp{}, err
	}
	return result, nil
}

func appendOllamaBudgetRecovery(messages []ollamaChatMsg) []ollamaChatMsg {
	recovered := cloneOllamaChatMessages(messages)
	const instruction = "\n\nOUTPUT BUDGET RECOVERY: Your previous attempt ended before producing " +
		"assistant content or a function call. Do not narrate internal reasoning. " +
		"Respond immediately with the next function call, or a concise final answer."
	if len(recovered) > 0 && recovered[0].Role == models.RoleSystem {
		recovered[0].Content += instruction
		return recovered
	}
	return append(
		[]ollamaChatMsg{{Role: models.RoleSystem, Content: strings.TrimSpace(instruction)}},
		recovered...,
	)
}

// compactOllamaChatMessages bounds the complete native chat payload rather
// than merely counting turns. It reserves maxTokens for generation, accounts
// for native tool schemas, preserves the latest user turn or native
// assistant/tool pair, and stores only this compacted input for pending replay.
func compactOllamaChatMessages(
	messages []ollamaChatMsg,
	tools []ollamaTool,
	contextSize int,
	maxTokens int,
) ([]ollamaChatMsg, error) {
	if contextSize <= 0 || len(messages) == 0 {
		return messages, nil
	}

	generationReserve := maxTokens
	if generationReserve <= 0 {
		generationReserve = min(1024, max(1, contextSize/4))
	}
	promptBudget := contextSize - generationReserve - ollamaPromptSafetyTokens
	if promptBudget < 256 {
		return nil, fmt.Errorf(
			"ollama context_size %d cannot reserve the configured %d output tokens plus prompt safety margin; lower model.max_output_tokens or increase model.context_size",
			contextSize,
			generationReserve,
		)
	}
	messageBudget := promptBudget - estimateOllamaToolTokens(tools)
	if messageBudget < 256 {
		return nil, fmt.Errorf(
			"ollama context_size %d is too small for the native tool schemas while reserving %d output tokens; increase model.context_size",
			contextSize,
			generationReserve,
		)
	}
	groupStart := 0
	if messages[0].Role == models.RoleSystem {
		groupStart = 1
	}
	_, malformedProtocol := groupOllamaMessages(messages[groupStart:])
	if estimateOllamaMessagesTokens(messages) <= messageBudget && !malformedProtocol {
		return messages, nil
	}

	start := 0
	compacted := make([]ollamaChatMsg, 0, len(messages))
	used := 128 // chat-template prefix/suffix and generation marker
	if messages[0].Role == models.RoleSystem {
		system := cloneOllamaChatMessage(messages[0])
		systemCost := estimateOllamaMessageTokens(system)
		if used+systemCost > messageBudget {
			return nil, fmt.Errorf(
				"ollama context_size %d cannot fit Axiom's system context and native tool schemas while reserving %d output tokens; increase model.context_size",
				contextSize,
				generationReserve,
			)
		}
		compacted = append(compacted, system)
		used += systemCost
		start = 1
	}

	const omissionText = "[CONTEXT COMPACTED: older messages were omitted to reserve space for the response.]"
	omissionCost := estimateOllamaMessageTokens(ollamaChatMsg{
		Role:    models.RoleSystem,
		Content: omissionText,
	})
	groups, hadOrphan := groupOllamaMessages(messages[start:])
	if len(groups) == 0 {
		if hadOrphan {
			compacted = append(compacted, ollamaChatMsg{
				Role:    models.RoleSystem,
				Content: omissionText,
			})
		}
		return compacted, nil
	}
	available := messageBudget - used - omissionCost
	if available < 64 {
		return nil, fmt.Errorf(
			"ollama context_size %d leaves no room for the latest user request after system context and tool schemas; increase model.context_size",
			contextSize,
		)
	}

	latestUser := -1
	latestPair := -1
	for index, group := range groups {
		if group.nativeToolPair {
			latestPair = index
		}
		for _, message := range group.messages {
			if isGenuineOllamaUserMessage(message) {
				latestUser = index
			}
		}
	}

	pinned := make([]int, 0, 3)
	addPinned := func(index int) {
		if index < 0 {
			return
		}
		for _, existing := range pinned {
			if existing == index {
				return
			}
		}
		pinned = append(pinned, index)
	}
	// Preserve the task before its active tool protocol, then the newest exact
	// native pair. If neither is last, retain the immediate final turn too.
	addPinned(latestUser)
	addPinned(latestPair)
	addPinned(len(groups) - 1)

	selected := make(map[int][]ollamaChatMsg, len(groups))
	for _, index := range pinned {
		group := groups[index]
		cost := estimateOllamaMessagesTokensWithoutTemplate(group.messages)
		if index == latestUser && cost > available {
			return nil, fmt.Errorf(
				"latest user request needs an estimated %d tokens but only %d remain in Ollama context_size %d; shorten the request or increase model.context_size",
				cost,
				available,
				contextSize,
			)
		}
		fitted, fitCost, err := fitMandatoryOllamaGroup(group, available)
		if err != nil {
			return nil, fmt.Errorf(
				"latest native tool turn cannot fit Ollama context_size %d: %w",
				contextSize,
				err,
			)
		}
		selected[index] = fitted
		available -= fitCost
	}

	// Fill newest-first using complete groups. Stop at the first group that
	// does not fit so the retained history remains a contiguous recent window.
	for index := len(groups) - 1; index >= 0; index-- {
		if _, exists := selected[index]; exists {
			continue
		}
		cost := estimateOllamaMessagesTokensWithoutTemplate(groups[index].messages)
		if cost > available {
			break
		}
		selected[index] = cloneOllamaChatMessages(groups[index].messages)
		available -= cost
	}

	omitted := hadOrphan || len(selected) < len(groups)
	if omitted {
		compacted = append(compacted, ollamaChatMsg{
			Role:    models.RoleSystem,
			Content: omissionText,
		})
	}
	for index := range groups {
		if selectedGroup, exists := selected[index]; exists {
			compacted = append(compacted, selectedGroup...)
		}
	}
	if estimateOllamaMessagesTokens(compacted) > messageBudget {
		return nil, fmt.Errorf(
			"internal Ollama context compaction estimate %d exceeds message budget %d",
			estimateOllamaMessagesTokens(compacted),
			messageBudget,
		)
	}
	return compacted, nil
}

type ollamaMessageGroup struct {
	messages       []ollamaChatMsg
	nativeToolPair bool
}

func groupOllamaMessages(messages []ollamaChatMsg) ([]ollamaMessageGroup, bool) {
	groups := make([]ollamaMessageGroup, 0, len(messages))
	hadOrphan := false
	for index := 0; index < len(messages); index++ {
		message := messages[index]
		if message.Role == models.RoleAssistant && len(message.ToolCalls) > 0 {
			if index+1 < len(messages) && messages[index+1].Role == models.RoleTool {
				groups = append(groups, ollamaMessageGroup{
					messages:       cloneOllamaChatMessages(messages[index : index+2]),
					nativeToolPair: true,
				})
				index++
				continue
			}
			hadOrphan = true
			continue
		}
		if message.Role == models.RoleTool {
			// Never send a native tool result without its assistant tool call.
			hadOrphan = true
			continue
		}
		groups = append(groups, ollamaMessageGroup{
			messages: []ollamaChatMsg{cloneOllamaChatMessage(message)},
		})
	}
	return groups, hadOrphan
}

func isGenuineOllamaUserMessage(message ollamaChatMsg) bool {
	return message.Role == models.RoleUser && !message.SyntheticToolResult
}

func fitMandatoryOllamaGroup(
	group ollamaMessageGroup,
	tokenBudget int,
) ([]ollamaChatMsg, int, error) {
	full := cloneOllamaChatMessages(group.messages)
	fullCost := estimateOllamaMessagesTokensWithoutTemplate(full)
	if fullCost <= tokenBudget {
		return full, fullCost, nil
	}
	if !group.nativeToolPair {
		if len(full) == 1 {
			fitted := []ollamaChatMsg{
				truncateOllamaMessage(full[0], tokenBudget),
			}
			cost := estimateOllamaMessagesTokensWithoutTemplate(fitted)
			if cost <= tokenBudget {
				return fitted, cost, nil
			}
		}
		return nil, 0, fmt.Errorf(
			"mandatory message needs an estimated %d tokens, only %d remain",
			fullCost,
			tokenBudget,
		)
	}
	if len(full) != 2 {
		return nil, 0, fmt.Errorf("native tool group has %d messages, want 2", len(full))
	}

	assistant := full[0]
	toolResult := full[1]
	minAssistant := assistant
	minAssistant.Content = ""
	minAssistant.Thinking = ""
	minTool := toolResult
	minTool.Content = ""
	minimum := estimateOllamaMessagesTokensWithoutTemplate(
		[]ollamaChatMsg{minAssistant, minTool},
	)
	if minimum > tokenBudget {
		return nil, 0, fmt.Errorf(
			"untruncated function name and arguments need an estimated %d tokens, only %d remain",
			minimum,
			tokenBudget,
		)
	}

	remaining := tokenBudget - minimum
	assistantExtra := min(remaining/4, max(0, estimateOllamaMessageTokens(assistant)-estimateOllamaMessageTokens(minAssistant)))
	assistantBudget := estimateOllamaMessageTokens(minAssistant) + assistantExtra
	toolBudget := tokenBudget - assistantBudget
	assistant = truncateOllamaMessage(assistant, assistantBudget)
	toolResult = truncateOllamaMessage(toolResult, toolBudget)
	fitted := []ollamaChatMsg{assistant, toolResult}
	cost := estimateOllamaMessagesTokensWithoutTemplate(fitted)
	if cost > tokenBudget {
		return nil, 0, fmt.Errorf(
			"compacted native pair estimate %d exceeds its %d-token budget",
			cost,
			tokenBudget,
		)
	}
	return fitted, cost, nil
}

func estimateOllamaToolTokens(tools []ollamaTool) int {
	if len(tools) == 0 {
		return 0
	}
	encoded, err := json.Marshal(tools)
	if err != nil {
		return len(tools) * 128
	}
	return estimateOllamaBytesTokens(len(encoded)) + 32 + len(tools)*8
}

func estimateOllamaMessagesTokens(messages []ollamaChatMsg) int {
	total := 128 // chat-template prefix/suffix and generation marker
	return total + estimateOllamaMessagesTokensWithoutTemplate(messages)
}

func estimateOllamaMessagesTokensWithoutTemplate(messages []ollamaChatMsg) int {
	total := 0
	for _, message := range messages {
		total += estimateOllamaMessageTokens(message)
	}
	return total
}

func estimateOllamaMessageTokens(message ollamaChatMsg) int {
	size := len(message.Content) + len(message.Thinking) + len(message.ToolName)
	if len(message.ToolCalls) > 0 {
		if encoded, err := json.Marshal(message.ToolCalls); err == nil {
			size += len(encoded)
		}
	}
	return 12 + estimateOllamaBytesTokens(size)
}

func estimateOllamaBytesTokens(size int) int {
	return (size + ollamaEstimatedBytesPerToken - 1) / ollamaEstimatedBytesPerToken
}

func truncateOllamaMessage(message ollamaChatMsg, tokenBudget int) ollamaChatMsg {
	fixedSize := len(message.ToolName)
	if len(message.ToolCalls) > 0 {
		if encoded, err := json.Marshal(message.ToolCalls); err == nil {
			fixedSize += len(encoded)
		}
	}
	contentBytes := max(
		0,
		(tokenBudget-12)*ollamaEstimatedBytesPerToken-fixedSize,
	)
	if len(message.Thinking) > 0 {
		thinkingBytes := min(len(message.Thinking), contentBytes/3)
		message.Thinking = compactUTF8(message.Thinking, thinkingBytes)
		contentBytes -= len(message.Thinking)
	}
	message.Content = compactUTF8(message.Content, contentBytes)
	return message
}

func compactUTF8(content string, maxBytes int) string {
	if maxBytes <= 0 {
		return ""
	}
	if len(content) <= maxBytes {
		return content
	}
	const marker = "\n... [content compacted] ...\n"
	if maxBytes <= len(marker)+8 {
		return validUTF8Prefix(content, maxBytes)
	}
	available := maxBytes - len(marker)
	headBytes := available * 2 / 3
	tailBytes := available - headBytes
	head := validUTF8Prefix(content, headBytes)
	tail := validUTF8Suffix(content, tailBytes)
	return head + marker + tail
}

func validUTF8Prefix(content string, maxBytes int) string {
	if maxBytes >= len(content) {
		return content
	}
	end := max(0, maxBytes)
	for end > 0 && end < len(content) && (content[end]&0xc0) == 0x80 {
		end--
	}
	return content[:end]
}

func validUTF8Suffix(content string, maxBytes int) string {
	if maxBytes >= len(content) {
		return content
	}
	start := max(0, len(content)-maxBytes)
	for start < len(content) && (content[start]&0xc0) == 0x80 {
		start++
	}
	return content[start:]
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
				Role:                models.RoleUser,
				Content:             "[Tool Result]\n" + msg.Content,
				SyntheticToolResult: true,
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
	if len(message.ToolCalls) == 0 {
		return message
	}
	cloned := make([]ollamaToolCall, len(message.ToolCalls))
	for index, call := range message.ToolCalls {
		cloned[index] = call
		if call.Function.Index != nil {
			value := *call.Function.Index
			cloned[index].Function.Index = &value
		}
		if call.Function.Arguments != nil {
			cloned[index].Function.Arguments = cloneJSONMap(call.Function.Arguments)
		}
	}
	message.ToolCalls = cloned
	return message
}

func cloneJSONMap(input map[string]interface{}) map[string]interface{} {
	output := make(map[string]interface{}, len(input))
	for key, value := range input {
		output[key] = cloneJSONValue(value)
	}
	return output
}

func cloneJSONValue(value interface{}) interface{} {
	switch typed := value.(type) {
	case map[string]interface{}:
		return cloneJSONMap(typed)
	case []interface{}:
		cloned := make([]interface{}, len(typed))
		for index, item := range typed {
			cloned[index] = cloneJSONValue(item)
		}
		return cloned
	default:
		return value
	}
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
