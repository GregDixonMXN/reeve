package cognitive

import (
	"context"
	"fmt"
	"strings"
	"unicode"

	"herald/pkg/models"
)

const classifierPrompt = `You are a task classifier. Analyze the user's request and output EXACTLY one word.

Output LOCAL if the task is:
- Reading, listing, or searching files
- Basic math or simple questions
- Running a single command or script
- Small edits to one file
- System info or status checks
- Simple web searches or lookups

Output CLOUD if the task is:
- Architecting or refactoring across multiple files
- Complex code generation (new features, full implementations)
- Multi-step reasoning or planning
- Writing tests, documentation, or boilerplate for a whole module
- Debugging subtle logic errors across a codebase
- Creative writing or nuanced analysis

Output ONLY the word LOCAL or CLOUD. Nothing else.`

// RouteDecision indicates which runner should handle the task.
type RouteDecision int

const (
	RouteLocal RouteDecision = iota
	RouteCloud
)

func (d RouteDecision) String() string {
	if d == RouteCloud {
		return "CLOUD"
	}
	return "LOCAL"
}

// cloudKeywords are strong signals that a task needs cloud-level reasoning.
var cloudKeywords = []string{
	"architect", "refactor", "redesign", "implement", "build", "create",
	"analyze", "explain", "review", "debug", "test", "write", "generate",
	"multiple files", "codebase", "complex", "plan", "strategy", "design",
	"optimize", "improve", "fix", "rewrite", "integrate", "migrate",
	"documentation", "boilerplate", "scaffold", "structure", "pipeline",
}

// localKeywords are fast-path signals that don't need cloud.
var localKeywords = []string{
	"list", "show", "what is", "status", "version", "ls", "cat",
	"read", "open", "print", "display", "check", "get", "find",
}

// classifyByHeuristic routes tasks using fast keyword and length heuristics.
// This replaces the old approach of calling the local LLM just to classify —
// that approach added a full inference round-trip (up to 5 seconds) before
// every request. Heuristic classification is ~0ms.
func classifyByHeuristic(msg string) RouteDecision {
	normalized := normalizeIntent(msg)
	words := strings.Fields(normalized)
	wordCount := len(words)

	// Strong cloud intent wins even for concise requests such as
	// "analyze the herald project".
	for _, kw := range cloudKeywords {
		if containsIntentKeyword(normalized, kw) {
			return RouteCloud
		}
	}

	// Very short messages without a cloud signal are usually simple.
	if wordCount < 6 {
		return RouteLocal
	}

	// Local fast-path: if it starts with a local keyword, skip cloud.
	for _, kw := range localKeywords {
		if normalized == kw || strings.HasPrefix(normalized, kw+" ") {
			return RouteLocal
		}
	}

	// Long messages typically imply complex intent — send to cloud
	if wordCount > 30 {
		return RouteCloud
	}

	// Default: handle locally — cheaper and faster
	return RouteLocal
}

func normalizeIntent(msg string) string {
	words := strings.FieldsFunc(strings.ToLower(msg), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_'
	})
	return strings.Join(words, " ")
}

func containsIntentKeyword(normalized, keyword string) bool {
	return strings.Contains(" "+normalized+" ", " "+keyword+" ")
}

// HybridRouter holds references to both runners and routes prompts
// to the appropriate one based on fast heuristic classification.
type HybridRouter struct {
	local LLMRunner
	cloud LLMRunner
}

// HybridRouterConfig configures the hybrid router.
type HybridRouterConfig struct {
	LocalRunner LLMRunner
	CloudRunner LLMRunner
	// ClassifyTimeout is kept for backwards compat but unused — classification
	// is now heuristic-only (no LLM call, no timeout needed).
	ClassifyTimeout interface{}
}

// NewHybridRouter creates a router that classifies tasks via heuristic before dispatching.
func NewHybridRouter(cfg HybridRouterConfig) *HybridRouter {
	return &HybridRouter{
		local: cfg.LocalRunner,
		cloud: cfg.CloudRunner,
	}
}

// Classify returns a RouteDecision using a fast heuristic — no LLM inference.
func (h *HybridRouter) Classify(_ context.Context, userMessage string) RouteDecision {
	if h.cloud == nil {
		return RouteLocal
	}
	return classifyByHeuristic(userMessage)
}

// Route classifies the task and returns the appropriate runner + the decision.
func (h *HybridRouter) Route(ctx context.Context, userMessage string) (LLMRunner, RouteDecision) {
	decision := h.Classify(ctx, userMessage)

	switch decision {
	case RouteCloud:
		if h.cloud != nil {
			return h.cloud, RouteCloud
		}
		return h.local, RouteLocal
	default:
		return h.local, RouteLocal
	}
}

// ── HybridRunner implements LLMRunner so it can be set on the Engine ────────

// HybridRunner wraps HybridRouter to satisfy the LLMRunner interface.
// When the engine calls Complete(), it classifies and dispatches automatically.
type HybridRunner struct {
	router *HybridRouter
}

// NewHybridRunner creates a LLMRunner that auto-routes between local and cloud.
func NewHybridRunner(router *HybridRouter) *HybridRunner {
	return &HybridRunner{router: router}
}

// Complete classifies the prompt, picks the right runner, and executes.
func (hr *HybridRunner) Complete(ctx context.Context, prompt string, maxTokens int) (string, error) {
	runner, _ := hr.router.Route(ctx, latestUserIntentFromPrompt(prompt))
	return runner.Complete(ctx, prompt, maxTokens)
}

// CompleteWithTools routes and dispatches with tool support.
// systemContext is the system prompt plus repo and memory context;
// messages is the structured conversation history.
func (hr *HybridRunner) CompleteWithTools(ctx context.Context, systemContext string, messages []models.Message, maxTokens int, tools []models.ToolDefinition) (string, error) {
	return hr.completeWithToolsForConversation(ctx, "", systemContext, messages, maxTokens, tools)
}

// CompleteWithToolsForConversation preserves provider-native tool state when a
// hybrid route selects a runner such as OpenAI Responses. Without forwarding
// the conversation key, a function result would lose its call_id linkage on
// the next iteration.
func (hr *HybridRunner) CompleteWithToolsForConversation(
	ctx context.Context,
	conversationID string,
	systemContext string,
	messages []models.Message,
	maxTokens int,
	tools []models.ToolDefinition,
) (string, error) {
	return hr.completeWithToolsForConversation(ctx, conversationID, systemContext, messages, maxTokens, tools)
}

func (hr *HybridRunner) completeWithToolsForConversation(
	ctx context.Context,
	conversationID string,
	systemContext string,
	messages []models.Message,
	maxTokens int,
	tools []models.ToolDefinition,
) (string, error) {
	// Route on user intent, not on the most recent tool result. This keeps one
	// agent turn on the same brain throughout its Think→Tool→Result cycle.
	routeSignal := latestUserIntent(messages)
	if routeSignal == "" {
		routeSignal = systemContext
	}
	runner, decision := hr.router.Route(ctx, routeSignal)

	effectiveSystem := systemContext
	effectiveTools := tools
	if decision == RouteCloud {
		// The cloud model is already the active brain for this turn. Hiding the
		// delegator prevents a redundant cloud→cloud tool call (and possible loop).
		effectiveTools = withoutTool(tools, "ask_cloud_model")
		effectiveSystem = withoutToolListing(systemContext, tools, "ask_cloud_model")
		effectiveSystem = cloudRouteSystemRules(effectiveSystem)
		effectiveSystem += "\nHYBRID ROUTING: This turn is already running on the cloud model. " +
			"Do not call ask_cloud_model; use the available tools directly.\n"
	}

	if ctar, ok := runner.(ConversationToolAwareRunner); ok && strings.TrimSpace(conversationID) != "" {
		return ctar.CompleteWithToolsForConversation(
			ctx,
			conversationID,
			effectiveSystem,
			messages,
			maxTokens,
			effectiveTools,
		)
	}
	if tar, ok := runner.(ToolAwareRunner); ok {
		return tar.CompleteWithTools(ctx, effectiveSystem, messages, maxTokens, effectiveTools)
	}
	// Fallback: flatten to a single prompt
	effectiveSystem = appendTextualToolListing(effectiveSystem, effectiveTools)
	prompt := effectiveSystem + "\n"
	for _, msg := range messages {
		prompt += fmt.Sprintf("[%s]: %s\n", msg.Role, msg.Content)
	}
	return runner.Complete(ctx, prompt, maxTokens)
}

// latestUserIntent returns the newest actual user turn. Tool output and
// assistant messages are deliberately ignored because they describe execution
// state, not the task that should determine local-vs-cloud routing.
func latestUserIntent(messages []models.Message) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role == models.RoleUser {
			return stripInjectedUserContext(messages[i].Content)
		}
	}
	return ""
}

// latestUserIntentFromPrompt recovers the latest user turn from the flattened
// prompt format used for no-tool planning/reflection requests. A JSON-format
// correction appended by Engine.Generate is skipped so a retry keeps the same
// routing decision as the original request.
func latestUserIntentFromPrompt(prompt string) string {
	const (
		userMarker = "<|user|>\n"
		endMarker  = "\n<|end|>"
	)

	searchEnd := len(prompt)
	for searchEnd > 0 {
		markerAt := strings.LastIndex(prompt[:searchEnd], userMarker)
		if markerAt < 0 {
			break
		}
		contentStart := markerAt + len(userMarker)
		contentEnd := strings.Index(prompt[contentStart:], endMarker)
		if contentEnd < 0 {
			contentEnd = len(prompt) - contentStart
		}
		candidate := strings.TrimSpace(prompt[contentStart : contentStart+contentEnd])
		if !strings.HasPrefix(candidate, "Your last response was not valid JSON.") {
			return stripInjectedUserContext(candidate)
		}
		searchEnd = markerAt
	}

	return strings.TrimSpace(prompt)
}

func stripInjectedUserContext(content string) string {
	markers := []string{
		"\n\n[TASK PLAN — EXECUTE THIS EXACTLY]",
		"\n[PROJECT CONTEXT:",
		"\n[GIT STATUS:",
		"\n\n[SYSTEM MEMORY RECALL]",
	}
	end := len(content)
	for _, marker := range markers {
		if at := strings.Index(content, marker); at >= 0 && at < end {
			end = at
		}
	}
	return strings.TrimSpace(content[:end])
}

func withoutTool(tools []models.ToolDefinition, name string) []models.ToolDefinition {
	filtered := make([]models.ToolDefinition, 0, len(tools))
	for _, tool := range tools {
		if tool.Name != name {
			filtered = append(filtered, tool)
		}
	}
	return filtered
}

func withoutToolListing(systemContext string, tools []models.ToolDefinition, name string) string {
	for _, tool := range tools {
		if tool.Name != name {
			continue
		}
		for _, schema := range []string{tool.SchemaJSON(), tool.ArgsSchema} {
			if schema == "" {
				continue
			}
			listing := fmt.Sprintf("- %s: %s\n  Args: %s\n", tool.Name, tool.Description, schema)
			updated := strings.Replace(systemContext, listing, "", 1)
			if updated != systemContext {
				return updated
			}
		}
		return systemContext
	}
	return systemContext
}

func cloudRouteSystemRules(systemContext string) string {
	return strings.Replace(systemContext, modeRule2("hybrid"), modeRule2("cloud"), 1)
}

// Unload releases both runners.
func (hr *HybridRunner) Unload() error {
	var errs []string
	if err := hr.router.local.Unload(); err != nil {
		errs = append(errs, fmt.Sprintf("local: %v", err))
	}
	if hr.router.cloud != nil {
		if err := hr.router.cloud.Unload(); err != nil {
			errs = append(errs, fmt.Sprintf("cloud: %v", err))
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf("unload errors: %s", strings.Join(errs, "; "))
	}
	return nil
}
