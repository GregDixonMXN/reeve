package cognitive

import (
	"context"
	"fmt"
	"strings"

	"axiom/pkg/models"
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
	lower := strings.ToLower(msg)
	words := strings.Fields(msg)
	wordCount := len(words)

	// Very short messages are almost always simple — keep local
	if wordCount < 6 {
		return RouteLocal
	}

	// Local fast-path: if it starts with a local keyword, skip cloud
	for _, kw := range localKeywords {
		if strings.HasPrefix(lower, kw) {
			return RouteLocal
		}
	}

	// Check for cloud keywords anywhere in the message
	for _, kw := range cloudKeywords {
		if strings.Contains(lower, kw) {
			return RouteCloud
		}
	}

	// Long messages typically imply complex intent — send to cloud
	if wordCount > 30 {
		return RouteCloud
	}

	// Default: handle locally — cheaper and faster
	return RouteLocal
}

// HybridRouter holds references to both runners and routes prompts
// to the appropriate one based on fast heuristic classification.
type HybridRouter struct {
	local LLMRunner // Ollama
	cloud LLMRunner // Claude
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
	runner, decision := hr.router.Route(ctx, prompt)
	_ = decision
	return runner.Complete(ctx, prompt, maxTokens)
}

// CompleteWithTools routes and dispatches with tool support.
// systemContext is the system prompt + repo/tools/memory block;
// messages is the structured conversation history.
func (hr *HybridRunner) CompleteWithTools(ctx context.Context, systemContext string, messages []models.Message, maxTokens int, tools []models.ToolDefinition) (string, error) {
	// Use the last user message as the routing signal.
	routeSignal := systemContext
	if len(messages) > 0 {
		routeSignal = messages[len(messages)-1].Content
	}
	runner, _ := hr.router.Route(ctx, routeSignal)

	if tar, ok := runner.(ToolAwareRunner); ok && len(tools) > 0 {
		return tar.CompleteWithTools(ctx, systemContext, messages, maxTokens, tools)
	}
	// Fallback: flatten to a single prompt
	prompt := systemContext + "\n"
	for _, msg := range messages {
		prompt += fmt.Sprintf("[%s]: %s\n", msg.Role, msg.Content)
	}
	return runner.Complete(ctx, prompt, maxTokens)
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
