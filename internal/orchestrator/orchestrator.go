package orchestrator

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"axiom/internal/cognitive"
	"axiom/internal/guardrail"
	"axiom/internal/memory"
	"axiom/internal/tools"
	"axiom/pkg/logger"
	"axiom/pkg/models"
)

const MaxToolIterations = 10

// TaskCompleteToken is the sentinel the LLM emits to break the agent loop early.
const TaskCompleteToken = "<TASK_COMPLETE>"

// maxContextMessages is the sliding-window ceiling for conversation history
// passed to the LLM. Older messages beyond this are dropped to prevent
// context overflow and keep inference fast.
const maxContextMessages = 40

// maxToolResultBytes is the max size of a single tool result injected into
// the context. Larger results are truncated with a notice.
const maxToolResultBytes = 3000

// complexityThreshold is the minimum word count before planning is attempted.
const complexityThreshold = 10

// LoopEventKind describes what kind of progress event occurred.
type LoopEventKind string

const (
	LoopEventThinking   LoopEventKind = "thinking"   // LLM reasoning step started
	LoopEventToolCall   LoopEventKind = "tool_call"   // LLM wants to call a tool
	LoopEventToolResult LoopEventKind = "tool_result" // tool returned a result
	LoopEventBlocked    LoopEventKind = "blocked"     // guardrail blocked a tool call
	LoopEventDone       LoopEventKind = "done"        // loop completed (clean)
	LoopEventMaxIter    LoopEventKind = "max_iter"    // loop hit iteration ceiling
	LoopEventError      LoopEventKind = "error"       // unrecoverable error
)

// LoopEvent is a real-time progress update emitted during AgentLoop execution.
type LoopEvent struct {
	Kind      LoopEventKind `json:"kind"`
	Iteration int           `json:"iteration"`
	Message   string        `json:"message"`
	ToolName  string        `json:"tool_name,omitempty"`
}

// LoopProgressFn is called after each meaningful step in the agent loop.
// Use it to stream events to the frontend via Wails runtime.EventsEmit.
type LoopProgressFn func(event LoopEvent)

type Config struct {
	Logger      *logger.Logger
	Memory      *memory.Store
	Cognitive   *cognitive.Engine
	Tools       *tools.Registry
	Guardrail   *guardrail.Guard
	ModeManager *ModeManager
}

type Orchestrator struct {
	ctx         context.Context
	cancel      context.CancelFunc
	log         *logger.Logger
	memory      *memory.Store
	cognitive   *cognitive.Engine
	tools       *tools.Registry
	guardrail   *guardrail.Guard
	modeManager *ModeManager

	mu            sync.RWMutex
	conversations map[string]*models.Conversation
}

func New(cfg Config) *Orchestrator {
	return &Orchestrator{
		log:           cfg.Logger,
		memory:        cfg.Memory,
		cognitive:     cfg.Cognitive,
		tools:         cfg.Tools,
		guardrail:     cfg.Guardrail,
		modeManager:   cfg.ModeManager,
		conversations: make(map[string]*models.Conversation),
	}
}

func (o *Orchestrator) SetContext(ctx context.Context) {
	o.ctx, o.cancel = context.WithCancel(ctx)
}

func (o *Orchestrator) Start() {
	o.log.Info("Orchestrator online — Think-Verify-Act loop ready")

	// Report memory stats on startup
	totalMem, _ := o.memory.MemoryCount()
	vecMem, _ := o.memory.VectorCount()
	o.log.Info("Memory loaded: %d total memories, %d with vectors", totalMem, vecMem)
}

func (o *Orchestrator) Shutdown() {
	o.log.Info("Orchestrator shutting down...")
	if o.cancel != nil {
		o.cancel()
	}
}

// SendMessage runs the full Think-Verify-Act loop with semantic memory injection.
func (o *Orchestrator) SendMessage(conversationID, userMessage string) (*models.AgentResponse, error) {
	start := time.Now()
	o.log.Debug("Ingest: [%s] %s", conversationID, userMessage)

	conv := o.getOrCreateConversation(conversationID)
	conv.AddMessage(models.RoleUser, userMessage)

	// ── RECALL: Semantic memory search ──────────────────────────────────
	memories, err := o.memory.Search(o.ctx, userMessage, 5)
	if err != nil {
		o.log.Warn("Memory recall failed: %v", err)
	}
	if len(memories) > 0 {
		o.log.Debug("Recalled %d memories (top score: %.3f)", len(memories), memories[0].Score)
	}

	// Pre-build memory injection once — avoids rebuilding the string every iteration
	memInjection := buildMemoryInjection(memories)

	// ── THINK-VERIFY-ACT LOOP ───────────────────────────────────────────
	var finalResponse *cognitive.Response
	var toolsUsedThisTurn []string

	for i := 0; i < MaxToolIterations; i++ {
		// Snapshot current messages (trimmed for context window safety)
		rawMsgs := make([]models.Message, len(conv.Messages))
		copy(rawMsgs, conv.Messages)
		llmMessages := trimContextMessages(rawMsgs)

		// Inject memory into the last user message on the first iteration only
		if i == 0 && memInjection != "" {
			for j := len(llmMessages) - 1; j >= 0; j-- {
				if llmMessages[j].Role == models.RoleUser {
					llmMessages[j].Content += memInjection
					break
				}
			}
		}

		resp, err := o.cognitive.Generate(o.ctx, cognitive.Request{
			Messages:      llmMessages,
			MemoryContext: memories,
			Tools:         o.tools.Definitions(),
		})
		if err != nil {
			return nil, fmt.Errorf("cognitive error: %w", err)
		}

		if resp.ToolCall == nil {
			finalResponse = resp
			break
		}

		o.log.Debug("Tool call: %s(%v)", resp.ToolCall.Name, resp.ToolCall.Args)
		toolsUsedThisTurn = append(toolsUsedThisTurn, resp.ToolCall.Name)

		assistantIntent := resp.Content
		if assistantIntent == "" {
			argsBytes, _ := json.Marshal(resp.ToolCall.Args)
			cleanReasoning := fmt.Sprintf("%q", resp.Reasoning)
			assistantIntent = fmt.Sprintf(`{"reasoning": %s, "tool_call": {"name": "%s", "args": %s}, "content": ""}`,
				cleanReasoning, resp.ToolCall.Name, string(argsBytes))
		}
		conv.AddMessage(models.RoleAssistant, assistantIntent)

		if err := o.guardrail.Check(resp.ToolCall); err != nil {
			o.log.Warn("Guardrail blocked: %v", err)
			conv.AddMessage(models.RoleTool, fmt.Sprintf("[BLOCKED] %s: %s", resp.ToolCall.Name, err))
			continue
		}

		result, err := o.tools.Execute(o.ctx, resp.ToolCall)
		if err != nil {
			conv.AddMessage(models.RoleTool, fmt.Sprintf("[ERROR] %s: %s", resp.ToolCall.Name, err))
			continue
		}

		formattedResult := fmt.Sprintf("[TOOL RESULT]\n%s\n[END TOOL RESULT]", result)
		conv.AddMessage(models.RoleTool, formattedResult)
	}

	if finalResponse == nil {
		return nil, fmt.Errorf("max tool iterations (%d) exceeded", MaxToolIterations)
	}

	if finalResponse.Content == "" {
		finalResponse.Content = "Action completed successfully."
	}

	// ── PERSIST: Store exchange in long-term memory ─────────────────────
	conv.AddMessage(models.RoleAssistant, finalResponse.Content)
	if err := o.memory.Store(o.ctx, userMessage, finalResponse.Content); err != nil {
		o.log.Warn("Memory persist failed: %v", err)
	}

	elapsed := time.Since(start)
	o.log.Info("Response in %s | %d memories recalled", elapsed, len(memories))

	return &models.AgentResponse{
		Content:        finalResponse.Content,
		ToolsUsed:      toolsUsedThisTurn,
		MemoryRecalled: len(memories),
		LatencyMs:      elapsed.Milliseconds(),
		Reasoning:      finalResponse.Reasoning,
	}, nil
}

// AgentLoop is the autonomous execution engine. It takes a single user prompt,
// feeds it to the HybridEngine, and continues the Think→Tool→Result cycle
// automatically until one of three conditions is met:
//
//  1. The LLM emits <TASK_COMPLETE> in its content — clean early exit.
//  2. The LLM returns no tool call — it has nothing left to do.
//  3. MaxIterations (10) is reached — safety ceiling, returns control to user.
//
// onProgress is called after each step so the frontend can render live updates.
// Pass nil to suppress progress events (e.g. in tests).
func (o *Orchestrator) AgentLoop(conversationID, userPrompt string, onProgress LoopProgressFn) (*models.AgentResponse, error) {
	start := time.Now()

	emit := func(e LoopEvent) {
		if onProgress != nil {
			onProgress(e)
		}
	}

	conv := o.getOrCreateConversation(conversationID)
	conv.AddMessage(models.RoleUser, userPrompt)

	// ── Semantic memory recall ───────────────────────────────────────────
	memories, err := o.memory.Search(o.ctx, userPrompt, 5)
	if err != nil {
		o.log.Warn("Memory recall failed: %v", err)
	}

	// Pre-build memory injection string once — used in iteration 1 only
	memInjection := buildMemoryInjection(memories)

	// ── Planning step (complex tasks only) ───────────────────────────────
	// Only run a planning inference when the request is clearly multi-step.
	// Simple requests skip this to save a full LLM round-trip.
	var taskPlan string
	if isComplexRequest(userPrompt) {
		emit(LoopEvent{Kind: LoopEventThinking, Iteration: 0, Message: "Planning task..."})
		planResp, planErr := o.cognitive.Generate(o.ctx, cognitive.Request{
			Messages: []models.Message{
				{
					Role: models.RoleUser,
					Content: fmt.Sprintf(
						"Before acting, briefly outline your plan to complete the following task. "+
							"List 1-3 concrete steps only. Be concise.\n\nTASK: %s", userPrompt),
				},
			},
			MemoryContext: memories,
			Tools:         nil, // no tools during planning — think only
		})
		if planErr != nil {
			o.log.Warn("Planning step failed (non-fatal): %v", planErr)
		} else if planResp != nil && planResp.Content != "" {
			taskPlan = planResp.Content
			o.log.Info("Task plan: %s", taskPlan)
			emit(LoopEvent{Kind: LoopEventThinking, Iteration: 0, Message: fmt.Sprintf("Plan: %s", taskPlan)})
		}
	}

	var (
		finalContent   string
		finalReasoning string
		toolsUsed      []string
		iteration      int
	)

	for iteration = 1; iteration <= MaxToolIterations; iteration++ {
		emit(LoopEvent{
			Kind:      LoopEventThinking,
			Iteration: iteration,
			Message:   fmt.Sprintf("Iteration %d/%d — asking LLM", iteration, MaxToolIterations),
		})

		// Snapshot + trim context window to prevent overflow on long conversations
		rawMsgs := make([]models.Message, len(conv.Messages))
		copy(rawMsgs, conv.Messages)
		llmMessages := trimContextMessages(rawMsgs)

		// On iteration 1: inject plan anchor and memory context into the last user message.
		// Only done once — no need to re-inject on subsequent iterations.
		if iteration == 1 {
			for j := len(llmMessages) - 1; j >= 0; j-- {
				if llmMessages[j].Role == models.RoleUser {
					if taskPlan != "" {
						llmMessages[j].Content += fmt.Sprintf("\n\n[TASK PLAN]\n%s\n[END TASK PLAN]", taskPlan)
					}
					if memInjection != "" {
						llmMessages[j].Content += memInjection
					}
					break
				}
			}
		}

		resp, err := o.cognitive.Generate(o.ctx, cognitive.Request{
			Messages:      llmMessages,
			MemoryContext: memories,
			Tools:         o.tools.Definitions(),
		})
		if err != nil {
			emit(LoopEvent{Kind: LoopEventError, Iteration: iteration, Message: err.Error()})
			return nil, fmt.Errorf("cognitive error at iteration %d: %w", iteration, err)
		}

		// ── No tool call: LLM is done ────────────────────────────────────
		if resp.ToolCall == nil {
			finalContent = resp.Content
			finalReasoning = resp.Reasoning

			// Check for explicit completion sentinel
			if strings.Contains(finalContent, TaskCompleteToken) {
				finalContent = strings.ReplaceAll(finalContent, TaskCompleteToken, "")
				finalContent = strings.TrimSpace(finalContent)
				emit(LoopEvent{
					Kind:      LoopEventDone,
					Iteration: iteration,
					Message:   "Task complete — LLM signalled <TASK_COMPLETE>",
				})
			} else {
				emit(LoopEvent{
					Kind:      LoopEventDone,
					Iteration: iteration,
					Message:   "Task complete — no further tool calls",
				})
			}
			break
		}

		// ── Tool call: execute and loop back ─────────────────────────────
		toolName := resp.ToolCall.Name
		toolsUsed = append(toolsUsed, toolName)

		emit(LoopEvent{
			Kind:      LoopEventToolCall,
			Iteration: iteration,
			ToolName:  toolName,
			Message:   fmt.Sprintf("Calling tool: %s", toolName),
		})

		// Record assistant's intent before executing
		assistantMsg := resp.Content
		if assistantMsg == "" {
			argsBytes, _ := json.Marshal(resp.ToolCall.Args)
			assistantMsg = fmt.Sprintf(`{"reasoning":%q,"tool_call":{"name":%q,"args":%s},"content":""}`,
				resp.Reasoning, toolName, string(argsBytes))
		}
		conv.AddMessage(models.RoleAssistant, assistantMsg)

		// Guardrail check
		if err := o.guardrail.Check(resp.ToolCall); err != nil {
			o.log.Warn("Guardrail blocked %s: %v", toolName, err)
			blocked := fmt.Sprintf("[BLOCKED] %s: %s", toolName, err)
			conv.AddMessage(models.RoleTool, blocked)
			emit(LoopEvent{Kind: LoopEventBlocked, Iteration: iteration, ToolName: toolName, Message: blocked})
			continue
		}

		// Execute tool
		result, err := o.tools.Execute(o.ctx, resp.ToolCall)
		if err != nil {
			errMsg := fmt.Sprintf("[ERROR] %s: %s", toolName, err)
			conv.AddMessage(models.RoleTool, errMsg)
			emit(LoopEvent{Kind: LoopEventToolResult, Iteration: iteration, ToolName: toolName, Message: errMsg})
			continue
		}

		toolResult := fmt.Sprintf("[TOOL RESULT]\n%s\n[END TOOL RESULT]", result)
		conv.AddMessage(models.RoleTool, toolResult)
		emit(LoopEvent{
			Kind:      LoopEventToolResult,
			Iteration: iteration,
			ToolName:  toolName,
			Message:   result,
		})
	}

	// ── Hit the ceiling ──────────────────────────────────────────────────
	if iteration > MaxToolIterations && finalContent == "" {
		finalContent = fmt.Sprintf("Agent loop reached the %d-iteration safety limit. The task may be partially complete. Tools used: %s",
			MaxToolIterations, strings.Join(toolsUsed, ", "))
		emit(LoopEvent{
			Kind:      LoopEventMaxIter,
			Iteration: MaxToolIterations,
			Message:   finalContent,
		})
	}

	// ── Persist exchange in long-term memory ─────────────────────────────
	conv.AddMessage(models.RoleAssistant, finalContent)
	if finalContent != "" {
		if err := o.memory.Store(o.ctx, userPrompt, finalContent); err != nil {
			o.log.Warn("Memory persist failed: %v", err)
		}
	}

	elapsed := time.Since(start)
	o.log.Info("AgentLoop: %d iterations, %d tools, %s elapsed", iteration-1, len(toolsUsed), elapsed)

	return &models.AgentResponse{
		Content:        finalContent,
		Reasoning:      finalReasoning,
		ToolsUsed:      toolsUsed,
		MemoryRecalled: len(memories),
		LatencyMs:      elapsed.Milliseconds(),
	}, nil
}

// WipeMemory clears both long-term vector storage and short-term conversation context.
// This is the "Nuclear Option" callable from the UI.
func (o *Orchestrator) WipeMemory() error {
	// 1. Wipe the persistent vector DB
	if err := o.memory.WipeAll(); err != nil {
		return err
	}

	// 2. Clear the active short-term context by re-initializing the map
	o.mu.Lock()
	o.conversations = make(map[string]*models.Conversation)
	o.mu.Unlock()

	return nil
}

func (o *Orchestrator) GetConversations() []models.ConversationSummary {
	o.mu.RLock()
	defer o.mu.RUnlock()

	summaries := make([]models.ConversationSummary, 0, len(o.conversations))
	for id, conv := range o.conversations {
		summaries = append(summaries, models.ConversationSummary{
			ID:           id,
			Title:        conv.Title,
			MessageCount: len(conv.Messages),
			LastActivity: conv.LastActivity,
		})
	}
	return summaries
}

func (o *Orchestrator) getOrCreateConversation(id string) *models.Conversation {
	o.mu.Lock()
	defer o.mu.Unlock()

	if conv, ok := o.conversations[id]; ok {
		return conv
	}
	conv := models.NewConversation(id)
	o.conversations[id] = conv
	return conv
}

// isComplexRequest returns true when a prompt is likely multi-step or requires
// deep reasoning. Used to gate the planning step — simple requests skip it.
func isComplexRequest(prompt string) bool {
	words := strings.Fields(prompt)
	if len(words) < complexityThreshold {
		return false
	}
	lower := strings.ToLower(prompt)
	indicators := []string{
		"implement", "build", "create", "architect", "refactor", "analyze",
		"across", "multiple", "design", "strategy", "pipeline", "step",
		"rewrite", "migrate", "integrate", "optimize", "restructure",
	}
	for _, kw := range indicators {
		if strings.Contains(lower, kw) {
			return true
		}
	}
	// Long prompts are usually complex
	return len(words) > 30
}

// trimContextMessages applies a sliding window to the message history and
// truncates oversized tool results. This prevents context overflow and keeps
// the LLM inference fast as conversations grow.
func trimContextMessages(messages []models.Message) []models.Message {
	// First pass: truncate any tool results that are too large
	for i := range messages {
		if messages[i].Role == models.RoleTool && len(messages[i].Content) > maxToolResultBytes {
			messages[i].Content = messages[i].Content[:maxToolResultBytes] +
				fmt.Sprintf("\n... [TRUNCATED — %d bytes omitted]", len(messages[i].Content)-maxToolResultBytes)
		}
	}

	// Second pass: apply sliding window — keep first message + last N
	if len(messages) <= maxContextMessages {
		return messages
	}

	// Always keep the first user message for context anchor, then last window
	tail := messages[len(messages)-maxContextMessages+1:]
	return append([]models.Message{messages[0]}, tail...)
}

// buildMemoryInjection pre-builds the memory injection string once so it
// doesn't get reconstructed on every loop iteration.
func buildMemoryInjection(memories []memory.MemoryEntry) string {
	if len(memories) == 0 {
		return ""
	}
	var sb strings.Builder
	for _, m := range memories {
		if m.Score >= 0.5 {
			sb.WriteString(m.Content)
			sb.WriteString("\n---\n")
		}
	}
	if sb.Len() == 0 {
		return ""
	}
	return "\n\n[SYSTEM MEMORY RECALL]\n" + sb.String()
}
