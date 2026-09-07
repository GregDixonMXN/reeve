package orchestrator

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"herald/internal/cognitive"
	"herald/internal/guardrail"
	"herald/internal/memory"
	"herald/internal/tools"
	"herald/pkg/logger"
	"herald/pkg/models"
)

// MaxToolIterationsLocal is the agent loop ceiling for local and hybrid modes.
const MaxToolIterationsLocal = 25

// MaxToolIterationsCloud is the agent loop ceiling for cloud mode.
// Cloud models can sustain longer tool sequences, but this remains a hard cost
// and runaway-loop boundary.
const MaxToolIterationsCloud = 50

// Deprecated: use MaxToolIterationsLocal or MaxToolIterationsCloud.
const MaxToolIterations = MaxToolIterationsLocal

// TaskCompleteToken is the sentinel the LLM emits to break the agent loop early.
const TaskCompleteToken = "<TASK_COMPLETE>"

// maxContextMessages is the sliding-window ceiling for conversation history
// passed to the LLM. Older messages beyond this are dropped to prevent
// context overflow and keep inference fast.
const maxContextMessages = 40

// maxToolResultBytes is the max size of a single tool result injected into
// the context. Larger results are truncated with a notice.
const maxToolResultBytes = 8000

// maxCloudToolResultBytes is the limit for ask_cloud_model and execute_code
// results — build output and stack traces need room to be fully readable.
const maxCloudToolResultBytes = 32000

// complexityThreshold is the minimum word count before planning is attempted.
const complexityThreshold = 10

// LoopEventKind describes what kind of progress event occurred.
type LoopEventKind string

const (
	LoopEventThinking   LoopEventKind = "thinking"    // LLM reasoning step started
	LoopEventToolCall   LoopEventKind = "tool_call"   // LLM wants to call a tool
	LoopEventToolResult LoopEventKind = "tool_result" // tool returned a result
	LoopEventBlocked    LoopEventKind = "blocked"     // guardrail blocked a tool call
	LoopEventDone       LoopEventKind = "done"        // loop completed (clean)
	LoopEventMaxIter    LoopEventKind = "max_iter"    // loop hit iteration ceiling
	LoopEventError      LoopEventKind = "error"       // unrecoverable error
)

// LoopEvent is a real-time progress update emitted during AgentLoop execution.
type LoopEvent struct {
	ConversationID string        `json:"conversation_id"`
	RunID          string        `json:"run_id"`
	Kind           LoopEventKind `json:"kind"`
	Iteration      int           `json:"iteration"`
	MaxIterations  int           `json:"max_iterations"`
	Message        string        `json:"message"`
	ToolName       string        `json:"tool_name,omitempty"`
}

// LoopProgressFn is called after each meaningful step in the agent loop.
// Use it to stream events to the frontend via Wails runtime.EventsEmit.
type LoopProgressFn func(event LoopEvent)

type Config struct {
	Logger            *logger.Logger
	Memory            *memory.Store
	Cognitive         *cognitive.Engine
	Tools             *tools.Registry
	Guardrail         *guardrail.Guard
	ModeManager       *ModeManager
	ReflectionEnabled bool
	WorkspaceDirs     []string // allowed dirs — used for project context loading
	MaxIterations     int      // 0 = use mode defaults (25 local, 50 cloud)
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

	mu                sync.RWMutex
	conversations     map[string]*models.Conversation
	conversationLocks map[string]*sync.Mutex
	reflectionEnabled bool
	workspaceDirs     []string
	iterationLimit    int // 0 = use mode defaults

	// lifecycleMu makes wipe a barrier: it cancels active runs, waits for all
	// mutating operations to leave, then clears memory and conversations. That
	// prevents a run holding an old conversation pointer from saving it again.
	lifecycleMu    sync.RWMutex
	runMu          sync.Mutex
	runs           map[string]*activeRun
	runByConv      map[string]string
	pendingCancels map[string]string
	wiping         bool
}

type activeRun struct {
	conversationID string
	runID          string
	cancel         context.CancelFunc
}

func New(cfg Config) *Orchestrator {
	return &Orchestrator{
		log:               cfg.Logger,
		memory:            cfg.Memory,
		cognitive:         cfg.Cognitive,
		tools:             cfg.Tools,
		guardrail:         cfg.Guardrail,
		modeManager:       cfg.ModeManager,
		conversations:     make(map[string]*models.Conversation),
		conversationLocks: make(map[string]*sync.Mutex),
		runs:              make(map[string]*activeRun),
		runByConv:         make(map[string]string),
		pendingCancels:    make(map[string]string),
		reflectionEnabled: cfg.ReflectionEnabled,
		workspaceDirs:     cfg.WorkspaceDirs,
		iterationLimit:    cfg.MaxIterations,
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

	// Prune stale memories on startup — keeps recall quality high and DB lean
	if pruned, err := o.memory.Prune(); err != nil {
		o.log.Warn("Memory prune failed (non-fatal): %v", err)
	} else if pruned > 0 {
		o.log.Info("Memory pruned: removed %d stale entries", pruned)
	}
}

func (o *Orchestrator) Shutdown() {
	o.log.Info("Orchestrator shutting down...")
	if o.cancel != nil {
		o.cancel()
	}
	o.runMu.Lock()
	for _, run := range o.runs {
		run.cancel()
	}
	o.runMu.Unlock()
}

func (o *Orchestrator) rootContext() context.Context {
	if o.ctx != nil {
		return o.ctx
	}
	return context.Background()
}

func (o *Orchestrator) beginRun(conversationID, runID string) (context.Context, *activeRun, error) {
	conversationID = strings.TrimSpace(conversationID)
	runID = strings.TrimSpace(runID)
	if conversationID == "" {
		return nil, nil, fmt.Errorf("conversation id is required")
	}
	if runID == "" {
		return nil, nil, fmt.Errorf("run id is required")
	}

	o.runMu.Lock()
	defer o.runMu.Unlock()
	if o.wiping {
		return nil, nil, fmt.Errorf("memory purge is in progress")
	}
	if _, exists := o.runs[runID]; exists {
		return nil, nil, fmt.Errorf("run id %q is already active", runID)
	}
	if existing, exists := o.runByConv[conversationID]; exists {
		return nil, nil, fmt.Errorf("conversation %q already has active run %q", conversationID, existing)
	}

	runCtx, cancel := context.WithCancel(o.rootContext())
	run := &activeRun{conversationID: conversationID, runID: runID, cancel: cancel}
	o.runs[runID] = run
	o.runByConv[conversationID] = runID
	if pendingConversation, pending := o.pendingCancels[runID]; pending && pendingConversation == conversationID {
		delete(o.pendingCancels, runID)
		cancel()
	}
	return runCtx, run, nil
}

func (o *Orchestrator) finishRun(run *activeRun) {
	if run == nil {
		return
	}
	o.runMu.Lock()
	if current, ok := o.runs[run.runID]; ok && current == run {
		delete(o.runs, run.runID)
		delete(o.runByConv, run.conversationID)
	}
	delete(o.pendingCancels, run.runID)
	o.runMu.Unlock()
	run.cancel()
}

// QueueAgentLoopCancellation closes the small admission race between the UI
// accepting a run and AgentLoop registering it. App calls this only after
// validating that the exact run is its active scope.
func (o *Orchestrator) QueueAgentLoopCancellation(conversationID, runID string) error {
	conversationID = strings.TrimSpace(conversationID)
	runID = strings.TrimSpace(runID)
	if conversationID == "" || runID == "" {
		return fmt.Errorf("conversation id and run id are required")
	}
	o.runMu.Lock()
	defer o.runMu.Unlock()
	if run, ok := o.runs[runID]; ok {
		if run.conversationID != conversationID {
			return fmt.Errorf("active run %q belongs to a different conversation", runID)
		}
		run.cancel()
		return nil
	}
	o.pendingCancels[runID] = conversationID
	return nil
}

func (o *Orchestrator) ClearQueuedAgentLoopCancellation(conversationID, runID string) {
	conversationID = strings.TrimSpace(conversationID)
	runID = strings.TrimSpace(runID)
	o.runMu.Lock()
	if pendingConversation, ok := o.pendingCancels[runID]; ok && pendingConversation == conversationID {
		delete(o.pendingCancels, runID)
	}
	o.runMu.Unlock()
}

// CancelAgentLoop cancels exactly one active run. Requiring both identifiers
// prevents a stale UI from cancelling a newer run that happens to reuse one of
// them.
func (o *Orchestrator) CancelAgentLoop(conversationID, runID string) error {
	conversationID = strings.TrimSpace(conversationID)
	runID = strings.TrimSpace(runID)
	o.runMu.Lock()
	run, ok := o.runs[runID]
	if !ok || run.conversationID != conversationID {
		o.runMu.Unlock()
		return fmt.Errorf("active run %q for conversation %q was not found", runID, conversationID)
	}
	run.cancel()
	o.runMu.Unlock()
	return nil
}

// HasActiveRuns reports whether any autonomous run is in progress.
func (o *Orchestrator) HasActiveRuns() bool {
	o.runMu.Lock()
	defer o.runMu.Unlock()
	return len(o.runs) > 0
}

// SendMessage runs the full Think-Verify-Act loop with semantic memory injection.
func (o *Orchestrator) SendMessage(conversationID, userMessage string) (*models.AgentResponse, error) {
	conversationID = strings.TrimSpace(conversationID)
	if conversationID == "" {
		return nil, fmt.Errorf("conversation id is required")
	}
	o.lifecycleMu.RLock()
	defer o.lifecycleMu.RUnlock()
	conversationMu := o.conversationMutex(conversationID)
	conversationMu.Lock()
	defer conversationMu.Unlock()
	ctx := o.rootContext()

	start := time.Now()
	o.log.Debug("Ingest: [%s] %s", conversationID, userMessage)

	conv, err := o.getOrCreateConversation(conversationID)
	if err != nil {
		return nil, err
	}
	beforeTurn := conv.Clone()
	addUserMessage(conv, userMessage)
	if err := o.persistConversation(ctx, conversationID, conv); err != nil {
		*conv = *beforeTurn
		return nil, fmt.Errorf("persist user message: %w", err)
	}

	// ── RECALL: Semantic memory search ──────────────────────────────────
	memories, err := o.memory.Search(ctx, userMessage, 5)
	if err != nil {
		o.log.Warn("Memory recall failed: %v", err)
	}
	if len(memories) > 0 {
		o.log.Debug("Recalled %d memories (top score: %.3f)", len(memories), memories[0].Score)
	}

	// ── THINK-VERIFY-ACT LOOP ───────────────────────────────────────────
	loop, err := o.runToolLoop(
		ctx,
		conversationID,
		conv,
		memories,
		o.maxIterations(),
		toolLoopPolicy{
			onToolCall: func(_ int, response *cognitive.Response, _ bool) {
				o.log.Debug("Tool call: %s(%v)", response.ToolCall.Name, response.ToolCall.Args)
			},
		},
	)
	if err != nil {
		status := fmt.Sprintf("[SYSTEM ERROR] agent loop failed at iteration %d: %s", loop.Iterations, err)
		if persistErr := o.persistTerminalStatus(conversationID, conv, status); persistErr != nil {
			return nil, fmt.Errorf(
				"agent loop error at iteration %d: %w (persist terminal status: %v)",
				loop.Iterations,
				err,
				persistErr,
			)
		}
		return nil, fmt.Errorf("agent loop error at iteration %d: %w", loop.Iterations, err)
	}
	if loop.HitLimit {
		if err := o.persistTerminalStatus(
			conversationID,
			conv,
			fmt.Sprintf("[SYSTEM ERROR] max tool iterations (%d) exceeded", o.maxIterations()),
		); err != nil {
			return nil, fmt.Errorf("max tool iterations (%d) exceeded (persist terminal status: %v)", o.maxIterations(), err)
		}
		return nil, fmt.Errorf("max tool iterations (%d) exceeded", o.maxIterations())
	}
	finalResponse := loop.FinalResponse
	toolsUsedThisTurn := loop.ToolsUsed

	if finalResponse.Content == "" {
		finalResponse.Content = "Action completed successfully."
	}

	// ── REFLECTION: Optional self-critique step ─────────────────────────
	finalResponse.Content = o.reflectFinalResponse(ctx, userMessage, finalResponse.Content)

	// ── PERSIST: Store exchange in long-term memory ─────────────────────
	conv.AddMessage(models.RoleAssistant, finalResponse.Content)
	if err := o.memory.Store(ctx, userMessage, finalResponse.Content); err != nil {
		o.log.Warn("Memory persist failed: %v", err)
	}

	// Persist conversation to SQLite
	if err := o.persistConversation(ctx, conversationID, conv); err != nil {
		return nil, fmt.Errorf("persist completed conversation: %w", err)
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
//  3. The configured or mode-specific iteration ceiling is reached.
//
// onProgress is called after each step so the frontend can render live updates.
// Pass nil to suppress progress events (e.g. in tests).
func (o *Orchestrator) AgentLoop(conversationID, runID, userPrompt string, onProgress LoopProgressFn) (*models.AgentResponse, error) {
	conversationID = strings.TrimSpace(conversationID)
	runID = strings.TrimSpace(runID)
	o.lifecycleMu.RLock()
	defer o.lifecycleMu.RUnlock()

	runCtx, run, err := o.beginRun(conversationID, runID)
	if err != nil {
		return nil, err
	}
	defer o.finishRun(run)

	conversationMu := o.conversationMutex(conversationID)
	conversationMu.Lock()
	defer conversationMu.Unlock()

	start := time.Now()
	maxIter := o.maxIterations()

	emit := func(e LoopEvent) {
		e.ConversationID = conversationID
		e.RunID = runID
		e.MaxIterations = maxIter
		if onProgress != nil {
			onProgress(e)
		}
	}

	conv, err := o.getOrCreateConversation(conversationID)
	if err != nil {
		return nil, err
	}
	beforeTurn := conv.Clone()
	addUserMessage(conv, userPrompt)
	if err := o.persistRunSnapshot(conversationID, conv); err != nil {
		*conv = *beforeTurn
		return nil, fmt.Errorf("persist user message: %w", err)
	}
	if err := runCtx.Err(); err != nil {
		emit(LoopEvent{Kind: LoopEventError, Iteration: 0, Message: "Run cancelled"})
		if persistErr := o.persistTerminalStatus(conversationID, conv, "[RUN CANCELLED]"); persistErr != nil {
			return nil, fmt.Errorf("run cancelled before start: %w (persist terminal status: %v)", err, persistErr)
		}
		return nil, err
	}

	// ── Semantic memory recall ───────────────────────────────────────────
	memories, err := o.memory.Search(runCtx, userPrompt, 5)
	if err != nil {
		o.log.Warn("Memory recall failed: %v", err)
	}

	// ── Project context loading ───────────────────────────────────────────
	// Scan workspace dirs for context files relevant to this prompt and inject
	// them once on iteration 1. Gives the LLM the same "lay of the land" a
	// human engineer would have before touching a project.
	projectContext := o.loadProjectContext(runCtx, userPrompt)
	if projectContext != "" {
		o.log.Info("Project context loaded (%d bytes)", len(projectContext))
	}

	// ── Planning step (complex tasks only) ───────────────────────────────
	// Only run a planning inference when the request is clearly multi-step.
	// Simple requests skip this to save a full LLM round-trip.
	var taskPlan string
	if isComplexRequest(userPrompt) {
		emit(LoopEvent{Kind: LoopEventThinking, Iteration: 0, Message: "Planning task..."})

		// For multi-file projects, require a full file manifest before acting.
		// This forces the LLM to think about the complete structure up front and
		// prevents the premature-execution spiral that burns 15 iterations.
		planPrompt := fmt.Sprintf(
			"Before acting, produce a complete FILE MANIFEST for this task. "+
				"List EVERY file you will create or modify, with its exact path and one-line description. "+
				"Then list the steps in order. Be specific — paths must be absolute.\n\n"+
				"CRITICAL REMINDER: You will use write_file tool calls to create every file. "+
				"You must NEVER output file contents as text or code blocks — that is a hard failure. "+
				"One file = one write_file call. The file manifest you produce here will be executed as write_file calls.\n\n"+
				"TASK: %s", userPrompt)

		if projectContext != "" {
			planPrompt += projectContext
		}

		planResp, planErr := o.cognitive.Generate(runCtx, cognitive.Request{
			Messages: []models.Message{
				{Role: models.RoleUser, Content: planPrompt},
			},
			MemoryContext: memories,
			Tools:         nil, // think only — no tools during planning
		})
		if planErr != nil {
			o.log.Warn("Planning step failed (non-fatal): %v", planErr)
		} else if planResp != nil && planResp.Content != "" {
			taskPlan = planResp.Content
			o.log.Info("Task plan: %s", taskPlan)
			emit(LoopEvent{Kind: LoopEventThinking, Iteration: 0, Message: fmt.Sprintf("Plan: %s", taskPlan)})
		}
	}

	loop, loopErr := o.runToolLoop(
		runCtx,
		conversationID,
		conv,
		memories,
		maxIter,
		toolLoopPolicy{
			beforeIteration: func(iteration int) {
				emit(LoopEvent{
					Kind:      LoopEventThinking,
					Iteration: iteration,
					Message:   fmt.Sprintf("Iteration %d/%d — asking LLM", iteration, maxIter),
				})
			},
			prepareMessages: func(iteration int, llmMessages []models.Message, lastToolResult string) {
				// Recalled memory is supplied through the engine's explicitly
				// untrusted reference section rather than duplicated here.
				if iteration == 1 {
					for j := len(llmMessages) - 1; j >= 0; j-- {
						if llmMessages[j].Role != models.RoleUser {
							continue
						}
						if taskPlan != "" {
							llmMessages[j].Content += fmt.Sprintf(
								"\n\n[TASK PLAN — EXECUTE THIS EXACTLY]\n%s\n[END TASK PLAN]"+
									"\n\n[MANDATORY EXECUTION RULE]\n"+
									"Execute the plan above using write_file tool calls — one per file. "+
									"Do NOT output any file contents as text or code blocks in your response. "+
									"The ONLY correct action for creating a file is a write_file tool call. "+
									"Start with the first file in the manifest now.",
								taskPlan,
							)
						}
						if projectContext != "" {
							llmMessages[j].Content += projectContext
						}
						break
					}
				}
				if iteration > 1 && lastToolResult != "" {
					if recovery := errorRecoveryInjection(lastToolResult); recovery != "" {
						for j := len(llmMessages) - 1; j >= 0; j-- {
							if llmMessages[j].Role == models.RoleTool {
								llmMessages[j].Content += recovery
								break
							}
						}
					}
				}
			},
			intercept: func(response *cognitive.Response) (*cognitive.Response, bool) {
				extractedPath, extractedCode := extractCodeBlock(response.Content)
				if extractedPath == "" {
					return response, false
				}
				o.log.Warn("LLM dumped code to content instead of write_file — intercepting (path: %s)", extractedPath)
				return &cognitive.Response{
					Reasoning: response.Reasoning,
					Content:   fmt.Sprintf("Writing %s", extractedPath),
					ToolCall: &models.ToolCall{
						Name: "write_file",
						Args: map[string]interface{}{"path": extractedPath, "content": extractedCode},
					},
				}, true
			},
			onToolCall: func(iteration int, response *cognitive.Response, intercepted bool) {
				message := fmt.Sprintf("Calling tool: %s", response.ToolCall.Name)
				if intercepted {
					message = fmt.Sprintf("Auto-intercepted: writing %s to disk", response.ToolCall.Args["path"])
				}
				emit(LoopEvent{
					Kind:      LoopEventToolCall,
					Iteration: iteration,
					ToolName:  response.ToolCall.Name,
					Message:   message,
				})
			},
			onToolOutcome: func(iteration int, outcome toolTurnOutcome) {
				eventKind := LoopEventToolResult
				if outcome.Status == toolTurnBlocked {
					eventKind = LoopEventBlocked
				}
				emit(LoopEvent{
					Kind:      eventKind,
					Iteration: iteration,
					ToolName:  outcome.ToolName,
					Message:   outcome.EventMessage,
				})
			},
		},
	)
	if loopErr != nil {
		iteration := loop.Iterations
		message := loopErr.Error()
		status := fmt.Sprintf("[SYSTEM ERROR] %s", loopErr)
		if errors.Is(loopErr, context.Canceled) {
			message = "Run cancelled"
			status = "[RUN CANCELLED]"
		}
		emit(LoopEvent{Kind: LoopEventError, Iteration: iteration, Message: message})
		if persistErr := o.persistTerminalStatus(conversationID, conv, status); persistErr != nil {
			return nil, fmt.Errorf("agent loop error at iteration %d: %w (persist terminal status: %v)", iteration, loopErr, persistErr)
		}
		return nil, fmt.Errorf("agent loop error at iteration %d: %w", iteration, loopErr)
	}

	toolsUsed := loop.ToolsUsed
	var finalContent, finalReasoning string
	if loop.HitLimit {
		finalContent = fmt.Sprintf("Agent loop reached the %d-iteration safety limit. The task may be partially complete. Tools used: %s",
			maxIter, strings.Join(toolsUsed, ", "))
		emit(LoopEvent{
			Kind:      LoopEventMaxIter,
			Iteration: maxIter,
			Message:   finalContent,
		})
	} else {
		finalContent = loop.FinalResponse.Content
		finalReasoning = loop.FinalResponse.Reasoning
		if strings.Contains(finalContent, TaskCompleteToken) {
			finalContent = strings.TrimSpace(strings.ReplaceAll(finalContent, TaskCompleteToken, ""))
			emit(LoopEvent{
				Kind:      LoopEventDone,
				Iteration: loop.Iterations,
				Message:   "Task complete — LLM signalled <TASK_COMPLETE>",
			})
		} else {
			emit(LoopEvent{
				Kind:      LoopEventDone,
				Iteration: loop.Iterations,
				Message:   "Task complete — no further tool calls",
			})
		}
		if finalContent == "" {
			finalContent = "Action completed successfully."
		}
	}

	finalContent = o.reflectFinalResponse(runCtx, userPrompt, finalContent)

	// ── Persist exchange in long-term memory ─────────────────────────────
	conv.AddMessage(models.RoleAssistant, finalContent)
	if finalContent != "" {
		if err := o.memory.Store(runCtx, userPrompt, finalContent); err != nil {
			o.log.Warn("Memory persist failed: %v", err)
		}
	}

	// Persist conversation to SQLite
	if err := o.persistRunSnapshot(conversationID, conv); err != nil {
		return nil, fmt.Errorf("persist completed conversation: %w", err)
	}

	elapsed := time.Since(start)
	o.log.Info("AgentLoop: %d iterations, %d tools, %s elapsed", loop.Iterations, len(toolsUsed), elapsed)

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
	// Reject new runs and cancel current ones before waiting at the lifecycle
	// barrier. Every chat mutation holds a lifecycle read lock, so once the
	// write lock is acquired no stale conversation pointer can be saved later.
	o.runMu.Lock()
	o.wiping = true
	for _, run := range o.runs {
		run.cancel()
	}
	o.runMu.Unlock()

	o.lifecycleMu.Lock()
	defer o.lifecycleMu.Unlock()
	defer func() {
		o.runMu.Lock()
		o.pendingCancels = make(map[string]string)
		o.wiping = false
		o.runMu.Unlock()
	}()

	if err := o.memory.WipeAll(); err != nil {
		return err
	}

	o.mu.Lock()
	o.conversations = make(map[string]*models.Conversation)
	o.conversationLocks = make(map[string]*sync.Mutex)
	o.mu.Unlock()
	return nil
}

func (o *Orchestrator) GetConversations() ([]models.ConversationSummary, error) {
	o.lifecycleMu.RLock()
	defer o.lifecycleMu.RUnlock()

	merged := make(map[string]models.ConversationSummary)
	stored, err := o.memory.ListConversations(o.rootContext())
	if err != nil {
		return nil, fmt.Errorf("list persisted conversations: %w", err)
	}
	for _, summary := range stored {
		merged[summary.ID] = models.ConversationSummary{
			ID:           summary.ID,
			Title:        summary.Title,
			MessageCount: summary.MessageCount,
			LastActivity: summary.LastActivity,
		}
	}

	type liveConversation struct {
		id   string
		conv *models.Conversation
		mu   *sync.Mutex
	}
	o.mu.RLock()
	live := make([]liveConversation, 0, len(o.conversations))
	for id, conv := range o.conversations {
		live = append(live, liveConversation{id: id, conv: conv, mu: o.conversationLocks[id]})
	}
	o.mu.RUnlock()
	for _, item := range live {
		if item.mu != nil {
			item.mu.Lock()
		}
		merged[item.id] = models.ConversationSummary{
			ID:           item.id,
			Title:        item.conv.Title,
			MessageCount: item.conv.VisibleMessageCount(),
			LastActivity: item.conv.LastActivity,
		}
		if item.mu != nil {
			item.mu.Unlock()
		}
	}

	summaries := make([]models.ConversationSummary, 0, len(merged))
	for _, summary := range merged {
		summaries = append(summaries, summary)
	}
	sort.Slice(summaries, func(i, j int) bool {
		if summaries[i].LastActivity.Equal(summaries[j].LastActivity) {
			return summaries[i].ID < summaries[j].ID
		}
		return summaries[i].LastActivity.After(summaries[j].LastActivity)
	})
	return summaries, nil
}

// CreateConversation creates and persists an empty chat so it appears in the
// sidebar before its first message is sent.
func (o *Orchestrator) CreateConversation(id string) (*models.Conversation, error) {
	id = strings.TrimSpace(id)
	if id == "" {
		return nil, fmt.Errorf("conversation id is required")
	}
	o.lifecycleMu.RLock()
	defer o.lifecycleMu.RUnlock()
	conversationMu := o.conversationMutex(id)
	conversationMu.Lock()
	defer conversationMu.Unlock()
	conv, err := o.getOrCreateConversation(id)
	if err != nil {
		return nil, err
	}
	if err := o.persistConversation(o.rootContext(), id, conv); err != nil {
		return nil, err
	}
	return conv.Clone(), nil
}

// GetConversation loads one complete conversation, including persisted
// messages after an application restart.
func (o *Orchestrator) GetConversation(id string) (*models.Conversation, error) {
	id = strings.TrimSpace(id)
	if id == "" {
		return nil, fmt.Errorf("conversation id is required")
	}
	o.lifecycleMu.RLock()
	defer o.lifecycleMu.RUnlock()
	conversationMu := o.conversationMutex(id)
	conversationMu.Lock()
	defer conversationMu.Unlock()
	conv, found, err := o.findConversation(id)
	if err != nil {
		return nil, err
	}
	if !found {
		// A read must not create live state or a phantom sidebar entry.
		return models.NewConversation(id), nil
	}
	return conv.Clone(), nil
}

func (o *Orchestrator) conversationMutex(id string) *sync.Mutex {
	o.mu.Lock()
	defer o.mu.Unlock()
	mu := o.conversationLocks[id]
	if mu == nil {
		mu = &sync.Mutex{}
		o.conversationLocks[id] = mu
	}
	return mu
}

func (o *Orchestrator) findConversation(id string) (*models.Conversation, bool, error) {
	o.mu.RLock()
	if conv, ok := o.conversations[id]; ok {
		o.mu.RUnlock()
		return conv, true, nil
	}
	o.mu.RUnlock()

	// Try loading from persistent storage
	stored, err := o.memory.LoadConversationRecord(o.rootContext(), id)
	if err == nil && stored != nil {
		conv := models.NewConversation(id)
		conv.Title = stored.Title
		for _, msg := range stored.Messages {
			conv.Messages = append(conv.Messages, models.Message{
				Role:      msg.Role,
				Content:   msg.Content,
				Timestamp: msg.Timestamp,
				Internal:  msg.Internal,
			})
		}
		if !stored.LastActivity.IsZero() {
			conv.LastActivity = stored.LastActivity
		}
		o.mu.Lock()
		o.conversations[id] = conv
		o.mu.Unlock()
		o.log.Debug("Loaded conversation %s from storage (%d messages)", id, len(stored.Messages))
		return conv, true, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("load conversation %s: %w", id, err)
	}
	return nil, false, nil
}

func (o *Orchestrator) getOrCreateConversation(id string) (*models.Conversation, error) {
	if conv, found, err := o.findConversation(id); err != nil {
		return nil, err
	} else if found {
		return conv, nil
	}
	conv := models.NewConversation(id)
	o.mu.Lock()
	o.conversations[id] = conv
	o.mu.Unlock()
	return conv, nil
}

func (o *Orchestrator) persistConversation(ctx context.Context, id string, conv *models.Conversation) error {
	messages := make([]memory.ConversationMessage, len(conv.Messages))
	for i, msg := range conv.Messages {
		messages[i] = memory.ConversationMessage{
			Role:      msg.Role,
			Content:   msg.Content,
			Timestamp: msg.Timestamp,
			Internal:  msg.Internal,
		}
	}
	return o.memory.SaveConversationRecord(ctx, memory.ConversationRecord{
		ID:           id,
		Title:        conv.Title,
		Messages:     messages,
		LastActivity: conv.LastActivity,
	})
}

func (o *Orchestrator) persistRunSnapshot(id string, conv *models.Conversation) error {
	ctx, cancel := context.WithTimeout(o.rootContext(), 5*time.Second)
	defer cancel()
	if err := o.persistConversation(ctx, id, conv); err != nil {
		return fmt.Errorf("persist conversation snapshot: %w", err)
	}
	return nil
}

func (o *Orchestrator) persistTerminalStatus(id string, conv *models.Conversation, status string) error {
	conv.AddMessage(models.RoleSystem, status)
	return o.persistRunSnapshot(id, conv)
}

func addUserMessage(conv *models.Conversation, content string) {
	if conv.Title == "" || conv.Title == "New Conversation" {
		conv.Title = deriveConversationTitle(content)
	}
	conv.AddMessage(models.RoleUser, content)
}

func deriveConversationTitle(prompt string) string {
	title := strings.Join(strings.Fields(prompt), " ")
	title = strings.Trim(title, " \t\r\n\"'`")
	if title == "" {
		return "New Conversation"
	}
	const maxRunes = 60
	runes := []rune(title)
	if len(runes) <= maxRunes {
		return title
	}
	short := strings.TrimSpace(string(runes[:maxRunes-1]))
	return short + "…"
}

// maxIterations returns the agent loop ceiling for the current mode.
// If a custom limit is configured (> 0), it takes precedence over mode defaults.
// Set to a very large number (e.g. 9999) in herald.toml to effectively uncap.
func (o *Orchestrator) maxIterations() int {
	if o.iterationLimit > 0 {
		return o.iterationLimit
	}
	if o.modeManager != nil && o.modeManager.CurrentMode() == ModeCloud {
		return MaxToolIterationsCloud
	}
	return MaxToolIterationsLocal
}

// loadProjectContext scans workspace dirs for context files (README.md, HERALD.md,
// go.mod, package.json, Cargo.toml) in paths referenced by the user prompt and
// injects their contents as a system prefix. This gives the LLM the same "lay of
// the land" a human engineer would have before touching a project.
func (o *Orchestrator) loadProjectContext(ctx context.Context, prompt string) string {
	if len(o.workspaceDirs) == 0 {
		return ""
	}

	// Look for any workspace dir mentioned in the prompt
	lowerPrompt := strings.ToLower(prompt)
	var targetDirs []string
	for _, dir := range o.workspaceDirs {
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if e.IsDir() && strings.Contains(lowerPrompt, strings.ToLower(e.Name())) {
				targetDirs = append(targetDirs, filepath.Join(dir, e.Name()))
			}
		}
		// Also include the workspace dir itself if mentioned
		if strings.Contains(lowerPrompt, strings.ToLower(filepath.Base(dir))) {
			targetDirs = append(targetDirs, dir)
		}
	}

	if len(targetDirs) == 0 {
		return ""
	}

	contextFiles := []string{
		"README.md", "HERALD.md", "go.mod", "package.json",
		"Cargo.toml", "pyproject.toml", "requirements.txt", "Makefile",
	}

	var sb strings.Builder
	for _, dir := range targetDirs {
		// Read context files
		for _, cf := range contextFiles {
			path := filepath.Join(dir, cf)
			data, err := os.ReadFile(path)
			if err != nil {
				continue
			}
			content := string(data)
			if len(content) > 4000 {
				content = content[:4000] + "\n... [truncated]"
			}
			sb.WriteString(fmt.Sprintf("\n[PROJECT CONTEXT: %s]\n%s\n[END CONTEXT]\n", path, content))
		}

		// Git status — gives the LLM real situational awareness and stops it
		// clobbering unstaged work or re-creating files that already exist.
		if gitStatus := runGitStatus(ctx, dir); gitStatus != "" {
			sb.WriteString(fmt.Sprintf("\n[GIT STATUS: %s]\n%s\n[END GIT STATUS]\n", dir, gitStatus))
		}
	}

	return sb.String()
}

// runGitStatus runs `git status --short` in a directory and returns the output.
// Returns empty string if git is unavailable, the dir is not a repo, or it times out.
func runGitStatus(parent context.Context, dir string) string {
	ctx, cancel := context.WithTimeout(parent, 3*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "git", "status", "--short", "--branch")
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	s := strings.TrimSpace(string(out))
	if s == "" {
		return ""
	}
	// Cap output — a repo with 1000 untracked files doesn't need to flood context
	if len(s) > 1500 {
		s = s[:1500] + "\n... [truncated]"
	}
	return s
}

// errorRecoveryInjection examines the last tool result and — if it contains an
// error — appends a targeted recovery instruction to steer the next LLM call
// toward a surgical fix rather than a full rewrite.
func errorRecoveryInjection(toolResult string) string {
	lower := strings.ToLower(toolResult)
	isError := strings.Contains(lower, "[error]") ||
		strings.Contains(lower, "[stderr]") ||
		strings.Contains(lower, "exit code") && !strings.Contains(lower, "exit code] 0") ||
		strings.Contains(lower, "traceback") ||
		strings.Contains(lower, "syntaxerror") ||
		strings.Contains(lower, "nameerror") ||
		strings.Contains(lower, "importerror") ||
		strings.Contains(lower, "panic:") ||
		strings.Contains(lower, "undefined:")

	if !isError {
		return ""
	}

	return "\n\n[RECOVERY INSTRUCTION] The last tool call produced an error. " +
		"Read the FULL error message above carefully. " +
		"Identify the SPECIFIC file and line causing it. " +
		"Fix ONLY that — do not rewrite the entire file unless the whole structure is wrong. " +
		"If it is a missing import or dependency, add only that. " +
		"If it is a syntax error, fix only that line."
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
	// First pass: truncate any tool results that are too large.
	// execute_code and ask_cloud_model get the higher limit — build output and
	// stack traces must be fully visible for the LLM to diagnose errors correctly.
	for i := range messages {
		if messages[i].Role != models.RoleTool {
			continue
		}
		limit := maxToolResultBytes
		content := messages[i].Content
		if strings.Contains(content, "[SUCCESS]") ||
			strings.Contains(content, "ask_cloud_model") ||
			strings.Contains(content, "[STDOUT]") ||
			strings.Contains(content, "[STDERR]") ||
			strings.Contains(content, "[EXIT CODE]") {
			limit = maxCloudToolResultBytes
		}
		if len(content) > limit {
			messages[i].Content = content[:limit] +
				fmt.Sprintf("\n... [TRUNCATED — %d bytes omitted]", len(content)-limit)
		}
	}

	// Second pass: sliding window — if within limit return as-is
	if len(messages) <= maxContextMessages {
		return messages
	}

	// Over limit: instead of silently dropping history, preserve continuity by:
	//   1. Keeping the first user message as a context anchor
	//   2. Injecting a summary stub describing what was condensed
	//   3. Keeping the last N messages verbatim for immediate context
	keepTail := maxContextMessages / 2 // keep last ~20 messages verbatim
	if keepTail < 10 {
		keepTail = 10
	}
	tail := messages[len(messages)-keepTail:]
	anchor := messages[0]
	dropped := len(messages) - keepTail - 1

	summary := models.Message{
		Role: models.RoleUser,
		Content: fmt.Sprintf(
			"[CONTEXT SUMMARY: %d earlier messages condensed. "+
				"Conversation began with: %q — continue from the recent messages below.]",
			dropped, truncateStr(anchor.Content, 200)),
		Timestamp: anchor.Timestamp,
	}

	result := make([]models.Message, 0, keepTail+2)
	result = append(result, anchor, summary)
	result = append(result, tail...)
	return result
}

// extractCodeBlock scans LLM content for a pattern like:
//
//	# /some/path/file.py
//	```python
//	<code>
//	```
//
// or:
//
//	**`/path/to/file.py`**
//	```
//	<code>
//	```
//
// Returns (path, code) if found, or ("", "") if not.
// This is the intercept layer for local models that dump file contents into
// content text instead of calling write_file.
func extractCodeBlock(content string) (path string, code string) {
	if content == "" {
		return "", ""
	}
	lines := strings.Split(content, "\n")

	// Scan for a line that looks like a file path annotation
	for i, line := range lines {
		line = strings.TrimSpace(line)

		// Pattern: line starting with # /path or ## /path
		if strings.HasPrefix(line, "#") {
			candidate := strings.TrimSpace(strings.TrimPrefix(strings.TrimPrefix(line, "##"), "#"))
			if looksLikeFilePath(candidate) && i+1 < len(lines) {
				code, ok := collectFencedBlock(lines[i+1:])
				if ok {
					return candidate, code
				}
			}
		}

		// Pattern: **`/path`** or `/path`
		stripped := strings.Trim(line, "*`")
		if looksLikeFilePath(stripped) && i+1 < len(lines) {
			code, ok := collectFencedBlock(lines[i+1:])
			if ok {
				return stripped, code
			}
		}
	}
	return "", ""
}

// looksLikeFilePath returns true if s is an absolute path with a file extension.
func looksLikeFilePath(s string) bool {
	if len(s) < 4 {
		return false
	}
	if !strings.HasPrefix(s, "/") {
		return false
	}
	ext := filepath.Ext(s)
	if ext == "" {
		return false
	}
	validExts := []string{".py", ".go", ".js", ".ts", ".tsx", ".jsx", ".json", ".toml", ".yaml", ".yml", ".md", ".sh", ".txt", ".html", ".css"}
	for _, e := range validExts {
		if ext == e {
			return true
		}
	}
	return false
}

// collectFencedBlock extracts content between ``` fences from a slice of lines.
func collectFencedBlock(lines []string) (string, bool) {
	// Skip the opening fence (may include language hint)
	if len(lines) == 0 {
		return "", false
	}
	first := strings.TrimSpace(lines[0])
	if !strings.HasPrefix(first, "```") {
		return "", false
	}
	var buf strings.Builder
	for _, l := range lines[1:] {
		if strings.TrimSpace(l) == "```" {
			return buf.String(), true
		}
		buf.WriteString(l)
		buf.WriteByte('\n')
	}
	return "", false
}

func truncateStr(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}
