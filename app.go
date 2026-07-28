package main

import (
	"axiom/internal/orchestrator"
	"axiom/pkg/models"
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/wailsapp/wails/v2/pkg/runtime"
)

// App is the main Wails application struct. Bound to the frontend.
type App struct {
	ctx         context.Context
	orch        *orchestrator.Orchestrator
	modeManager *orchestrator.ModeManager
	runtimeInfo RuntimeInfo
	operationMu sync.Mutex
	streamMu    sync.RWMutex
	activeRun   *appRunScope
	purging     bool
	runCancels  runCancellationCoordinator
}

type appRunScope struct {
	conversationID string
	runID          string
}

// RuntimeInfo is safe, non-secret provider metadata exposed to the frontend.
// It deliberately carries only whether credentials were resolved, never the
// credential value or its source contents.
type RuntimeInfo struct {
	CloudProvider   string
	CloudModel      string
	LocalProvider   string
	LocalModel      string
	CloudConfigured bool
	LocalEnabled    bool
	LocalConfigured bool
	NetworkAllowed  bool
}

type runCancellationCoordinator interface {
	QueueAgentLoopCancellation(conversationID, runID string) error
	ClearQueuedAgentLoopCancellation(conversationID, runID string)
}

func NewApp(orch *orchestrator.Orchestrator, mm *orchestrator.ModeManager, info RuntimeInfo) *App {
	return &App{
		orch:        orch,
		modeManager: mm,
		runtimeInfo: info,
		runCancels:  orch,
	}
}

func (a *App) Startup(ctx context.Context) {
	a.ctx = ctx
	a.orch.SetContext(ctx)
	a.orch.Start()
}

func (a *App) Shutdown(ctx context.Context) {
	a.orch.Shutdown()
}

// ── Chat ────────────────────────────────────────────────────────────────────

func (a *App) SendMessage(conversationID, message string) (map[string]interface{}, error) {
	if !a.operationMu.TryLock() {
		return nil, fmt.Errorf("another agent operation is already active")
	}
	defer a.operationMu.Unlock()
	resp, err := a.orch.SendMessage(conversationID, message)
	if err != nil {
		return nil, err
	}
	return map[string]interface{}{
		"content":         resp.Content,
		"reasoning":       resp.Reasoning,
		"tools_used":      resp.ToolsUsed,
		"memory_recalled": resp.MemoryRecalled,
		"latency_ms":      resp.LatencyMs,
		"mode":            string(a.modeManager.CurrentMode()),
	}, nil
}

// ── Mode Toggle (exposed to frontend) ───────────────────────────────────────

// SetMode switches between "local", "hybrid", and "cloud".
// Call from frontend: window.go.main.App.SetMode("cloud")
// app.go

func (a *App) SetMode(mode string) error {
	if !a.operationMu.TryLock() {
		return fmt.Errorf("cannot change mode while an agent operation is active")
	}
	defer a.operationMu.Unlock()
	err := a.modeManager.SetMode(orchestrator.Mode(mode))
	if err != nil {
		return err
	}

	// Notify frontend of mode change
	runtime.EventsEmit(a.ctx, "axiom:mode_changed", mode)
	return nil
}

// GetMode returns the current operating mode.
// Call from frontend: window.go.main.App.GetMode()
func (a *App) GetMode() string {
	return string(a.modeManager.CurrentMode())
}

// GetAvailableModes returns which modes are usable based on config.
func (a *App) GetAvailableModes() []string {
	modes := a.modeManager.AvailableModes()
	result := make([]string, len(modes))
	for i, m := range modes {
		result[i] = string(m)
	}
	return result
}

// GetRuntimeStatus reports provider readiness without exposing credentials.
func (a *App) GetRuntimeStatus() map[string]interface{} {
	available := a.GetAvailableModes()
	activeMode := a.GetMode()
	ready := activeMode != "" && len(available) > 0

	provider := a.runtimeInfo.CloudProvider
	model := a.runtimeInfo.CloudModel
	switch activeMode {
	case "local":
		provider = a.runtimeInfo.LocalProvider
		model = a.runtimeInfo.LocalModel
	case "hybrid":
		provider = "hybrid"
		model = a.runtimeInfo.LocalModel
		if a.runtimeInfo.CloudConfigured && a.runtimeInfo.CloudModel != "" {
			model += " + " + a.runtimeInfo.CloudModel
		}
	case "":
		if a.runtimeInfo.LocalEnabled {
			provider = a.runtimeInfo.LocalProvider
			model = a.runtimeInfo.LocalModel
		}
	}

	setupHint := ""
	if !ready && a.runtimeInfo.LocalEnabled && !a.runtimeInfo.LocalConfigured {
		setupHint = fmt.Sprintf(
			"Start Ollama and install %s, then restart Axiom.",
			a.runtimeInfo.LocalModel,
		)
	} else if !ready && !a.runtimeInfo.NetworkAllowed {
		setupHint = "Cloud inference is disabled. Set security.allow_network = true, then restart Axiom."
	} else if !ready && a.runtimeInfo.CloudProvider == "openai" && !a.runtimeInfo.CloudConfigured {
		setupHint = "Set AXIOM_OPENAI_API_KEY or OPENAI_API_KEY, then restart Axiom."
	} else if !ready {
		setupHint = "Configure an available inference provider, then restart Axiom."
	}

	return map[string]interface{}{
		"provider":         provider,
		"model":            model,
		"cloud_provider":   a.runtimeInfo.CloudProvider,
		"cloud_model":      a.runtimeInfo.CloudModel,
		"local_provider":   a.runtimeInfo.LocalProvider,
		"local_model":      a.runtimeInfo.LocalModel,
		"cloud_configured": a.runtimeInfo.CloudConfigured,
		"local_enabled":    a.runtimeInfo.LocalEnabled,
		"local_configured": a.runtimeInfo.LocalConfigured,
		"network_allowed":  a.runtimeInfo.NetworkAllowed,
		"active_mode":      activeMode,
		"available_modes":  available,
		"ready":            ready,
		"setup_hint":       setupHint,
	}
}

func (a *App) GetConversations() ([]map[string]interface{}, error) {
	summaries, err := a.orch.GetConversations()
	if err != nil {
		return nil, err
	}
	result := make([]map[string]interface{}, len(summaries))
	for i, s := range summaries {
		result[i] = map[string]interface{}{
			"id":            s.ID,
			"title":         s.Title,
			"message_count": s.MessageCount,
			"last_activity": s.LastActivity,
		}
	}
	return result, nil
}

// CreateConversation persists an empty conversation immediately so a newly
// created chat is stable across sidebar refreshes and application restarts.
func (a *App) CreateConversation(id string) (map[string]interface{}, error) {
	conversation, err := a.orch.CreateConversation(id)
	if err != nil {
		return nil, err
	}
	return conversationPayload(conversation), nil
}

// GetConversation returns a complete conversation with its stored messages.
func (a *App) GetConversation(id string) (map[string]interface{}, error) {
	conversation, err := a.orch.GetConversation(id)
	if err != nil {
		return nil, err
	}
	return conversationPayload(conversation), nil
}

// ── Agent Loop ──────────────────────────────────────────────────────────────

// RunAgentLoop starts the autonomous agent loop for a given prompt.
// It adds planning, cancellation, recovery guidance, and live events to the
// shared Think→Guard→Tool→Result protocol used by SendMessage.
//
// Frontend usage:
//
//	window.runtime.EventsOn("axiom:loop_event", (event) => { ... })
//	await window.go.main.App.RunAgentLoop(conversationID, runID, prompt)
func (a *App) RunAgentLoop(conversationID, runID, message string) (map[string]interface{}, error) {
	conversationID = strings.TrimSpace(conversationID)
	runID = strings.TrimSpace(runID)
	if !a.operationMu.TryLock() {
		return nil, fmt.Errorf("another agent operation is already active")
	}
	defer a.operationMu.Unlock()
	if err := a.beginRunScope(conversationID, runID); err != nil {
		return nil, err
	}
	defer a.endRunScope(conversationID, runID)

	resp, err := a.orch.AgentLoop(conversationID, runID, message, func(event orchestrator.LoopEvent) {
		runtime.EventsEmit(a.ctx, "axiom:loop_event", map[string]interface{}{
			"conversation_id": event.ConversationID,
			"run_id":          event.RunID,
			"kind":            string(event.Kind),
			"iteration":       event.Iteration,
			"max_iterations":  event.MaxIterations,
			"message":         event.Message,
			"tool_name":       event.ToolName,
		})
	})
	if err != nil {
		return nil, err
	}
	return map[string]interface{}{
		"conversation_id": conversationID,
		"run_id":          runID,
		"content":         resp.Content,
		"reasoning":       resp.Reasoning,
		"tools_used":      resp.ToolsUsed,
		"memory_recalled": resp.MemoryRecalled,
		"latency_ms":      resp.LatencyMs,
		"mode":            string(a.modeManager.CurrentMode()),
	}, nil
}

// CancelAgentLoop cancels one exact conversation/run pair.
func (a *App) CancelAgentLoop(conversationID, runID string) error {
	conversationID = strings.TrimSpace(conversationID)
	runID = strings.TrimSpace(runID)
	a.streamMu.Lock()
	defer a.streamMu.Unlock()
	if a.activeRun == nil || a.activeRun.conversationID != conversationID || a.activeRun.runID != runID {
		return fmt.Errorf("active run %q for conversation %q was not found", runID, conversationID)
	}
	// QueueAgentLoopCancellation atomically cancels an orchestrator-registered
	// run or records cancellation for one still crossing admission. Keeping the
	// app scope locked through that decision prevents endRunScope from clearing
	// the scope just before a stale cancellation is queued for reused IDs.
	return a.runCancels.QueueAgentLoopCancellation(conversationID, runID)
}

// ── Streaming ───────────────────────────────────────────────────────────────

func (a *App) emitToken(token string) {
	a.streamMu.RLock()
	run := a.activeRun
	if run == nil {
		a.streamMu.RUnlock()
		return
	}
	payload := map[string]string{
		"conversation_id": run.conversationID,
		"run_id":          run.runID,
		"token":           token,
	}
	a.streamMu.RUnlock()
	runtime.EventsEmit(a.ctx, "axiom:token", payload)
}

// ── Memory Management ───────────────────────────────────────────────────────

// WipeMemory is the nuclear option to clear all memories and context.
// Exposed to the frontend via Wails.
func (a *App) WipeMemory() (string, error) {
	a.streamMu.Lock()
	if a.purging {
		a.streamMu.Unlock()
		return "", fmt.Errorf("memory purge is already in progress")
	}
	a.purging = true
	var scopedRun *appRunScope
	if a.activeRun != nil {
		copy := *a.activeRun
		scopedRun = &copy
	}
	a.streamMu.Unlock()
	defer func() {
		a.streamMu.Lock()
		a.purging = false
		a.streamMu.Unlock()
	}()

	// A run becomes visible to the UI just before AgentLoop registers it with
	// the orchestrator. Queueing cancellation first closes that admission gap;
	// taking operationMu then waits for the cancelled run (or any SendMessage)
	// to leave before the destructive lifecycle barrier starts.
	if scopedRun != nil {
		if err := a.orch.QueueAgentLoopCancellation(scopedRun.conversationID, scopedRun.runID); err != nil {
			return "", fmt.Errorf("cancel active run before purge: %w", err)
		}
	}
	a.operationMu.Lock()
	defer a.operationMu.Unlock()

	err := a.orch.WipeMemory()
	if err != nil {
		return "", err
	}
	a.streamMu.Lock()
	a.activeRun = nil
	a.streamMu.Unlock()
	return "Memory successfully purged.", nil
}

func (a *App) beginRunScope(conversationID, runID string) error {
	conversationID = strings.TrimSpace(conversationID)
	runID = strings.TrimSpace(runID)
	if conversationID == "" || runID == "" {
		return fmt.Errorf("conversation id and run id are required")
	}
	a.streamMu.Lock()
	defer a.streamMu.Unlock()
	if a.purging {
		return fmt.Errorf("memory purge is in progress")
	}
	if a.activeRun != nil {
		return fmt.Errorf("agent run %q is already active for conversation %q", a.activeRun.runID, a.activeRun.conversationID)
	}
	a.activeRun = &appRunScope{conversationID: conversationID, runID: runID}
	return nil
}

func (a *App) endRunScope(conversationID, runID string) {
	conversationID = strings.TrimSpace(conversationID)
	runID = strings.TrimSpace(runID)
	a.streamMu.Lock()
	defer a.streamMu.Unlock()
	if a.activeRun != nil && a.activeRun.conversationID == conversationID && a.activeRun.runID == runID {
		a.runCancels.ClearQueuedAgentLoopCancellation(conversationID, runID)
		a.activeRun = nil
	}
}

func conversationPayload(conversation *models.Conversation) map[string]interface{} {
	visibleMessages := make([]models.Message, 0, len(conversation.Messages))
	for _, message := range conversation.Messages {
		if !message.Internal {
			visibleMessages = append(visibleMessages, message)
		}
	}
	return map[string]interface{}{
		"id":            conversation.ID,
		"title":         conversation.Title,
		"messages":      visibleMessages,
		"last_activity": conversation.LastActivity,
	}
}
