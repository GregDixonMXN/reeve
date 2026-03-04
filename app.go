package main

import (
	"axiom/internal/orchestrator"
	"context"
	"fmt"

	"github.com/wailsapp/wails/v2/pkg/runtime"
)

// App is the main Wails application struct. Bound to the frontend.
type App struct {
	ctx         context.Context
	orch        *orchestrator.Orchestrator
	modeManager *orchestrator.ModeManager
}

func NewApp(orch *orchestrator.Orchestrator, mm *orchestrator.ModeManager) *App {
	return &App{
		orch:        orch,
		modeManager: mm,
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
	// Add this loud print statement:
	fmt.Printf("🔄 [SYSTEM] UI requested mode switch to: %s\n", mode)

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

func (a *App) GetConversations() []map[string]interface{} {
	summaries := a.orch.GetConversations()
	result := make([]map[string]interface{}, len(summaries))
	for i, s := range summaries {
		result[i] = map[string]interface{}{
			"id":            s.ID,
			"title":         s.Title,
			"message_count": s.MessageCount,
			"last_activity": s.LastActivity,
		}
	}
	return result
}

// ── Agent Loop ──────────────────────────────────────────────────────────────

// RunAgentLoop starts the autonomous agent loop for a given prompt.
// Unlike SendMessage (single round-trip), this drives the full
// Think→Tool→Result cycle automatically, streaming live events to the
// frontend via "axiom:loop_event" until <TASK_COMPLETE> or max iterations.
//
// Frontend usage:
//
//	window.runtime.EventsOn("axiom:loop_event", (event) => { ... })
//	await window.go.main.App.RunAgentLoop(conversationID, prompt)
func (a *App) RunAgentLoop(conversationID, message string) (map[string]interface{}, error) {
	resp, err := a.orch.AgentLoop(conversationID, message, func(event orchestrator.LoopEvent) {
		runtime.EventsEmit(a.ctx, "axiom:loop_event", map[string]interface{}{
			"kind":      string(event.Kind),
			"iteration": event.Iteration,
			"message":   event.Message,
			"tool_name": event.ToolName,
		})
	})
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

// ── Streaming ───────────────────────────────────────────────────────────────

func (a *App) EmitToken(token string) {
	runtime.EventsEmit(a.ctx, "axiom:token", map[string]string{"token": token})
}

// ── Memory Management ───────────────────────────────────────────────────────

// WipeMemory is the nuclear option to clear all memories and context.
// Exposed to the frontend via Wails.
func (a *App) WipeMemory() (string, error) {
	err := a.orch.WipeMemory()
	if err != nil {
		return "", err
	}
	return "Memory successfully purged.", nil
}
