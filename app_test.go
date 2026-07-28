package main

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"axiom/internal/cognitive"
	"axiom/internal/config"
	"axiom/internal/guardrail"
	"axiom/internal/memory"
	"axiom/internal/orchestrator"
	"axiom/internal/tools"
	"axiom/pkg/logger"
	"axiom/pkg/models"
)

type appTestRunner struct{ called bool }

func (r *appTestRunner) Complete(context.Context, string, int) (string, error) {
	r.called = true
	return `{"reasoning":"","tool_call":null,"content":"unexpected"}`, nil
}

func (*appTestRunner) Unload() error { return nil }

type appBlockingRunner struct{}

func (*appBlockingRunner) Complete(ctx context.Context, _ string, _ int) (string, error) {
	<-ctx.Done()
	return "", ctx.Err()
}

func (*appBlockingRunner) Unload() error { return nil }

type blockingRunCancellationCoordinator struct {
	delegate runCancellationCoordinator
	reached  chan struct{}
	release  chan struct{}
	once     sync.Once
}

func (c *blockingRunCancellationCoordinator) QueueAgentLoopCancellation(conversationID, runID string) error {
	c.once.Do(func() {
		close(c.reached)
		<-c.release
	})
	return c.delegate.QueueAgentLoopCancellation(conversationID, runID)
}

func (c *blockingRunCancellationCoordinator) ClearQueuedAgentLoopCancellation(conversationID, runID string) {
	c.delegate.ClearQueuedAgentLoopCancellation(conversationID, runID)
}

func TestFindConfigPathMarksExplicitSelectionRequired(t *testing.T) {
	explicit := filepath.Join(t.TempDir(), "missing.toml")
	t.Setenv("AXIOM_CONFIG", "  "+explicit+"  ")

	path, required := findConfigPath()
	if path != explicit {
		t.Fatalf("findConfigPath path = %q, want %q", path, explicit)
	}
	if !required {
		t.Fatal("explicit AXIOM_CONFIG was not marked required")
	}
}

func TestConversationPayloadOmitsInternalProtocol(t *testing.T) {
	conversation := models.NewConversation("conv-visible")
	conversation.AddMessage(models.RoleUser, "repair it")
	conversation.AddInternalMessage(models.RoleAssistant, `{"tool_call":{"name":"system_info"}}`)
	conversation.AddInternalMessage(models.RoleTool, "secret tool output")
	conversation.AddMessage(models.RoleAssistant, "Repaired")

	payload := conversationPayload(conversation)
	messages, ok := payload["messages"].([]models.Message)
	if !ok {
		t.Fatalf("messages payload type = %T", payload["messages"])
	}
	if len(messages) != 2 {
		t.Fatalf("visible messages = %#v", messages)
	}
	if messages[0].Content != "repair it" || messages[1].Content != "Repaired" {
		t.Fatalf("visible messages = %#v", messages)
	}
}

func TestRuntimeStatusNeverExposesCredentials(t *testing.T) {
	engine := cognitive.NewEngine(config.ModelConfig{})
	manager := orchestrator.NewModeManager(orchestrator.ModeManagerConfig{
		Engine:        engine,
		CloudRunner:   &appTestRunner{},
		CloudProvider: "openai",
	})
	if err := manager.SetMode(orchestrator.ModeCloud); err != nil {
		t.Fatal(err)
	}

	app := NewApp(nil, manager, RuntimeInfo{
		CloudProvider:   "openai",
		CloudModel:      "gpt-5.6-sol",
		CloudConfigured: true,
		NetworkAllowed:  true,
	})
	status := app.GetRuntimeStatus()

	if status["ready"] != true || status["active_mode"] != "cloud" {
		t.Fatalf("runtime status = %#v", status)
	}
	if status["provider"] != "openai" || status["model"] != "gpt-5.6-sol" {
		t.Fatalf("provider status = %#v", status)
	}
	for key := range status {
		if key == "api_key" || key == "openai_key" {
			t.Fatalf("runtime status exposed secret field %q", key)
		}
	}
}

func TestRuntimeStatusReportsActiveLocalBrain(t *testing.T) {
	engine := cognitive.NewEngine(config.ModelConfig{})
	manager := orchestrator.NewModeManager(orchestrator.ModeManagerConfig{
		Engine:      engine,
		LocalRunner: &appTestRunner{},
	})
	if err := manager.SetMode(orchestrator.ModeLocal); err != nil {
		t.Fatal(err)
	}

	app := NewApp(nil, manager, RuntimeInfo{
		CloudProvider:   "openai",
		CloudModel:      "gpt-5.6-sol",
		LocalProvider:   "ollama",
		LocalModel:      "qwen3.6:27b-mtp-q4_K_M",
		LocalEnabled:    true,
		LocalConfigured: true,
		NetworkAllowed:  true,
	})
	status := app.GetRuntimeStatus()
	if status["provider"] != "ollama" || status["model"] != "qwen3.6:27b-mtp-q4_K_M" {
		t.Fatalf("local runtime status = %#v", status)
	}
}

func TestRuntimeStatusExplainsMissingLocalModel(t *testing.T) {
	manager := orchestrator.NewModeManager(orchestrator.ModeManagerConfig{
		Engine: cognitive.NewEngine(config.ModelConfig{}),
	})
	app := NewApp(nil, manager, RuntimeInfo{
		CloudProvider: "openai",
		LocalProvider: "ollama",
		LocalModel:    "qwen3.6:27b-mtp-q4_K_M",
		LocalEnabled:  true,
	})

	status := app.GetRuntimeStatus()
	hint, _ := status["setup_hint"].(string)
	if status["ready"] != false || !strings.Contains(hint, "qwen3.6:27b-mtp-q4_K_M") {
		t.Fatalf("runtime status = %#v, want exact local-model guidance", status)
	}
}

func TestRuntimeStatusExplainsDisabledNetworkBeforeMissingKey(t *testing.T) {
	manager := orchestrator.NewModeManager(orchestrator.ModeManagerConfig{
		Engine:        cognitive.NewEngine(config.ModelConfig{}),
		CloudProvider: "openai",
	})
	app := NewApp(nil, manager, RuntimeInfo{
		CloudProvider:  "openai",
		NetworkAllowed: false,
	})

	status := app.GetRuntimeStatus()
	hint, _ := status["setup_hint"].(string)
	if status["ready"] != false || !strings.Contains(hint, "security.allow_network") {
		t.Fatalf("runtime status = %#v, want network-policy guidance", status)
	}
	if strings.Contains(hint, "API_KEY") {
		t.Fatalf("network-disabled status incorrectly requested a credential: %q", hint)
	}
}

func TestCancelAgentLoopQueuesDuringAdmission(t *testing.T) {
	store, err := memory.NewStore(config.DatabaseConfig{
		Path:   filepath.Join(t.TempDir(), "memory.db"),
		VecDim: 4,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	runner := &appTestRunner{}
	engine := cognitive.NewEngine(config.ModelConfig{ContextSize: 1024, MaxOutputTokens: 128})
	engine.SetRunner(runner)
	orch := orchestrator.New(orchestrator.Config{
		Logger:        logger.New("error"),
		Memory:        store,
		Cognitive:     engine,
		Tools:         tools.NewRegistry(config.ToolsConfig{}, nil, nil),
		Guardrail:     guardrail.New(config.SecurityConfig{}),
		MaxIterations: 1,
	})
	orch.SetContext(context.Background())
	app := NewApp(orch, nil, RuntimeInfo{})
	if err := app.beginRunScope(" conversation ", " run "); err != nil {
		t.Fatal(err)
	}
	defer app.endRunScope("conversation", "run")
	if err := app.CancelAgentLoop("conversation", "run"); err != nil {
		t.Fatalf("CancelAgentLoop: %v", err)
	}
	if _, err := orch.AgentLoop("conversation", "run", "stop now", nil); !errors.Is(err, context.Canceled) {
		t.Fatalf("AgentLoop error = %v, want context.Canceled", err)
	}
	if runner.called {
		t.Fatal("runner started despite admission-time cancellation")
	}
}

func TestCancelAgentLoopCannotLeaveStaleCancellationForReusedIDs(t *testing.T) {
	store, err := memory.NewStore(config.DatabaseConfig{
		Path:   filepath.Join(t.TempDir(), "memory.db"),
		VecDim: 4,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	runner := &appTestRunner{}
	engine := cognitive.NewEngine(config.ModelConfig{ContextSize: 1024, MaxOutputTokens: 128})
	engine.SetRunner(runner)
	orch := orchestrator.New(orchestrator.Config{
		Logger:        logger.New("error"),
		Memory:        store,
		Cognitive:     engine,
		Tools:         tools.NewRegistry(config.ToolsConfig{}, nil, nil),
		Guardrail:     guardrail.New(config.SecurityConfig{}),
		MaxIterations: 1,
	})
	orch.SetContext(context.Background())
	app := NewApp(orch, nil, RuntimeInfo{})
	coordinator := &blockingRunCancellationCoordinator{
		delegate: orch,
		reached:  make(chan struct{}),
		release:  make(chan struct{}),
	}
	app.runCancels = coordinator

	if err := app.beginRunScope(" conversation ", " run "); err != nil {
		t.Fatal(err)
	}
	cancelDone := make(chan error, 1)
	go func() {
		cancelDone <- app.CancelAgentLoop("conversation", "run")
	}()
	select {
	case <-coordinator.reached:
	case <-time.After(2 * time.Second):
		t.Fatal("cancellation did not reach the synchronized admission gap")
	}
	if app.streamMu.TryLock() {
		app.streamMu.Unlock()
		t.Fatal("cancellation released app admission state before cancel-or-queue completed")
	}

	endDone := make(chan struct{})
	go func() {
		app.endRunScope(" conversation ", " run ")
		close(endDone)
	}()
	close(coordinator.release)
	select {
	case err := <-cancelDone:
		if err != nil {
			t.Fatalf("CancelAgentLoop: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("cancellation did not finish")
	}
	select {
	case <-endDone:
	case <-time.After(2 * time.Second):
		t.Fatal("run scope did not finish after cancellation decision")
	}

	// Reusing both canonical identifiers must start normally. Before the atomic
	// handoff, the late queue could survive endRunScope and cancel this run.
	if err := app.beginRunScope("conversation", "run"); err != nil {
		t.Fatalf("reuse run scope: %v", err)
	}
	defer app.endRunScope("conversation", "run")
	if _, err := orch.AgentLoop(" conversation ", " run ", "run normally", nil); err != nil {
		t.Fatalf("reused identifiers inherited stale cancellation: %v", err)
	}
	if !runner.called {
		t.Fatal("runner did not start with reused identifiers")
	}
}

func TestWipeMemoryCancelsRunBeforeOrchestratorAdmission(t *testing.T) {
	store, err := memory.NewStore(config.DatabaseConfig{
		Path:   filepath.Join(t.TempDir(), "memory.db"),
		VecDim: 4,
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	engine := cognitive.NewEngine(config.ModelConfig{ContextSize: 1024, MaxOutputTokens: 128})
	engine.SetRunner(&appBlockingRunner{})
	orch := orchestrator.New(orchestrator.Config{
		Logger:        logger.New("error"),
		Memory:        store,
		Cognitive:     engine,
		Tools:         tools.NewRegistry(config.ToolsConfig{}, nil, nil),
		Guardrail:     guardrail.New(config.SecurityConfig{}),
		MaxIterations: 1,
	})
	orch.SetContext(context.Background())
	app := NewApp(orch, nil, RuntimeInfo{})

	app.operationMu.Lock()
	operationLocked := true
	defer func() {
		if operationLocked {
			app.operationMu.Unlock()
		}
	}()
	if err := app.beginRunScope(" conversation ", " run "); err != nil {
		t.Fatal(err)
	}

	wipeDone := make(chan error, 1)
	go func() {
		_, err := app.WipeMemory()
		wipeDone <- err
	}()
	deadline := time.Now().Add(2 * time.Second)
	for {
		app.streamMu.RLock()
		purging := app.purging
		app.streamMu.RUnlock()
		if purging {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("purge did not enter admission barrier")
		}
		time.Sleep(time.Millisecond)
	}

	runDone := make(chan error, 1)
	go func() {
		_, err := orch.AgentLoop(" conversation ", " run ", "do not resurrect", nil)
		runDone <- err
	}()
	select {
	case err := <-runDone:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("AgentLoop error = %v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("pre-admission run was not cancelled")
	}
	select {
	case err := <-wipeDone:
		t.Fatalf("purge crossed operation barrier early: %v", err)
	default:
	}

	app.endRunScope(" conversation ", " run ")
	app.operationMu.Unlock()
	operationLocked = false
	select {
	case err := <-wipeDone:
		if err != nil {
			t.Fatalf("WipeMemory: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("purge did not finish after the run left")
	}
	if summaries, err := orch.GetConversations(); err != nil || len(summaries) != 0 {
		t.Fatalf("conversation resurrected after purge: %#v, %v", summaries, err)
	}
}
