package orchestrator

import (
	"context"
	"errors"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"herald/internal/cognitive"
	"herald/internal/config"
	"herald/internal/guardrail"
	"herald/internal/memory"
	"herald/internal/tools"
	"herald/pkg/logger"
)

type lifecycleRunner struct {
	started chan struct{}
	once    sync.Once
	block   bool
}

func (r *lifecycleRunner) Complete(ctx context.Context, _ string, _ int) (string, error) {
	if r.started != nil {
		r.once.Do(func() { close(r.started) })
	}
	if r.block {
		<-ctx.Done()
		return "", ctx.Err()
	}
	return `{"reasoning":"","tool_call":null,"content":"Lifecycle repaired"}`, nil
}

func (r *lifecycleRunner) Unload() error { return nil }

func newLifecycleTestStore(t *testing.T, path string) *memory.Store {
	t.Helper()
	store, err := memory.NewStore(config.DatabaseConfig{Path: path, VecDim: 4}, nil)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	return store
}

func newLifecycleTestOrchestrator(store *memory.Store, runner cognitive.LLMRunner) *Orchestrator {
	engine := cognitive.NewEngine(config.ModelConfig{ContextSize: 4096, MaxOutputTokens: 256})
	engine.SetRunner(runner)
	return New(Config{
		Logger:        logger.New("error"),
		Memory:        store,
		Cognitive:     engine,
		Tools:         tools.NewRegistry(config.ToolsConfig{}, nil, nil),
		Guardrail:     guardrail.New(config.SecurityConfig{}),
		MaxIterations: 2,
	})
}

func TestConversationLifecycleRestoresTitleAndMessages(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "memory.db")
	store := newLifecycleTestStore(t, dbPath)
	orch := newLifecycleTestOrchestrator(store, &lifecycleRunner{})
	orch.SetContext(context.Background())

	var events []LoopEvent
	response, err := orch.AgentLoop("conv-restore", "run-1", "Repair conversation restoration in Herald", func(event LoopEvent) {
		events = append(events, event)
	})
	if err != nil {
		t.Fatalf("AgentLoop: %v", err)
	}
	if response.Content != "Lifecycle repaired" {
		t.Fatalf("response content = %q", response.Content)
	}
	if len(events) == 0 {
		t.Fatal("expected progress events")
	}
	for _, event := range events {
		if event.ConversationID != "conv-restore" || event.RunID != "run-1" || event.MaxIterations != 2 {
			t.Fatalf("unscoped event: %#v", event)
		}
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	reopened := newLifecycleTestStore(t, dbPath)
	defer reopened.Close()
	restarted := newLifecycleTestOrchestrator(reopened, &lifecycleRunner{})
	restarted.SetContext(context.Background())

	summaries, err := restarted.GetConversations()
	if err != nil {
		t.Fatalf("GetConversations: %v", err)
	}
	if len(summaries) != 1 {
		t.Fatalf("summaries len = %d, want 1", len(summaries))
	}
	if summaries[0].Title != "Repair conversation restoration in Herald" || summaries[0].MessageCount != 2 {
		t.Fatalf("summary = %#v", summaries[0])
	}
	conversation, err := restarted.GetConversation("conv-restore")
	if err != nil {
		t.Fatalf("GetConversation: %v", err)
	}
	if conversation.Title != summaries[0].Title || len(conversation.Messages) != 2 {
		t.Fatalf("conversation = %#v", conversation)
	}
	if conversation.Messages[0].Content != "Repair conversation restoration in Herald" || conversation.Messages[1].Content != "Lifecycle repaired" {
		t.Fatalf("messages = %#v", conversation.Messages)
	}
}

func TestCancellationRequiresExactConversationAndRun(t *testing.T) {
	store := newLifecycleTestStore(t, filepath.Join(t.TempDir(), "memory.db"))
	defer store.Close()
	runner := &lifecycleRunner{started: make(chan struct{}), block: true}
	orch := newLifecycleTestOrchestrator(store, runner)
	orch.SetContext(context.Background())

	var events []LoopEvent
	var eventMu sync.Mutex
	done := make(chan error, 1)
	go func() {
		_, err := orch.AgentLoop("conv-cancel", "run-cancel", "wait here", func(event LoopEvent) {
			eventMu.Lock()
			events = append(events, event)
			eventMu.Unlock()
		})
		done <- err
	}()

	select {
	case <-runner.started:
	case <-time.After(2 * time.Second):
		t.Fatal("runner did not start")
	}
	if _, err := orch.AgentLoop("conv-cancel", "another-run", "overlap", nil); err == nil {
		t.Fatal("second run for the same conversation was accepted")
	}
	if _, err := orch.AgentLoop("another-conversation", "run-cancel", "duplicate", nil); err == nil {
		t.Fatal("duplicate active run id was accepted")
	}
	if err := orch.CancelAgentLoop("wrong-conversation", "run-cancel"); err == nil {
		t.Fatal("mismatched conversation unexpectedly cancelled run")
	}
	select {
	case err := <-done:
		t.Fatalf("run stopped after mismatched cancellation: %v", err)
	default:
	}
	if err := orch.CancelAgentLoop("conv-cancel", "run-cancel"); err != nil {
		t.Fatalf("CancelAgentLoop: %v", err)
	}
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("AgentLoop error = %v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("cancelled run did not stop")
	}

	eventMu.Lock()
	defer eventMu.Unlock()
	if len(events) < 2 {
		t.Fatalf("events = %#v", events)
	}
	last := events[len(events)-1]
	if last.Kind != LoopEventError || last.Message != "Run cancelled" || last.ConversationID != "conv-cancel" || last.RunID != "run-cancel" {
		t.Fatalf("last event = %#v", last)
	}
	conversation, getErr := orch.GetConversation("conv-cancel")
	if getErr != nil {
		t.Fatalf("GetConversation after cancellation: %v", getErr)
	}
	if len(conversation.Messages) != 2 || conversation.Messages[1].Content != "[RUN CANCELLED]" {
		t.Fatalf("cancelled conversation = %#v", conversation.Messages)
	}
	record, loadErr := store.LoadConversationRecord(context.Background(), "conv-cancel")
	if loadErr != nil {
		t.Fatalf("LoadConversationRecord after cancellation: %v", loadErr)
	}
	if record == nil || len(record.Messages) != 2 || record.Messages[1].Content != "[RUN CANCELLED]" {
		t.Fatalf("persisted cancelled conversation = %#v", record)
	}
}

func TestQueuedCancellationClosesAdmissionRace(t *testing.T) {
	store := newLifecycleTestStore(t, filepath.Join(t.TempDir(), "memory.db"))
	defer store.Close()
	runner := &lifecycleRunner{started: make(chan struct{}), block: true}
	orch := newLifecycleTestOrchestrator(store, runner)
	orch.SetContext(context.Background())
	if err := orch.QueueAgentLoopCancellation("conv-pending", "run-pending"); err != nil {
		t.Fatalf("QueueAgentLoopCancellation: %v", err)
	}
	_, err := orch.AgentLoop("conv-pending", "run-pending", "cancel immediately", nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("AgentLoop error = %v, want context.Canceled", err)
	}
	select {
	case <-runner.started:
		t.Fatal("runner started despite queued cancellation")
	default:
	}
	conversation, err := orch.GetConversation("conv-pending")
	if err != nil {
		t.Fatal(err)
	}
	if len(conversation.Messages) != 2 || conversation.Messages[1].Content != "[RUN CANCELLED]" {
		t.Fatalf("conversation = %#v", conversation.Messages)
	}
}

func TestWipeCancelsRunAndCannotResurrectConversation(t *testing.T) {
	store := newLifecycleTestStore(t, filepath.Join(t.TempDir(), "memory.db"))
	defer store.Close()
	runner := &lifecycleRunner{started: make(chan struct{}), block: true}
	orch := newLifecycleTestOrchestrator(store, runner)
	orch.SetContext(context.Background())

	done := make(chan error, 1)
	go func() {
		_, err := orch.AgentLoop("conv-wipe", "run-wipe", "erase this", nil)
		done <- err
	}()
	select {
	case <-runner.started:
	case <-time.After(2 * time.Second):
		t.Fatal("runner did not start")
	}

	if err := orch.WipeMemory(); err != nil {
		t.Fatalf("WipeMemory: %v", err)
	}
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("AgentLoop error = %v, want context.Canceled", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("wipe did not stop active run")
	}
	if summaries, err := orch.GetConversations(); err != nil || len(summaries) != 0 {
		t.Fatalf("conversations resurrected after wipe: %#v", summaries)
	}
	record, err := store.LoadConversationRecord(context.Background(), "conv-wipe")
	if err != nil {
		t.Fatalf("LoadConversationRecord: %v", err)
	}
	if record != nil {
		t.Fatalf("persisted conversation resurrected after wipe: %#v", record)
	}
}

func TestDeriveConversationTitle(t *testing.T) {
	if got := deriveConversationTitle("  Analyze   the Herald project\ncarefully  "); got != "Analyze the Herald project carefully" {
		t.Fatalf("title = %q", got)
	}
	long := deriveConversationTitle("This is a deliberately long conversation prompt that should be shortened without producing an enormous sidebar title")
	if len([]rune(long)) > 60 || long[len(long)-len("…"):] != "…" {
		t.Fatalf("long title = %q", long)
	}
}

func TestGetConversationDoesNotCreatePhantomHistory(t *testing.T) {
	store := newLifecycleTestStore(t, filepath.Join(t.TempDir(), "memory.db"))
	defer store.Close()
	orch := newLifecycleTestOrchestrator(store, &lifecycleRunner{})
	orch.SetContext(context.Background())
	conversation, err := orch.GetConversation("missing")
	if err != nil {
		t.Fatal(err)
	}
	if conversation.ID != "missing" || len(conversation.Messages) != 0 {
		t.Fatalf("missing conversation = %#v", conversation)
	}
	if summaries, err := orch.GetConversations(); err != nil || len(summaries) != 0 {
		t.Fatalf("read created phantom summaries: %#v", summaries)
	}
}

func TestGetConversationPropagatesStorageFailure(t *testing.T) {
	store := newLifecycleTestStore(t, filepath.Join(t.TempDir(), "memory.db"))
	orch := newLifecycleTestOrchestrator(store, &lifecycleRunner{})
	orch.SetContext(context.Background())
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := orch.GetConversation("unreadable"); err == nil {
		t.Fatal("GetConversation converted storage failure into an empty conversation")
	}
	if summaries, err := orch.GetConversations(); err == nil {
		t.Fatalf("GetConversations swallowed storage failure and returned %#v", summaries)
	}
}
