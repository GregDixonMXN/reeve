package orchestrator

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"herald/internal/cognitive"
	"herald/internal/config"
	"herald/internal/guardrail"
	"herald/internal/tools"
	"herald/pkg/logger"
	"herald/pkg/models"
)

type sequenceRunner struct {
	mu        sync.Mutex
	responses []string
	calls     int
}

func (r *sequenceRunner) Complete(context.Context, string, int) (string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.calls >= len(r.responses) {
		return "", fmt.Errorf("unexpected completion call %d", r.calls+1)
	}
	response := r.responses[r.calls]
	r.calls++
	return response, nil
}

func (*sequenceRunner) Unload() error { return nil }

func TestEncodeAssistantToolIntentUsesCanonicalEnvelope(t *testing.T) {
	response := &cognitive.Response{
		Reasoning: "inspect",
		Content:   "I will inspect the host.",
		ToolCall: &models.ToolCall{
			Name: "system_info",
			Args: map[string]interface{}{},
		},
	}
	encoded, err := encodeAssistantToolIntent(response)
	if err != nil {
		t.Fatal(err)
	}
	var decoded models.LLMResponse
	if err := json.Unmarshal([]byte(encoded), &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Reasoning != response.Reasoning ||
		decoded.Content != response.Content ||
		decoded.ToolCall == nil ||
		decoded.ToolCall.Name != response.ToolCall.Name {
		t.Fatalf("canonical intent = %#v", decoded)
	}
}

func TestPublicLoopsShareCanonicalToolProtocol(t *testing.T) {
	tests := []struct {
		name string
		run  func(*Orchestrator) (*models.AgentResponse, error)
	}{
		{
			name: "send_message",
			run: func(orchestrator *Orchestrator) (*models.AgentResponse, error) {
				return orchestrator.SendMessage("conversation", "inspect host")
			},
		},
		{
			name: "agent_loop",
			run: func(orchestrator *Orchestrator) (*models.AgentResponse, error) {
				return orchestrator.AgentLoop("conversation", "run", "inspect host", nil)
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			store := newLifecycleTestStore(t, filepath.Join(t.TempDir(), "memory.db"))
			defer store.Close()
			runner := &sequenceRunner{responses: []string{
				`{"reasoning":"inspect","tool_call":{"name":"system_info","args":{}},"content":"Checking the host."}`,
				`{"reasoning":"done","tool_call":null,"content":"Host inspected."}`,
			}}
			orchestrator := newLifecycleTestOrchestrator(store, runner)
			orchestrator.SetContext(context.Background())

			response, err := test.run(orchestrator)
			if err != nil {
				t.Fatal(err)
			}
			if response.Content != "Host inspected." ||
				len(response.ToolsUsed) != 1 ||
				response.ToolsUsed[0] != "system_info" {
				t.Fatalf("response = %#v", response)
			}
			conversation, err := orchestrator.GetConversation("conversation")
			if err != nil {
				t.Fatal(err)
			}
			if len(conversation.Messages) != 4 {
				t.Fatalf("messages = %#v", conversation.Messages)
			}
			var intent models.LLMResponse
			if err := json.Unmarshal([]byte(conversation.Messages[1].Content), &intent); err != nil {
				t.Fatalf("persisted intent is not canonical JSON: %v", err)
			}
			if !conversation.Messages[1].Internal ||
				intent.ToolCall == nil ||
				intent.ToolCall.Name != "system_info" {
				t.Fatalf("persisted intent = %#v", conversation.Messages[1])
			}
			if !conversation.Messages[2].Internal ||
				!strings.HasPrefix(conversation.Messages[2].Content, "[TOOL RESULT]\n") ||
				!strings.HasSuffix(conversation.Messages[2].Content, "\n[END TOOL RESULT]") {
				t.Fatalf("persisted tool result = %#v", conversation.Messages[2])
			}
		})
	}
}

func TestAgentLoopNeverPersistsEmptyFinalResponse(t *testing.T) {
	store := newLifecycleTestStore(t, filepath.Join(t.TempDir(), "memory.db"))
	defer store.Close()
	orchestrator := newLifecycleTestOrchestrator(store, &sequenceRunner{responses: []string{
		`{"reasoning":"","tool_call":null,"content":""}`,
	}})
	orchestrator.SetContext(context.Background())

	response, err := orchestrator.AgentLoop("conversation", "run", "finish", nil)
	if err != nil {
		t.Fatal(err)
	}
	if response.Content != "Action completed successfully." {
		t.Fatalf("response content = %q", response.Content)
	}
	conversation, err := orchestrator.GetConversation("conversation")
	if err != nil {
		t.Fatal(err)
	}
	if len(conversation.Messages) != 2 ||
		conversation.Messages[1].Content != "Action completed successfully." {
		t.Fatalf("messages = %#v", conversation.Messages)
	}
}

func TestAgentLoopRunsConfiguredReflection(t *testing.T) {
	store := newLifecycleTestStore(t, filepath.Join(t.TempDir(), "memory.db"))
	defer store.Close()
	original := strings.Repeat("A", 220)
	runner := &sequenceRunner{responses: []string{
		fmt.Sprintf(`{"reasoning":"","tool_call":null,"content":%q}`, original),
		`{"reasoning":"","tool_call":null,"content":"Improved response."}`,
	}}
	orchestrator := newLifecycleTestOrchestrator(store, runner)
	orchestrator.reflectionEnabled = true
	orchestrator.SetContext(context.Background())

	response, err := orchestrator.AgentLoop("conversation", "run", "review", nil)
	if err != nil {
		t.Fatal(err)
	}
	if response.Content != "Improved response." {
		t.Fatalf("reflected content = %q", response.Content)
	}
	if runner.calls != 2 {
		t.Fatalf("completion calls = %d, want 2", runner.calls)
	}
}

func TestLoopCeilingPoliciesRemainExplicit(t *testing.T) {
	toolCall := `{"reasoning":"","tool_call":{"name":"system_info","args":{}},"content":""}`

	t.Run("send_message_returns_error", func(t *testing.T) {
		store := newLifecycleTestStore(t, filepath.Join(t.TempDir(), "memory.db"))
		defer store.Close()
		orchestrator := newLifecycleTestOrchestrator(store, &sequenceRunner{
			responses: []string{toolCall, toolCall},
		})
		orchestrator.SetContext(context.Background())

		if _, err := orchestrator.SendMessage("conversation", "loop"); err == nil ||
			!strings.Contains(err.Error(), "max tool iterations") {
			t.Fatalf("SendMessage error = %v", err)
		}
	})

	t.Run("agent_loop_returns_partial_status", func(t *testing.T) {
		store := newLifecycleTestStore(t, filepath.Join(t.TempDir(), "memory.db"))
		defer store.Close()
		orchestrator := newLifecycleTestOrchestrator(store, &sequenceRunner{
			responses: []string{toolCall, toolCall},
		})
		orchestrator.SetContext(context.Background())
		var events []LoopEvent

		response, err := orchestrator.AgentLoop("conversation", "run", "loop", func(event LoopEvent) {
			events = append(events, event)
		})
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(response.Content, "2-iteration safety limit") {
			t.Fatalf("response content = %q", response.Content)
		}
		if len(events) == 0 || events[len(events)-1].Kind != LoopEventMaxIter {
			t.Fatalf("events = %#v", events)
		}
	})
}

func TestInterceptedWriteUsesSharedGuardedProtocolAndTracksTool(t *testing.T) {
	project := t.TempDir()
	outputPath := filepath.Join(project, "generated.txt")
	runner := &sequenceRunner{responses: []string{
		fmt.Sprintf(
			`{"reasoning":"write it","tool_call":null,"content":%q}`,
			fmt.Sprintf("# %s\n```text\nwritten through the tool\n```", outputPath),
		),
		`{"reasoning":"","tool_call":null,"content":"File written."}`,
	}}
	engine := cognitive.NewEngine(config.ModelConfig{ContextSize: 4096, MaxOutputTokens: 512})
	engine.SetRunner(runner)
	registry := tools.NewRegistry(config.ToolsConfig{AllowedDirs: []string{project}}, nil, nil)
	guard := guardrail.New(config.SecurityConfig{EnableGuardrails: true, AllowNetwork: true})
	registry.SetGuardrail(guard)
	store := newLifecycleTestStore(t, filepath.Join(t.TempDir(), "memory.db"))
	defer store.Close()
	orchestrator := New(Config{
		Logger:        logger.New("error"),
		Memory:        store,
		Cognitive:     engine,
		Tools:         registry,
		Guardrail:     guard,
		MaxIterations: 3,
	})
	orchestrator.SetContext(context.Background())

	var events []LoopEvent
	response, err := orchestrator.AgentLoop("conversation", "run", "write file", func(event LoopEvent) {
		events = append(events, event)
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(response.ToolsUsed) != 1 || response.ToolsUsed[0] != "write_file" {
		t.Fatalf("tools used = %v", response.ToolsUsed)
	}
	content, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "written through the tool\n" {
		t.Fatalf("file content = %q", content)
	}
	var sawIntercept, sawResult bool
	for _, event := range events {
		if event.Kind == LoopEventToolCall &&
			event.ToolName == "write_file" &&
			strings.Contains(event.Message, "Auto-intercepted") {
			sawIntercept = true
		}
		if event.Kind == LoopEventToolResult && event.ToolName == "write_file" {
			sawResult = true
		}
	}
	if !sawIntercept || !sawResult {
		t.Fatalf("events = %#v", events)
	}
}
