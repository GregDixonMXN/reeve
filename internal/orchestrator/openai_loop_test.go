package orchestrator

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"axiom/internal/cognitive"
	"axiom/internal/cognitive/adapters"
	"axiom/internal/config"
	"axiom/internal/guardrail"
	"axiom/internal/tools"
	"axiom/pkg/logger"
)

func TestAgentLoopPreservesOpenAICallIDReplay(t *testing.T) {
	project := t.TempDir()
	notePath := filepath.Join(project, "note.txt")
	if err := os.WriteFile(notePath, []byte("orchestrator replay works"), 0o600); err != nil {
		t.Fatal(err)
	}
	arguments, err := json.Marshal(map[string]interface{}{"path": notePath})
	if err != nil {
		t.Fatal(err)
	}

	var requestCount atomic.Int32
	secondRequest := make(chan []byte, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, request *http.Request) {
		body, readErr := io.ReadAll(request.Body)
		if readErr != nil {
			t.Errorf("read request: %v", readErr)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if requestCount.Add(1) == 1 {
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"id":     "response_tool",
				"status": "completed",
				"output": []map[string]interface{}{
					{
						"id":        "function_1",
						"type":      "function_call",
						"status":    "completed",
						"call_id":   "call_from_orchestrator",
						"name":      "read_file",
						"arguments": string(arguments),
					},
				},
			})
			return
		}
		secondRequest <- body
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"id":     "response_final",
			"status": "completed",
			"output": []map[string]interface{}{
				{
					"id":   "message_1",
					"type": "message",
					"role": "assistant",
					"content": []map[string]interface{}{
						{"type": "output_text", "text": "The note was read."},
					},
				},
			},
		})
	}))
	defer server.Close()

	runner := adapters.NewOpenAIRunner(adapters.OpenAIRunnerConfig{
		APIKey:  "test-api-key",
		Model:   "test-model",
		BaseURL: server.URL,
	})
	engine := cognitive.NewEngine(config.ModelConfig{ContextSize: 4096, MaxOutputTokens: 512})
	engine.SetRunner(runner)
	registry := tools.NewRegistry(config.ToolsConfig{
		AllowedDirs: []string{project},
		MaxFileSize: 1024 * 1024,
	}, nil, nil)
	guard := guardrail.New(config.SecurityConfig{
		EnableGuardrails: true,
		AllowNetwork:     true,
	})
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

	response, err := orchestrator.AgentLoop("conversation", "run", "read the note", nil)
	if err != nil {
		t.Fatal(err)
	}
	if response.Content != "The note was read." ||
		len(response.ToolsUsed) != 1 ||
		response.ToolsUsed[0] != "read_file" {
		t.Fatalf("response = %#v", response)
	}

	var replay struct {
		Input []json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(<-secondRequest, &replay); err != nil {
		t.Fatal(err)
	}
	var sawCall, sawResult bool
	for _, raw := range replay.Input {
		var item struct {
			Type   string `json:"type"`
			CallID string `json:"call_id"`
			Output string `json:"output"`
		}
		if err := json.Unmarshal(raw, &item); err != nil {
			t.Fatal(err)
		}
		switch item.Type {
		case "function_call":
			sawCall = item.CallID == "call_from_orchestrator"
		case "function_call_output":
			sawResult = item.CallID == "call_from_orchestrator" &&
				strings.Contains(item.Output, "orchestrator replay works")
		}
	}
	if !sawCall || !sawResult {
		t.Fatalf("OpenAI replay lost native linkage: call=%v result=%v input=%s", sawCall, sawResult, replay.Input)
	}
}
