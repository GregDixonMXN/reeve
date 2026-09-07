package cognitive_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"reeve/internal/cognitive"
	"reeve/internal/cognitive/adapters"
	"reeve/internal/config"
	"reeve/pkg/models"
)

type hybridLocalFailRunner struct{}

func (*hybridLocalFailRunner) Complete(context.Context, string, int) (string, error) {
	return "", errors.New("local runner should not handle cloud-routed test turn")
}

func (*hybridLocalFailRunner) Unload() error { return nil }

func TestHybridOpenAIToolLoopPreservesResponsesCallLinkage(t *testing.T) {
	t.Parallel()

	var (
		requestMu sync.Mutex
		requests  [][]byte
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body json.RawMessage
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		requestMu.Lock()
		requests = append(requests, append([]byte(nil), body...))
		call := len(requests)
		requestMu.Unlock()

		w.Header().Set("Content-Type", "application/json")
		if call == 1 {
			_, _ = w.Write([]byte(`{
				"id":"resp_tool",
				"status":"completed",
				"output":[
					{"type":"reasoning","id":"rs_1","encrypted_content":"opaque-state","summary":[]},
					{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read_file","arguments":"{\"path\":\"/tmp/reeve.txt\"}"}
				]
			}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"id":"resp_final",
			"status":"completed",
			"output":[
				{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}
			]
		}`))
	}))
	defer server.Close()

	cloud := adapters.NewOpenAIRunner(adapters.OpenAIRunnerConfig{
		APIKey:  "sk-hybrid-test",
		Model:   "gpt-test",
		BaseURL: server.URL,
	})
	hybrid := cognitive.NewHybridRunner(cognitive.NewHybridRouter(cognitive.HybridRouterConfig{
		LocalRunner: &hybridLocalFailRunner{},
		CloudRunner: cloud,
	}))
	engine := cognitive.NewEngine(config.ModelConfig{MaxOutputTokens: 512})
	engine.SetRunner(hybrid)
	engine.SetMode("hybrid")

	tools := []models.ToolDefinition{{
		Name:       "read_file",
		ArgsSchema: `{"path":"string"}`,
	}}
	messages := []models.Message{{
		Role:    models.RoleUser,
		Content: "analyze and implement this change",
	}}
	first, err := engine.Generate(context.Background(), cognitive.Request{
		ConversationID: "conversation-1",
		Messages:       messages,
		Tools:          tools,
	})
	if err != nil {
		t.Fatalf("first Generate(): %v", err)
	}
	if first.ToolCall == nil || first.ToolCall.Name != "read_file" {
		t.Fatalf("first response = %#v, want read_file call", first)
	}

	args, err := json.Marshal(first.ToolCall.Args)
	if err != nil {
		t.Fatal(err)
	}
	assistantIntent := fmt.Sprintf(
		`{"reasoning":%q,"tool_call":{"name":%q,"args":%s},"content":""}`,
		first.Reasoning,
		first.ToolCall.Name,
		args,
	)
	messages = append(messages,
		models.Message{Role: models.RoleAssistant, Content: assistantIntent},
		models.Message{Role: models.RoleTool, Content: "file contents"},
	)
	second, err := engine.Generate(context.Background(), cognitive.Request{
		ConversationID: "conversation-1",
		Messages:       messages,
		Tools:          tools,
	})
	if err != nil {
		t.Fatalf("second Generate(): %v", err)
	}
	if second.Content != "done" || second.ToolCall != nil {
		t.Fatalf("second response = %#v, want final done response", second)
	}

	requestMu.Lock()
	captured := append([][]byte(nil), requests...)
	requestMu.Unlock()
	if len(captured) != 2 {
		t.Fatalf("request count = %d, want 2", len(captured))
	}

	var followup struct {
		Input []json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(captured[1], &followup); err != nil {
		t.Fatal(err)
	}
	var (
		sawReasoning      bool
		sawFunctionCall   bool
		sawFunctionOutput bool
	)
	for _, raw := range followup.Input {
		var item struct {
			Type             string `json:"type"`
			CallID           string `json:"call_id"`
			EncryptedContent string `json:"encrypted_content"`
			Output           string `json:"output"`
		}
		if err := json.Unmarshal(raw, &item); err != nil {
			t.Fatal(err)
		}
		switch item.Type {
		case "reasoning":
			sawReasoning = item.EncryptedContent == "opaque-state"
		case "function_call":
			sawFunctionCall = item.CallID == "call_1"
		case "function_call_output":
			sawFunctionOutput = item.CallID == "call_1" && item.Output == "file contents"
		}
	}
	if !sawReasoning || !sawFunctionCall || !sawFunctionOutput {
		t.Fatalf(
			"follow-up linkage reasoning/function/output = %v/%v/%v",
			sawReasoning,
			sawFunctionCall,
			sawFunctionOutput,
		)
	}
}
