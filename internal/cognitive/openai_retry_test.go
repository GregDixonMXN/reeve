package cognitive_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"axiom/internal/cognitive"
	"axiom/internal/cognitive/adapters"
	"axiom/internal/config"
	"axiom/pkg/models"
)

func TestOpenAICompatibilityFallbackRetainsJSONRetry(t *testing.T) {
	t.Parallel()

	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		attempt := calls.Add(1)
		text := "not valid structured output"
		if attempt == 2 {
			text = `{"reasoning":"","tool_call":{"name":"read_file","args":{"path":"/tmp/file"}},"content":""}`
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]string{{"text": text}},
		})
	}))
	defer server.Close()

	runner := adapters.NewRemoteRunner(adapters.RemoteRunnerConfig{
		BaseURL:  server.URL,
		Model:    "test",
		Protocol: adapters.ProtocolOpenAI,
	})
	engine := cognitive.NewEngine(config.ModelConfig{ContextSize: 1024, MaxOutputTokens: 128})
	engine.SetRunner(runner)
	response, err := engine.Generate(context.Background(), cognitive.Request{
		Messages: []models.Message{{Role: models.RoleUser, Content: "read the file"}},
		Tools:    []models.ToolDefinition{{Name: "read_file", ArgsSchema: `{"path":"string"}`}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 2 {
		t.Fatalf("completion calls = %d, want 2", calls.Load())
	}
	if response.ToolCall == nil || response.ToolCall.Name != "read_file" {
		t.Fatalf("retry response = %#v", response)
	}
}
