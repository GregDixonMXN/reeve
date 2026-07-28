package adapters

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"axiom/pkg/models"
)

// Run explicitly with:
//
//	AXIOM_OLLAMA_LIVE=1 go test ./internal/cognitive/adapters -run TestLiveOllamaNativeToolLoop -v
func TestLiveOllamaNativeToolLoop(t *testing.T) {
	if os.Getenv("AXIOM_OLLAMA_LIVE") != "1" {
		t.Skip("set AXIOM_OLLAMA_LIVE=1 to run against the configured local Ollama server")
	}

	baseURL := os.Getenv("AXIOM_OLLAMA_URL")
	if baseURL == "" {
		baseURL = "http://127.0.0.1:11435"
	}
	model := os.Getenv("AXIOM_OLLAMA_MODEL")
	if model == "" {
		model = "qwen3.6:27b-mtp-q4_K_M"
	}

	runner := NewRemoteRunner(RemoteRunnerConfig{
		BaseURL:         baseURL,
		Model:           model,
		Protocol:        ProtocolOllama,
		TimeoutS:        600,
		ContextSize:     65536,
		Temperature:     0.6,
		TopP:            0.95,
		TopK:            20,
		MinP:            0,
		PresencePenalty: 0,
		RepeatPenalty:   1,
		Threads:         12,
		Think:           true,
	})
	t.Cleanup(func() {
		if err := runner.Unload(); err != nil {
			t.Errorf("unload live model: %v", err)
		}
	})

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()

	tool := models.ToolDefinition{
		Name:        "lookup_secret",
		Description: "Return a test value for a key",
		ArgsSchema:  `{"key":"string"}`,
	}
	user := models.Message{
		Role:    models.RoleUser,
		Content: "Use lookup_secret exactly once with key alpha. Then reply with only its returned value.",
	}
	firstRaw, err := runner.CompleteWithToolsForConversation(
		ctx,
		"live-qwen-tool-loop",
		"You are testing a native tool loop. Follow the user exactly.",
		[]models.Message{user},
		256,
		[]models.ToolDefinition{tool},
	)
	if err != nil {
		t.Fatalf("first live completion: %v", err)
	}

	var first models.LLMResponse
	if err := json.Unmarshal([]byte(firstRaw), &first); err != nil {
		t.Fatalf("decode first live completion: %v", err)
	}
	if first.ToolCall == nil ||
		first.ToolCall.Name != "lookup_secret" ||
		first.ToolCall.Args["key"] != "alpha" {
		t.Fatalf("live tool call = %#v", first.ToolCall)
	}

	const marker = "AXIOM_GO_RUNNER_TOOL_LOOP_OK"
	secondRaw, err := runner.CompleteWithToolsForConversation(
		ctx,
		"live-qwen-tool-loop",
		"You are testing a native tool loop. Follow the user exactly.",
		[]models.Message{
			user,
			{Role: models.RoleAssistant, Content: firstRaw},
			{Role: models.RoleTool, Content: marker},
		},
		256,
		[]models.ToolDefinition{tool},
	)
	if err != nil {
		t.Fatalf("second live completion: %v", err)
	}

	var second models.LLMResponse
	if err := json.Unmarshal([]byte(secondRaw), &second); err != nil {
		t.Fatalf("decode second live completion: %v", err)
	}
	if second.ToolCall != nil || !strings.Contains(second.Content, marker) {
		t.Fatalf("live final response = %#v", second)
	}
}
