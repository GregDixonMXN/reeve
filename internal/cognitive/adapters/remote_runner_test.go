package adapters

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"axiom/pkg/models"
)

func TestRemoteRunnerOllamaChatWithoutToolsUsesThinkingAndConfiguredOptions(t *testing.T) {
	t.Parallel()

	requests := make(chan struct {
		path string
		body ollamaChatReq
	}, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var body ollamaChatReq
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		requests <- struct {
			path string
			body ollamaChatReq
		}{path: req.URL.Path, body: body}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"message":{"role":"assistant","thinking":"native thought","content":"Hello"},
			"done":true
		}`)
	}))
	defer server.Close()

	runner := NewRemoteRunner(RemoteRunnerConfig{
		BaseURL:         server.URL,
		Model:           "qwen-test",
		Protocol:        ProtocolOllama,
		ContextSize:     65536,
		Temperature:     0.6,
		TopP:            0.95,
		TopK:            20,
		MinP:            0.05,
		PresencePenalty: 0.25,
		RepeatPenalty:   1.1,
		Threads:         12,
		Think:           true,
	})
	raw, err := runner.CompleteWithTools(
		context.Background(),
		"system",
		[]models.Message{{Role: models.RoleUser, Content: "hello"}},
		8192,
		nil,
	)
	if err != nil {
		t.Fatalf("CompleteWithTools() error = %v", err)
	}

	var output models.LLMResponse
	if err := json.Unmarshal([]byte(raw), &output); err != nil {
		t.Fatalf("decode Axiom output: %v", err)
	}
	if output.Reasoning != "native thought" || output.Content != "Hello" || output.ToolCall != nil {
		t.Fatalf("output = %#v", output)
	}

	captured := <-requests
	if captured.path != "/api/chat" {
		t.Fatalf("request path = %q, want /api/chat", captured.path)
	}
	if captured.body.Model != "qwen-test" || captured.body.Stream {
		t.Fatalf("request identity = %#v", captured.body)
	}
	if captured.body.Think == nil || !*captured.body.Think {
		t.Fatalf("think = %#v, want explicit true", captured.body.Think)
	}
	if len(captured.body.Tools) != 0 {
		t.Fatalf("tools = %#v, want none", captured.body.Tools)
	}
	if len(captured.body.Messages) != 2 ||
		captured.body.Messages[0].Role != models.RoleSystem ||
		captured.body.Messages[0].Content != "system" ||
		captured.body.Messages[1].Role != models.RoleUser ||
		captured.body.Messages[1].Content != "hello" {
		t.Fatalf("messages = %#v", captured.body.Messages)
	}

	wantOptions := map[string]float64{
		"num_predict":      8192,
		"num_ctx":          65536,
		"temperature":      0.6,
		"top_p":            0.95,
		"top_k":            20,
		"min_p":            0.05,
		"presence_penalty": 0.25,
		"repeat_penalty":   1.1,
		"num_thread":       12,
	}
	for name, want := range wantOptions {
		if got := captured.body.Options[name]; got != want {
			t.Errorf("option %s = %#v, want %v", name, got, want)
		}
	}
}

func TestRemoteRunnerOllamaChatSendsExplicitThinkFalse(t *testing.T) {
	t.Parallel()

	requests := make(chan ollamaChatReq, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var body ollamaChatReq
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		requests <- body
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"message":{"role":"assistant","content":"done"},"done":true}`)
	}))
	defer server.Close()

	runner := NewRemoteRunner(RemoteRunnerConfig{
		BaseURL:  server.URL,
		Model:    "qwen-test",
		Protocol: ProtocolOllama,
		Think:    false,
	})
	if _, err := runner.CompleteWithTools(
		context.Background(),
		"system",
		[]models.Message{{Role: models.RoleUser, Content: "hello"}},
		512,
		nil,
	); err != nil {
		t.Fatalf("CompleteWithTools() error = %v", err)
	}

	body := <-requests
	if body.Think == nil || *body.Think {
		t.Fatalf("think = %#v, want explicit false", body.Think)
	}
	if body.Options["temperature"] != defaultRemoteTemperature ||
		body.Options["top_p"] != defaultRemoteTopP {
		t.Fatalf("default options = %#v", body.Options)
	}
	if got := body.Options["min_p"]; got != float64(0) {
		t.Fatalf("default min_p = %#v, want explicit zero", got)
	}
	if got := body.Options["presence_penalty"]; got != float64(0) {
		t.Fatalf("default presence_penalty = %#v, want explicit zero", got)
	}
	if got := body.Options["repeat_penalty"]; got != defaultRemoteRepeat {
		t.Fatalf("default repeat_penalty = %#v, want %v", got, defaultRemoteRepeat)
	}
	if _, exists := body.Options["top_k"]; exists {
		t.Fatalf("zero top_k should use Ollama default: %#v", body.Options)
	}
	if _, exists := body.Options["num_thread"]; exists {
		t.Fatalf("zero threads should use Ollama auto-selection: %#v", body.Options)
	}
}

func TestRemoteRunnerPreservesNativeOllamaToolTurnForConversation(t *testing.T) {
	t.Parallel()

	requests := make(chan ollamaChatReq, 2)
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var body ollamaChatReq
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		requests <- body
		w.Header().Set("Content-Type", "application/json")
		if calls.Add(1) == 1 {
			_, _ = io.WriteString(w, `{
				"message":{
					"role":"assistant",
					"thinking":"Need the file",
					"content":"I will inspect it.",
					"tool_calls":[{"type":"function","function":{
						"index":0,
						"name":"read_file",
						"arguments":{"path":"notes.txt","options":{"lines":5}}
					}}]
				},
				"done":true
			}`)
			return
		}
		_, _ = io.WriteString(w, `{
			"message":{
				"role":"assistant",
				"thinking":"The tool returned the answer",
				"content":"The file says hello."
			},
			"done":true
		}`)
	}))
	defer server.Close()

	runner := NewRemoteRunner(RemoteRunnerConfig{
		BaseURL:  server.URL,
		Model:    "qwen-test",
		Protocol: ProtocolOllama,
		Think:    true,
	})
	tool := models.ToolDefinition{
		Name:       "read_file",
		ArgsSchema: `{"path":"string"}`,
	}
	userMessage := models.Message{Role: models.RoleUser, Content: "What is in notes.txt?"}

	firstRaw, err := runner.CompleteWithToolsForConversation(
		context.Background(),
		"conversation-tool-loop",
		"system",
		[]models.Message{userMessage},
		512,
		[]models.ToolDefinition{tool},
	)
	if err != nil {
		t.Fatalf("first completion error = %v", err)
	}
	var first models.LLMResponse
	if err := json.Unmarshal([]byte(firstRaw), &first); err != nil {
		t.Fatalf("decode first output: %v", err)
	}
	if first.ToolCall == nil || first.ToolCall.Name != "read_file" {
		t.Fatalf("first tool call = %#v", first.ToolCall)
	}
	if first.Reasoning != "Need the file\n\nI will inspect it." {
		t.Fatalf("first reasoning = %q", first.Reasoning)
	}

	const toolResult = "[TOOL RESULT]\nhello\n[END TOOL RESULT]"
	secondRaw, err := runner.CompleteWithToolsForConversation(
		context.Background(),
		"conversation-tool-loop",
		"system changed but must not replace the active loop",
		[]models.Message{
			userMessage,
			{Role: models.RoleAssistant, Content: firstRaw},
			{Role: models.RoleTool, Content: toolResult},
		},
		512,
		[]models.ToolDefinition{tool},
	)
	if err != nil {
		t.Fatalf("second completion error = %v", err)
	}
	var second models.LLMResponse
	if err := json.Unmarshal([]byte(secondRaw), &second); err != nil {
		t.Fatalf("decode second output: %v", err)
	}
	if second.Reasoning != "The tool returned the answer" ||
		second.Content != "The file says hello." ||
		second.ToolCall != nil {
		t.Fatalf("second output = %#v", second)
	}

	firstRequest := <-requests
	if len(firstRequest.Messages) != 2 {
		t.Fatalf("first request messages = %#v", firstRequest.Messages)
	}
	secondRequest := <-requests
	if len(secondRequest.Messages) != 4 {
		t.Fatalf("second request has %d messages, want system + user + native assistant + tool", len(secondRequest.Messages))
	}
	if secondRequest.Messages[0].Content != "system" {
		t.Fatalf("active-loop system message = %q, want original system", secondRequest.Messages[0].Content)
	}

	assistant := secondRequest.Messages[2]
	if assistant.Role != models.RoleAssistant ||
		assistant.Thinking != "Need the file" ||
		assistant.Content != "I will inspect it." ||
		len(assistant.ToolCalls) != 1 ||
		assistant.ToolCalls[0].Type != "function" ||
		assistant.ToolCalls[0].Function.Index == nil ||
		*assistant.ToolCalls[0].Function.Index != 0 ||
		assistant.ToolCalls[0].Function.Name != "read_file" {
		t.Fatalf("replayed native assistant = %#v", assistant)
	}
	args, err := json.Marshal(assistant.ToolCalls[0].Function.Arguments)
	if err != nil {
		t.Fatalf("encode replayed arguments: %v", err)
	}
	if string(args) != `{"options":{"lines":5},"path":"notes.txt"}` {
		t.Fatalf("replayed arguments = %s", args)
	}

	toolMessage := secondRequest.Messages[3]
	if toolMessage.Role != models.RoleTool ||
		toolMessage.ToolName != "read_file" ||
		toolMessage.Content != toolResult ||
		toolMessage.Thinking != "" ||
		len(toolMessage.ToolCalls) != 0 {
		t.Fatalf("native tool result = %#v", toolMessage)
	}

	runner.mu.Lock()
	_, stillPending := runner.pending["conversation-tool-loop"]
	runner.mu.Unlock()
	if stillPending {
		t.Fatal("final assistant response did not clear pending tool state")
	}
}

func TestRemoteRunnerConversationStateIsKeyedAndMismatchClears(t *testing.T) {
	t.Parallel()

	requests := make(chan ollamaChatReq, 3)
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var body ollamaChatReq
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		requests <- body
		w.Header().Set("Content-Type", "application/json")
		if calls.Add(1) == 1 {
			_, _ = io.WriteString(w, `{
				"message":{
					"role":"assistant",
					"thinking":"private state",
					"content":"",
					"tool_calls":[{"function":{"name":"read_file","arguments":{"path":"a"}}}]
				},
				"done":true
			}`)
			return
		}
		_, _ = io.WriteString(w, `{"message":{"role":"assistant","content":"done"},"done":true}`)
	}))
	defer server.Close()

	runner := NewRemoteRunner(RemoteRunnerConfig{
		BaseURL:  server.URL,
		Model:    "qwen-test",
		Protocol: ProtocolOllama,
	})
	user := models.Message{Role: models.RoleUser, Content: "read a"}
	firstRaw, err := runner.CompleteWithToolsForConversation(
		context.Background(),
		"conversation-a",
		"system",
		[]models.Message{user},
		512,
		[]models.ToolDefinition{{Name: "read_file"}},
	)
	if err != nil {
		t.Fatalf("first completion error = %v", err)
	}
	<-requests

	followup := []models.Message{
		user,
		{Role: models.RoleAssistant, Content: firstRaw},
		{Role: models.RoleTool, Content: "result"},
	}
	if _, err := runner.CompleteWithToolsForConversation(
		context.Background(),
		"conversation-b",
		"system b",
		followup,
		512,
		[]models.ToolDefinition{{Name: "read_file"}},
	); err != nil {
		t.Fatalf("conversation-b completion error = %v", err)
	}
	keyedRequest := <-requests
	if containsNativeOllamaToolTurn(keyedRequest.Messages) {
		t.Fatalf("conversation-b replayed conversation-a provider state: %#v", keyedRequest.Messages)
	}
	if got := keyedRequest.Messages[len(keyedRequest.Messages)-1]; got.Role != models.RoleUser ||
		!strings.HasPrefix(got.Content, "[Tool Result]\n") {
		t.Fatalf("unlinked tool output = %#v, want safe user context", got)
	}

	var mismatched models.LLMResponse
	if err := json.Unmarshal([]byte(firstRaw), &mismatched); err != nil {
		t.Fatalf("decode first output: %v", err)
	}
	mismatched.ToolCall.Name = "write_file"
	mismatchJSON, _ := json.Marshal(mismatched)
	if _, err := runner.CompleteWithToolsForConversation(
		context.Background(),
		"conversation-a",
		"replacement system",
		[]models.Message{
			user,
			{Role: models.RoleAssistant, Content: string(mismatchJSON)},
			{Role: models.RoleTool, Content: "result"},
		},
		512,
		[]models.ToolDefinition{{Name: "read_file"}},
	); err != nil {
		t.Fatalf("mismatch completion error = %v", err)
	}
	mismatchRequest := <-requests
	if containsNativeOllamaToolTurn(mismatchRequest.Messages) {
		t.Fatalf("mismatch replayed stale provider state: %#v", mismatchRequest.Messages)
	}
	if mismatchRequest.Messages[0].Content != "replacement system" {
		t.Fatalf("mismatch did not build fresh input: %#v", mismatchRequest.Messages)
	}

	runner.mu.Lock()
	_, stillPending := runner.pending["conversation-a"]
	runner.mu.Unlock()
	if stillPending {
		t.Fatal("mismatched turn did not clear pending tool state")
	}
}

func TestRemoteRunnerDoesNotSerializeDifferentConversations(t *testing.T) {
	t.Parallel()

	firstStarted := make(chan struct{})
	secondStarted := make(chan struct{})
	releaseFirst := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var body ollamaChatReq
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		switch body.Messages[len(body.Messages)-1].Content {
		case "first":
			close(firstStarted)
			<-releaseFirst
		case "second":
			close(secondStarted)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"message":{"role":"assistant","content":"done"},"done":true}`)
	}))
	defer server.Close()

	runner := NewRemoteRunner(RemoteRunnerConfig{
		BaseURL:  server.URL,
		Model:    "qwen-test",
		Protocol: ProtocolOllama,
	})
	errs := make(chan error, 2)
	run := func(conversationID, content string) {
		_, err := runner.CompleteWithToolsForConversation(
			context.Background(),
			conversationID,
			"system",
			[]models.Message{{Role: models.RoleUser, Content: content}},
			512,
			nil,
		)
		errs <- err
	}

	go run("conversation-first", "first")
	<-firstStarted
	go run("conversation-second", "second")

	var timedOut bool
	select {
	case <-secondStarted:
	case <-time.After(2 * time.Second):
		timedOut = true
	}
	close(releaseFirst)
	for range 2 {
		if err := <-errs; err != nil {
			t.Fatalf("completion error = %v", err)
		}
	}
	if timedOut {
		t.Fatal("second conversation was blocked behind the first conversation's HTTP request")
	}
}

func TestRemoteRunnerRejectsMultipleOllamaToolCalls(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"message":{
				"role":"assistant",
				"tool_calls":[
					{"function":{"name":"read_file","arguments":{"path":"one"}}},
					{"function":{"name":"read_file","arguments":{"path":"two"}}}
				]
			},
			"done":true
		}`)
	}))
	defer server.Close()

	runner := NewRemoteRunner(RemoteRunnerConfig{
		BaseURL:  server.URL,
		Model:    "qwen-test",
		Protocol: ProtocolOllama,
	})
	_, err := runner.CompleteWithToolsForConversation(
		context.Background(),
		"conversation",
		"system",
		[]models.Message{{Role: models.RoleUser, Content: "read both"}},
		512,
		[]models.ToolDefinition{{Name: "read_file"}},
	)
	if err == nil {
		t.Fatal("expected multiple tool calls to be rejected")
	}
	if !strings.Contains(err.Error(), "2 function calls") {
		t.Fatalf("error = %v, want multiple-call detail", err)
	}
}

func TestRemoteRunnerPendingStateTTLAndUnloadClearing(t *testing.T) {
	t.Parallel()

	var unloads atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path != "/api/generate" {
			t.Errorf("request path = %q, want /api/generate", req.URL.Path)
		}
		unloads.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"done":true}`)
	}))
	defer server.Close()

	runner := NewRemoteRunner(RemoteRunnerConfig{
		BaseURL:  server.URL,
		Model:    "qwen-test",
		Protocol: ProtocolOllama,
	})
	now := time.Now()
	runner.pending["expired"] = ollamaPendingToolLoop{updated: now.Add(-ollamaPendingStateTTL - time.Second)}
	runner.pending["fresh"] = ollamaPendingToolLoop{updated: now}
	runner.prunePendingLocked(now)
	if _, exists := runner.pending["expired"]; exists {
		t.Fatal("expired pending state was not pruned")
	}
	if _, exists := runner.pending["fresh"]; !exists {
		t.Fatal("fresh pending state was pruned")
	}

	if err := runner.Unload(); err != nil {
		t.Fatalf("Unload() error = %v", err)
	}
	if unloads.Load() != 1 {
		t.Fatalf("unload requests = %d, want 1", unloads.Load())
	}
	runner.mu.Lock()
	pendingCount := len(runner.pending)
	runner.mu.Unlock()
	if pendingCount != 0 {
		t.Fatalf("pending state count after unload = %d, want 0", pendingCount)
	}
}

func TestRemoteRunnerLegacyGenerateUsesConfiguredOptionsAndThinking(t *testing.T) {
	t.Parallel()

	requests := make(chan map[string]interface{}, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var body map[string]interface{}
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		requests <- body
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"thinking":"legacy thought",
			"response":"{\"reasoning\":\"\",\"tool_call\":null,\"content\":\"done\"}",
			"done":true
		}`)
	}))
	defer server.Close()

	runner := NewRemoteRunner(RemoteRunnerConfig{
		BaseURL:         server.URL,
		Model:           "qwen-test",
		Protocol:        ProtocolOllama,
		ContextSize:     32768,
		Temperature:     0.6,
		TopP:            0.95,
		TopK:            20,
		MinP:            0.05,
		PresencePenalty: 0.25,
		RepeatPenalty:   1.1,
		Threads:         12,
		Think:           true,
	})
	raw, err := runner.Complete(context.Background(), "prompt", 4096)
	if err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	const thinkPrefix = "<think>legacy thought</think>"
	if !strings.HasPrefix(raw, thinkPrefix) {
		t.Fatalf("legacy output = %q, want native thinking prefix", raw)
	}
	var output models.LLMResponse
	if err := json.Unmarshal([]byte(strings.TrimPrefix(raw, thinkPrefix)), &output); err != nil {
		t.Fatalf("decode output: %v", err)
	}
	if output.Reasoning != "" || output.Content != "done" {
		t.Fatalf("output = %#v", output)
	}

	body := <-requests
	if body["think"] != true {
		t.Fatalf("think = %#v, want true", body["think"])
	}
	options, ok := body["options"].(map[string]interface{})
	if !ok {
		t.Fatalf("options = %#v", body["options"])
	}
	for name, want := range map[string]float64{
		"num_predict":      4096,
		"num_ctx":          32768,
		"temperature":      0.6,
		"top_p":            0.95,
		"top_k":            20,
		"min_p":            0.05,
		"presence_penalty": 0.25,
		"repeat_penalty":   1.1,
		"num_thread":       12,
	} {
		if got := options[name]; got != want {
			t.Errorf("option %s = %#v, want %v", name, got, want)
		}
	}
}

func containsNativeOllamaToolTurn(messages []ollamaChatMsg) bool {
	for _, message := range messages {
		if message.Thinking != "" || len(message.ToolCalls) > 0 || message.ToolName != "" {
			return true
		}
	}
	return false
}
