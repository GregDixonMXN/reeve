package adapters

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
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

func TestRemoteRunnerCompactsOversizedOllamaChatBeforePOST(t *testing.T) {
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
		_, _ = io.WriteString(w, `{
			"message":{"role":"assistant","content":"compacted"},
			"done":true,
			"done_reason":"stop"
		}`)
	}))
	defer server.Close()

	const (
		contextSize    = 4096
		maxTokens      = 512
		systemSentinel = "SYSTEM SENTINEL — preserve this instruction."
		latestSentinel = "LATEST USER SENTINEL — preserve exactly."
		oldSentinel    = "OLDEST HISTORY SENTINEL — omit this."
		toolSentinel   = "OVERSIZED TOOL TAIL — omit this."
	)
	messages := []models.Message{{
		Role:    models.RoleUser,
		Content: oldSentinel + strings.Repeat("a", 32*1024),
	}}
	for index := 0; index < 12; index++ {
		messages = append(messages,
			models.Message{
				Role:    models.RoleAssistant,
				Content: strings.Repeat("assistant-history-", 900),
			},
			models.Message{
				Role:    models.RoleUser,
				Content: strings.Repeat("user-history-", 900),
			},
		)
	}
	messages = append(messages,
		models.Message{
			Role:    models.RoleTool,
			Content: toolSentinel + strings.Repeat("z", 256*1024),
		},
		models.Message{Role: models.RoleUser, Content: latestSentinel},
	)
	tool := models.ToolDefinition{
		Name:        "write_file",
		Description: "Write one complete file inside the approved workspace.",
		ArgsSchema:  `{"path":"string","content":"string"}`,
	}
	runner := NewRemoteRunner(RemoteRunnerConfig{
		BaseURL:     server.URL,
		Model:       "qwen-test",
		Protocol:    ProtocolOllama,
		ContextSize: contextSize,
		Think:       true,
	})
	if _, err := runner.CompleteWithTools(
		context.Background(),
		systemSentinel,
		messages,
		maxTokens,
		[]models.ToolDefinition{tool},
	); err != nil {
		t.Fatalf("CompleteWithTools() error = %v", err)
	}

	request := <-requests
	if len(request.Messages) < 3 {
		t.Fatalf("compacted messages = %#v", request.Messages)
	}
	if !strings.HasPrefix(request.Messages[0].Content, systemSentinel) {
		t.Fatalf("system message = %q, want sentinel prefix", request.Messages[0].Content)
	}
	if got := request.Messages[len(request.Messages)-1]; got.Role != models.RoleUser ||
		got.Content != latestSentinel {
		t.Fatalf("latest message = %#v, want exact latest user sentinel", got)
	}
	var joined strings.Builder
	for _, message := range request.Messages {
		joined.WriteString(message.Content)
	}
	contents := joined.String()
	if strings.Contains(contents, oldSentinel) || strings.Contains(contents, toolSentinel) {
		t.Fatalf("compacted request retained oversized history sentinel")
	}
	if !strings.Contains(contents, "CONTEXT COMPACTED") {
		t.Fatalf("compacted request has no omission notice: %#v", request.Messages)
	}
	promptBudget := contextSize - maxTokens - ollamaPromptSafetyTokens
	estimate := estimateOllamaMessagesTokens(request.Messages) +
		estimateOllamaToolTokens(request.Tools)
	if estimate > promptBudget {
		t.Fatalf("estimated prompt tokens = %d, budget = %d", estimate, promptBudget)
	}
	if request.Options["num_ctx"] != float64(contextSize) ||
		request.Options["num_predict"] != float64(maxTokens) {
		t.Fatalf("options = %#v", request.Options)
	}
}

func TestRemoteRunnerRecoversThinkingOnlyLengthResponse(t *testing.T) {
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
					"thinking":"I am still planning...",
					"content":""
				},
				"done":true,
				"done_reason":"length",
				"eval_count":256
			}`)
			return
		}
		_, _ = io.WriteString(w, `{
			"message":{"role":"assistant","content":"Recovered answer."},
			"done":true,
			"done_reason":"stop"
		}`)
	}))
	defer server.Close()

	runner := NewRemoteRunner(RemoteRunnerConfig{
		BaseURL:     server.URL,
		Model:       "qwen-test",
		Protocol:    ProtocolOllama,
		ContextSize: 4096,
		Think:       true,
	})
	raw, err := runner.CompleteWithTools(
		context.Background(),
		"system",
		[]models.Message{{Role: models.RoleUser, Content: "do the task"}},
		256,
		[]models.ToolDefinition{{Name: "write_file"}},
	)
	if err != nil {
		t.Fatalf("CompleteWithTools() error = %v", err)
	}
	var output models.LLMResponse
	if err := json.Unmarshal([]byte(raw), &output); err != nil {
		t.Fatalf("decode output: %v", err)
	}
	if output.Content != "Recovered answer." {
		t.Fatalf("output = %#v", output)
	}
	if calls.Load() != 2 {
		t.Fatalf("HTTP calls = %d, want one recovery retry", calls.Load())
	}
	first := <-requests
	second := <-requests
	if first.Think == nil || !*first.Think {
		t.Fatalf("first think = %#v, want true", first.Think)
	}
	if second.Think == nil || *second.Think {
		t.Fatalf("recovery think = %#v, want false", second.Think)
	}
	if second.Options["num_ctx"] != first.Options["num_ctx"] ||
		second.Options["num_predict"] != first.Options["num_predict"] ||
		len(second.Tools) != len(first.Tools) {
		t.Fatalf("recovery changed options/tools: first=%#v second=%#v", first, second)
	}
	if !strings.Contains(second.Messages[0].Content, "OUTPUT BUDGET RECOVERY") {
		t.Fatalf("recovery system message = %q", second.Messages[0].Content)
	}
}

func TestRemoteRunnerRejectsThinkingOnlyStopResponseWithoutRetry(t *testing.T) {
	t.Parallel()

	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"message":{"role":"assistant","thinking":"unfinished","content":""},
			"done":true,
			"done_reason":"stop"
		}`)
	}))
	defer server.Close()

	runner := NewRemoteRunner(RemoteRunnerConfig{
		BaseURL:  server.URL,
		Model:    "qwen-test",
		Protocol: ProtocolOllama,
		Think:    true,
	})
	_, err := runner.CompleteWithTools(
		context.Background(),
		"system",
		[]models.Message{{Role: models.RoleUser, Content: "hello"}},
		256,
		nil,
	)
	if err == nil || !strings.Contains(err.Error(), "reasoning but no assistant content") {
		t.Fatalf("error = %v, want thinking-only detail", err)
	}
	if calls.Load() != 1 {
		t.Fatalf("HTTP calls = %d, want no retry for done_reason=stop", calls.Load())
	}
}

func TestRemoteRunnerLengthRecoveryToolCallReplaysRetriedInput(t *testing.T) {
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
		switch calls.Add(1) {
		case 1:
			_, _ = io.WriteString(w, `{
				"message":{"role":"assistant","thinking":"too much planning","content":""},
				"done":true,
				"done_reason":"length",
				"eval_count":512
			}`)
		case 2:
			_, _ = io.WriteString(w, `{
				"message":{
					"role":"assistant",
					"content":"Acting now.",
					"tool_calls":[{"type":"function","function":{
						"index":4,
						"name":"read_file",
						"arguments":{"path":"recovered.txt","options":{"lines":7}}
					}}]
				},
				"done":true,
				"done_reason":"stop"
			}`)
		default:
			_, _ = io.WriteString(w, `{
				"message":{"role":"assistant","content":"Recovery replay complete."},
				"done":true,
				"done_reason":"stop"
			}`)
		}
	}))
	defer server.Close()

	runner := NewRemoteRunner(RemoteRunnerConfig{
		BaseURL:     server.URL,
		Model:       "qwen-test",
		Protocol:    ProtocolOllama,
		ContextSize: 4096,
		Think:       true,
	})
	tool := models.ToolDefinition{
		Name:       "read_file",
		ArgsSchema: `{"path":"string"}`,
	}
	user := models.Message{Role: models.RoleUser, Content: "read recovered.txt"}
	firstRaw, err := runner.CompleteWithToolsForConversation(
		context.Background(),
		"recovered-tool-loop",
		"system",
		[]models.Message{user},
		512,
		[]models.ToolDefinition{tool},
	)
	if err != nil {
		t.Fatalf("recovery completion error = %v", err)
	}
	var first models.LLMResponse
	if err := json.Unmarshal([]byte(firstRaw), &first); err != nil {
		t.Fatalf("decode recovered tool response: %v", err)
	}
	if first.ToolCall == nil ||
		first.ToolCall.Name != "read_file" ||
		first.ToolCall.Args["path"] != "recovered.txt" {
		t.Fatalf("recovered tool call = %#v", first.ToolCall)
	}
	initialRequest := <-requests
	recoveryRequest := <-requests
	if initialRequest.Think == nil || !*initialRequest.Think ||
		recoveryRequest.Think == nil || *recoveryRequest.Think {
		t.Fatalf(
			"initial/recovery think = %#v/%#v",
			initialRequest.Think,
			recoveryRequest.Think,
		)
	}
	runner.mu.Lock()
	pending := runner.pending["recovered-tool-loop"]
	runner.mu.Unlock()
	if !reflect.DeepEqual(pending.input, recoveryRequest.Messages) {
		t.Fatal("pending state did not store the retried recovery input")
	}

	const toolResult = "[TOOL RESULT]\nrecovered data\n[END TOOL RESULT]"
	if _, err := runner.CompleteWithToolsForConversation(
		context.Background(),
		"recovered-tool-loop",
		"replacement system",
		[]models.Message{
			user,
			{Role: models.RoleAssistant, Content: firstRaw},
			{Role: models.RoleTool, Content: toolResult},
		},
		512,
		[]models.ToolDefinition{tool},
	); err != nil {
		t.Fatalf("recovered replay completion error = %v", err)
	}
	replayRequest := <-requests
	if len(replayRequest.Messages) < 2 {
		t.Fatalf("replay messages = %#v", replayRequest.Messages)
	}
	assistant := replayRequest.Messages[len(replayRequest.Messages)-2]
	result := replayRequest.Messages[len(replayRequest.Messages)-1]
	if len(assistant.ToolCalls) != 1 ||
		assistant.ToolCalls[0].Function.Index == nil ||
		*assistant.ToolCalls[0].Function.Index != 4 ||
		assistant.ToolCalls[0].Function.Name != "read_file" {
		t.Fatalf("replayed recovery assistant = %#v", assistant)
	}
	options, ok := assistant.ToolCalls[0].Function.Arguments["options"].(map[string]interface{})
	if !ok || options["lines"] != float64(7) {
		t.Fatalf("replayed recovery arguments = %#v", assistant.ToolCalls[0].Function.Arguments)
	}
	if result.Role != models.RoleTool ||
		result.ToolName != "read_file" ||
		result.Content != toolResult {
		t.Fatalf("replayed recovery tool result = %#v", result)
	}
}

func TestRemoteRunnerCompactionPreservesPendingNativeToolPair(t *testing.T) {
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
					"thinking":"Need the exact file.",
					"content":"Reading it now.",
					"tool_calls":[{"type":"function","function":{
						"index":0,
						"name":"read_file",
						"arguments":{"path":"notes.txt"}
					}}]
				},
				"done":true,
				"done_reason":"stop"
			}`)
			return
		}
		_, _ = io.WriteString(w, `{
			"message":{"role":"assistant","content":"The file was read."},
			"done":true,
			"done_reason":"stop"
		}`)
	}))
	defer server.Close()

	const oldSentinel = "HUGE OLD CONTEXT MUST STAY OMITTED"
	runner := NewRemoteRunner(RemoteRunnerConfig{
		BaseURL:     server.URL,
		Model:       "qwen-test",
		Protocol:    ProtocolOllama,
		ContextSize: 4096,
		Think:       true,
	})
	tool := models.ToolDefinition{
		Name:       "read_file",
		ArgsSchema: `{"path":"string"}`,
	}
	user := models.Message{Role: models.RoleUser, Content: "Read notes.txt."}
	firstRaw, err := runner.CompleteWithToolsForConversation(
		context.Background(),
		"compact-tool-loop",
		"system",
		[]models.Message{
			{Role: models.RoleUser, Content: oldSentinel + strings.Repeat("x", 180*1024)},
			user,
		},
		512,
		[]models.ToolDefinition{tool},
	)
	if err != nil {
		t.Fatalf("first completion error = %v", err)
	}
	firstRequest := <-requests
	runner.mu.Lock()
	pending := runner.pending["compact-tool-loop"]
	runner.mu.Unlock()
	if !reflect.DeepEqual(pending.input, firstRequest.Messages) {
		t.Fatalf("pending input differs from compacted request")
	}
	for _, message := range pending.input {
		if strings.Contains(message.Content, oldSentinel) {
			t.Fatalf("pending state retained oversized old context")
		}
	}

	const toolResult = "[TOOL RESULT]\nhello\n[END TOOL RESULT]"
	if _, err := runner.CompleteWithToolsForConversation(
		context.Background(),
		"compact-tool-loop",
		"replacement system must not replace active provider state",
		[]models.Message{
			user,
			{Role: models.RoleAssistant, Content: firstRaw},
			{Role: models.RoleTool, Content: toolResult},
		},
		512,
		[]models.ToolDefinition{tool},
	); err != nil {
		t.Fatalf("second completion error = %v", err)
	}
	secondRequest := <-requests
	if len(secondRequest.Messages) < 3 {
		t.Fatalf("second request = %#v", secondRequest.Messages)
	}
	assistant := secondRequest.Messages[len(secondRequest.Messages)-2]
	result := secondRequest.Messages[len(secondRequest.Messages)-1]
	if assistant.Role != models.RoleAssistant ||
		assistant.Thinking != "Need the exact file." ||
		assistant.Content != "Reading it now." ||
		len(assistant.ToolCalls) != 1 ||
		assistant.ToolCalls[0].Function.Name != "read_file" {
		t.Fatalf("replayed native assistant = %#v", assistant)
	}
	if result.Role != models.RoleTool ||
		result.ToolName != "read_file" ||
		result.Content != toolResult {
		t.Fatalf("replayed tool result = %#v", result)
	}
	promptBudget := 4096 - 512 - ollamaPromptSafetyTokens
	estimate := estimateOllamaMessagesTokens(secondRequest.Messages) +
		estimateOllamaToolTokens(secondRequest.Tools)
	if estimate > promptBudget {
		t.Fatalf("second request estimate = %d, budget = %d", estimate, promptBudget)
	}
}

func TestCompactOllamaChatMessagesKeepsTaskAndWholeNativePairs(t *testing.T) {
	t.Parallel()

	nativeCall := func(name string) ollamaChatMsg {
		var call ollamaToolCall
		call.Type = "function"
		call.Function.Name = name
		call.Function.Arguments = map[string]interface{}{"path": name + ".txt"}
		return ollamaChatMsg{
			Role:      models.RoleAssistant,
			Thinking:  "use " + name,
			ToolCalls: []ollamaToolCall{call},
		}
	}
	const (
		taskSentinel    = "LATEST GENUINE USER TASK"
		droppedSentinel = "DROP COMPLETE OLD PAIR"
		latestResult    = "LATEST TOOL RESULT"
	)
	messages := []ollamaChatMsg{
		{Role: models.RoleSystem, Content: "system"},
		{Role: models.RoleUser, Content: taskSentinel},
		nativeCall("old_tool"),
		{
			Role:     models.RoleTool,
			ToolName: "old_tool",
			Content:  droppedSentinel + strings.Repeat("x", 64*1024),
		},
		nativeCall("latest_tool"),
		{Role: models.RoleTool, ToolName: "latest_tool", Content: latestResult},
	}
	compacted, err := compactOllamaChatMessages(messages, nil, 4096, 512)
	if err != nil {
		t.Fatalf("compactOllamaChatMessages() error = %v", err)
	}

	var foundTask, foundLatestPair bool
	for index, message := range compacted {
		if message.Content == taskSentinel {
			foundTask = true
		}
		if strings.Contains(message.Content, droppedSentinel) {
			t.Fatal("oversized old native pair was partially retained")
		}
		if message.Role != models.RoleTool {
			continue
		}
		if index == 0 ||
			compacted[index-1].Role != models.RoleAssistant ||
			len(compacted[index-1].ToolCalls) == 0 {
			t.Fatalf("orphaned native tool result at index %d: %#v", index, compacted)
		}
		if message.ToolName == "latest_tool" &&
			message.Content == latestResult &&
			compacted[index-1].ToolCalls[0].Function.Name == "latest_tool" {
			foundLatestPair = true
		}
	}
	if !foundTask || !foundLatestPair {
		t.Fatalf(
			"compacted task/latest pair = %v/%v; messages=%#v",
			foundTask,
			foundLatestPair,
			compacted,
		)
	}
}

func TestCompactOllamaChatMessagesRejectsMandatoryFixedContextOverflow(t *testing.T) {
	t.Parallel()

	_, err := compactOllamaChatMessages(
		[]ollamaChatMsg{
			{Role: models.RoleSystem, Content: strings.Repeat("system-", 5000)},
			{Role: models.RoleUser, Content: "latest task"},
		},
		[]ollamaTool{{
			Type: "function",
			Function: ollamaToolFunction{
				Name:        "write_file",
				Description: strings.Repeat("schema-", 2000),
			},
		}},
		4096,
		512,
	)
	if err == nil {
		t.Fatal("expected fixed system/tool context overflow error")
	}
	if !strings.Contains(err.Error(), "context_size") ||
		!strings.Contains(err.Error(), "increase") {
		t.Fatalf("error = %v, want actionable context_size guidance", err)
	}
}

func TestCompactOllamaChatMessagesTruncatesFinalSyntheticToolResult(t *testing.T) {
	t.Parallel()

	const task = "KEEP THIS USER TASK EXACTLY"
	compacted, err := compactOllamaChatMessages(
		[]ollamaChatMsg{
			{Role: models.RoleSystem, Content: "system"},
			{Role: models.RoleUser, Content: task},
			{
				Role:                models.RoleUser,
				Content:             "[Tool Result]\n" + strings.Repeat("very large output ", 20*1024),
				SyntheticToolResult: true,
			},
		},
		nil,
		4096,
		512,
	)
	if err != nil {
		t.Fatalf("compactOllamaChatMessages() error = %v", err)
	}
	var foundTask, foundCompactedResult bool
	for _, message := range compacted {
		if message.Content == task {
			foundTask = true
		}
		if strings.HasPrefix(message.Content, "[Tool Result]") &&
			strings.Contains(message.Content, "content compacted") {
			foundCompactedResult = true
		}
	}
	if !foundTask || !foundCompactedResult {
		t.Fatalf(
			"task/compacted synthetic result = %v/%v; messages=%#v",
			foundTask,
			foundCompactedResult,
			compacted,
		)
	}
}

func TestCompactOllamaChatMessagesPreservesUserWhoQuotesToolResultPrefix(t *testing.T) {
	t.Parallel()

	const quotedPrompt = "[Tool Result]\nThis is user-provided quoted text, not a protocol message."
	compacted, err := compactOllamaChatMessages(
		[]ollamaChatMsg{
			{Role: models.RoleSystem, Content: "system"},
			{Role: models.RoleAssistant, Content: strings.Repeat("old", 64*1024)},
			{Role: models.RoleUser, Content: quotedPrompt},
		},
		nil,
		4096,
		512,
	)
	if err != nil {
		t.Fatalf("compactOllamaChatMessages() error = %v", err)
	}
	if got := compacted[len(compacted)-1]; got.Role != models.RoleUser ||
		got.Content != quotedPrompt {
		t.Fatalf("quoted current user prompt = %#v, want exact preservation", got)
	}
}

func TestCompactOllamaChatMessagesDropsMalformedNativeOrphansEvenWhenSmall(t *testing.T) {
	t.Parallel()

	var call ollamaToolCall
	call.Function.Name = "read_file"
	call.Function.Arguments = map[string]interface{}{"path": "orphan.txt"}
	compacted, err := compactOllamaChatMessages(
		[]ollamaChatMsg{
			{Role: models.RoleSystem, Content: "system"},
			{Role: models.RoleUser, Content: "current task"},
			{Role: models.RoleAssistant, ToolCalls: []ollamaToolCall{call}},
			{Role: models.RoleAssistant, Content: "unrelated assistant"},
			{Role: models.RoleTool, ToolName: "read_file", Content: "orphan result"},
		},
		nil,
		4096,
		512,
	)
	if err != nil {
		t.Fatalf("compactOllamaChatMessages() error = %v", err)
	}
	for _, message := range compacted {
		if message.Role == models.RoleTool || len(message.ToolCalls) > 0 {
			t.Fatalf("malformed native orphan survived: %#v", compacted)
		}
	}
	var joined strings.Builder
	for _, message := range compacted {
		joined.WriteString(message.Content)
	}
	if !strings.Contains(joined.String(), "CONTEXT COMPACTED") {
		t.Fatalf("orphan sanitization did not add a compaction notice: %#v", compacted)
	}
}

func TestCompactOllamaChatMessagesReservesFullConfiguredOutput(t *testing.T) {
	t.Parallel()

	const (
		contextSize = 8192
		maxTokens   = 6144
	)
	compacted, err := compactOllamaChatMessages(
		[]ollamaChatMsg{
			{Role: models.RoleSystem, Content: "s"},
			{Role: models.RoleAssistant, Content: strings.Repeat("old", 20*1024)},
			{Role: models.RoleUser, Content: "latest"},
		},
		nil,
		contextSize,
		maxTokens,
	)
	if err != nil {
		t.Fatalf("compactOllamaChatMessages() error = %v", err)
	}
	estimate := estimateOllamaMessagesTokens(compacted)
	if estimate+maxTokens+ollamaPromptSafetyTokens > contextSize {
		t.Fatalf(
			"prompt=%d + output=%d + safety=%d exceeds context=%d",
			estimate,
			maxTokens,
			ollamaPromptSafetyTokens,
			contextSize,
		)
	}

	_, err = compactOllamaChatMessages(
		[]ollamaChatMsg{{Role: models.RoleUser, Content: "latest"}},
		nil,
		1024,
		1024,
	)
	if err == nil || !strings.Contains(err.Error(), "max_output_tokens") {
		t.Fatalf("impossible output reserve error = %v", err)
	}
}

func TestCloneOllamaChatMessageDetachesNestedToolArguments(t *testing.T) {
	t.Parallel()

	index := 3
	original := ollamaChatMsg{
		Role: models.RoleAssistant,
		ToolCalls: []ollamaToolCall{{
			Function: struct {
				Index     *int                   `json:"index,omitempty"`
				Name      string                 `json:"name"`
				Arguments map[string]interface{} `json:"arguments"`
			}{
				Index: &index,
				Name:  "write_file",
				Arguments: map[string]interface{}{
					"nested": map[string]interface{}{
						"items": []interface{}{"original"},
					},
				},
			},
		}},
	}
	cloned := cloneOllamaChatMessage(original)
	*cloned.ToolCalls[0].Function.Index = 9
	nested := cloned.ToolCalls[0].Function.Arguments["nested"].(map[string]interface{})
	nested["items"].([]interface{})[0] = "changed"

	if *original.ToolCalls[0].Function.Index != 3 {
		t.Fatalf("clone mutated original index")
	}
	originalNested := original.ToolCalls[0].Function.Arguments["nested"].(map[string]interface{})
	if got := originalNested["items"].([]interface{})[0]; got != "original" {
		t.Fatalf("clone mutated original nested arguments: %#v", got)
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
