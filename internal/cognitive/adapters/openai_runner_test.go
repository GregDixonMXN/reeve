package adapters

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"herald/pkg/models"
)

type capturedOpenAIRequest struct {
	path          string
	authorization string
	contentType   string
	body          []byte
}

func TestOpenAIRunnerBuildsResponsesRequestAndParsesText(t *testing.T) {
	t.Parallel()

	requests := make(chan capturedOpenAIRequest, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		body, _ := io.ReadAll(req.Body)
		requests <- capturedOpenAIRequest{
			path:          req.URL.Path,
			authorization: req.Header.Get("Authorization"),
			contentType:   req.Header.Get("Content-Type"),
			body:          body,
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"resp_text",
			"status":"completed",
			"output":[{
				"id":"msg_1",
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"done","annotations":[]}]
			}]
		}`)
	}))
	defer server.Close()

	const apiKey = "sk-request-shape-secret"
	runner := NewOpenAIRunner(OpenAIRunnerConfig{
		APIKey:          apiKey,
		Model:           "gpt-test",
		BaseURL:         server.URL,
		MaxTokens:       777,
		ReasoningEffort: "high",
		TimeoutS:        5,
	})
	raw, err := runner.CompleteWithToolsForConversation(
		context.Background(),
		"conversation-1",
		"system context",
		[]models.Message{{Role: models.RoleUser, Content: "hello"}},
		123,
		[]models.ToolDefinition{{
			Name:        "read_file",
			Description: "Read a file",
			ArgsSchema:  `{"path":"string","limit":"int (optional, default 5)"}`,
		}},
	)
	if err != nil {
		t.Fatalf("CompleteWithToolsForConversation() error = %v", err)
	}

	var output openAIHeraldOutput
	if err := json.Unmarshal([]byte(raw), &output); err != nil {
		t.Fatalf("decode Herald output: %v", err)
	}
	if output.Content != "done" || output.ToolCall != nil {
		t.Fatalf("output = %#v, want final content %q", output, "done")
	}

	captured := <-requests
	if captured.path != "/v1/responses" {
		t.Fatalf("request path = %q, want /v1/responses", captured.path)
	}
	if captured.authorization != "Bearer "+apiKey {
		t.Fatalf("Authorization header = %q", captured.authorization)
	}
	if captured.contentType != "application/json" {
		t.Fatalf("Content-Type header = %q", captured.contentType)
	}
	if strings.Contains(string(captured.body), apiKey) {
		t.Fatal("API key leaked into request body")
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(captured.body, &payload); err != nil {
		t.Fatalf("decode request payload: %v", err)
	}
	if payload["model"] != "gpt-test" {
		t.Fatalf("model = %#v, want gpt-test", payload["model"])
	}
	if payload["instructions"] != "system context" {
		t.Fatalf("instructions = %#v", payload["instructions"])
	}
	if payload["max_output_tokens"] != float64(777) {
		t.Fatalf("max_output_tokens = %#v, want 777", payload["max_output_tokens"])
	}
	if store, exists := payload["store"]; !exists || store != false {
		t.Fatalf("store = %#v (present %v), want explicit false", store, exists)
	}
	if parallel, exists := payload["parallel_tool_calls"]; !exists || parallel != false {
		t.Fatalf("parallel_tool_calls = %#v (present %v), want explicit false", parallel, exists)
	}

	reasoning, ok := payload["reasoning"].(map[string]interface{})
	if !ok || reasoning["effort"] != "high" {
		t.Fatalf("reasoning = %#v, want effort high", payload["reasoning"])
	}
	include, ok := payload["include"].([]interface{})
	if !ok || len(include) != 1 || include[0] != "reasoning.encrypted_content" {
		t.Fatalf("include = %#v", payload["include"])
	}

	tools, ok := payload["tools"].([]interface{})
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %#v, want one native function", payload["tools"])
	}
	tool, ok := tools[0].(map[string]interface{})
	if !ok {
		t.Fatalf("tool = %#v, want object", tools[0])
	}
	if tool["type"] != "function" || tool["name"] != "read_file" || tool["description"] != "Read a file" {
		t.Fatalf("tool identity = %#v", tool)
	}
	if _, nested := tool["function"]; nested {
		t.Fatalf("Responses tool must be flat, got nested function: %#v", tool)
	}
	if strict, exists := tool["strict"]; !exists || strict != false {
		t.Fatalf("strict = %#v (present %v), want explicit false", strict, exists)
	}

	parameters, ok := tool["parameters"].(map[string]interface{})
	if !ok {
		t.Fatalf("parameters = %#v, want object", tool["parameters"])
	}
	if parameters["type"] != "object" || parameters["additionalProperties"] != false {
		t.Fatalf("parameters = %#v", parameters)
	}
	requiredValues, ok := parameters["required"].([]interface{})
	if !ok {
		t.Fatalf("required = %#v, want array", parameters["required"])
	}
	required := make([]string, 0, len(requiredValues))
	for _, value := range requiredValues {
		required = append(required, value.(string))
	}
	slices.Sort(required)
	if !slices.Equal(required, []string{"path"}) {
		t.Fatalf("required = %v, want [path]", required)
	}
}

func TestOpenAIRunnerPreservesRawOutputAcrossToolLoop(t *testing.T) {
	t.Parallel()

	requests := make(chan []byte, 2)
	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		body, _ := io.ReadAll(req.Body)
		requests <- body
		w.Header().Set("Content-Type", "application/json")
		if calls.Add(1) == 1 {
			_, _ = io.WriteString(w, `{
				"id":"resp_tool",
				"status":"completed",
				"output":[
					{
						"id":"reasoning_1",
						"type":"reasoning",
						"summary":[{"type":"summary_text","text":"Need the file"}],
						"encrypted_content":"opaque-ciphertext"
					},
					{
						"id":"function_1",
						"type":"function_call",
						"status":"completed",
						"call_id":"call_1",
						"name":"read_file",
						"arguments":"{\"path\":\"notes.txt\"}"
					}
				]
			}`)
			return
		}
		_, _ = io.WriteString(w, `{
			"id":"resp_final",
			"status":"completed",
			"output":[{
				"id":"msg_2",
				"type":"message",
				"role":"assistant",
				"content":[{"type":"output_text","text":"The file says hello."}]
			}]
		}`)
	}))
	defer server.Close()

	runner := NewOpenAIRunner(OpenAIRunnerConfig{
		APIKey:  "sk-tool-loop",
		Model:   "gpt-test",
		BaseURL: server.URL,
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
	var first openAIHeraldOutput
	if err := json.Unmarshal([]byte(firstRaw), &first); err != nil {
		t.Fatalf("decode first output: %v", err)
	}
	if first.ToolCall == nil || first.ToolCall.Name != "read_file" {
		t.Fatalf("first tool call = %#v", first.ToolCall)
	}
	if first.Reasoning != "Need the file" {
		t.Fatalf("first reasoning = %q", first.Reasoning)
	}

	const toolResult = "[TOOL RESULT]\nhello\n[END TOOL RESULT]"
	secondRaw, err := runner.CompleteWithToolsForConversation(
		context.Background(),
		"conversation-tool-loop",
		"system",
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
	var second openAIHeraldOutput
	if err := json.Unmarshal([]byte(secondRaw), &second); err != nil {
		t.Fatalf("decode second output: %v", err)
	}
	if second.Content != "The file says hello." || second.ToolCall != nil {
		t.Fatalf("second output = %#v", second)
	}

	<-requests // First request is covered by the request-shape test.
	var secondRequest struct {
		Input []json.RawMessage `json:"input"`
	}
	if err := json.Unmarshal(<-requests, &secondRequest); err != nil {
		t.Fatalf("decode second request: %v", err)
	}
	if len(secondRequest.Input) != 4 {
		t.Fatalf("second input has %d items, want original message + 2 raw outputs + tool result", len(secondRequest.Input))
	}

	var reasoningItem map[string]interface{}
	if err := json.Unmarshal(secondRequest.Input[1], &reasoningItem); err != nil {
		t.Fatalf("decode replayed reasoning item: %v", err)
	}
	if reasoningItem["type"] != "reasoning" || reasoningItem["encrypted_content"] != "opaque-ciphertext" {
		t.Fatalf("replayed reasoning item = %#v", reasoningItem)
	}

	var functionItem map[string]interface{}
	if err := json.Unmarshal(secondRequest.Input[2], &functionItem); err != nil {
		t.Fatalf("decode replayed function item: %v", err)
	}
	if functionItem["type"] != "function_call" ||
		functionItem["call_id"] != "call_1" ||
		functionItem["name"] != "read_file" ||
		functionItem["arguments"] != `{"path":"notes.txt"}` {
		t.Fatalf("replayed function item = %#v", functionItem)
	}

	var functionOutput map[string]interface{}
	if err := json.Unmarshal(secondRequest.Input[3], &functionOutput); err != nil {
		t.Fatalf("decode function output: %v", err)
	}
	if functionOutput["type"] != "function_call_output" ||
		functionOutput["call_id"] != "call_1" ||
		functionOutput["output"] != toolResult {
		t.Fatalf("function output = %#v", functionOutput)
	}
}

func TestOpenAIRunnerRejectsMultipleFunctionCalls(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{
			"id":"resp_parallel",
			"status":"completed",
			"output":[
				{"type":"function_call","call_id":"call_1","name":"read_file","arguments":"{\"path\":\"one\"}"},
				{"type":"function_call","call_id":"call_2","name":"read_file","arguments":"{\"path\":\"two\"}"}
			]
		}`)
	}))
	defer server.Close()

	runner := NewOpenAIRunner(OpenAIRunnerConfig{
		APIKey:  "sk-parallel",
		Model:   "gpt-test",
		BaseURL: server.URL,
	})
	_, err := runner.CompleteWithTools(
		context.Background(),
		"system",
		[]models.Message{{Role: models.RoleUser, Content: "read both"}},
		512,
		[]models.ToolDefinition{{Name: "read_file", ArgsSchema: `{"path":"string"}`}},
	)
	if err == nil {
		t.Fatal("expected multiple function calls to be rejected")
	}
	if !strings.Contains(err.Error(), "2 function calls") {
		t.Fatalf("error = %v, want multiple-call detail", err)
	}
}

func TestOpenAIRunnerBoundsAndRedactsHTTPError(t *testing.T) {
	t.Parallel()

	const apiKey = "sk-super-secret"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, "upstream exposed "+apiKey+" "+strings.Repeat("x", 512))
	}))
	defer server.Close()

	runner := NewOpenAIRunner(OpenAIRunnerConfig{
		APIKey:            apiKey,
		Model:             "gpt-test",
		BaseURL:           server.URL,
		MaxErrorBodyBytes: 64,
	})
	_, err := runner.Complete(context.Background(), "hello", 512)
	if err == nil {
		t.Fatal("expected HTTP error")
	}
	message := err.Error()
	if strings.Contains(message, apiKey) {
		t.Fatalf("error leaked API key: %s", message)
	}
	if !strings.Contains(message, "[REDACTED]") {
		t.Fatalf("error did not redact API key: %s", message)
	}
	if !strings.Contains(message, "[truncated]") {
		t.Fatalf("error was not marked truncated: %s", message)
	}
	if len(message) > 160 {
		t.Fatalf("error length = %d, want bounded message: %s", len(message), message)
	}
}

func TestOpenAIRunnerRedactsKeyAcrossErrorBoundary(t *testing.T) {
	t.Parallel()

	const (
		apiKey    = "sk-boundary-secret"
		bodyLimit = int64(24)
	)
	prefix := strings.Repeat("x", int(bodyLimit)-4)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(w, prefix+apiKey+strings.Repeat("z", 128))
	}))
	defer server.Close()

	runner := NewOpenAIRunner(OpenAIRunnerConfig{
		APIKey:            apiKey,
		Model:             "gpt-test",
		BaseURL:           server.URL,
		MaxErrorBodyBytes: bodyLimit,
	})
	_, err := runner.Complete(context.Background(), "hello", 512)
	if err == nil {
		t.Fatal("expected HTTP error")
	}
	message := err.Error()
	if strings.Contains(message, apiKey) || strings.Contains(message, apiKey[:6]) {
		t.Fatalf("error leaked API key material: %s", message)
	}
	if !strings.Contains(message, "[REDACTED]") {
		t.Fatalf("error did not redact boundary-spanning API key: %s", message)
	}
}

func TestOpenAIRunnerDoesNotShiftPartialKeyPastWhitespaceBoundary(t *testing.T) {
	t.Parallel()

	const (
		apiKey    = "sk-whitespace-boundary-secret"
		bodyLimit = int64(24)
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = io.WriteString(
			w,
			strings.Repeat(" ", int(bodyLimit)+2)+apiKey+strings.Repeat("z", 64),
		)
	}))
	defer server.Close()

	runner := NewOpenAIRunner(OpenAIRunnerConfig{
		APIKey:            apiKey,
		Model:             "gpt-test",
		BaseURL:           server.URL,
		MaxErrorBodyBytes: bodyLimit,
	})
	_, err := runner.Complete(context.Background(), "hello", 512)
	if err == nil {
		t.Fatal("expected HTTP error")
	}
	message := err.Error()
	for _, secretMaterial := range []string{apiKey, apiKey[:len(apiKey)-1]} {
		if strings.Contains(message, secretMaterial) {
			t.Fatalf("error shifted partial API key into output: %q", message)
		}
	}
}

func TestOpenAIRunnerMissingKeyFailsBeforeNetwork(t *testing.T) {
	t.Parallel()

	var calls atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	runner := NewOpenAIRunner(OpenAIRunnerConfig{
		Model:   "gpt-test",
		BaseURL: server.URL,
	})
	_, err := runner.Complete(context.Background(), "hello", 512)
	if err == nil || !strings.Contains(err.Error(), "API key is required") {
		t.Fatalf("error = %v, want missing-key validation", err)
	}
	if calls.Load() != 0 {
		t.Fatalf("server received %d requests, want zero", calls.Load())
	}
}
