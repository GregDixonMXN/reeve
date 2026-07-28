package adapters

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"axiom/pkg/models"
)

func TestOllamaRequestsSeparateContextAndOutputLimits(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		run  func(context.Context, string) error
	}{
		{
			name: "non-streaming generate",
			run: func(ctx context.Context, baseURL string) error {
				runner := NewRemoteRunner(RemoteRunnerConfig{
					BaseURL: baseURL, Model: "test", Protocol: ProtocolOllama, ContextSize: 32768,
				})
				_, err := runner.Complete(ctx, "prompt", 1234)
				return err
			},
		},
		{
			name: "non-streaming chat",
			run: func(ctx context.Context, baseURL string) error {
				runner := NewRemoteRunner(RemoteRunnerConfig{
					BaseURL: baseURL, Model: "test", Protocol: ProtocolOllama, ContextSize: 32768,
				})
				_, err := runner.CompleteWithTools(ctx, "system", []models.Message{{Role: models.RoleUser, Content: "prompt"}}, 1234, []models.ToolDefinition{{Name: "read_file"}})
				return err
			},
		},
		{
			name: "streaming generate",
			run: func(ctx context.Context, baseURL string) error {
				runner := NewStreamingRunner(StreamingRunnerConfig{
					BaseURL: baseURL, Model: "test", Protocol: ProtocolOllama, ContextSize: 32768,
				})
				_, err := runner.Complete(ctx, "prompt", 1234)
				return err
			},
		},
		{
			name: "streaming chat",
			run: func(ctx context.Context, baseURL string) error {
				runner := NewStreamingRunner(StreamingRunnerConfig{
					BaseURL: baseURL, Model: "test", Protocol: ProtocolOllama, ContextSize: 32768,
				})
				_, err := runner.CompleteWithTools(ctx, "system", []models.Message{{Role: models.RoleUser, Content: "prompt"}}, 1234, []models.ToolDefinition{{Name: "read_file"}})
				return err
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			requests := make(chan map[string]interface{}, 1)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				var body map[string]interface{}
				if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
					t.Errorf("decode request: %v", err)
					return
				}
				requests <- body

				streaming, _ := body["stream"].(bool)
				w.Header().Set("Content-Type", "application/json")
				if strings.HasSuffix(req.URL.Path, "/api/chat") {
					response := `{"message":{"role":"assistant","content":"ok"},"done":true}`
					if streaming {
						fmt.Fprintln(w, response)
					} else {
						fmt.Fprint(w, response)
					}
					return
				}

				response := `{"response":"ok","done":true}`
				if streaming {
					fmt.Fprintln(w, response)
				} else {
					fmt.Fprint(w, response)
				}
			}))
			defer server.Close()

			if err := tt.run(context.Background(), server.URL); err != nil {
				t.Fatalf("runner error = %v", err)
			}

			body := <-requests
			options, ok := body["options"].(map[string]interface{})
			if !ok {
				t.Fatalf("options = %#v, want object", body["options"])
			}
			if got := options["num_ctx"]; got != float64(32768) {
				t.Fatalf("num_ctx = %#v, want 32768", got)
			}
			if got := options["num_predict"]; got != float64(1234) {
				t.Fatalf("num_predict = %#v, want 1234", got)
			}
		})
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestAnthropicUsesConfiguredCloudOutputLimit(t *testing.T) {
	t.Parallel()

	var limits []int
	runner := NewAnthropicRunner(AnthropicRunnerConfig{
		APIKey: "test", Model: "test", MaxTokens: 777,
	})
	runner.client.Transport = roundTripFunc(func(req *http.Request) (*http.Response, error) {
		body, err := io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		var payload anthropicRequest
		if err := json.Unmarshal(body, &payload); err != nil {
			return nil, err
		}
		limits = append(limits, payload.MaxTokens)
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(`{"content":[{"type":"text","text":"ok"}]}`)),
		}, nil
	})

	if _, err := runner.Complete(context.Background(), "prompt", 32768); err != nil {
		t.Fatalf("Complete() error = %v", err)
	}
	if _, err := runner.CompleteWithTools(
		context.Background(),
		"system",
		[]models.Message{{Role: models.RoleUser, Content: "prompt"}},
		32768,
		[]models.ToolDefinition{{Name: "read_file"}},
	); err != nil {
		t.Fatalf("CompleteWithTools() error = %v", err)
	}

	if len(limits) != 2 {
		t.Fatalf("captured %d requests, want 2", len(limits))
	}
	for i, got := range limits {
		if got != 777 {
			t.Fatalf("request %d max_tokens = %d, want 777", i, got)
		}
	}
}
