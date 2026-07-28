package adapters

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestCheckOllamaModelRequiresExactInstalledTag(t *testing.T) {
	t.Parallel()

	var shown string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		switch req.URL.Path {
		case "/api/version":
			_, _ = io.WriteString(w, `{"version":"test"}`)
		case "/api/show":
			body, _ := io.ReadAll(req.Body)
			shown = string(body)
			_, _ = io.WriteString(w, `{"details":{"family":"qwen35"}}`)
		default:
			http.NotFound(w, req)
		}
	}))
	defer server.Close()

	const model = "qwen3.6:27b-mtp-q4_K_M"
	if err := CheckOllamaModel(context.Background(), server.URL, model, time.Second); err != nil {
		t.Fatalf("CheckOllamaModel() error = %v", err)
	}
	if !strings.Contains(shown, model) {
		t.Fatalf("show request = %q, want exact model %q", shown, model)
	}
}

func TestCheckOllamaModelReportsMissingModel(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if req.URL.Path == "/api/version" {
			_, _ = io.WriteString(w, `{"version":"test"}`)
			return
		}
		http.Error(w, `{"error":"model not found"}`, http.StatusNotFound)
	}))
	defer server.Close()

	err := CheckOllamaModel(context.Background(), server.URL, "missing:model", time.Second)
	if err == nil || !strings.Contains(err.Error(), "model not found") {
		t.Fatalf("CheckOllamaModel() error = %v, want missing-model detail", err)
	}
}
