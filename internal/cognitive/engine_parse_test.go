package cognitive

import (
	"encoding/json"
	"testing"

	"reeve/pkg/models"
)

func TestParseResponsePreservesCodeFenceInsideJSONContent(t *testing.T) {
	content := "# /tmp/generated.txt\n```text\nhello\n```"
	raw, err := json.Marshal(models.LLMResponse{Content: content})
	if err != nil {
		t.Fatal(err)
	}

	response, err := (&Engine{}).parseResponse(string(raw))
	if err != nil {
		t.Fatal(err)
	}
	if response.Content != content {
		t.Fatalf("content = %q, want %q", response.Content, content)
	}
}

func TestCleanJSONStillAcceptsMarkdownWrappedObject(t *testing.T) {
	raw := "```json\n{\"reasoning\":\"\",\"tool_call\":null,\"content\":\"done\"}\n```"
	cleaned := cleanJSON(raw)
	if !json.Valid([]byte(cleaned)) {
		t.Fatalf("cleaned response is not JSON: %q", cleaned)
	}
}
