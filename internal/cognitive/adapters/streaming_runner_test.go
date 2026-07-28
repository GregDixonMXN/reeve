package adapters

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestSafeStreamEmitterHidesProtocolAndEmitsDecodedContent(t *testing.T) {
	var visible strings.Builder
	emitter := safeStreamEmitter{callback: func(token string) { visible.WriteString(token) }}
	for _, chunk := range []string{
		`{"reasoning":"private`,
		` reasoning","tool_call":null,"content":"Hello`,
		`\nworld"}`,
	} {
		emitter.Write(chunk)
	}
	if visible.Len() != 0 {
		t.Fatalf("protocol leaked before completion: %q", visible.String())
	}
	emitter.Finish(true)
	if visible.String() != "Hello\nworld" {
		t.Fatalf("visible content = %q", visible.String())
	}
}

func TestSafeStreamEmitterDiscardsStructuredToolTurn(t *testing.T) {
	var visible strings.Builder
	emitter := safeStreamEmitter{callback: func(token string) { visible.WriteString(token) }}
	emitter.Write(`{"reasoning":"private","content":"must not leak"}`)
	emitter.Finish(false)
	if visible.Len() != 0 {
		t.Fatalf("tool turn leaked: %q", visible.String())
	}
}

func TestSafeStreamEmitterStreamsPlainUserContent(t *testing.T) {
	var visible strings.Builder
	emitter := safeStreamEmitter{callback: func(token string) { visible.WriteString(token) }}
	emitter.Write("Hello")
	emitter.Write(" world")
	emitter.Finish(true)
	if visible.String() != "Hello world" {
		t.Fatalf("plain content = %q", visible.String())
	}
}

func TestNormalizeNativeContentProducesAxiomContract(t *testing.T) {
	t.Parallel()

	plain := normalizeNativeContent("Hello")
	var decoded struct {
		Reasoning string          `json:"reasoning"`
		Content   string          `json:"content"`
		ToolCall  json.RawMessage `json:"tool_call"`
	}
	if err := json.Unmarshal([]byte(plain), &decoded); err != nil {
		t.Fatalf("normalized plain content is not JSON: %v", err)
	}
	if decoded.Reasoning != "" || decoded.Content != "Hello" || string(decoded.ToolCall) != "null" {
		t.Fatalf("normalized response = %s", plain)
	}

	withReasoning := normalizeNativeContent("<think>private</think>Hello")
	if err := json.Unmarshal([]byte(withReasoning), &decoded); err != nil {
		t.Fatalf("normalized reasoning content is not JSON: %v", err)
	}
	if decoded.Reasoning != "private" || decoded.Content != "Hello" || string(decoded.ToolCall) != "null" {
		t.Fatalf("normalized reasoning response = %s", withReasoning)
	}
	if strings.Contains(withReasoning, "<think>") || strings.Contains(withReasoning, "</think>") {
		t.Fatalf("reasoning tags leaked into normalized response: %s", withReasoning)
	}

	schemaReasoning := `Follow {"reasoning":"string","tool_call":null,"content":"string"}`
	withSchemaReasoning := normalizeNativeContent("<think>" + schemaReasoning + "</think>Hello")
	if err := json.Unmarshal([]byte(withSchemaReasoning), &decoded); err != nil {
		t.Fatalf("normalized schema reasoning content is not JSON: %v", err)
	}
	if decoded.Reasoning != schemaReasoning || decoded.Content != "Hello" || string(decoded.ToolCall) != "null" {
		t.Fatalf("schema text in reasoning changed response detection: %s", withSchemaReasoning)
	}

	structuredWithReasoning := normalizeNativeContent(
		`<think>private</think>{"reasoning":"","tool_call":null,"content":"Hello"}`,
	)
	if err := json.Unmarshal([]byte(structuredWithReasoning), &decoded); err != nil {
		t.Fatalf("normalized structured reasoning content is not JSON: %v", err)
	}
	if decoded.Reasoning != "private" || decoded.Content != "Hello" || string(decoded.ToolCall) != "null" {
		t.Fatalf("structured visible response was not preserved: %s", structuredWithReasoning)
	}

	structured := `{"reasoning":"done","tool_call":null,"content":"Hello"}`
	if got := normalizeNativeContent(structured); got != structured {
		t.Fatalf("structured response changed to %q", got)
	}

	arbitraryJSON := `{"answer":42}`
	if got := normalizeNativeContent(arbitraryJSON); !strings.Contains(got, `"content":"{\"answer\":42}"`) {
		t.Fatalf("arbitrary JSON was not preserved as visible content: %s", got)
	}
}
