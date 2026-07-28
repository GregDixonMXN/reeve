package adapters

import (
	"encoding/json"
	"slices"
	"strings"
	"testing"
)

func TestAnthropicRunnerRejectsMultipleToolUses(t *testing.T) {
	t.Parallel()

	var response anthropicResponse
	if err := json.Unmarshal([]byte(`{
		"stop_reason":"tool_use",
		"content":[
			{"type":"tool_use","id":"tool-1","name":"read_file","input":{"path":"one.txt"}},
			{"type":"tool_use","id":"tool-2","name":"read_file","input":{"path":"two.txt"}}
		]
	}`), &response); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}

	runner := &AnthropicRunner{}
	tests := []struct {
		name string
		run  func() error
	}{
		{
			name: "native response",
			run: func() error {
				_, err := runner.parseResponse(&response)
				return err
			},
		},
		{
			name: "tool-aware serialization",
			run: func() error {
				_, err := runner.serializeToAxiomJSON(&response)
				return err
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.run()
			if err == nil {
				t.Fatal("expected parallel tool_use blocks to be rejected")
			}
			if !strings.Contains(err.Error(), "2 tool_use blocks") {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestAnthropicRunnerParseSchemaMarksRequiredFields(t *testing.T) {
	t.Parallel()

	runner := &AnthropicRunner{}
	schema := runner.parseSchema(`{
		"path":"string",
		"limit":"int (optional, default 5)",
		"enabled":"bool"
	}`)

	required := slices.Clone(schema.Required)
	slices.Sort(required)
	want := []string{"enabled", "path"}
	if !slices.Equal(required, want) {
		t.Fatalf("required = %v, want %v", required, want)
	}

	if schema.Properties["limit"].Type != "integer" {
		t.Fatalf("limit type = %q, want integer", schema.Properties["limit"].Type)
	}
	if schema.Properties["enabled"].Type != "boolean" {
		t.Fatalf("enabled type = %q, want boolean", schema.Properties["enabled"].Type)
	}
}
