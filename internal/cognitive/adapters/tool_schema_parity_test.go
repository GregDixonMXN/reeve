package adapters

import (
	"encoding/json"
	"testing"

	"reeve/pkg/models"
)

func TestProviderToolSchemaParity(t *testing.T) {
	t.Parallel()

	minBudget := float64(1)
	maxBudget := float64(12)
	definitions := []models.ToolDefinition{{
		Name:        "run_agent",
		Description: "Run an agent",
		Parameters: models.ObjectSchema(map[string]models.JSONSchema{
			"mode": {
				Type: "string",
				Enum: []string{"cloud", "hybrid", "local"},
			},
			"budget": {
				Type:    "integer",
				Minimum: &minBudget,
				Maximum: &maxBudget,
			},
			"options": {
				Type: "object",
				Properties: map[string]models.JSONSchema{
					"trace": {Type: "boolean"},
					"labels": {
						Type:  "array",
						Items: &models.JSONSchema{Type: "string"},
					},
				},
				Required: []string{"trace"},
			},
		}, "options", "mode"),
	}}

	openAI := convertToOpenAIResponseTools(definitions)[0].Parameters
	ollama := convertToOllamaTools(definitions)[0].Function.Parameters
	anthropic := (&AnthropicRunner{}).convertTools(definitions)[0].InputSchema

	openAIJSON := mustMarshalSchema(t, openAI)
	ollamaJSON := mustMarshalSchema(t, ollama)
	anthropicJSON := mustMarshalSchema(t, anthropic)
	if openAIJSON != ollamaJSON || openAIJSON != anthropicJSON {
		t.Fatalf("provider schemas diverged:\nOpenAI: %s\nOllama: %s\nAnthropic: %s", openAIJSON, ollamaJSON, anthropicJSON)
	}
}

func TestLegacyToolSchemaProviderParity(t *testing.T) {
	t.Parallel()

	definitions := []models.ToolDefinition{{
		Name:       "legacy",
		ArgsSchema: `{"query":"string","limit":"int (optional, default 5)"}`,
	}}
	openAI := convertToOpenAIResponseTools(definitions)[0].Parameters
	ollama := convertToOllamaTools(definitions)[0].Function.Parameters
	anthropic := (&AnthropicRunner{}).convertTools(definitions)[0].InputSchema

	openAIJSON := mustMarshalSchema(t, openAI)
	if got := mustMarshalSchema(t, ollama); got != openAIJSON {
		t.Fatalf("Ollama legacy schema = %s, want %s", got, openAIJSON)
	}
	if got := mustMarshalSchema(t, anthropic); got != openAIJSON {
		t.Fatalf("Anthropic legacy schema = %s, want %s", got, openAIJSON)
	}
}

func mustMarshalSchema(t *testing.T, schema models.JSONSchema) string {
	t.Helper()
	encoded, err := json.Marshal(schema)
	if err != nil {
		t.Fatalf("marshal schema: %v", err)
	}
	return string(encoded)
}
