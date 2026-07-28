package cognitive

import (
	"strings"
	"testing"

	"axiom/internal/config"
	"axiom/pkg/models"
)

func TestSystemPromptRendersCanonicalTypedToolSchema(t *testing.T) {
	t.Parallel()

	engine := NewEngine(config.ModelConfig{})
	definition := models.ToolDefinition{
		Name:        "route",
		Description: "Choose an execution route",
		ArgsSchema:  `{"legacy":"string"}`,
		Parameters: models.ObjectSchema(map[string]models.JSONSchema{
			"mode": {
				Type: "string",
				Enum: []string{"local", "cloud"},
			},
		}, "mode"),
	}

	system := engine.buildSystemSection(Request{Tools: []models.ToolDefinition{definition}})
	for _, wanted := range []string{
		`"enum":["cloud","local"]`,
		`"required":["mode"]`,
		`"additionalProperties":false`,
	} {
		if !strings.Contains(system, wanted) {
			t.Fatalf("system prompt does not contain %s:\n%s", wanted, system)
		}
	}
	if strings.Contains(system, `"legacy"`) {
		t.Fatalf("system prompt used legacy schema instead of typed contract:\n%s", system)
	}
}
