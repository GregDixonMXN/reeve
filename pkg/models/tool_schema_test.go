package models

import (
	"reflect"
	"testing"
)

func TestParseLegacyArgsSchemaPreservesRequiredOptionalAndTypes(t *testing.T) {
	t.Parallel()

	schema := ParseLegacyArgsSchema(`{
		"provider":"string (claude|gemini)",
		"limit":"int (optional, default 5)",
		"temperature":"number (optional)",
		"ratio":"float (optional)",
		"enabled":"bool"
	}`)

	if !reflect.DeepEqual(schema.Required, []string{"enabled", "provider"}) {
		t.Fatalf("required = %v", schema.Required)
	}
	if schema.Properties["limit"].Type != "integer" {
		t.Fatalf("limit type = %q", schema.Properties["limit"].Type)
	}
	if schema.Properties["enabled"].Type != "boolean" {
		t.Fatalf("enabled type = %q", schema.Properties["enabled"].Type)
	}
	for _, property := range []string{"temperature", "ratio"} {
		if schema.Properties[property].Type != "number" {
			t.Fatalf("%s type = %q, want number", property, schema.Properties[property].Type)
		}
	}
	if !reflect.DeepEqual(schema.Properties["provider"].Enum, []string{"claude", "gemini"}) {
		t.Fatalf("provider enum = %v", schema.Properties["provider"].Enum)
	}
	if schema.AdditionalProperties == nil || *schema.AdditionalProperties {
		t.Fatal("tool argument object is not closed")
	}
}

func TestCanonicalSchemaPrefersTypedRecursiveContract(t *testing.T) {
	t.Parallel()

	definition := ToolDefinition{
		ArgsSchema: `{"legacy":"string"}`,
		Parameters: ObjectSchema(map[string]JSONSchema{
			"settings": {
				Type: "object",
				Properties: map[string]JSONSchema{
					"retries": {Type: "integer"},
				},
				Required: []string{"retries"},
			},
		}, "settings"),
	}

	schema := definition.CanonicalSchema()
	if _, present := schema.Properties["legacy"]; present {
		t.Fatal("legacy schema overrode typed parameters")
	}
	settings := schema.Properties["settings"]
	if settings.Type != "object" || settings.Properties["retries"].Type != "integer" {
		t.Fatalf("nested schema = %#v", settings)
	}
	if settings.AdditionalProperties == nil || *settings.AdditionalProperties {
		t.Fatal("nested object schema is not closed")
	}
}

func TestParseStructuredRecursiveJSONSchema(t *testing.T) {
	t.Parallel()

	schema := ParseLegacyArgsSchema(`{
		"type":"object",
		"properties":{
			"steps":{"type":"array","items":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}},
			"retries":{"type":"integer","minimum":1,"maximum":3}
		},
		"required":["steps","retries"]
	}`)

	steps := schema.Properties["steps"]
	if steps.Type != "array" || steps.Items == nil {
		t.Fatalf("steps schema = %#v", steps)
	}
	if steps.Items.Properties["name"].Type != "string" {
		t.Fatalf("nested item schema = %#v", steps.Items)
	}
	retries := schema.Properties["retries"]
	if retries.Minimum == nil || *retries.Minimum != 1 || retries.Maximum == nil || *retries.Maximum != 3 {
		t.Fatalf("numeric bounds = %#v", retries)
	}
}
