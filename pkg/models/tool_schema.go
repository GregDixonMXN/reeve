package models

import (
	"encoding/json"
	"sort"
	"strings"
	"unicode"
)

// JSONSchema is the provider-neutral subset of JSON Schema used for tool
// arguments. It is recursive so object properties and array items retain the
// same contract at every depth.
type JSONSchema struct {
	Type                 string                `json:"type,omitempty"`
	Description          string                `json:"description,omitempty"`
	Properties           map[string]JSONSchema `json:"properties,omitempty"`
	Required             []string              `json:"required,omitempty"`
	Items                *JSONSchema           `json:"items,omitempty"`
	Enum                 []string              `json:"enum,omitempty"`
	Minimum              *float64              `json:"minimum,omitempty"`
	Maximum              *float64              `json:"maximum,omitempty"`
	AdditionalProperties *bool                 `json:"additionalProperties,omitempty"`
}

// ObjectSchema constructs a closed object schema. Tool argument objects are
// closed by default so providers and local validation enforce the same set of
// named parameters.
func ObjectSchema(properties map[string]JSONSchema, required ...string) *JSONSchema {
	allowAdditional := false
	schema := JSONSchema{
		Type:                 "object",
		Properties:           properties,
		Required:             required,
		AdditionalProperties: &allowAdditional,
	}
	canonical := schema.Canonical()
	return &canonical
}

// Canonical returns a detached, deterministic schema. Object schemas are
// closed unless the author explicitly opts into additional properties.
func (s JSONSchema) Canonical() JSONSchema {
	result := JSONSchema{
		Type:        strings.ToLower(strings.TrimSpace(s.Type)),
		Description: strings.TrimSpace(s.Description),
		Enum:        append([]string(nil), s.Enum...),
	}
	if result.Type == "" && (s.Properties != nil || len(s.Required) > 0) {
		result.Type = "object"
	}
	if result.Type == "" {
		result.Type = "object"
	}

	if s.AdditionalProperties != nil {
		value := *s.AdditionalProperties
		result.AdditionalProperties = &value
	} else if result.Type == "object" {
		allowAdditional := false
		result.AdditionalProperties = &allowAdditional
	}
	if s.Minimum != nil {
		value := *s.Minimum
		result.Minimum = &value
	}
	if s.Maximum != nil {
		value := *s.Maximum
		result.Maximum = &value
	}

	if s.Properties != nil {
		result.Properties = make(map[string]JSONSchema, len(s.Properties))
		for name, property := range s.Properties {
			result.Properties[name] = property.Canonical()
		}
	}
	if s.Items != nil {
		items := s.Items.Canonical()
		result.Items = &items
	}

	if len(s.Required) > 0 {
		seen := make(map[string]struct{}, len(s.Required))
		for _, name := range s.Required {
			name = strings.TrimSpace(name)
			if name == "" {
				continue
			}
			if _, exists := seen[name]; exists {
				continue
			}
			seen[name] = struct{}{}
			result.Required = append(result.Required, name)
		}
		sort.Strings(result.Required)
	}
	if len(result.Enum) > 0 {
		sort.Strings(result.Enum)
	}
	return result
}

// CanonicalSchema resolves a typed definition or adapts the legacy ArgsSchema
// format into the same recursive representation.
func (d ToolDefinition) CanonicalSchema() JSONSchema {
	if d.Parameters != nil {
		return d.Parameters.Canonical()
	}
	return ParseLegacyArgsSchema(d.ArgsSchema)
}

// SchemaJSON renders the canonical schema deterministically for prompts and
// diagnostics. encoding/json sorts map keys.
func (d ToolDefinition) SchemaJSON() string {
	encoded, err := json.Marshal(d.CanonicalSchema())
	if err != nil {
		return `{"type":"object","additionalProperties":false}`
	}
	return string(encoded)
}

// ParseLegacyArgsSchema converts Reeve's historical
// {"name":"type (optional, description)"} representation into JSON Schema.
// It also accepts an already-structured JSON Schema for compatibility with
// callers that adopted the standard shape before ToolDefinition.Parameters.
func ParseLegacyArgsSchema(source string) JSONSchema {
	source = strings.TrimSpace(source)
	if source == "" {
		return *ObjectSchema(nil)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal([]byte(source), &raw); err != nil {
		return *ObjectSchema(nil)
	}
	if isStructuredJSONSchema(raw) {
		var schema JSONSchema
		if err := json.Unmarshal([]byte(source), &schema); err != nil {
			return *ObjectSchema(nil)
		}
		return schema.Canonical()
	}

	properties := make(map[string]JSONSchema, len(raw))
	required := make([]string, 0, len(raw))
	for name, encoded := range raw {
		var description string
		if err := json.Unmarshal(encoded, &description); err == nil {
			property := legacyProperty(description)
			properties[name] = property
			if !containsWordFold(description, "optional") {
				required = append(required, name)
			}
			continue
		}

		var property JSONSchema
		if err := json.Unmarshal(encoded, &property); err != nil {
			property = JSONSchema{Type: "string"}
		}
		properties[name] = property.Canonical()
	}
	return *ObjectSchema(properties, required...)
}

func isStructuredJSONSchema(raw map[string]json.RawMessage) bool {
	for _, key := range []string{
		"type", "properties", "required", "items", "enum",
		"minimum", "maximum", "additionalProperties",
	} {
		if _, exists := raw[key]; exists {
			return true
		}
	}
	return false
}

func legacyProperty(description string) JSONSchema {
	property := JSONSchema{
		Type:        "string",
		Description: strings.TrimSpace(description),
	}
	switch {
	case containsAnyWordFold(description, "int", "integer", "count"):
		property.Type = "integer"
	case containsAnyWordFold(description, "number", "float", "decimal"):
		property.Type = "number"
	case containsAnyWordFold(description, "bool", "boolean"):
		property.Type = "boolean"
	}
	property.Enum = legacyStringEnum(description)
	return property
}

func legacyStringEnum(description string) []string {
	start := strings.IndexRune(description, '(')
	end := strings.IndexRune(description, ')')
	if start < 0 || end <= start+1 {
		return nil
	}
	candidate := description[start+1 : end]
	if !strings.Contains(candidate, "|") {
		return nil
	}
	parts := strings.Split(candidate, "|")
	values := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" || strings.ContainsAny(part, " ,;") {
			return nil
		}
		values = append(values, part)
	}
	return values
}

func containsAnyWordFold(source string, words ...string) bool {
	for _, word := range words {
		if containsWordFold(source, word) {
			return true
		}
	}
	return false
}

func containsWordFold(source, wanted string) bool {
	wanted = strings.ToLower(wanted)
	for _, word := range strings.FieldsFunc(strings.ToLower(source), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_'
	}) {
		if word == wanted {
			return true
		}
	}
	return false
}
