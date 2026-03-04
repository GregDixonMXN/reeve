package guardrail

import (
	"encoding/json"
	"fmt"
	"strings"

	"axiom/internal/config"
	"axiom/pkg/models"
)

type Guard struct {
	cfg     config.SecurityConfig
	schemas map[string]string
}

func New(cfg config.SecurityConfig) *Guard {
	return &Guard{
		cfg:     cfg,
		schemas: make(map[string]string),
	}
}

// RegisterSchema stores the argument schema for a tool for validation.
func (g *Guard) RegisterSchema(name, schema string) {
	g.schemas[name] = schema
}

// Check validates a tool call. Returns nil if allowed.
func (g *Guard) Check(call *models.ToolCall) error {
	if !g.cfg.EnableGuardrails {
		return nil
	}

	// Schema validation: check required args are present
	if schema, ok := g.schemas[call.Name]; ok {
		if err := g.validateSchema(call, schema); err != nil {
			return err
		}
	}

	// Check arguments for blocked patterns
	for key, val := range call.Args {
		strVal, ok := val.(string)
		if !ok {
			continue
		}
		for _, blocked := range g.cfg.BlockedCommands {
			if strings.Contains(strings.ToLower(strVal), strings.ToLower(blocked)) {
				return fmt.Errorf("blocked pattern '%s' in argument '%s'", blocked, key)
			}
		}
	}

	// Block network tools if disabled
	if !g.cfg.AllowNetwork {
		networkTools := map[string]bool{"wolfram": true, "http_request": true}
		if networkTools[call.Name] {
			return fmt.Errorf("tool '%s' needs network access (disabled in config)", call.Name)
		}
	}

	return nil
}

// validateSchema checks that all required args (those without 'optional' in their definition) are present.
func (g *Guard) validateSchema(call *models.ToolCall, schema string) error {
	var schemaMap map[string]string
	if err := json.Unmarshal([]byte(schema), &schemaMap); err != nil {
		// If schema isn't valid JSON, skip validation
		return nil
	}

	for key, typeDef := range schemaMap {
		// If the type definition doesn't contain 'optional', the arg is required
		if !strings.Contains(strings.ToLower(typeDef), "optional") {
			if _, exists := call.Args[key]; !exists {
				return fmt.Errorf("missing required argument '%s' for tool '%s'", key, call.Name)
			}
		}
	}

	return nil
}
