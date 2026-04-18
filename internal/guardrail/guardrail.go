package guardrail

import (
	"encoding/json"
	"fmt"
	"path/filepath"
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

	for key, val := range call.Args {
		strVal, ok := val.(string)
		if !ok {
			continue
		}
		normalized := strings.ToLower(strings.TrimSpace(strVal))
		for _, blocked := range g.cfg.BlockedCommands {
			if strings.Contains(normalized, strings.ToLower(blocked)) {
				return fmt.Errorf("blocked pattern '%s' in argument '%s'", blocked, key)
			}
		}
		if strings.Contains(normalized, "../") || strings.Contains(normalized, "~/.ssh") || strings.Contains(normalized, "/etc/") {
			return fmt.Errorf("suspicious path or system access pattern in argument '%s'", key)
		}
	}

	if err := g.validateToolSemantics(call); err != nil {
		return err
	}

	if !g.cfg.AllowNetwork {
		networkTools := map[string]bool{"wolfram": true, "http_request": true, "web_search": true, "web_scrape": true}
		if networkTools[call.Name] {
			return fmt.Errorf("tool '%s' needs network access (disabled in config)", call.Name)
		}
	}

	return nil
}

func (g *Guard) validateToolSemantics(call *models.ToolCall) error {
	switch call.Name {
	case "write_file", "edit_file", "read_file":
		path, _ := call.Args["path"].(string)
		if path == "" {
			return nil
		}
		clean := filepath.Clean(path)
		if !filepath.IsAbs(clean) {
			return fmt.Errorf("tool '%s' requires an absolute path", call.Name)
		}
	case "execute_code":
		command, _ := call.Args["command"].(string)
		dir, _ := call.Args["dir"].(string)
		if strings.TrimSpace(command) == "" {
			return fmt.Errorf("execute_code requires a non-empty command")
		}
		if !filepath.IsAbs(filepath.Clean(dir)) {
			return fmt.Errorf("execute_code requires an absolute working directory")
		}
		for _, token := range []string{"&&", "||", ";", "|", "$(", "`"} {
			if strings.Contains(command, token) {
				return fmt.Errorf("execute_code command contains unsafe shell operator '%s'", token)
			}
		}
	case "git_ops":
		cwd, _ := call.Args["cwd"].(string)
		if cwd != "" && !filepath.IsAbs(filepath.Clean(cwd)) {
			return fmt.Errorf("git_ops requires an absolute cwd")
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
