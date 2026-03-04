package guardrail

import (
	"fmt"
	"strings"

	"axiom/internal/config"
	"axiom/pkg/models"
)

type Guard struct {
	cfg config.SecurityConfig
}

func New(cfg config.SecurityConfig) *Guard {
	return &Guard{cfg: cfg}
}

// Check validates a tool call. Returns nil if allowed.
func (g *Guard) Check(call *models.ToolCall) error {
	if !g.cfg.EnableGuardrails {
		return nil
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
