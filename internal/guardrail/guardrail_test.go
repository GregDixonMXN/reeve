package guardrail

import (
	"strings"
	"sync"
	"testing"

	"herald/internal/config"
	"herald/pkg/models"
)

func TestNetworkPolicyFailsClosedWhenGuardrailsDisabled(t *testing.T) {
	g := New(config.SecurityConfig{
		EnableGuardrails: false,
		AllowNetwork:     false,
	})

	tests := []struct {
		name string
		call *models.ToolCall
	}{
		{name: "web search", call: &models.ToolCall{Name: "web_search", Args: map[string]interface{}{"query": "test"}}},
		{name: "web scrape", call: &models.ToolCall{Name: "web_scrape", Args: map[string]interface{}{"url": "https://example.com"}}},
		{name: "wolfram", call: &models.ToolCall{Name: "wolfram", Args: map[string]interface{}{"query": "1+1"}}},
		{name: "cloud delegation", call: &models.ToolCall{Name: "ask_cloud_model", Args: map[string]interface{}{"provider": "claude", "prompt": "hello"}}},
		{name: "cloud vision", call: &models.ToolCall{Name: "analyze_image", Args: map[string]interface{}{"image_path": "/tmp/image.png"}}},
		{name: "arbitrary execution", call: &models.ToolCall{Name: "execute_code", Args: map[string]interface{}{"command": "python3 script.py", "dir": "/tmp"}}},
		{name: "git push", call: &models.ToolCall{Name: "git_ops", Args: map[string]interface{}{"action": " PUSH ", "cwd": "/tmp"}}},
		{name: "git helper path", call: &models.ToolCall{Name: "git_ops", Args: map[string]interface{}{"action": "status", "cwd": "/tmp"}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := g.Check(tt.call); err == nil {
				t.Fatalf("%s was allowed while network access was disabled", tt.call.Name)
			}
		})
	}
}

func TestNetworkPolicyAllowsLocalOnlyTools(t *testing.T) {
	g := New(config.SecurityConfig{
		EnableGuardrails: false,
		AllowNetwork:     false,
	})

	for _, call := range []*models.ToolCall{
		{Name: "read_file", Args: map[string]interface{}{"path": "/tmp/file"}},
		{Name: "write_file", Args: map[string]interface{}{"path": "/tmp/file", "content": "safe"}},
	} {
		if err := g.Check(call); err != nil {
			t.Errorf("%s(%v) rejected: %v", call.Name, call.Args, err)
		}
	}
}

func TestNetworkToolsAreHiddenWhenDisabled(t *testing.T) {
	g := New(config.SecurityConfig{AllowNetwork: false})
	for _, name := range []string{
		"web_search", "web_scrape", "wolfram", "http_request",
		"ask_cloud_model", "analyze_image", "execute_code", "git_ops",
	} {
		if g.ToolAvailable(name) {
			t.Errorf("ToolAvailable(%q) = true, want false", name)
		}
	}
	for _, name := range []string{"read_file", "write_file"} {
		if !g.ToolAvailable(name) {
			t.Errorf("ToolAvailable(%q) = false, want true", name)
		}
	}
}

func TestNetworkPolicyAllowsExecuteCodeOnlyWithEnforcedIsolation(t *testing.T) {
	g := New(config.SecurityConfig{EnableGuardrails: false, AllowNetwork: false})
	call := &models.ToolCall{
		Name: "execute_code",
		Args: map[string]interface{}{"command": "go test ./...", "dir": "/tmp"},
	}
	if err := g.Check(call); err == nil {
		t.Fatal("execute_code was allowed before isolation capability was enabled")
	}
	if g.ToolAvailable("execute_code") {
		t.Fatal("execute_code was visible before isolation capability was enabled")
	}

	g.SetIsolatedExecutionAvailable(true)
	if err := g.Check(call); err != nil {
		t.Fatalf("isolated execute_code was rejected: %v", err)
	}
	if !g.ToolAvailable("execute_code") {
		t.Fatal("isolated execute_code remained hidden")
	}
	gitStatus := &models.ToolCall{Name: "git_ops", Args: map[string]interface{}{"action": "status", "cwd": "/tmp"}}
	if err := g.Check(gitStatus); err != nil {
		t.Fatalf("isolated local git action was rejected: %v", err)
	}
	if !g.ToolAvailable("git_ops") {
		t.Fatal("isolated git_ops remained hidden")
	}
	gitPush := &models.ToolCall{Name: "git_ops", Args: map[string]interface{}{"action": "push", "cwd": "/tmp"}}
	if err := g.Check(gitPush); err == nil {
		t.Fatal("git push was allowed with networking disabled")
	}
}

func TestConcurrentSchemaRegistrationAndCheck(t *testing.T) {
	g := New(config.SecurityConfig{EnableGuardrails: true, AllowNetwork: true})
	call := &models.ToolCall{Name: "read_file", Args: map[string]interface{}{"path": "/tmp/file"}}

	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				g.RegisterSchema("read_file", `{"path":"string"}`)
			}
		}()
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				if err := g.Check(call); err != nil && !strings.Contains(err.Error(), "missing required") {
					t.Errorf("Check: %v", err)
					return
				}
			}
		}()
	}
	wg.Wait()
}

func TestSchemaValidationRejectsWrongArgumentTypes(t *testing.T) {
	t.Parallel()

	g := New(config.SecurityConfig{EnableGuardrails: true, AllowNetwork: true})
	g.RegisterSchema("example", `{"path":"string","limit":"int (optional)","enabled":"bool (optional)"}`)

	valid := &models.ToolCall{Name: "example", Args: map[string]interface{}{
		"path":    "file.txt",
		"limit":   float64(5),
		"enabled": true,
	}}
	if err := g.Check(valid); err != nil {
		t.Fatalf("valid arguments rejected: %v", err)
	}

	tests := []struct {
		name  string
		key   string
		value interface{}
	}{
		{name: "string", key: "path", value: float64(1)},
		{name: "integer", key: "limit", value: "five"},
		{name: "fractional integer", key: "limit", value: 1.5},
		{name: "boolean", key: "enabled", value: "true"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			args := map[string]interface{}{"path": "file.txt"}
			args[tt.key] = tt.value
			if err := g.Check(&models.ToolCall{Name: "example", Args: args}); err == nil {
				t.Fatalf("%s=%#v was accepted", tt.key, tt.value)
			}
		})
	}
}

func TestTypedSchemaValidationRequiredOptionalEnumIntegerAndNested(t *testing.T) {
	t.Parallel()

	g := New(config.SecurityConfig{EnableGuardrails: true, AllowNetwork: true})
	minRetries := float64(0)
	maxRetries := float64(5)
	minTemperature := float64(0)
	maxTemperature := float64(2)
	maxTokens := float64(8192)
	g.RegisterJSONSchema("deploy", *models.ObjectSchema(map[string]models.JSONSchema{
		"provider": {
			Type: "string",
			Enum: []string{"local", "openai"},
		},
		"retries": {
			Type:    "integer",
			Minimum: &minRetries,
			Maximum: &maxRetries,
		},
		"temperature": {
			Type:    "number",
			Minimum: &minTemperature,
			Maximum: &maxTemperature,
		},
		"note": {Type: "string"},
		"config": {
			Type: "object",
			Properties: map[string]models.JSONSchema{
				"enabled": {Type: "boolean"},
				"limits": {
					Type: "object",
					Properties: map[string]models.JSONSchema{
						"tokens": {Type: "integer", Maximum: &maxTokens},
					},
					Required: []string{"tokens"},
				},
			},
			Required: []string{"enabled", "limits"},
		},
	}, "provider", "retries", "config"))

	valid := &models.ToolCall{Name: "deploy", Args: map[string]interface{}{
		"provider":    "openai",
		"retries":     float64(2),
		"temperature": 0.7,
		"config": map[string]interface{}{
			"enabled": true,
			"limits": map[string]interface{}{
				"tokens": float64(4096),
			},
		},
	}}
	if err := g.Check(valid); err != nil {
		t.Fatalf("valid typed arguments rejected: %v", err)
	}

	tests := []struct {
		name string
		args map[string]interface{}
		want string
	}{
		{
			name: "missing required",
			args: map[string]interface{}{},
			want: "missing required argument",
		},
		{
			name: "enum",
			args: map[string]interface{}{
				"provider": "anthropic", "retries": float64(2),
				"config": map[string]interface{}{"enabled": true, "limits": map[string]interface{}{"tokens": float64(1)}},
			},
			want: "must be one of",
		},
		{
			name: "integer",
			args: map[string]interface{}{
				"provider": "local", "retries": 1.5,
				"config": map[string]interface{}{"enabled": true, "limits": map[string]interface{}{"tokens": float64(1)}},
			},
			want: "must be an integer",
		},
		{
			name: "integer minimum",
			args: map[string]interface{}{
				"provider": "local", "retries": float64(-1),
				"config": map[string]interface{}{"enabled": true, "limits": map[string]interface{}{"tokens": float64(1)}},
			},
			want: "must be at least 0",
		},
		{
			name: "number maximum",
			args: map[string]interface{}{
				"provider": "local", "retries": float64(1), "temperature": 2.1,
				"config": map[string]interface{}{"enabled": true, "limits": map[string]interface{}{"tokens": float64(1)}},
			},
			want: "must be at most 2",
		},
		{
			name: "nested required",
			args: map[string]interface{}{
				"provider": "local", "retries": float64(1),
				"config": map[string]interface{}{"enabled": true, "limits": map[string]interface{}{}},
			},
			want: "config.limits.tokens",
		},
		{
			name: "nested maximum",
			args: map[string]interface{}{
				"provider": "local", "retries": float64(1),
				"config": map[string]interface{}{"enabled": true, "limits": map[string]interface{}{"tokens": float64(8193)}},
			},
			want: "must be at most 8192",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := g.Check(&models.ToolCall{Name: "deploy", Args: tt.args})
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Check() error = %v, want substring %q", err, tt.want)
			}
		})
	}
}
