package guardrail

import (
	"fmt"
	"math"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"sync"

	"axiom/internal/config"
	"axiom/pkg/models"
)

type Guard struct {
	cfg                        config.SecurityConfig
	mu                         sync.RWMutex
	schemas                    map[string]models.JSONSchema
	isolatedExecutionAvailable bool
}

// SetIsolatedExecutionAvailable records whether execute_code is backed by an
// enforceable OS filesystem/network boundary. Startup sets this only after the
// host capability probe succeeds.
func (g *Guard) SetIsolatedExecutionAvailable(available bool) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.isolatedExecutionAvailable = available
}

func (g *Guard) hasIsolatedExecution() bool {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.isolatedExecutionAvailable
}

func New(cfg config.SecurityConfig) *Guard {
	return &Guard{
		cfg:     cfg,
		schemas: make(map[string]models.JSONSchema),
	}
}

// RegisterSchema stores a legacy argument schema for validation. New code
// should register the ToolDefinition so typed schemas do not need to be
// serialized and parsed again.
func (g *Guard) RegisterSchema(name, schema string) {
	g.RegisterJSONSchema(name, models.ParseLegacyArgsSchema(schema))
}

// RegisterDefinition stores a tool's canonical argument contract.
func (g *Guard) RegisterDefinition(def models.ToolDefinition) {
	g.RegisterJSONSchema(def.Name, def.CanonicalSchema())
}

// RegisterJSONSchema stores an explicit argument contract.
func (g *Guard) RegisterJSONSchema(name string, schema models.JSONSchema) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.schemas[name] = schema.Canonical()
}

// Check validates a tool call. Returns nil if allowed.
func (g *Guard) Check(call *models.ToolCall) error {
	if call == nil {
		return fmt.Errorf("tool call is nil")
	}

	// Network policy is a capability boundary, not an optional content
	// guardrail. Enforce it even when heuristic guardrails are disabled.
	if err := g.checkNetworkPolicy(call); err != nil {
		return err
	}

	if !g.cfg.EnableGuardrails {
		return nil
	}

	// Schema validation: check required args are present
	g.mu.RLock()
	schema, ok := g.schemas[call.Name]
	g.mu.RUnlock()
	if ok {
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

	return nil
}

// ToolAvailable reports whether a tool can be exposed under the configured
// network policy.
func (g *Guard) ToolAvailable(name string) bool {
	if g.cfg.AllowNetwork {
		return true
	}
	switch name {
	case "execute_code", "git_ops":
		return g.hasIsolatedExecution()
	case "wolfram", "http_request", "web_search", "web_scrape",
		"ask_cloud_model", "analyze_image":
		return false
	default:
		return true
	}
}

func (g *Guard) checkNetworkPolicy(call *models.ToolCall) error {
	if g.cfg.AllowNetwork {
		return nil
	}

	switch call.Name {
	case "wolfram", "http_request", "web_search", "web_scrape",
		"ask_cloud_model", "analyze_image":
		return fmt.Errorf("tool '%s' needs network access (disabled in config)", call.Name)
	case "execute_code":
		if !g.hasIsolatedExecution() {
			// A command can open sockets through an interpreter, compiler, package
			// manager, or child process. A command-name policy cannot prevent that,
			// so fail closed unless execution is placed behind a real OS boundary.
			return fmt.Errorf("tool 'execute_code' is disabled while network access is disabled because this execution path has no enforced network isolation")
		}
	case "git_ops":
		action, _ := call.Args["action"].(string)
		if strings.EqualFold(strings.TrimSpace(action), "push") {
			return fmt.Errorf("git push needs network access (disabled in config)")
		}
		if !g.hasIsolatedExecution() {
			// Even nominally local Git actions can execute repository-controlled
			// hooks, filters, fsmonitor programs, or external diff drivers.
			return fmt.Errorf("tool 'git_ops' is disabled while network access is disabled because repository-controlled Git helpers are not network-isolated")
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

func (g *Guard) validateSchema(call *models.ToolCall, schema models.JSONSchema) error {
	return validateObject(call.Name, "", call.Args, schema.Canonical())
}

func validateObject(toolName, path string, value map[string]interface{}, schema models.JSONSchema) error {
	for _, key := range schema.Required {
		if _, exists := value[key]; !exists {
			return fmt.Errorf("missing required argument '%s' for tool '%s'", joinArgumentPath(path, key), toolName)
		}
	}

	keys := make([]string, 0, len(value))
	for key := range value {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		property, exists := schema.Properties[key]
		if !exists {
			if schema.AdditionalProperties != nil && !*schema.AdditionalProperties {
				return fmt.Errorf("unknown argument '%s' for tool '%s'", joinArgumentPath(path, key), toolName)
			}
			continue
		}
		if err := validateJSONValue(toolName, joinArgumentPath(path, key), value[key], property); err != nil {
			return err
		}
	}
	return nil
}

func validateJSONValue(toolName, path string, value interface{}, schema models.JSONSchema) error {
	if len(schema.Enum) > 0 {
		text, ok := value.(string)
		if !ok || !containsString(schema.Enum, text) {
			return fmt.Errorf("argument '%s' for tool '%s' must be one of %v", path, toolName, schema.Enum)
		}
	}

	switch schema.Type {
	case "", "any":
		return nil
	case "string":
		if _, ok := value.(string); !ok {
			return schemaTypeError(toolName, path, "a string")
		}
	case "integer":
		if !isInteger(value) {
			return schemaTypeError(toolName, path, "an integer")
		}
		if err := validateNumberBounds(toolName, path, value, schema); err != nil {
			return err
		}
	case "number":
		if !isNumber(value) {
			return schemaTypeError(toolName, path, "a number")
		}
		if err := validateNumberBounds(toolName, path, value, schema); err != nil {
			return err
		}
	case "boolean":
		if _, ok := value.(bool); !ok {
			return schemaTypeError(toolName, path, "a boolean")
		}
	case "object":
		object, ok := value.(map[string]interface{})
		if !ok {
			return schemaTypeError(toolName, path, "an object")
		}
		if err := validateObject(toolName, path, object, schema); err != nil {
			return err
		}
	case "array":
		array := reflect.ValueOf(value)
		if !array.IsValid() || (array.Kind() != reflect.Array && array.Kind() != reflect.Slice) {
			return schemaTypeError(toolName, path, "an array")
		}
		if schema.Items != nil {
			for i := 0; i < array.Len(); i++ {
				itemPath := fmt.Sprintf("%s[%d]", path, i)
				if err := validateJSONValue(toolName, itemPath, array.Index(i).Interface(), *schema.Items); err != nil {
					return err
				}
			}
		}
	default:
		return fmt.Errorf("argument '%s' for tool '%s' has unsupported schema type %q", path, toolName, schema.Type)
	}
	return nil
}

func schemaTypeError(toolName, path, wanted string) error {
	return fmt.Errorf("argument '%s' for tool '%s' must be %s", path, toolName, wanted)
}

func joinArgumentPath(parent, child string) string {
	if parent == "" {
		return child
	}
	return parent + "." + child
}

func containsString(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}

func isInteger(value interface{}) bool {
	switch number := value.(type) {
	case float64:
		return !math.IsNaN(number) && !math.IsInf(number, 0) && math.Trunc(number) == number
	case float32:
		converted := float64(number)
		return !math.IsNaN(converted) && !math.IsInf(converted, 0) && math.Trunc(converted) == converted
	}
	kind := reflect.TypeOf(value)
	return kind != nil && kind.Kind() >= reflect.Int && kind.Kind() <= reflect.Uint64
}

func isNumber(value interface{}) bool {
	_, ok := numberValue(value)
	return ok
}

func validateNumberBounds(toolName, path string, value interface{}, schema models.JSONSchema) error {
	number, ok := numberValue(value)
	if !ok {
		return nil
	}
	if schema.Minimum != nil && number < *schema.Minimum {
		return fmt.Errorf("argument '%s' for tool '%s' must be at least %g", path, toolName, *schema.Minimum)
	}
	if schema.Maximum != nil && number > *schema.Maximum {
		return fmt.Errorf("argument '%s' for tool '%s' must be at most %g", path, toolName, *schema.Maximum)
	}
	return nil
}

func numberValue(value interface{}) (float64, bool) {
	switch number := value.(type) {
	case float64:
		return number, !math.IsNaN(number) && !math.IsInf(number, 0)
	case float32:
		converted := float64(number)
		return converted, !math.IsNaN(converted) && !math.IsInf(converted, 0)
	}
	reflected := reflect.ValueOf(value)
	if !reflected.IsValid() {
		return 0, false
	}
	switch reflected.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return float64(reflected.Int()), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return float64(reflected.Uint()), true
	default:
		return 0, false
	}
}
