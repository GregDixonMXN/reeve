package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode"

	"reeve/internal/config"
	"reeve/internal/guardrail"
	"reeve/pkg/models"
	reevescripts "reeve/scripts"
)

type ToolFunc func(ctx context.Context, args map[string]interface{}) (string, error)

// MemorySearcher is the interface for semantic memory search.
type MemorySearcher interface {
	Search(ctx context.Context, query string, limit int) ([]MemoryResult, error)
}

// MemoryResult represents a single memory search result.
type MemoryResult struct {
	Content string
	Score   float64
}

type staticTool struct {
	def models.ToolDefinition
	fn  ToolFunc
}

type dynamicTool struct {
	def     models.ToolDefinition
	runtime string
	script  string
}

// toolCacheEntry holds a cached tool result and when it was stored.
type toolCacheEntry struct {
	result string
	at     time.Time
}

// toolCacheTTL is how long cached tool results remain valid.
const toolCacheTTL = 5 * time.Minute

// cacheableTools lists tools whose results are safe to cache (idempotent reads).
// Mutating tools (write_file, execute_code) are intentionally excluded.
var cacheableTools = map[string]bool{
	"read_file":     true,
	"list_dir":      true,
	"web_search":    true,
	"web_scrape":    true,
	"wolfram":       true,
	"system_info":   true,
	"analyze_image": true,
	"search_memory": true,
}

// parallelSafeTools lists tools that are safe to execute concurrently.
// These are read-only tools that don't have side effects.
var parallelSafeTools = map[string]bool{
	"read_file":     true,
	"list_dir":      true,
	"system_info":   true,
	"search_memory": true,
}

// ToolResult holds the result of a parallel tool execution.
type ToolResult struct {
	Name   string
	Result string
	Err    error
}

type Registry struct {
	cfg     config.ToolsConfig
	static  map[string]staticTool
	dynamic map[string]dynamicTool
	cloud   *CloudDelegator
	sandbox *Sandbox
	mem     MemorySearcher
	guard   *guardrail.Guard

	// Cloud tool toggle for mode switching
	mu           sync.RWMutex
	cloudEnabled bool

	// Tool result cache for idempotent tools
	cacheMu sync.RWMutex
	cache   map[string]toolCacheEntry
}

func NewRegistry(cfg config.ToolsConfig, cloudCfg *CloudConfig, sandboxCfg *SandboxConfig) *Registry {
	if cfg.MaxFileSize <= 0 {
		cfg.MaxFileSize = 10 * 1024 * 1024
	}
	if cfg.MaxOutputBytes <= 0 {
		cfg.MaxOutputBytes = 1024 * 1024
	}
	r := &Registry{
		cfg:          cfg,
		static:       make(map[string]staticTool),
		dynamic:      make(map[string]dynamicTool),
		cloudEnabled: true,
		cache:        make(map[string]toolCacheEntry),
	}

	if cloudCfg != nil {
		r.cloud = NewCloudDelegator(*cloudCfg)
	}

	if sandboxCfg != nil && sandboxCfg.Enabled {
		r.sandbox = NewSandbox(*sandboxCfg)
	}

	r.registerBuiltins()
	return r
}

// SetCloudEnabled toggles the ask_cloud_model tool visibility.
// Called by ModeManager when switching modes.
func (r *Registry) SetCloudEnabled(enabled bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.cloudEnabled = enabled
}

// SetMemory sets the memory searcher for the search_memory tool.
func (r *Registry) SetMemory(m MemorySearcher) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.mem = m
	// Register the search_memory tool now that memory is available
	if m != nil {
		r.static["search_memory"] = staticTool{
			def: models.ToolDefinition{
				Name:        "search_memory",
				Description: "Search semantic memory for relevant past conversations and stored knowledge",
				Parameters: models.ObjectSchema(map[string]models.JSONSchema{
					"query": {Type: "string", Description: "Text to search for in semantic memory"},
					"limit": {Type: "integer", Description: "Maximum number of matches to return (default 5)"},
				}, "query"),
				ArgsSchema: `{"query": "string", "limit": "int (optional, default 5)"}`,
			},
			fn: r.searchMemory,
		}
		// Register schema with guardrail if available
		if r.guard != nil {
			r.guard.RegisterDefinition(r.static["search_memory"].def)
		}
	}
}

// SetGuardrail sets the guardrail for schema validation.
func (r *Registry) SetGuardrail(g *guardrail.Guard) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.guard = g
	// Register schemas for all tools. Dynamic tools cross a subprocess boundary,
	// so validating their required inputs is just as important as static tools.
	for _, tool := range r.static {
		g.RegisterDefinition(tool.def)
	}
	for _, tool := range r.dynamic {
		g.RegisterDefinition(tool.def)
	}
}

func (r *Registry) Definitions() []models.ToolDefinition {
	r.mu.RLock()
	defer r.mu.RUnlock()

	defs := make([]models.ToolDefinition, 0, len(r.static)+len(r.dynamic))
	for name, t := range r.static {
		// Skip cloud tool if disabled (Local mode or Cloud mode)
		if name == "ask_cloud_model" && !r.cloudEnabled {
			continue
		}
		if r.guard != nil && !r.guard.ToolAvailable(name) {
			continue
		}
		defs = append(defs, t.def)
	}
	for name, t := range r.dynamic {
		if r.guard != nil && !r.guard.ToolAvailable(name) {
			continue
		}
		defs = append(defs, t.def)
	}
	sort.Slice(defs, func(i, j int) bool {
		return defs[i].Name < defs[j].Name
	})
	return defs
}

// toolCacheKey builds a deterministic cache key from the tool name + args.
func toolCacheKey(name string, args map[string]interface{}) string {
	argsJSON, _ := json.Marshal(args)
	return name + ":" + string(argsJSON)
}

func (r *Registry) Execute(ctx context.Context, call *models.ToolCall) (string, error) {
	// Enforce the guard at the registry boundary as well as in the orchestrator.
	// This protects direct callers and, critically, runs before cache lookup so a
	// cached network result cannot bypass a later restrictive policy.
	r.mu.RLock()
	guard := r.guard
	r.mu.RUnlock()
	if guard != nil {
		if err := guard.Check(call); err != nil {
			return "", err
		}
	}

	// ── Cache check for idempotent tools ──────────────────────────────
	if cacheableTools[call.Name] {
		key := toolCacheKey(call.Name, call.Args)
		r.cacheMu.RLock()
		if entry, ok := r.cache[key]; ok && time.Since(entry.at) < toolCacheTTL {
			r.cacheMu.RUnlock()
			return entry.result + " [cached]", nil
		}
		r.cacheMu.RUnlock()

		// Execute and store result in cache
		if t, ok := r.static[call.Name]; ok {
			result, err := r.withTimeout(ctx, func(c context.Context) (string, error) {
				return t.fn(c, call.Args)
			})
			if err == nil {
				r.cacheMu.Lock()
				r.cache[key] = toolCacheEntry{result: result, at: time.Now()}
				r.cacheMu.Unlock()
			}
			return result, err
		}
	}

	if t, ok := r.static[call.Name]; ok {
		// Block cloud tool if disabled
		if call.Name == "ask_cloud_model" {
			r.mu.RLock()
			enabled := r.cloudEnabled
			r.mu.RUnlock()
			if !enabled {
				return "", fmt.Errorf("ask_cloud_model is disabled in current mode")
			}
		}
		return r.withTimeout(ctx, func(c context.Context) (string, error) {
			return t.fn(c, call.Args)
		})
	}
	if t, ok := r.dynamic[call.Name]; ok {
		return r.executeDynamic(ctx, t, call.Args)
	}
	return "", fmt.Errorf("unknown tool: %s", call.Name)
}

// InvalidateCache clears all cached tool results.
// Call this when files are written or the environment changes.
func (r *Registry) InvalidateCache() {
	r.cacheMu.Lock()
	r.cache = make(map[string]toolCacheEntry)
	r.cacheMu.Unlock()
}

// ExecuteParallel runs multiple tool calls concurrently for safe tools.
// Non-safe tools are executed sequentially. Results are returned in the same order as calls.
func (r *Registry) ExecuteParallel(ctx context.Context, calls []*models.ToolCall) []ToolResult {
	results := make([]ToolResult, len(calls))
	var wg sync.WaitGroup
	var mu sync.Mutex

	for i, call := range calls {
		if parallelSafeTools[call.Name] {
			// Safe to execute in parallel
			wg.Add(1)
			go func(idx int, c *models.ToolCall) {
				defer wg.Done()
				result, err := r.Execute(ctx, c)
				mu.Lock()
				results[idx] = ToolResult{Name: c.Name, Result: result, Err: err}
				mu.Unlock()
			}(i, call)
		} else {
			// Execute sequentially for unsafe tools
			result, err := r.Execute(ctx, call)
			results[i] = ToolResult{Name: call.Name, Result: result, Err: err}
		}
	}

	wg.Wait()
	return results
}

// ── Registration ────────────────────────────────────────────────────────────

func (r *Registry) registerBuiltins() {
	// ── File Tools ──────────────────────────────────────────────────────
	r.static["read_file"] = staticTool{
		def: models.ToolDefinition{
			Name:        "read_file",
			Description: "Read file contents from an allowed directory",
			Parameters: models.ObjectSchema(map[string]models.JSONSchema{
				"path": {Type: "string", Description: "Absolute path to the file"},
			}, "path"),
			ArgsSchema: `{"path": "string"}`,
		},
		fn: r.readFile,
	}

	r.static["write_file"] = staticTool{
		def: models.ToolDefinition{
			Name:        "write_file",
			Description: "Write content to a file in an allowed directory",
			Parameters: models.ObjectSchema(map[string]models.JSONSchema{
				"path":    {Type: "string", Description: "Absolute path to the file"},
				"content": {Type: "string", Description: "Complete content to write"},
			}, "path", "content"),
			ArgsSchema: `{"path": "string", "content": "string"}`,
		},
		fn: r.writeFile,
	}

	r.static["edit_file"] = staticTool{
		def: models.ToolDefinition{
			Name:        "edit_file",
			Description: "Replace the first occurrence of old_text with new_text in a file",
			Parameters: models.ObjectSchema(map[string]models.JSONSchema{
				"path":     {Type: "string", Description: "Absolute path to the file"},
				"old_text": {Type: "string", Description: "Exact text to replace"},
				"new_text": {Type: "string", Description: "Replacement text"},
			}, "path", "old_text", "new_text"),
			ArgsSchema: `{"path": "string", "old_text": "string", "new_text": "string"}`,
		},
		fn: r.editFile,
	}

	r.static["list_dir"] = staticTool{
		def: models.ToolDefinition{
			Name:        "list_dir",
			Description: "List files and directories at a path",
			Parameters: models.ObjectSchema(map[string]models.JSONSchema{
				"path": {Type: "string", Description: "Absolute directory path"},
			}, "path"),
			ArgsSchema: `{"path": "string"}`,
		},
		fn: r.listDir,
	}

	r.static["system_info"] = staticTool{
		def: models.ToolDefinition{
			Name:        "system_info",
			Description: "Get system information",
			Parameters:  models.ObjectSchema(nil),
			ArgsSchema:  `{}`,
		},
		fn: r.systemInfo,
	}

	// ── Code Execution ──────────────────────────────────────────────────
	if r.sandbox != nil {
		r.static["execute_code"] = staticTool{
			def: models.ToolDefinition{
				Name: "execute_code",
				Description: `Execute a single binary command in a sandboxed directory. Returns stdout, stderr, and exit code.
Use this to run code, build projects, and run tests. Shell chaining, pipes, redirects, and subshells are intentionally blocked.`,
				Parameters: models.ObjectSchema(map[string]models.JSONSchema{
					"command": {Type: "string", Description: "Single command with no shell operators"},
					"dir":     {Type: "string", Description: "Absolute working directory"},
				}, "command", "dir"),
				ArgsSchema: `{"command": "string (single command, no shell operators)", "dir": "string (absolute working directory)"}`,
			},
			fn: r.executeCode,
		}
	}

	// ── Web Tools ───────────────────────────────────────────────────────
	r.dynamic["web_search"] = dynamicTool{
		def: models.ToolDefinition{
			Name:        "web_search",
			Description: "Search the internet via DuckDuckGo. Returns title, URL, snippet for top results.",
			Parameters: models.ObjectSchema(map[string]models.JSONSchema{
				"query":       {Type: "string", Description: "Search query"},
				"max_results": {Type: "integer", Description: "Maximum results to return (default 5)"},
			}, "query"),
			ArgsSchema: `{"query": "string", "max_results": "int (optional, default 5)"}`,
		},
		runtime: "python",
		script:  "scripts/web_search.py",
	}

	r.dynamic["web_scrape"] = dynamicTool{
		def: models.ToolDefinition{
			Name:        "web_scrape",
			Description: "Extract clean text from a URL. Strips HTML/ads. Truncated to 8000 chars.",
			Parameters: models.ObjectSchema(map[string]models.JSONSchema{
				"url":       {Type: "string", Description: "URL to scrape"},
				"max_chars": {Type: "integer", Description: "Maximum characters to return (default 8000)"},
			}, "url"),
			ArgsSchema: `{"url": "string", "max_chars": "int (optional, default 8000)"}`,
		},
		runtime: "python",
		script:  "scripts/web_scrape.py",
	}

	// Git may execute repository-controlled hooks, filters, diff drivers, and
	// fsmonitor helpers. Expose it only when the Python wrapper and every child
	// process inherit the same OS capability boundary as execute_code.
	if r.sandbox != nil && r.sandbox.HasOSIsolation() {
		r.dynamic["git_ops"] = dynamicTool{
			def: models.ToolDefinition{
				Name:        "git_ops",
				Description: "Run git operations in a project directory. Actions: status, diff, log, add, commit, push, branch, checkout",
				Parameters: models.ObjectSchema(map[string]models.JSONSchema{
					"action": {
						Type:        "string",
						Description: "Git operation to perform",
						Enum:        []string{"status", "diff", "log", "add", "commit", "push", "branch", "checkout"},
					},
					"args": {Type: "string", Description: "Optional arguments for the selected action"},
					"cwd":  {Type: "string", Description: "Absolute project directory"},
				}, "action", "cwd"),
				ArgsSchema: `{"action": "string (status|diff|log|add|commit|push|branch|checkout)", "args": "string (optional)", "cwd": "string (absolute project directory)"}`,
			},
			runtime: "python",
			script:  "scripts/git_ops.py",
		}
	}

	// ── Wolfram Oracle ──────────────────────────────────────────────────
	if r.cfg.WolframAppID != "" {
		r.dynamic["wolfram"] = dynamicTool{
			def: models.ToolDefinition{
				Name:        "wolfram",
				Description: "Query Wolfram|Alpha for verified math, science, and factual data",
				Parameters: models.ObjectSchema(map[string]models.JSONSchema{
					"query": {Type: "string", Description: "Question or expression to evaluate"},
				}, "query"),
				ArgsSchema: `{"query": "string"}`,
			},
			runtime: "python",
			script:  "scripts/wolfram_query.py",
		}
	}

	// ── Image Analysis ──────────────────────────────────────────────────
	if r.cloud != nil && r.cloud.cfg.AnthropicKey != "" {
		r.static["analyze_image"] = staticTool{
			def: models.ToolDefinition{
				Name:        "analyze_image",
				Description: "Analyze an image using Claude vision. Describe contents, read text, identify objects, answer questions about the image.",
				Parameters: models.ObjectSchema(map[string]models.JSONSchema{
					"image_path": {Type: "string", Description: "Path to the image file"},
					"question":   {Type: "string", Description: "Question about the image (defaults to a detailed description)"},
				}, "image_path"),
				ArgsSchema: `{"image_path": "string (path to image file)", "question": "string (optional, default: Describe this image in detail)"}`,
			},
			fn: r.analyzeImage,
		}
	}

	// ── Cloud Delegator (with workspace bypass) ─────────────────────────
	if r.cloud != nil && r.cloud.Available() {
		r.static["ask_cloud_model"] = staticTool{
			def: models.ToolDefinition{
				Name: "ask_cloud_model",
				Description: `Delegate code generation or complex reasoning to Claude or Gemini.
MANDATORY: For ANY code generation task, you MUST provide 'output_path' — the exact file path to write the result to (e.g. /home/shki/projects/myapp/main.py).
If you omit output_path on a code task, the call will fail.
Use 'claude' for coding/reasoning, 'gemini' for large documents.
Returns: "[SUCCESS] Written to <path> (N lines)" — do NOT ask Claude to repeat the code, just verify with execute_code.`,
				Parameters: models.ObjectSchema(map[string]models.JSONSchema{
					"provider": {
						Type:        "string",
						Description: "Cloud provider to use",
						Enum:        []string{"claude", "gemini"},
					},
					"prompt":      {Type: "string", Description: "Task for the cloud model"},
					"context":     {Type: "string", Description: "Optional supporting context"},
					"output_path": {Type: "string", Description: "Optional output path; required for code generation"},
				}, "provider", "prompt"),
				ArgsSchema: `{"provider": "string (claude|gemini)", "prompt": "string", "context": "string (optional)", "output_path": "string (optional; required for code generation — exact file path on disk)"}`,
			},
			fn: r.askCloudModel,
		}
	}
}

// ── analyze_image ───────────────────────────────────────────────────────────

func (r *Registry) analyzeImage(ctx context.Context, args map[string]interface{}) (string, error) {
	imagePath, _ := args["image_path"].(string)
	question, _ := args["question"].(string)
	if imagePath == "" {
		return "", fmt.Errorf("'image_path' required")
	}
	root, relativePath, displayPath, err := r.openAllowedRootPath(imagePath)
	if err != nil {
		return "", fmt.Errorf("'%s' outside allowed directories: %w", imagePath, err)
	}
	defer root.Close()

	data, err := readFileFromRoot(root, relativePath, r.cfg.MaxFileSize)
	if err != nil {
		return "", fmt.Errorf("read image: %w", err)
	}

	// Detect MIME type from extension
	ext := strings.ToLower(filepath.Ext(displayPath))
	mimeType := map[string]string{
		".jpg":  "image/jpeg",
		".jpeg": "image/jpeg",
		".png":  "image/png",
		".gif":  "image/gif",
		".webp": "image/webp",
	}[ext]
	if mimeType == "" {
		return "", fmt.Errorf("unsupported image type '%s' (supported: jpg, png, gif, webp)", ext)
	}

	return r.withTimeout(ctx, func(c context.Context) (string, error) {
		return r.cloud.AnalyzeImage(c, data, mimeType, question)
	})
}

// ── ask_cloud_model with workspace bypass ───────────────────────────────────

// codeKeywords are phrases that signal a code-generation intent.
// When any are present in the prompt and output_path is missing, we reject the call.
var (
	inherentCodeActions = []string{
		"implement", "implements", "implemented", "implementing", "implementation",
		"refactor", "refactors", "refactored", "refactoring",
		"rewrite", "rewrites", "rewritten", "rewriting",
		"scaffold", "scaffolds", "scaffolded", "scaffolding",
	}
	codeActions = []string{
		"write", "writes", "wrote", "written", "writing",
		"create", "creates", "created", "creating",
		"build", "builds", "built", "building",
		"generate", "generates", "generated", "generating",
		"fix", "fixes", "fixed", "fixing",
		"add", "adds", "added", "adding",
		"modify", "modifies", "modified", "modifying",
		"update", "updates", "updated", "updating",
	}
	codeObjects = []string{
		"code", "function", "functions", "class", "classes", "script", "scripts",
		"file", "files", "module", "modules", "program", "programs", "app", "apps",
		"application", "applications", "component", "components", "package", "packages",
	}
	codeExtensions = []string{".go", ".py", ".js", ".ts", ".tsx", ".jsx", ".rs", ".java", ".c", ".cpp", ".cs", ".rb", ".php", ".swift", ".kt"}
)

func wordIn(word string, choices []string) bool {
	for _, choice := range choices {
		if word == choice {
			return true
		}
	}
	return false
}

func promptLooksLikeCodeTask(prompt string) bool {
	lower := strings.ToLower(prompt)
	words := strings.FieldsFunc(lower, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_'
	})
	hasAction := false
	hasCodeObject := false
	for _, extension := range codeExtensions {
		if strings.Contains(lower, extension) {
			hasCodeObject = true
			break
		}
	}
	for _, word := range words {
		if wordIn(word, inherentCodeActions) {
			return true
		}
		if wordIn(word, codeActions) {
			hasAction = true
		}
		if wordIn(word, codeObjects) {
			hasCodeObject = true
		}
	}
	return hasAction && hasCodeObject
}

func (r *Registry) askCloudModel(ctx context.Context, args map[string]interface{}) (string, error) {
	provider, _ := args["provider"].(string)
	prompt, _ := args["prompt"].(string)
	codeContext, _ := args["context"].(string)
	outputPath, _ := args["output_path"].(string)

	if provider == "" {
		return "", fmt.Errorf("'provider' required (claude or gemini)")
	}
	if prompt == "" {
		return "", fmt.Errorf("'prompt' required")
	}

	// Enforce output_path for code-generation calls.
	// If the prompt looks like a coding task and no output_path was given, fail fast
	// so the local LLM is forced to provide a real file path on its next attempt.
	if outputPath == "" && promptLooksLikeCodeTask(prompt) {
		return "", fmt.Errorf(
			"output_path is required for code tasks — retry this call with " +
				"\"output_path\": \"/home/shki/projects/<project>/<filename>\" " +
				"so the result is written directly to disk",
		)
	}

	// Validate output_path is in allowed directories
	if outputPath != "" {
		root, relativePath, displayPath, err := r.openAllowedRootPath(outputPath)
		if err != nil {
			return "", fmt.Errorf("output_path '%s' outside allowed directories: %w", outputPath, err)
		}
		if err := validateRootTarget(root, relativePath); err != nil {
			root.Close()
			return "", fmt.Errorf("output_path '%s' is unsafe: %w", outputPath, err)
		}
		root.Close()
		outputPath = displayPath
	}

	// Keep disk writes in the registry so authorization can be repeated after
	// the potentially long network call. The delegator only returns content.
	result, err := r.cloud.Delegate(ctx, provider, prompt, codeContext)
	if err != nil {
		return "", fmt.Errorf("cloud delegation: %w", err)
	}
	if outputPath != "" {
		if int64(len(result.Content)) > r.cfg.MaxFileSize {
			return "", fmt.Errorf("cloud output is too large (%d bytes, max %d)", len(result.Content), r.cfg.MaxFileSize)
		}
		root, relativePath, displayPath, err := r.openAllowedRootPath(outputPath)
		if err != nil {
			return "", fmt.Errorf("output_path '%s' changed outside allowed directories during cloud delegation: %w", outputPath, err)
		}
		defer root.Close()
		if err := writeFileToRoot(root, relativePath, []byte(result.Content), 0644); err != nil {
			return "", fmt.Errorf("write cloud output: %w", err)
		}
		result.OutputPath = displayPath
		result.Content = ""
		r.InvalidateCache()
	}

	return result.FormatToolResult(), nil
}

// ── execute_code ────────────────────────────────────────────────────────────

func (r *Registry) executeCode(ctx context.Context, args map[string]interface{}) (string, error) {
	command, _ := args["command"].(string)
	dir, _ := args["dir"].(string)
	if command == "" {
		return "", fmt.Errorf("'command' required")
	}
	if dir == "" {
		return "", fmt.Errorf("'dir' required")
	}
	result := r.sandbox.Execute(ctx, command, dir)
	return result.String(), nil
}

// ── File Tools ──────────────────────────────────────────────────────────────

func (r *Registry) readFile(_ context.Context, args map[string]interface{}) (string, error) {
	path, _ := args["path"].(string)
	if path == "" {
		return "", fmt.Errorf("'path' required")
	}
	root, relativePath, _, err := r.openAllowedRootPath(path)
	if err != nil {
		return "", fmt.Errorf("'%s' outside allowed directories: %w", path, err)
	}
	defer root.Close()
	data, err := readFileFromRoot(root, relativePath, r.cfg.MaxFileSize)
	return string(data), err
}

func (r *Registry) writeFile(_ context.Context, args map[string]interface{}) (string, error) {
	path, _ := args["path"].(string)
	content, contentOK := args["content"].(string)
	if path == "" || !contentOK {
		return "", fmt.Errorf("'path' and 'content' required")
	}
	if int64(len(content)) > r.cfg.MaxFileSize {
		return "", fmt.Errorf("content too large (%d bytes, max %d)", len(content), r.cfg.MaxFileSize)
	}
	root, relativePath, displayPath, err := r.openAllowedRootPath(path)
	if err != nil {
		return "", fmt.Errorf("'%s' outside allowed directories: %w", path, err)
	}
	defer root.Close()
	if err := writeFileToRoot(root, relativePath, []byte(content), 0644); err != nil {
		return "", err
	}
	// Invalidate cache — file contents have changed
	r.InvalidateCache()
	return fmt.Sprintf("Wrote %d bytes to %s", len(content), displayPath), nil
}

func (r *Registry) editFile(_ context.Context, args map[string]interface{}) (string, error) {
	path, _ := args["path"].(string)
	oldText, _ := args["old_text"].(string)
	newText, _ := args["new_text"].(string)
	if path == "" || oldText == "" {
		return "", fmt.Errorf("'path' and 'old_text' required")
	}
	root, relativePath, displayPath, err := r.openAllowedRootPath(path)
	if err != nil {
		return "", fmt.Errorf("'%s' outside allowed directories: %w", path, err)
	}
	defer root.Close()
	data, err := readFileFromRoot(root, relativePath, r.cfg.MaxFileSize)
	if err != nil {
		return "", err
	}
	content := string(data)
	if !strings.Contains(content, oldText) {
		return "", fmt.Errorf("old_text not found in file")
	}
	newContent := strings.Replace(content, oldText, newText, 1)
	if int64(len(newContent)) > r.cfg.MaxFileSize {
		return "", fmt.Errorf("edited content too large (%d bytes, max %d)", len(newContent), r.cfg.MaxFileSize)
	}
	if err := writeFileToRoot(root, relativePath, []byte(newContent), 0644); err != nil {
		return "", err
	}
	r.InvalidateCache()
	return fmt.Sprintf("Edited %s: replaced %d bytes with %d bytes", displayPath, len(oldText), len(newText)), nil
}

func (r *Registry) listDir(_ context.Context, args map[string]interface{}) (string, error) {
	path, _ := args["path"].(string)
	if path == "" {
		return "", fmt.Errorf("'path' required")
	}
	root, relativePath, _, err := r.openAllowedRootPath(path)
	if err != nil {
		return "", fmt.Errorf("'%s' outside allowed directories: %w", path, err)
	}
	defer root.Close()
	directory, err := root.Open(relativePath)
	if err != nil {
		return "", err
	}
	defer directory.Close()
	output := newBoundedBuffer(r.cfg.MaxOutputBytes)
	for {
		entries, readErr := directory.ReadDir(128)
		for _, entry := range entries {
			prefix := "  "
			if entry.IsDir() {
				prefix = "d "
			}
			_, _ = output.Write([]byte(fmt.Sprintf("%s%s\n", prefix, entry.Name())))
			if output.Len() >= r.cfg.MaxOutputBytes {
				// One additional write records that more data existed without
				// retaining it, causing String to include its truncation marker.
				_, _ = output.Write([]byte("..."))
				return output.String(), nil
			}
		}
		if readErr == io.EOF {
			return output.String(), nil
		}
		if readErr != nil {
			return "", readErr
		}
	}
}

func (r *Registry) systemInfo(_ context.Context, _ map[string]interface{}) (string, error) {
	hostname, _ := os.Hostname()
	return fmt.Sprintf("Hostname: %s\nOS: %s\nArch: %s\nCPUs: %d",
		hostname, runtime.GOOS, runtime.GOARCH, runtime.NumCPU()), nil
}

func (r *Registry) searchMemory(ctx context.Context, args map[string]interface{}) (string, error) {
	query, _ := args["query"].(string)
	if query == "" {
		return "", fmt.Errorf("'query' required")
	}
	limit := 5
	switch value := args["limit"].(type) {
	case float64:
		limit = int(value)
	case int:
		limit = value
	}
	if limit < 1 {
		limit = 1
	} else if limit > 20 {
		limit = 20
	}
	if r.mem == nil {
		return "", fmt.Errorf("memory searcher not configured")
	}
	results, err := r.mem.Search(ctx, query, limit)
	if err != nil {
		return "", fmt.Errorf("memory search: %w", err)
	}
	if len(results) == 0 {
		return "No relevant memories found.", nil
	}
	output := newBoundedBuffer(r.cfg.MaxOutputBytes)
	for _, res := range results {
		_, _ = output.Write([]byte(fmt.Sprintf("[score=%.2f] %s\n---\n", res.Score, res.Content)))
		if output.Len() >= r.cfg.MaxOutputBytes {
			_, _ = output.Write([]byte("..."))
			break
		}
	}
	return output.String(), nil
}

// ── Dynamic tool execution ──────────────────────────────────────────────────

func (r *Registry) executeDynamic(ctx context.Context, tool dynamicTool, args map[string]interface{}) (string, error) {
	var binary string
	switch tool.runtime {
	case "python":
		binary = r.cfg.PythonPath
		if binary == "" {
			binary = "python3"
		}
	case "mojo":
		binary = r.cfg.MojoPath
		if binary == "" {
			binary = "mojo"
		}
	default:
		return "", fmt.Errorf("unsupported runtime: %s", tool.runtime)
	}

	normalizedArgs, err := r.prepareDynamicArgs(tool, args)
	if err != nil {
		return "", err
	}

	return r.withTimeout(ctx, func(c context.Context) (string, error) {
		argsJSON, _ := json.Marshal(normalizedArgs)
		parts := []string{binary, tool.script}
		workingDir, dirErr := r.dynamicWorkingDirectory(tool, normalizedArgs)
		if dirErr != nil {
			return "", dirErr
		}
		if tool.runtime == "python" {
			trustedPython, resolveErr := r.resolveTrustedPythonRuntime(binary, workingDir)
			if resolveErr != nil {
				return "", resolveErr
			}
			source, readErr := reevescripts.ToolFiles.ReadFile(filepath.Base(tool.script))
			if readErr != nil {
				source, readErr = os.ReadFile(tool.script)
				if readErr != nil {
					return "", fmt.Errorf("load embedded tool %q: %w", tool.def.Name, readErr)
				}
			}
			// -I removes the workspace/current directory and user-controlled
			// Python paths from module resolution. Without it, a repository could
			// shadow json/socket/ssl before a trusted embedded tool starts.
			parts = []string{trustedPython, "-I", "-c", string(source)}
		}
		if r.sandbox != nil && r.sandbox.HasOSIsolation() {
			result := r.sandbox.ExecuteArgs(c, parts, workingDir, strings.NewReader(string(argsJSON)))
			if result.Error != "" || result.TimedOut || result.ExitCode != 0 {
				return "", fmt.Errorf("tool '%s': %s", tool.def.Name, result.String())
			}
			return strings.TrimSpace(result.Stdout), nil
		}

		cmd := exec.CommandContext(c, parts[0], parts[1:]...)
		cmd.Dir = workingDir
		cmd.Stdin = strings.NewReader(string(argsJSON))
		output := newBoundedBuffer(r.cfg.MaxOutputBytes)
		cmd.Stdout = output
		cmd.Stderr = output
		err := cmd.Run()
		if err != nil {
			return "", fmt.Errorf("tool '%s': %w\n%s", tool.def.Name, err, output.String())
		}
		return strings.TrimSpace(output.String()), nil
	})
}

// resolveTrustedPythonRuntime resolves the interpreter before running an
// embedded helper and rejects any interpreter or virtual environment that
// overlaps a writable workspace. Python's isolated mode still loads a venv's
// site-packages (including .pth and sitecustomize), so a workspace-owned venv
// would let repository content execute before Reeve's trusted helper source.
func (r *Registry) resolveTrustedPythonRuntime(binary, workingDir string) (string, error) {
	binary = strings.TrimSpace(binary)
	if strings.HasPrefix(binary, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("resolve trusted Python runtime: expand home: %w", err)
		}
		binary = filepath.Join(home, binary[2:])
	}
	invocationPath, err := exec.LookPath(binary)
	if err != nil {
		return "", fmt.Errorf("resolve trusted Python runtime %q: %w", binary, err)
	}
	invocationPath, err = filepath.Abs(invocationPath)
	if err != nil {
		return "", fmt.Errorf("canonicalize trusted Python runtime %q: %w", binary, err)
	}
	invocationPath = filepath.Clean(invocationPath)
	resolvedPath, err := filepath.EvalSymlinks(invocationPath)
	if err != nil {
		return "", fmt.Errorf("resolve trusted Python runtime symlinks %q: %w", invocationPath, err)
	}
	resolvedPath, err = filepath.Abs(resolvedPath)
	if err != nil {
		return "", fmt.Errorf("canonicalize trusted Python target %q: %w", resolvedPath, err)
	}

	writableRoots := append([]string(nil), r.cfg.AllowedDirs...)
	if r.sandbox != nil {
		writableRoots = append(writableRoots, r.sandbox.cfg.AllowedDirs...)
	}
	writableRoots = append(writableRoots, workingDir)
	writableRoots = canonicalWritableRoots(writableRoots)
	for _, root := range writableRoots {
		if pathWithin(invocationPath, root) || pathWithin(resolvedPath, root) {
			return "", fmt.Errorf("trusted Python runtime %q is inside writable workspace root %q; configure an interpreter outside all allowed directories", invocationPath, root)
		}
	}

	venvRoot := filepath.Dir(filepath.Dir(invocationPath))
	if _, err := os.Stat(filepath.Join(venvRoot, "pyvenv.cfg")); err == nil {
		canonicalVenv, err := filepath.EvalSymlinks(venvRoot)
		if err != nil {
			return "", fmt.Errorf("resolve trusted Python virtual environment %q: %w", venvRoot, err)
		}
		for _, root := range writableRoots {
			if pathWithin(canonicalVenv, root) || pathWithin(root, canonicalVenv) {
				return "", fmt.Errorf("trusted Python virtual environment %q overlaps writable workspace root %q; configure an environment outside all allowed directories", canonicalVenv, root)
			}
		}
	}
	return invocationPath, nil
}

func canonicalWritableRoots(roots []string) []string {
	seen := make(map[string]struct{}, len(roots))
	canonical := make([]string, 0, len(roots))
	for _, root := range roots {
		root = strings.TrimSpace(root)
		if root == "" {
			continue
		}
		if strings.HasPrefix(root, "~/") {
			home, err := os.UserHomeDir()
			if err != nil {
				continue
			}
			root = filepath.Join(home, root[2:])
		}
		resolved, err := canonicalizePath(root)
		if err != nil {
			continue
		}
		if _, exists := seen[resolved]; exists {
			continue
		}
		seen[resolved] = struct{}{}
		canonical = append(canonical, resolved)
	}
	return canonical
}

func (r *Registry) dynamicWorkingDirectory(tool dynamicTool, args map[string]interface{}) (string, error) {
	if tool.def.Name == "git_ops" {
		cwd, _ := args["cwd"].(string)
		return cwd, nil
	}
	candidates := r.cfg.AllowedDirs
	if r.sandbox != nil {
		candidates = r.sandbox.cfg.AllowedDirs
	}
	for _, candidate := range candidates {
		if strings.HasPrefix(candidate, "~/") {
			home, err := os.UserHomeDir()
			if err != nil {
				continue
			}
			candidate = filepath.Join(home, candidate[2:])
		}
		resolved, err := filepath.EvalSymlinks(candidate)
		if err != nil {
			continue
		}
		if info, err := os.Stat(resolved); err == nil && info.IsDir() {
			return resolved, nil
		}
	}
	if r.sandbox == nil {
		if cwd, err := os.Getwd(); err == nil {
			return cwd, nil
		}
	}
	return "", fmt.Errorf("tool '%s' has no available working directory", tool.def.Name)
}

// prepareDynamicArgs applies tool-specific authorization before input crosses
// the subprocess boundary. git_ops keeps its existing stdin contract, but cwd
// is canonicalized and constrained to the same configured directories as the
// built-in file tools.
func (r *Registry) prepareDynamicArgs(tool dynamicTool, args map[string]interface{}) (map[string]interface{}, error) {
	if tool.def.Name == "wolfram" {
		normalized := make(map[string]interface{}, len(args)+1)
		for key, value := range args {
			normalized[key] = value
		}
		normalized["app_id"] = r.cfg.WolframAppID
		return normalized, nil
	}
	if tool.def.Name != "git_ops" {
		return args, nil
	}

	cwd, _ := args["cwd"].(string)
	if cwd == "" {
		return nil, fmt.Errorf("git_ops: 'cwd' required")
	}
	resolvedCWD, err := r.resolveAllowedPath(cwd)
	if err != nil {
		return nil, fmt.Errorf("git_ops cwd '%s' outside allowed directories: %w", cwd, err)
	}
	info, err := os.Stat(resolvedCWD)
	if err != nil {
		return nil, fmt.Errorf("git_ops cwd: %w", err)
	}
	if !info.IsDir() {
		return nil, fmt.Errorf("git_ops cwd '%s' is not a directory", cwd)
	}

	normalized := make(map[string]interface{}, len(args))
	for key, value := range args {
		normalized[key] = value
	}
	normalized["cwd"] = resolvedCWD
	return normalized, nil
}

// ── Helpers ─────────────────────────────────────────────────────────────────

// openAllowedRootPath anchors subsequent I/O to an os.Root file descriptor.
// Unlike check-then-open path validation, Root resolves every component at the
// moment of use and rejects symlinks that leave the configured directory.
func (r *Registry) openAllowedRootPath(path string) (*os.Root, string, string, error) {
	if strings.HasPrefix(path, "~/") {
		home, err := os.UserHomeDir()
		if err != nil {
			return nil, "", "", err
		}
		path = filepath.Join(home, path[2:])
	}
	target, err := filepath.Abs(path)
	if err != nil {
		return nil, "", "", err
	}
	target = filepath.Clean(target)

	for _, configured := range r.cfg.AllowedDirs {
		if strings.HasPrefix(configured, "~/") {
			home, err := os.UserHomeDir()
			if err != nil {
				continue
			}
			configured = filepath.Join(home, configured[2:])
		}
		configured, err = filepath.Abs(configured)
		if err != nil {
			continue
		}
		configured = filepath.Clean(configured)
		canonical, err := filepath.EvalSymlinks(configured)
		if err != nil {
			continue
		}
		info, err := os.Stat(canonical)
		if err != nil || !info.IsDir() {
			continue
		}

		for _, alias := range []string{configured, canonical} {
			if !pathWithin(target, alias) {
				continue
			}
			relative, err := filepath.Rel(alias, target)
			if err != nil {
				continue
			}
			root, err := os.OpenRoot(canonical)
			if err != nil {
				return nil, "", "", err
			}
			return root, relative, filepath.Join(canonical, relative), nil
		}
	}
	return nil, "", "", fmt.Errorf("path %q is not beneath an allowed root", target)
}

// validateRootTarget verifies the deepest existing ancestor without leaving
// the anchored root. It catches dangling or escaping symlinks before a long
// cloud request while still permitting a not-yet-created output path.
func validateRootTarget(root *os.Root, relative string) error {
	cursor := filepath.Clean(relative)
	for {
		_, lstatErr := root.Lstat(cursor)
		if lstatErr == nil {
			_, statErr := root.Stat(cursor)
			return statErr
		}
		if !os.IsNotExist(lstatErr) {
			return lstatErr
		}
		parent := filepath.Dir(cursor)
		if parent == cursor {
			return lstatErr
		}
		cursor = parent
	}
}

func readFileFromRoot(root *os.Root, relative string, maxBytes int64) ([]byte, error) {
	file, err := root.Open(relative)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	info, err := file.Stat()
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("path is not a regular file")
	}
	if info.Size() > maxBytes {
		return nil, fmt.Errorf("file too large (%d bytes, max %d)", info.Size(), maxBytes)
	}
	data, err := io.ReadAll(io.LimitReader(file, maxBytes+1))
	if err != nil {
		return nil, err
	}
	if int64(len(data)) > maxBytes {
		return nil, fmt.Errorf("file grew beyond the %d-byte limit while reading", maxBytes)
	}
	return data, nil
}

func mkdirAllInRoot(root *os.Root, relative string, perm os.FileMode) error {
	relative = filepath.Clean(relative)
	if relative == "." {
		return nil
	}
	current := ""
	for _, component := range strings.Split(relative, string(filepath.Separator)) {
		if component == "" || component == "." {
			continue
		}
		current = filepath.Join(current, component)
		if err := root.Mkdir(current, perm); err != nil && !os.IsExist(err) {
			return err
		}
		info, err := root.Stat(current)
		if err != nil {
			return err
		}
		if !info.IsDir() {
			return fmt.Errorf("%s is not a directory", current)
		}
	}
	return nil
}

func writeFileToRoot(root *os.Root, relative string, data []byte, perm os.FileMode) error {
	if err := mkdirAllInRoot(root, filepath.Dir(relative), 0755); err != nil {
		return err
	}
	file, err := root.OpenFile(relative, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, perm)
	if err != nil {
		return err
	}
	if _, err := file.Write(data); err != nil {
		_ = file.Close()
		return err
	}
	return file.Close()
}

func (r *Registry) withTimeout(ctx context.Context, fn func(context.Context) (string, error)) (string, error) {
	timeout := time.Duration(r.cfg.MaxExecTimeSec) * time.Second
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	return fn(ctx)
}

// canonicalizePath resolves every existing path component, including the
// deepest existing ancestor of a not-yet-created write target. A dangling
// symlink is rejected because its eventual target cannot be authorized safely.
func canonicalizePath(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	abs = filepath.Clean(abs)

	cursor := abs
	var suffix []string
	for {
		_, err := os.Lstat(cursor)
		if err == nil {
			resolved, err := filepath.EvalSymlinks(cursor)
			if err != nil {
				return "", err
			}
			for i := len(suffix) - 1; i >= 0; i-- {
				resolved = filepath.Join(resolved, suffix[i])
			}
			return filepath.Clean(resolved), nil
		}
		if !os.IsNotExist(err) {
			return "", err
		}

		parent := filepath.Dir(cursor)
		if parent == cursor {
			return "", err
		}
		suffix = append(suffix, filepath.Base(cursor))
		cursor = parent
	}
}

func pathWithin(path, allowed string) bool {
	rel, err := filepath.Rel(allowed, path)
	if err != nil || filepath.IsAbs(rel) {
		return false
	}
	return rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
}

func (r *Registry) resolveAllowedPath(path string) (string, error) {
	resolved, err := canonicalizePath(path)
	if err != nil {
		return "", fmt.Errorf("resolve path: %w", err)
	}

	for _, dir := range r.cfg.AllowedDirs {
		if strings.HasPrefix(dir, "~/") {
			home, err := os.UserHomeDir()
			if err != nil {
				continue
			}
			dir = filepath.Join(home, dir[2:])
		}
		allowed, err := canonicalizePath(dir)
		if err != nil {
			continue
		}
		if pathWithin(resolved, allowed) {
			return resolved, nil
		}
	}
	return "", fmt.Errorf("path resolves to '%s'", resolved)
}
