package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"axiom/internal/config"
	"axiom/internal/guardrail"
	"axiom/pkg/models"
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
	"read_file":        true,
	"list_dir":         true,
	"web_search":       true,
	"web_scrape":       true,
	"wolfram":          true,
	"system_info":      true,
	"analyze_image":    true,
	"search_memory":    true,
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
				ArgsSchema:  `{"query": "string", "limit": "int (optional, default 5)"}`,
			},
			fn: r.searchMemory,
		}
		// Register schema with guardrail if available
		if r.guard != nil {
			r.guard.RegisterSchema("search_memory", `{"query": "string", "limit": "int (optional, default 5)"}`)
		}
	}
}

// SetGuardrail sets the guardrail for schema validation.
func (r *Registry) SetGuardrail(g *guardrail.Guard) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.guard = g
	// Register schemas for all static tools
	for name, tool := range r.static {
		g.RegisterSchema(name, tool.def.ArgsSchema)
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
		defs = append(defs, t.def)
	}
	for _, t := range r.dynamic {
		defs = append(defs, t.def)
	}
	return defs
}

// toolCacheKey builds a deterministic cache key from the tool name + args.
func toolCacheKey(name string, args map[string]interface{}) string {
	argsJSON, _ := json.Marshal(args)
	return name + ":" + string(argsJSON)
}

func (r *Registry) Execute(ctx context.Context, call *models.ToolCall) (string, error) {
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
			ArgsSchema:  `{"path": "string"}`,
		},
		fn: r.readFile,
	}

	r.static["write_file"] = staticTool{
		def: models.ToolDefinition{
			Name:        "write_file",
			Description: "Write content to a file in an allowed directory",
			ArgsSchema:  `{"path": "string", "content": "string"}`,
		},
		fn: r.writeFile,
	}

	r.static["edit_file"] = staticTool{
		def: models.ToolDefinition{
			Name:        "edit_file",
			Description: "Replace the first occurrence of old_text with new_text in a file",
			ArgsSchema:  `{"path": "string", "old_text": "string", "new_text": "string"}`,
		},
		fn: r.editFile,
	}

	r.static["list_dir"] = staticTool{
		def: models.ToolDefinition{
			Name:        "list_dir",
			Description: "List files and directories at a path",
			ArgsSchema:  `{"path": "string"}`,
		},
		fn: r.listDir,
	}

	r.static["system_info"] = staticTool{
		def: models.ToolDefinition{
			Name:        "system_info",
			Description: "Get system information",
			ArgsSchema:  `{}`,
		},
		fn: r.systemInfo,
	}

	// ── Code Execution ──────────────────────────────────────────────────
	if r.sandbox != nil {
		r.static["execute_code"] = staticTool{
			def: models.ToolDefinition{
				Name: "execute_code",
				Description: `Execute a shell command in a sandboxed directory. Returns stdout, stderr, and exit code.
Use this to run code, build projects, run tests. If code fails, read the error, fix with write_file, run again.`,
				ArgsSchema: `{"command": "string", "dir": "string (working directory)"}`,
			},
			fn: r.executeCode,
		}
	}

	// ── Web Tools ───────────────────────────────────────────────────────
	r.dynamic["web_search"] = dynamicTool{
		def: models.ToolDefinition{
			Name:        "web_search",
			Description: "Search the internet via DuckDuckGo. Returns title, URL, snippet for top results.",
			ArgsSchema:  `{"query": "string", "max_results": "int (optional, default 5)"}`,
		},
		runtime: "python",
		script:  "scripts/web_search.py",
	}

	r.dynamic["web_scrape"] = dynamicTool{
		def: models.ToolDefinition{
			Name:        "web_scrape",
			Description: "Extract clean text from a URL. Strips HTML/ads. Truncated to 8000 chars.",
			ArgsSchema:  `{"url": "string", "max_chars": "int (optional, default 8000)"}`,
		},
		runtime: "python",
		script:  "scripts/web_scrape.py",
	}

	r.dynamic["git_ops"] = dynamicTool{
		def: models.ToolDefinition{
			Name:        "git_ops",
			Description: "Run git operations in a project directory. Actions: status, diff, log, add, commit, push, branch, checkout",
			ArgsSchema:  `{"action": "string (status|diff|log|add|commit|push|branch|checkout)", "args": "string (optional)", "cwd": "string (project directory)"}`,
		},
		runtime: "python",
		script:  "scripts/git_ops.py",
	}

	// ── Wolfram Oracle ──────────────────────────────────────────────────
	if r.cfg.WolframAppID != "" {
		r.dynamic["wolfram"] = dynamicTool{
			def: models.ToolDefinition{
				Name:        "wolfram",
				Description: "Query Wolfram|Alpha for verified math, science, and factual data",
				ArgsSchema:  `{"query": "string"}`,
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
				ArgsSchema:  `{"image_path": "string (path to image file)", "question": "string (optional, default: Describe this image in detail)"}`,
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
				ArgsSchema: `{"provider": "string (claude|gemini)", "prompt": "string", "context": "string (optional)", "output_path": "string (REQUIRED for code generation — exact file path on disk)"}`,
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
	if !r.isAllowed(imagePath) {
		return "", fmt.Errorf("'%s' outside allowed directories", imagePath)
	}

	data, err := os.ReadFile(imagePath)
	if err != nil {
		return "", fmt.Errorf("read image: %w", err)
	}
	if int64(len(data)) > r.cfg.MaxFileSize {
		return "", fmt.Errorf("image too large (%d bytes, max %d)", len(data), r.cfg.MaxFileSize)
	}

	// Detect MIME type from extension
	ext := strings.ToLower(filepath.Ext(imagePath))
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
var codeKeywords = []string{
	"write", "create", "build", "implement", "generate", "code", "function",
	"class", "script", "file", "module", "program", "app", "refactor", "fix",
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
	if outputPath == "" {
		lower := strings.ToLower(prompt)
		for _, kw := range codeKeywords {
			if strings.Contains(lower, kw) {
				return "", fmt.Errorf(
					"output_path is required for code tasks — retry this call with " +
						"\"output_path\": \"/home/shki/projects/<project>/<filename>\" " +
						"so the result is written directly to disk",
				)
			}
		}
	}

	// Validate output_path is in allowed directories
	if outputPath != "" && !r.isAllowed(outputPath) {
		return "", fmt.Errorf("output_path '%s' outside allowed directories", outputPath)
	}

	result, err := r.cloud.Delegate(ctx, provider, prompt, codeContext, outputPath)
	if err != nil {
		return "", fmt.Errorf("cloud delegation: %w", err)
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
	if !r.isAllowed(path) {
		return "", fmt.Errorf("'%s' outside allowed directories", path)
	}
	info, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	if info.Size() > r.cfg.MaxFileSize {
		return "", fmt.Errorf("file too large (%d bytes)", info.Size())
	}
	data, err := os.ReadFile(path)
	return string(data), err
}

func (r *Registry) writeFile(_ context.Context, args map[string]interface{}) (string, error) {
	path, _ := args["path"].(string)
	content, _ := args["content"].(string)
	if path == "" || content == "" {
		return "", fmt.Errorf("'path' and 'content' required")
	}
	if !r.isAllowed(path) {
		return "", fmt.Errorf("'%s' outside allowed directories", path)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return "", err
	}
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return "", err
	}
	// Invalidate cache — file contents have changed
	r.InvalidateCache()
	return fmt.Sprintf("Wrote %d bytes to %s", len(content), path), nil
}

func (r *Registry) editFile(_ context.Context, args map[string]interface{}) (string, error) {
	path, _ := args["path"].(string)
	oldText, _ := args["old_text"].(string)
	newText, _ := args["new_text"].(string)
	if path == "" || oldText == "" {
		return "", fmt.Errorf("'path' and 'old_text' required")
	}
	if !r.isAllowed(path) {
		return "", fmt.Errorf("'%s' outside allowed directories", path)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	content := string(data)
	if !strings.Contains(content, oldText) {
		return "", fmt.Errorf("old_text not found in file")
	}
	newContent := strings.Replace(content, oldText, newText, 1)
	if err := os.WriteFile(path, []byte(newContent), 0644); err != nil {
		return "", err
	}
	r.InvalidateCache()
	return fmt.Sprintf("Edited %s: replaced %d bytes with %d bytes", path, len(oldText), len(newText)), nil
}

func (r *Registry) listDir(_ context.Context, args map[string]interface{}) (string, error) {
	path, _ := args["path"].(string)
	if path == "" {
		return "", fmt.Errorf("'path' required")
	}
	if !r.isAllowed(path) {
		return "", fmt.Errorf("'%s' outside allowed directories", path)
	}
	entries, err := os.ReadDir(path)
	if err != nil {
		return "", err
	}
	var sb strings.Builder
	for _, e := range entries {
		prefix := "  "
		if e.IsDir() {
			prefix = "d "
		}
		sb.WriteString(fmt.Sprintf("%s%s\n", prefix, e.Name()))
	}
	return sb.String(), nil
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
	if l, ok := args["limit"].(float64); ok {
		limit = int(l)
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
	var sb strings.Builder
	for _, res := range results {
		sb.WriteString(fmt.Sprintf("[score=%.2f] %s\n---\n", res.Score, res.Content))
	}
	return sb.String(), nil
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

	return r.withTimeout(ctx, func(c context.Context) (string, error) {
		argsJSON, _ := json.Marshal(args)
		cmd := exec.CommandContext(c, binary, tool.script)
		cmd.Stdin = strings.NewReader(string(argsJSON))
		output, err := cmd.CombinedOutput()
		if err != nil {
			return "", fmt.Errorf("tool '%s': %w\n%s", tool.def.Name, err, string(output))
		}
		return strings.TrimSpace(string(output)), nil
	})
}

// ── Helpers ─────────────────────────────────────────────────────────────────

func (r *Registry) withTimeout(ctx context.Context, fn func(context.Context) (string, error)) (string, error) {
	timeout := time.Duration(r.cfg.MaxExecTimeSec) * time.Second
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	return fn(ctx)
}

func (r *Registry) isAllowed(path string) bool {
	abs, err := filepath.Abs(path)
	if err != nil {
		return false
	}
	for _, dir := range r.cfg.AllowedDirs {
		if strings.HasPrefix(dir, "~/") {
			home, err := os.UserHomeDir()
			if err != nil {
				continue
			}
			dir = filepath.Join(home, dir[2:])
		}
		allowed, err := filepath.Abs(dir)
		if err != nil {
			continue
		}
		if strings.HasPrefix(abs, allowed+string(filepath.Separator)) || abs == allowed {
			return true
		}
	}
	return false
}
