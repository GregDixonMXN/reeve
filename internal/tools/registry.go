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
	"axiom/pkg/models"
)

type ToolFunc func(ctx context.Context, args map[string]interface{}) (string, error)

type staticTool struct {
	def models.ToolDefinition
	fn  ToolFunc
}

type dynamicTool struct {
	def     models.ToolDefinition
	runtime string
	script  string
}

type Registry struct {
	cfg     config.ToolsConfig
	static  map[string]staticTool
	dynamic map[string]dynamicTool
	cloud   *CloudDelegator
	sandbox *Sandbox

	// Cloud tool toggle for mode switching
	mu           sync.RWMutex
	cloudEnabled bool
}

func NewRegistry(cfg config.ToolsConfig, cloudCfg *CloudConfig, sandboxCfg *SandboxConfig) *Registry {
	r := &Registry{
		cfg:          cfg,
		static:       make(map[string]staticTool),
		dynamic:      make(map[string]dynamicTool),
		cloudEnabled: true,
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

func (r *Registry) Execute(ctx context.Context, call *models.ToolCall) (string, error) {
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
				Description: `Delegate complex tasks to Claude or Gemini. Use 'claude' for coding, 'gemini' for large documents.
IMPORTANT: For large code generation, provide 'output_path' to write the response directly to disk.
This bypasses the context window — you'll get a summary like "[SUCCESS] Written to /path (412 lines)".`,
				ArgsSchema: `{"provider": "string (claude|gemini)", "prompt": "string", "context": "string (optional)", "output_path": "string (optional, write response to this file)"}`,
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
	return fmt.Sprintf("Wrote %d bytes to %s", len(content), path), nil
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
