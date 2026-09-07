package config

import (
	"errors"
	"fmt"
	"net"
	"net/url"
	"os"
	"path/filepath"
	"strings"

	"github.com/BurntSushi/toml"
)

type AppConfig struct {
	LogLevel   string           `toml:"log_level"`
	Database   DatabaseConfig   `toml:"database"`
	Model      ModelConfig      `toml:"model"`
	Embedding  EmbeddingConfig  `toml:"embedding"`
	Cloud      CloudConfig      `toml:"cloud"`
	Tools      ToolsConfig      `toml:"tools"`
	Security   SecurityConfig   `toml:"security"`
	Reflection ReflectionConfig `toml:"reflection"`
}

type ReflectionConfig struct {
	Enabled bool `toml:"enabled"`
}

type DatabaseConfig struct {
	Path       string `toml:"path"`
	Encrypted  bool   `toml:"encrypted"`
	Passphrase string `toml:"passphrase"`
	VecDim     int    `toml:"vec_dim"`
}

type ModelConfig struct {
	Enabled         bool    `toml:"enabled"`
	ModelPath       string  `toml:"model_path"`
	ContextSize     int     `toml:"context_size"`
	MaxOutputTokens int     `toml:"max_output_tokens"`
	Quantization    string  `toml:"quantization"`
	GPULayers       int     `toml:"gpu_layers"`
	Threads         int     `toml:"threads"`
	Temperature     float64 `toml:"temperature"`
	TopP            float64 `toml:"top_p"`
	TopK            int     `toml:"top_k"`
	MinP            float64 `toml:"min_p"`
	PresencePenalty float64 `toml:"presence_penalty"`
	RepeatPenalty   float64 `toml:"repeat_penalty"`
	RunnerType      string  `toml:"runner_type"`
	RunnerURL       string  `toml:"runner_url"`
	RunnerModel     string  `toml:"runner_model"`
	RunnerTimeout   int     `toml:"runner_timeout"`
	EnableThinking  bool    `toml:"enable_thinking"`
	EnableStreaming bool    `toml:"enable_streaming"`
	DefaultMode     string  `toml:"default_mode"`   // "local", "hybrid", "cloud"
	ProjectRoot     string  `toml:"project_root"`   // auto-detected if empty
	MaxIterations   int     `toml:"max_iterations"` // 0 = use mode defaults (25/50)
}

type EmbeddingConfig struct {
	Enabled  bool   `toml:"enabled"`
	Provider string `toml:"provider"`
	Model    string `toml:"model"`
	URL      string `toml:"url"`
	Dim      int    `toml:"dim"`
}

type CloudConfig struct {
	Provider        string `toml:"provider"`
	OpenAIKey       string `toml:"openai_key"`
	OpenAIModel     string `toml:"openai_model"`
	ReasoningEffort string `toml:"reasoning_effort"`
	AnthropicKey    string `toml:"anthropic_key"`
	AnthropicModel  string `toml:"anthropic_model"`
	GeminiKey       string `toml:"gemini_key"`
	GeminiModel     string `toml:"gemini_model"`
	MaxTokens       int    `toml:"max_tokens"`
	TimeoutSec      int    `toml:"timeout_sec"`
}

type ToolsConfig struct {
	MojoPath       string   `toml:"mojo_path"`
	MojoKernelsDir string   `toml:"mojo_kernels_dir"`
	PythonPath     string   `toml:"python_path"`
	WolframAppID   string   `toml:"wolfram_app_id"`
	AllowedDirs    []string `toml:"allowed_dirs"`
	MaxExecTimeSec int      `toml:"max_exec_time_sec"`
	MaxFileSize    int64    `toml:"max_file_size"`
	MaxOutputBytes int      `toml:"max_output_bytes"`
}

type SecurityConfig struct {
	EnableGuardrails bool           `toml:"enable_guardrails"`
	BlockedCommands  []string       `toml:"blocked_commands"`
	AllowNetwork     bool           `toml:"allow_network"`
	Sandbox          SandboxSection `toml:"sandbox"`
}

type SandboxSection struct {
	Enabled            bool     `toml:"enabled"`
	RequireOSIsolation bool     `toml:"require_os_isolation"`
	AllowedDirs        []string `toml:"allowed_dirs"`
	BlockedPatterns    []string `toml:"blocked_patterns"`
	AllowedBinaries    []string `toml:"allowed_binaries"`
	TimeoutSec         int      `toml:"timeout_sec"`
	MaxOutputBytes     int      `toml:"max_output_bytes"`
	CPUTimeSec         int      `toml:"cpu_time_sec"`
	MaxMemoryBytes     int64    `toml:"max_memory_bytes"`
	MaxProcesses       int      `toml:"max_processes"`
	MaxFileSizeBytes   int64    `toml:"max_file_size_bytes"`
	MaxOpenFiles       int      `toml:"max_open_files"`
}

func Load(path string) (*AppConfig, error) {
	return load(path, false)
}

// LoadRequired loads a configuration whose base file must exist and be
// readable. It is used for explicit HERALD_CONFIG selections so a typo or
// permissions problem cannot silently fall back to defaults.
func LoadRequired(path string) (*AppConfig, error) {
	return load(path, true)
}

func load(path string, requireBase bool) (*AppConfig, error) {
	cfg := Defaults()

	if data, err := os.ReadFile(path); err == nil {
		if _, err := toml.Decode(string(data), cfg); err != nil {
			return nil, fmt.Errorf("decode %s: %w", path, err)
		}
	} else if requireBase || !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("read config %s: %w", path, err)
	}

	localPath := strings.TrimSuffix(path, filepath.Ext(path)) + ".local" + filepath.Ext(path)
	if data, err := os.ReadFile(localPath); err == nil {
		if _, err := toml.Decode(string(data), cfg); err != nil {
			return nil, fmt.Errorf("decode %s: %w", localPath, err)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("read config override %s: %w", localPath, err)
	}

	applyEnvOverrides(cfg)
	sanitizeConfig(cfg)
	resolveRelativePaths(cfg, path)
	if err := validateCloudConfig(cfg); err != nil {
		return nil, err
	}
	if err := validateNetworkPolicy(cfg); err != nil {
		return nil, err
	}

	if cfg.Model.ProjectRoot == "" {
		absPath, err := filepath.Abs(path)
		if err == nil {
			cfg.Model.ProjectRoot = filepath.Dir(absPath)
		}
	}

	return cfg, nil
}

func resolveRelativePaths(cfg *AppConfig, configPath string) {
	absConfig, err := filepath.Abs(configPath)
	if err != nil {
		return
	}
	base := filepath.Dir(absConfig)
	resolve := func(value string, commandNameAllowed bool) string {
		value = expandUserPath(value)
		if value == "" || value == ":memory:" || filepath.IsAbs(value) || (commandNameAllowed && !strings.ContainsAny(value, `/\`)) {
			return value
		}
		return filepath.Join(base, value)
	}
	cfg.Database.Path = resolve(cfg.Database.Path, false)
	cfg.Model.ModelPath = resolve(cfg.Model.ModelPath, false)
	cfg.Model.ProjectRoot = resolve(cfg.Model.ProjectRoot, false)
	cfg.Tools.PythonPath = resolve(cfg.Tools.PythonPath, true)
	cfg.Tools.MojoPath = resolve(cfg.Tools.MojoPath, true)
	cfg.Tools.MojoKernelsDir = resolve(cfg.Tools.MojoKernelsDir, false)
}

func expandUserPath(value string) string {
	if value != "~" && !strings.HasPrefix(value, "~/") && !strings.HasPrefix(value, `~\`) {
		return value
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return value
	}
	if value == "~" {
		return home
	}
	return filepath.Join(home, value[2:])
}

func validateNetworkPolicy(cfg *AppConfig) error {
	if cfg.Security.AllowNetwork {
		return nil
	}
	endpoints := []struct {
		name    string
		rawURL  string
		enabled bool
	}{
		{name: "model.runner_url", rawURL: cfg.Model.RunnerURL, enabled: cfg.Model.Enabled},
		{name: "embedding.url", rawURL: cfg.Embedding.URL, enabled: cfg.Embedding.Enabled},
	}
	for _, endpoint := range endpoints {
		if !endpoint.enabled {
			continue
		}
		if !isLoopbackHTTPURL(endpoint.rawURL) {
			return fmt.Errorf("%s must use a loopback HTTP(S) endpoint while security.allow_network is false", endpoint.name)
		}
	}
	return nil
}

func isLoopbackHTTPURL(rawURL string) bool {
	parsed, err := url.Parse(rawURL)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Hostname() == "" {
		return false
	}
	host := strings.TrimSuffix(strings.ToLower(parsed.Hostname()), ".")
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func validateCloudConfig(cfg *AppConfig) error {
	switch cfg.Cloud.Provider {
	case "openai", "anthropic":
	default:
		return fmt.Errorf("cloud.provider must be \"openai\" or \"anthropic\", got %q", cfg.Cloud.Provider)
	}

	switch cfg.Cloud.ReasoningEffort {
	case "none", "low", "medium", "high", "xhigh", "max":
	default:
		return fmt.Errorf(
			"cloud.reasoning_effort must be one of none, low, medium, high, xhigh, or max, got %q",
			cfg.Cloud.ReasoningEffort,
		)
	}
	return nil
}

func applyEnvOverrides(cfg *AppConfig) {
	if v := strings.TrimSpace(os.Getenv("HERALD_OPENAI_API_KEY")); v != "" {
		cfg.Cloud.OpenAIKey = v
	} else if v := strings.TrimSpace(os.Getenv("OPENAI_API_KEY")); v != "" {
		cfg.Cloud.OpenAIKey = v
	}
	if v := strings.TrimSpace(os.Getenv("HERALD_ANTHROPIC_KEY")); v != "" {
		cfg.Cloud.AnthropicKey = v
	} else if v := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY")); v != "" {
		cfg.Cloud.AnthropicKey = v
	}
	if v := strings.TrimSpace(os.Getenv("HERALD_GEMINI_KEY")); v != "" {
		cfg.Cloud.GeminiKey = v
	} else if v := strings.TrimSpace(os.Getenv("GEMINI_API_KEY")); v != "" {
		cfg.Cloud.GeminiKey = v
	} else if v := strings.TrimSpace(os.Getenv("GOOGLE_API_KEY")); v != "" {
		cfg.Cloud.GeminiKey = v
	}
	if v := os.Getenv("HERALD_WOLFRAM_APP_ID"); v != "" {
		cfg.Tools.WolframAppID = v
	}
	if v := os.Getenv("WOLFRAM_APP_ID"); v != "" && cfg.Tools.WolframAppID == "" {
		cfg.Tools.WolframAppID = v
	}
}

func sanitizeConfig(cfg *AppConfig) {
	cfg.Cloud.Provider = strings.ToLower(strings.TrimSpace(cfg.Cloud.Provider))
	if cfg.Cloud.Provider == "" {
		cfg.Cloud.Provider = "openai"
	}
	cfg.Cloud.OpenAIKey = strings.TrimSpace(cfg.Cloud.OpenAIKey)
	cfg.Cloud.AnthropicKey = strings.TrimSpace(cfg.Cloud.AnthropicKey)
	cfg.Cloud.GeminiKey = strings.TrimSpace(cfg.Cloud.GeminiKey)
	cfg.Cloud.OpenAIModel = strings.TrimSpace(cfg.Cloud.OpenAIModel)
	if cfg.Cloud.OpenAIModel == "" {
		cfg.Cloud.OpenAIModel = "gpt-5.6-sol"
	}
	cfg.Cloud.ReasoningEffort = strings.ToLower(strings.TrimSpace(cfg.Cloud.ReasoningEffort))
	if cfg.Cloud.ReasoningEffort == "" {
		cfg.Cloud.ReasoningEffort = "medium"
	}

	if cfg.Model.ContextSize <= 0 {
		cfg.Model.ContextSize = 65536
	}
	if cfg.Model.MaxOutputTokens <= 0 {
		cfg.Model.MaxOutputTokens = 8192
	}

	if cfg.Model.MaxIterations <= 0 {
		cfg.Model.MaxIterations = 0
	} else if cfg.Model.MaxIterations > 200 {
		cfg.Model.MaxIterations = 200
	}

	if cfg.Model.RunnerTimeout <= 0 {
		cfg.Model.RunnerTimeout = 1800
	}
	if cfg.Cloud.TimeoutSec <= 0 {
		cfg.Cloud.TimeoutSec = 120
	}
	if cfg.Tools.MaxExecTimeSec <= 0 {
		cfg.Tools.MaxExecTimeSec = 30
	}
	if cfg.Tools.MaxOutputBytes <= 0 {
		cfg.Tools.MaxOutputBytes = 1024 * 1024
	}
	if cfg.Security.Sandbox.TimeoutSec <= 0 {
		cfg.Security.Sandbox.TimeoutSec = 30
	}
	if cfg.Security.Sandbox.MaxOutputBytes <= 0 {
		cfg.Security.Sandbox.MaxOutputBytes = 65536
	}
	if cfg.Security.Sandbox.CPUTimeSec <= 0 {
		cfg.Security.Sandbox.CPUTimeSec = cfg.Security.Sandbox.TimeoutSec
	}
	if cfg.Security.Sandbox.MaxMemoryBytes <= 0 {
		cfg.Security.Sandbox.MaxMemoryBytes = 8 * 1024 * 1024 * 1024
	}
	if cfg.Security.Sandbox.MaxProcesses <= 0 {
		cfg.Security.Sandbox.MaxProcesses = 1024
	}
	if cfg.Security.Sandbox.MaxFileSizeBytes <= 0 {
		cfg.Security.Sandbox.MaxFileSizeBytes = 1024 * 1024 * 1024
	}
	if cfg.Security.Sandbox.MaxOpenFiles <= 0 {
		cfg.Security.Sandbox.MaxOpenFiles = 1024
	}
}

func Defaults() *AppConfig {
	return &AppConfig{
		LogLevel: "info",
		Database: DatabaseConfig{
			Path:   "herald_memory.db",
			VecDim: 768,
		},
		Model: ModelConfig{
			Enabled:         true,
			ContextSize:     65536,
			MaxOutputTokens: 8192,
			Quantization:    "q4_k_m",
			Threads:         12,
			Temperature:     0.6,
			TopP:            0.95,
			TopK:            20,
			MinP:            0,
			PresencePenalty: 0,
			RepeatPenalty:   1,
			RunnerType:      "ollama",
			RunnerURL:       "http://localhost:11434",
			RunnerModel:     "qwen3.6:27b-mtp-q4_K_M",
			RunnerTimeout:   1800,
			EnableThinking:  true,
			EnableStreaming: false,
			DefaultMode:     "hybrid",
		},
		Embedding: EmbeddingConfig{
			Enabled:  false,
			Provider: "ollama",
			Model:    "nomic-embed-text",
			URL:      "http://localhost:11434",
			Dim:      768,
		},
		Cloud: CloudConfig{
			Provider:        "openai",
			OpenAIModel:     "gpt-5.6-sol",
			ReasoningEffort: "medium",
			AnthropicModel:  "claude-sonnet-4-5-20250929",
			GeminiModel:     "gemini-2.0-flash",
			MaxTokens:       8192,
			TimeoutSec:      120,
		},
		Tools: ToolsConfig{
			PythonPath:     "~/.herald/venv/bin/python",
			MojoKernelsDir: "mojo/kernels",
			MaxExecTimeSec: 30,
			MaxFileSize:    10 * 1024 * 1024,
			MaxOutputBytes: 1024 * 1024,
		},
		Security: SecurityConfig{
			EnableGuardrails: true,
			BlockedCommands:  []string{"rm -rf /", "sudo", "mkfs", "dd if="},
			AllowNetwork:     true,
			Sandbox: SandboxSection{
				Enabled:            false,
				RequireOSIsolation: true,
				AllowedDirs:        []string{"~/projects", "~/Documents", "~/Desktop"},
				BlockedPatterns:    []string{"rm -rf", "sudo"},
				AllowedBinaries: []string{
					"python3", "python", "pip", "pip3",
					"node", "npm", "npx", "yarn", "bun",
					"go", "cargo", "rustc",
					"gcc", "g++", "make", "cmake",
					"mojo", "ls", "cat", "grep", "head", "tail", "wc", "echo", "mkdir", "git",
				},
				TimeoutSec:       30,
				MaxOutputBytes:   65536,
				CPUTimeSec:       30,
				MaxMemoryBytes:   8 * 1024 * 1024 * 1024,
				MaxProcesses:     1024,
				MaxFileSizeBytes: 1024 * 1024 * 1024,
				MaxOpenFiles:     1024,
			},
		},
	}
}
