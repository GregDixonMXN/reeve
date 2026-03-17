package config

import (
	"os"
	"path/filepath"

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
	ModelPath       string  `toml:"model_path"`
	ContextSize     int     `toml:"context_size"`
	Quantization    string  `toml:"quantization"`
	GPULayers       int     `toml:"gpu_layers"`
	Threads         int     `toml:"threads"`
	Temperature     float64 `toml:"temperature"`
	TopP            float64 `toml:"top_p"`
	RunnerType      string  `toml:"runner_type"`
	RunnerURL       string  `toml:"runner_url"`
	RunnerModel     string  `toml:"runner_model"`
	RunnerTimeout   int     `toml:"runner_timeout"`
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
	AnthropicKey   string `toml:"anthropic_key"`
	AnthropicModel string `toml:"anthropic_model"`
	GeminiKey      string `toml:"gemini_key"`
	GeminiModel    string `toml:"gemini_model"`
	MaxTokens      int    `toml:"max_tokens"`
	TimeoutSec     int    `toml:"timeout_sec"`
}

type ToolsConfig struct {
	MojoPath       string   `toml:"mojo_path"`
	MojoKernelsDir string   `toml:"mojo_kernels_dir"`
	PythonPath     string   `toml:"python_path"`
	WolframAppID   string   `toml:"wolfram_app_id"`
	AllowedDirs    []string `toml:"allowed_dirs"`
	MaxExecTimeSec int      `toml:"max_exec_time_sec"`
	MaxFileSize    int64    `toml:"max_file_size"`
}

type SecurityConfig struct {
	EnableGuardrails bool           `toml:"enable_guardrails"`
	BlockedCommands  []string       `toml:"blocked_commands"`
	AllowNetwork     bool           `toml:"allow_network"`
	Sandbox          SandboxSection `toml:"sandbox"`
}

type SandboxSection struct {
	Enabled         bool     `toml:"enabled"`
	AllowedDirs     []string `toml:"allowed_dirs"`
	BlockedPatterns []string `toml:"blocked_patterns"`
	AllowedBinaries []string `toml:"allowed_binaries"`
	TimeoutSec      int      `toml:"timeout_sec"`
	MaxOutputBytes  int      `toml:"max_output_bytes"`
}

func Load(path string) (*AppConfig, error) {
	cfg := Defaults()

	data, err := os.ReadFile(path)
	if err != nil {
		return cfg, nil
	}

	if _, err := toml.Decode(string(data), cfg); err != nil {
		return nil, err
	}

	// Auto-detect project root from config file location if not set
	if cfg.Model.ProjectRoot == "" {
		absPath, err := filepath.Abs(path)
		if err == nil {
			cfg.Model.ProjectRoot = filepath.Dir(absPath)
		}
	}

	return cfg, nil
}

func Defaults() *AppConfig {
	return &AppConfig{
		LogLevel: "info",
		Database: DatabaseConfig{
			Path:   "axiom_memory.db",
			VecDim: 768,
		},
		Model: ModelConfig{
			ContextSize:     4096,
			Quantization:    "q4_k_m",
			Threads:         4,
			Temperature:     0.7,
			TopP:            0.9,
			RunnerType:      "ollama",
			RunnerURL:       "http://localhost:11434",
			RunnerModel:     "qwen2.5-coder:7b",
			RunnerTimeout:   120,
			EnableStreaming: false,
			DefaultMode:     "hybrid",
		},
		Embedding: EmbeddingConfig{
			Enabled:  true,
			Provider: "ollama",
			Model:    "nomic-embed-text",
			URL:      "http://localhost:11434",
			Dim:      768,
		},
		Cloud: CloudConfig{
			AnthropicModel: "claude-sonnet-4-5-20250929",
			GeminiModel:    "gemini-2.0-flash",
			MaxTokens:      8192,
			TimeoutSec:     120,
		},
		Tools: ToolsConfig{
			PythonPath:     "python3",
			MojoKernelsDir: "mojo/kernels",
			MaxExecTimeSec: 30,
			MaxFileSize:    10 * 1024 * 1024,
		},
		Security: SecurityConfig{
			EnableGuardrails: true,
			BlockedCommands:  []string{"rm -rf /", "sudo", "mkfs", "dd if="},
			AllowNetwork:     true,
			Sandbox: SandboxSection{
				Enabled:         false,
				AllowedDirs:     []string{"~/projects", "~/Documents", "~/Desktop"},
				BlockedPatterns: []string{"rm -rf", "sudo"},
				AllowedBinaries: []string{
					"python3", "python", "pip", "pip3",
					"node", "npm", "npx", "yarn", "bun",
					"go", "cargo", "rustc",
					"gcc", "g++", "make", "cmake",
					"mojo", "ls", "cat", "grep", "head", "tail", "wc", "echo", "mkdir", "git",
				},
				TimeoutSec:     30,
				MaxOutputBytes: 65536,
			},
		},
	}
}
