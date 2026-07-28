package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDefaultsSeparateContextAndOutputLimits(t *testing.T) {
	t.Parallel()

	cfg := Defaults()
	if cfg.Model.ContextSize != 65536 {
		t.Fatalf("ContextSize = %d, want 65536", cfg.Model.ContextSize)
	}
	if cfg.Model.MaxOutputTokens != 8192 {
		t.Fatalf("MaxOutputTokens = %d, want 8192", cfg.Model.MaxOutputTokens)
	}
	if cfg.Tools.MaxOutputBytes != 1024*1024 {
		t.Fatalf("Tools.MaxOutputBytes = %d, want 1048576", cfg.Tools.MaxOutputBytes)
	}
	if !cfg.Security.Sandbox.RequireOSIsolation {
		t.Fatal("OS isolation must be required by default")
	}
	if cfg.Security.Sandbox.CPUTimeSec != 30 ||
		cfg.Security.Sandbox.MaxMemoryBytes != 8*1024*1024*1024 ||
		cfg.Security.Sandbox.MaxProcesses != 1024 ||
		cfg.Security.Sandbox.MaxFileSizeBytes != 1024*1024*1024 ||
		cfg.Security.Sandbox.MaxOpenFiles != 1024 {
		t.Fatalf("unexpected sandbox resource defaults: %+v", cfg.Security.Sandbox)
	}
	if cfg.Cloud.Provider != "openai" || cfg.Cloud.OpenAIModel != "gpt-5.6-sol" {
		t.Fatalf("OpenAI defaults = provider %q model %q", cfg.Cloud.Provider, cfg.Cloud.OpenAIModel)
	}
	if cfg.Cloud.ReasoningEffort != "medium" {
		t.Fatalf("ReasoningEffort = %q, want medium", cfg.Cloud.ReasoningEffort)
	}
	if cfg.Model.DefaultMode != "hybrid" {
		t.Fatalf("DefaultMode = %q, want hybrid", cfg.Model.DefaultMode)
	}
	if !cfg.Model.Enabled {
		t.Fatal("local Qwen inference must be enabled by default")
	}
	if cfg.Model.RunnerModel != "qwen3.6:27b-mtp-q4_K_M" {
		t.Fatalf("RunnerModel = %q, want pinned Qwen3.6-27B MTP Q4", cfg.Model.RunnerModel)
	}
	if cfg.Model.RunnerTimeout != 1800 {
		t.Fatalf("RunnerTimeout = %d, want 1800 for CPU-local 8K generations", cfg.Model.RunnerTimeout)
	}
	if !cfg.Model.EnableThinking {
		t.Fatal("Qwen thinking must be enabled by default")
	}
	if cfg.Model.Temperature != 0.6 || cfg.Model.TopP != 0.95 || cfg.Model.TopK != 20 {
		t.Fatalf(
			"sampling = temperature %v top_p %v top_k %d",
			cfg.Model.Temperature,
			cfg.Model.TopP,
			cfg.Model.TopK,
		)
	}
	if cfg.Model.MinP != 0 || cfg.Model.PresencePenalty != 0 || cfg.Model.RepeatPenalty != 1 {
		t.Fatalf(
			"penalties = min_p %v presence %v repeat %v",
			cfg.Model.MinP,
			cfg.Model.PresencePenalty,
			cfg.Model.RepeatPenalty,
		)
	}
	if cfg.Embedding.Enabled {
		t.Fatal("Ollama embeddings must remain opt-in independently of local inference")
	}
}

func TestLoadRequiredRejectsMissingBaseConfig(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "missing.toml")
	_, err := LoadRequired(path)
	if err == nil || !strings.Contains(err.Error(), "read config "+path) {
		t.Fatalf("LoadRequired error = %v, want readable-path failure", err)
	}
}

func TestLoadRequiredRejectsUnreadableBaseConfig(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "axiom.toml")
	if err := os.Mkdir(path, 0o700); err != nil {
		t.Fatal(err)
	}
	_, err := LoadRequired(path)
	if err == nil || !strings.Contains(err.Error(), "read config "+path) {
		t.Fatalf("LoadRequired error = %v, want readable-file failure", err)
	}
}

func TestLoadAllowsMissingImplicitBaseConfig(t *testing.T) {
	t.Parallel()

	if _, err := Load(filepath.Join(t.TempDir(), "missing.toml")); err != nil {
		t.Fatalf("Load missing implicit config: %v", err)
	}
}

func TestResolveRelativePathsUsesConfigDirectory(t *testing.T) {
	t.Parallel()
	configPath := filepath.Join(t.TempDir(), "config", "axiom.toml")
	cfg := Defaults()
	cfg.Database.Path = "data/memory.db"
	cfg.Tools.PythonPath = ".venv/bin/python"
	cfg.Tools.MojoKernelsDir = "mojo/kernels"
	resolveRelativePaths(cfg, configPath)
	base := filepath.Dir(configPath)
	if cfg.Database.Path != filepath.Join(base, "data/memory.db") {
		t.Fatalf("database path = %q", cfg.Database.Path)
	}
	if cfg.Tools.PythonPath != filepath.Join(base, ".venv/bin/python") {
		t.Fatalf("python path = %q", cfg.Tools.PythonPath)
	}
	if cfg.Tools.MojoKernelsDir != filepath.Join(base, "mojo/kernels") {
		t.Fatalf("kernels path = %q", cfg.Tools.MojoKernelsDir)
	}

	cfg.Tools.PythonPath = "python3"
	resolveRelativePaths(cfg, configPath)
	if cfg.Tools.PythonPath != "python3" {
		t.Fatalf("command-name Python path was rewritten: %q", cfg.Tools.PythonPath)
	}
}

func TestResolveRelativePathsExpandsTrustedPythonHome(t *testing.T) {
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}

	cfg := Defaults()
	cfg.Tools.PythonPath = "~/.axiom/venv/bin/python"
	resolveRelativePaths(cfg, filepath.Join(t.TempDir(), "axiom.toml"))

	want := filepath.Join(home, ".axiom", "venv", "bin", "python")
	if cfg.Tools.PythonPath != want {
		t.Fatalf("python path = %q, want %q", cfg.Tools.PythonPath, want)
	}
}

func TestSanitizeConfigRestoresTokenDefaults(t *testing.T) {
	t.Parallel()

	cfg := Defaults()
	cfg.Model.ContextSize = 0
	cfg.Model.MaxOutputTokens = -1
	cfg.Model.RunnerTimeout = 0
	cfg.Tools.MaxOutputBytes = 0
	cfg.Security.Sandbox.TimeoutSec = 45
	cfg.Security.Sandbox.CPUTimeSec = 0
	cfg.Security.Sandbox.MaxMemoryBytes = 0
	cfg.Security.Sandbox.MaxProcesses = 0
	cfg.Security.Sandbox.MaxFileSizeBytes = 0
	cfg.Security.Sandbox.MaxOpenFiles = 0
	sanitizeConfig(cfg)

	if cfg.Model.ContextSize != 65536 {
		t.Fatalf("ContextSize = %d, want sanitize fallback 65536", cfg.Model.ContextSize)
	}
	if cfg.Model.MaxOutputTokens != 8192 {
		t.Fatalf("MaxOutputTokens = %d, want 8192", cfg.Model.MaxOutputTokens)
	}
	if cfg.Model.RunnerTimeout != 1800 {
		t.Fatalf("RunnerTimeout = %d, want 1800", cfg.Model.RunnerTimeout)
	}
	if cfg.Tools.MaxOutputBytes != 1024*1024 {
		t.Fatalf("Tools.MaxOutputBytes = %d, want 1048576", cfg.Tools.MaxOutputBytes)
	}
	if cfg.Security.Sandbox.CPUTimeSec != 45 {
		t.Fatalf("Sandbox.CPUTimeSec = %d, want timeout-derived 45", cfg.Security.Sandbox.CPUTimeSec)
	}
	if cfg.Security.Sandbox.MaxMemoryBytes != 8*1024*1024*1024 ||
		cfg.Security.Sandbox.MaxProcesses != 1024 ||
		cfg.Security.Sandbox.MaxFileSizeBytes != 1024*1024*1024 ||
		cfg.Security.Sandbox.MaxOpenFiles != 1024 {
		t.Fatalf("unexpected sanitized sandbox resource limits: %+v", cfg.Security.Sandbox)
	}
}

func TestValidateNetworkPolicyAllowsOnlyLoopbackCoreEndpoints(t *testing.T) {
	t.Parallel()

	for _, endpoint := range []string{
		"http://localhost:11434",
		"https://localhost.:8443",
		"http://127.0.0.1:11434",
		"http://[::1]:11434",
	} {
		cfg := Defaults()
		cfg.Security.AllowNetwork = false
		cfg.Model.RunnerURL = endpoint
		cfg.Embedding.URL = endpoint
		if err := validateNetworkPolicy(cfg); err != nil {
			t.Errorf("loopback endpoint %q rejected: %v", endpoint, err)
		}
	}
}

func TestValidateNetworkPolicyRejectsExternalCoreEndpoints(t *testing.T) {
	t.Parallel()

	for _, endpoint := range []string{
		"https://api.example.com",
		"http://192.168.1.2:11434",
		"http://2130706433:11434",
		"file:///tmp/model.sock",
	} {
		cfg := Defaults()
		cfg.Security.AllowNetwork = false
		cfg.Model.Enabled = true
		cfg.Model.RunnerURL = endpoint
		err := validateNetworkPolicy(cfg)
		if err == nil || !strings.Contains(err.Error(), "model.runner_url") {
			t.Errorf("external endpoint %q error = %v, want model.runner_url rejection", endpoint, err)
		}
	}
}

func TestValidateNetworkPolicyIgnoresDisabledLocalEndpoints(t *testing.T) {
	t.Parallel()

	cfg := Defaults()
	cfg.Security.AllowNetwork = false
	cfg.Model.Enabled = false
	cfg.Model.RunnerURL = "https://api.example.com"
	cfg.Embedding.Enabled = false
	cfg.Embedding.URL = "https://api.example.com"

	if err := validateNetworkPolicy(cfg); err != nil {
		t.Fatalf("disabled local endpoints rejected: %v", err)
	}
}

func TestOpenAIEnvironmentOverridePrecedence(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "standard")
	t.Setenv("AXIOM_OPENAI_API_KEY", "axiom-specific")

	cfg, err := Load(filepath.Join(t.TempDir(), "missing.toml"))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Cloud.OpenAIKey != "axiom-specific" {
		t.Fatalf("OpenAIKey = %q, want Axiom-specific override", cfg.Cloud.OpenAIKey)
	}
}

func TestOpenAIEnvironmentOverridesConfigValue(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "from-environment")
	t.Setenv("AXIOM_OPENAI_API_KEY", "")

	path := filepath.Join(t.TempDir(), "axiom.toml")
	if err := os.WriteFile(path, []byte("[cloud]\nopenai_key = \"from-config\"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := LoadRequired(path)
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Cloud.OpenAIKey != "from-environment" {
		t.Fatalf("OpenAIKey = %q, want environment override", cfg.Cloud.OpenAIKey)
	}
}

func TestOpenAIWhitespaceKeyIsNotConfigured(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "   ")
	t.Setenv("AXIOM_OPENAI_API_KEY", "")

	cfg, err := Load(filepath.Join(t.TempDir(), "missing.toml"))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Cloud.OpenAIKey != "" {
		t.Fatalf("OpenAIKey = %q, want trimmed empty value", cfg.Cloud.OpenAIKey)
	}
}

func TestLoadRejectsInvalidOpenAIReasoningEffort(t *testing.T) {
	t.Parallel()

	path := filepath.Join(t.TempDir(), "axiom.toml")
	if err := os.WriteFile(path, []byte("[cloud]\nreasoning_effort = \"extreme\"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	_, err := LoadRequired(path)
	if err == nil || !strings.Contains(err.Error(), "cloud.reasoning_effort") {
		t.Fatalf("LoadRequired error = %v, want reasoning effort validation", err)
	}
}
