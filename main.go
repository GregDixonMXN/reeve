package main

import (
	"context"
	"embed"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"time"

	"axiom/internal/cognitive"
	"axiom/internal/cognitive/adapters"
	"axiom/internal/config"
	"axiom/internal/guardrail"
	"axiom/internal/memory"
	"axiom/internal/orchestrator"
	"axiom/internal/tools"
	"axiom/pkg/logger"

	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
)

// memorySearchAdapter wraps memory.Store to satisfy tools.MemorySearcher.
type memorySearchAdapter struct {
	store *memory.Store
}

func (a *memorySearchAdapter) Search(ctx context.Context, query string, limit int) ([]tools.MemoryResult, error) {
	entries, err := a.store.Search(ctx, query, limit)
	if err != nil {
		return nil, err
	}
	results := make([]tools.MemoryResult, len(entries))
	for i, e := range entries {
		results[i] = tools.MemoryResult{
			Content: e.Content,
			Score:   e.Score,
		}
	}
	return results, nil
}

//go:embed all:frontend/dist
var assets embed.FS

func main() {
	if tools.IsSandboxHelperInvocation(os.Args) {
		if err := tools.RunSandboxHelper(os.Args); err != nil {
			fmt.Fprintf(os.Stderr, "axiom sandbox helper: %v\n", err)
			os.Exit(125)
		}
		return
	}

	configPath, explicitConfig := findConfigPath()
	var cfg *config.AppConfig
	var err error
	if explicitConfig {
		cfg, err = config.LoadRequired(configPath)
	} else {
		cfg, err = config.Load(configPath)
	}
	if err != nil {
		log.Fatalf("[AXIOM] Config error: %v", err)
	}

	appLog := logger.New(cfg.LogLevel)
	appLog.Info("Axiom v1.0 — AI Agent Runtime")
	appLog.Info("Config: %s", configPath)

	// ── Embedding ───────────────────────────────────────────────────────
	var embedder *memory.Embedder
	if cfg.Embedding.Enabled {
		embedder = memory.NewEmbedder(memory.EmbedderConfig{
			BaseURL: cfg.Embedding.URL,
			Model:   cfg.Embedding.Model,
			Dim:     cfg.Embedding.Dim,
		})
		appLog.Info("Embedder: %s (dim=%d)", cfg.Embedding.Model, cfg.Embedding.Dim)
	}

	// ── Memory ──────────────────────────────────────────────────────────
	mem, err := memory.NewStore(cfg.Database, embedder)
	if err != nil {
		log.Fatalf("[AXIOM] Memory: %v", err)
	}
	defer mem.Close()

	// ── Cognitive Engine ────────────────────────────────────────────────
	engine := cognitive.NewEngine(cfg.Model)
	var app *App

	// ── Optional Local Runner ───────────────────────────────────────────
	var localRunner cognitive.LLMRunner
	localReady := cfg.Model.Enabled
	if cfg.Model.Enabled && cfg.Model.RunnerType == "ollama" {
		probeCtx, cancelProbe := context.WithTimeout(context.Background(), 5*time.Second)
		probeErr := adapters.CheckOllamaModel(
			probeCtx,
			cfg.Model.RunnerURL,
			cfg.Model.RunnerModel,
			5*time.Second,
		)
		cancelProbe()
		if probeErr != nil {
			localReady = false
			appLog.Warn("Local Ollama preflight failed: %v", probeErr)
		}
	}
	if localReady && cfg.Model.EnableStreaming && cfg.Model.RunnerType == "ollama" {
		// StreamingRunner for live token output when streaming is enabled for Ollama
		localRunner = adapters.NewStreamingRunner(adapters.StreamingRunnerConfig{
			BaseURL:     cfg.Model.RunnerURL,
			Model:       cfg.Model.RunnerModel,
			Protocol:    adapters.ProtocolOllama,
			TimeoutS:    cfg.Model.RunnerTimeout,
			ContextSize: cfg.Model.ContextSize,
			OnToken: func(token string) {
				if app != nil {
					app.emitToken(token)
				}
			},
		})
		appLog.Info("Streaming: enabled for live token output")
	} else if localReady {
		// Non-streaming fallback
		switch cfg.Model.RunnerType {
		case "openai", "openai-compatible":
			localRunner = adapters.NewRemoteRunner(adapters.RemoteRunnerConfig{
				BaseURL:         cfg.Model.RunnerURL,
				Model:           cfg.Model.RunnerModel,
				Protocol:        adapters.ProtocolOpenAI,
				TimeoutS:        cfg.Model.RunnerTimeout,
				ContextSize:     cfg.Model.ContextSize,
				Temperature:     cfg.Model.Temperature,
				TopP:            cfg.Model.TopP,
				TopK:            cfg.Model.TopK,
				MinP:            cfg.Model.MinP,
				PresencePenalty: cfg.Model.PresencePenalty,
				RepeatPenalty:   cfg.Model.RepeatPenalty,
				Threads:         cfg.Model.Threads,
				Think:           cfg.Model.EnableThinking,
			})
		default:
			localRunner = adapters.NewRemoteRunner(adapters.RemoteRunnerConfig{
				BaseURL:         cfg.Model.RunnerURL,
				Model:           cfg.Model.RunnerModel,
				Protocol:        adapters.ProtocolOllama,
				TimeoutS:        cfg.Model.RunnerTimeout,
				ContextSize:     cfg.Model.ContextSize,
				Temperature:     cfg.Model.Temperature,
				TopP:            cfg.Model.TopP,
				TopK:            cfg.Model.TopK,
				MinP:            cfg.Model.MinP,
				PresencePenalty: cfg.Model.PresencePenalty,
				RepeatPenalty:   cfg.Model.RepeatPenalty,
				Threads:         cfg.Model.Threads,
				Think:           cfg.Model.EnableThinking,
			})
		}
	}
	if localRunner != nil {
		engine.SetRunner(localRunner)
		appLog.Info("Local runner: %s @ %s (model: %s)", cfg.Model.RunnerType, cfg.Model.RunnerURL, cfg.Model.RunnerModel)
	} else {
		appLog.Info("Local runner: disabled")
	}

	// Inject workspace dirs so the LLM knows exact allowed paths
	if len(cfg.Tools.AllowedDirs) > 0 {
		engine.SetWorkspaceDirs(cfg.Tools.AllowedDirs)
		appLog.Info("Workspace dirs injected into system prompt: %v", cfg.Tools.AllowedDirs)
	}

	// ── Primary Cloud Runner ────────────────────────────────────────────
	var cloudRunner cognitive.LLMRunner
	cloudModel := ""
	if cfg.Security.AllowNetwork {
		switch cfg.Cloud.Provider {
		case "openai":
			cloudModel = cfg.Cloud.OpenAIModel
			if cfg.Cloud.OpenAIKey != "" {
				cloudRunner = adapters.NewOpenAIRunner(adapters.OpenAIRunnerConfig{
					APIKey:          cfg.Cloud.OpenAIKey,
					Model:           cfg.Cloud.OpenAIModel,
					MaxTokens:       cfg.Cloud.MaxTokens,
					ReasoningEffort: cfg.Cloud.ReasoningEffort,
					TimeoutS:        cfg.Cloud.TimeoutSec,
				})
				appLog.Info("Cloud runner: OpenAI (%s, reasoning=%s) — cloud mode available", cfg.Cloud.OpenAIModel, cfg.Cloud.ReasoningEffort)
			} else {
				appLog.Warn("OpenAI cloud mode is not configured — set AXIOM_OPENAI_API_KEY or OPENAI_API_KEY")
			}
		case "anthropic":
			cloudModel = cfg.Cloud.AnthropicModel
			if cfg.Cloud.AnthropicKey != "" {
				cloudRunner = adapters.NewAnthropicRunner(adapters.AnthropicRunnerConfig{
					APIKey:    cfg.Cloud.AnthropicKey,
					Model:     cfg.Cloud.AnthropicModel,
					MaxTokens: cfg.Cloud.MaxTokens,
					TimeoutS:  cfg.Cloud.TimeoutSec,
				})
				appLog.Info("Cloud runner: Anthropic (%s) — cloud mode available", cfg.Cloud.AnthropicModel)
			} else {
				appLog.Warn("Anthropic cloud mode is not configured — set AXIOM_ANTHROPIC_KEY or ANTHROPIC_API_KEY")
			}
		}
	}

	// ── Cloud Delegator ─────────────────────────────────────────────────
	var cloudCfg *tools.CloudConfig
	if cfg.Security.AllowNetwork && (cfg.Cloud.AnthropicKey != "" || cfg.Cloud.GeminiKey != "") {
		cloudCfg = &tools.CloudConfig{
			AnthropicKey:   cfg.Cloud.AnthropicKey,
			AnthropicModel: cfg.Cloud.AnthropicModel,
			GeminiKey:      cfg.Cloud.GeminiKey,
			GeminiModel:    cfg.Cloud.GeminiModel,
			MaxTokens:      cfg.Cloud.MaxTokens,
			TimeoutSec:     cfg.Cloud.TimeoutSec,
		}
	}
	if !cfg.Security.AllowNetwork && (cfg.Cloud.OpenAIKey != "" || cfg.Cloud.AnthropicKey != "" || cfg.Cloud.GeminiKey != "") {
		appLog.Info("Network access disabled — cloud runners, delegation, and vision are unavailable")
	}

	// ── Sandbox ─────────────────────────────────────────────────────────
	var sandboxCfg *tools.SandboxConfig
	isolationAvailable := false
	if cfg.Security.Sandbox.Enabled {
		candidate := &tools.SandboxConfig{
			Enabled:            true,
			RequireOSIsolation: cfg.Security.Sandbox.RequireOSIsolation,
			AllowNetwork:       cfg.Security.AllowNetwork,
			AllowedDirs:        cfg.Security.Sandbox.AllowedDirs,
			BlockedPatterns:    cfg.Security.Sandbox.BlockedPatterns,
			AllowedBinaries:    cfg.Security.Sandbox.AllowedBinaries,
			TimeoutSec:         cfg.Security.Sandbox.TimeoutSec,
			MaxOutputBytes:     cfg.Security.Sandbox.MaxOutputBytes,
			CPUTimeSec:         cfg.Security.Sandbox.CPUTimeSec,
			MaxMemoryBytes:     cfg.Security.Sandbox.MaxMemoryBytes,
			MaxProcesses:       cfg.Security.Sandbox.MaxProcesses,
			MaxFileSizeBytes:   cfg.Security.Sandbox.MaxFileSizeBytes,
			MaxOpenFiles:       cfg.Security.Sandbox.MaxOpenFiles,
		}
		if candidate.RequireOSIsolation {
			helperPath, helperErr := tools.IsolationHelperPath()
			isolationErr := tools.IsolationSupported(candidate.AllowNetwork)
			switch {
			case helperErr != nil:
				appLog.Warn("OS-isolated execution unavailable: resolve helper executable: %v — execute_code disabled", helperErr)
			case isolationErr != nil:
				appLog.Warn("OS-isolated execution unavailable: %v — execute_code disabled", isolationErr)
			default:
				candidate.HelperPath = helperPath
				sandboxCfg = candidate
				isolationAvailable = true
				appLog.Info(
					"Sandbox: Linux Landlock/seccomp/resource isolation enabled (network=%v, timeout=%ds, memory=%dMiB, processes=%d)",
					candidate.AllowNetwork,
					candidate.TimeoutSec,
					candidate.MaxMemoryBytes/(1024*1024),
					candidate.MaxProcesses,
				)
			}
		} else {
			sandboxCfg = candidate
			appLog.Warn("Sandbox: OS isolation explicitly disabled; directory and command policy only")
		}
	}

	// ── Tools ───────────────────────────────────────────────────────────
	toolReg := tools.NewRegistry(cfg.Tools, cloudCfg, sandboxCfg)
	guard := guardrail.New(cfg.Security)
	guard.SetIsolatedExecutionAvailable(isolationAvailable)

	// Wire memory searcher for search_memory tool
	toolReg.SetMemory(&memorySearchAdapter{store: mem})

	// Wire guardrail for schema validation
	toolReg.SetGuardrail(guard)

	// ── Mode Manager ────────────────────────────────────────────────────
	modeManager := orchestrator.NewModeManager(orchestrator.ModeManagerConfig{
		Engine:                    engine,
		LocalRunner:               localRunner,
		CloudRunner:               cloudRunner,
		CloudProvider:             cfg.Cloud.Provider,
		HybridDelegationAvailable: cloudCfg != nil,
		OnCloudToggle:             toolReg.SetCloudEnabled,
	})

	// Set initial mode
	initialMode := cfg.Model.DefaultMode
	if initialMode == "" {
		initialMode = "cloud"
	}
	if err := modeManager.SetMode(orchestrator.Mode(initialMode)); err != nil {
		appLog.Warn("Failed to set initial mode '%s': %v", initialMode, err)
		available := modeManager.AvailableModes()
		if len(available) > 0 {
			if fallbackErr := modeManager.SetMode(available[0]); fallbackErr != nil {
				appLog.Warn("Failed to activate fallback mode '%s': %v", available[0], fallbackErr)
			}
		} else {
			appLog.Warn("No inference backend is configured; the UI will show setup instructions")
		}
	}
	appLog.Info("Mode: %s (available: %v)", modeManager.CurrentMode(), modeManager.AvailableModes())

	// ── Orchestrator ────────────────────────────────────────────────────
	orch := orchestrator.New(orchestrator.Config{
		Logger:            appLog,
		Memory:            mem,
		Cognitive:         engine,
		Tools:             toolReg,
		Guardrail:         guard,
		ModeManager:       modeManager,
		ReflectionEnabled: cfg.Reflection.Enabled,
		WorkspaceDirs:     cfg.Tools.AllowedDirs,
		MaxIterations:     cfg.Model.MaxIterations,
	})

	// ── Wails App ───────────────────────────────────────────────────────
	app = NewApp(orch, modeManager, RuntimeInfo{
		CloudProvider:   cfg.Cloud.Provider,
		CloudModel:      cloudModel,
		LocalProvider:   cfg.Model.RunnerType,
		LocalModel:      cfg.Model.RunnerModel,
		CloudConfigured: cloudRunner != nil,
		LocalEnabled:    cfg.Model.Enabled,
		LocalConfigured: localRunner != nil,
		NetworkAllowed:  cfg.Security.AllowNetwork,
	})

	err = wails.Run(&options.App{
		Title:  "Axiom AI Agent",
		Width:  1280,
		Height: 800,
		AssetServer: &assetserver.Options{
			Assets: assets,
		},
		BackgroundColour: &options.RGBA{R: 13, G: 13, B: 18, A: 1},
		OnStartup:        app.Startup,
		OnShutdown:       app.Shutdown,
		Bind: []interface{}{
			app,
		},
	})
	if err != nil {
		log.Fatalf("[AXIOM] %v", err)
	}
}

func findConfigPath() (string, bool) {
	if explicit := strings.TrimSpace(os.Getenv("AXIOM_CONFIG")); explicit != "" {
		return explicit, true
	}
	candidates := []string{"axiom.toml"}
	if configDir, err := os.UserConfigDir(); err == nil {
		candidates = append(candidates, filepath.Join(configDir, "axiom", "axiom.toml"))
	}
	if executable, err := os.Executable(); err == nil {
		dir := filepath.Dir(executable)
		candidates = append(candidates,
			filepath.Join(dir, "axiom.toml"),
			filepath.Join(filepath.Dir(dir), "axiom.toml"),
			filepath.Join(filepath.Dir(filepath.Dir(dir)), "axiom.toml"),
		)
	}
	seen := make(map[string]struct{}, len(candidates))
	for _, candidate := range candidates {
		absolute, err := filepath.Abs(candidate)
		if err != nil {
			continue
		}
		if _, duplicate := seen[absolute]; duplicate {
			continue
		}
		seen[absolute] = struct{}{}
		if info, err := os.Stat(absolute); err == nil && !info.IsDir() {
			return absolute, false
		}
	}
	return "axiom.toml", false
}
