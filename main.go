package main

import (
	"context"
	"embed"
	"log"

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
	cfg, err := config.Load("axiom.toml")
	if err != nil {
		log.Fatalf("[AXIOM] Config error: %v", err)
	}

	appLog := logger.New(cfg.LogLevel)
	appLog.Info("Axiom v1.0 — Local-First AI Agent Runtime")

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

	// ── Local Runner (Ollama) ───────────────────────────────────────────
	var localRunner cognitive.LLMRunner
	if cfg.Model.EnableStreaming && cfg.Model.RunnerType == "ollama" {
		// StreamingRunner for live token output when streaming is enabled for Ollama
		localRunner = adapters.NewStreamingRunner(adapters.StreamingRunnerConfig{
			BaseURL:  cfg.Model.RunnerURL,
			Model:    cfg.Model.RunnerModel,
			Protocol: adapters.ProtocolOllama,
			TimeoutS: cfg.Model.RunnerTimeout,
			OnToken: func(token string) {
				if app != nil {
					app.EmitToken(token)
				}
			},
		})
		appLog.Info("Streaming: enabled for live token output")
	} else {
		// Non-streaming fallback
		switch cfg.Model.RunnerType {
		case "openai":
			localRunner = adapters.NewRemoteRunner(adapters.RemoteRunnerConfig{
				BaseURL:  cfg.Model.RunnerURL,
				Model:    cfg.Model.RunnerModel,
				Protocol: adapters.ProtocolOpenAI,
				TimeoutS: cfg.Model.RunnerTimeout,
			})
		default:
			localRunner = adapters.NewRemoteRunner(adapters.RemoteRunnerConfig{
				BaseURL:  cfg.Model.RunnerURL,
				Model:    cfg.Model.RunnerModel,
				Protocol: adapters.ProtocolOllama,
				TimeoutS: cfg.Model.RunnerTimeout,
			})
		}
	}
	engine.SetRunner(localRunner)
	appLog.Info("Local runner: %s @ %s (model: %s)", cfg.Model.RunnerType, cfg.Model.RunnerURL, cfg.Model.RunnerModel)

	// ── Cloud Runner (Anthropic) ────────────────────────────────────────
	var cloudRunner *adapters.AnthropicRunner
	if cfg.Cloud.AnthropicKey != "" {
		cloudRunner = adapters.NewAnthropicRunner(adapters.AnthropicRunnerConfig{
			APIKey:    cfg.Cloud.AnthropicKey,
			Model:     cfg.Cloud.AnthropicModel,
			MaxTokens: cfg.Cloud.MaxTokens,
			TimeoutS:  cfg.Cloud.TimeoutSec,
		})
		appLog.Info("Cloud runner: Claude (%s) — cloud mode available", cfg.Cloud.AnthropicModel)
	}

	// ── Cloud Delegator ─────────────────────────────────────────────────
	var cloudCfg *tools.CloudConfig
	if cfg.Cloud.AnthropicKey != "" || cfg.Cloud.GeminiKey != "" {
		cloudCfg = &tools.CloudConfig{
			AnthropicKey:   cfg.Cloud.AnthropicKey,
			AnthropicModel: cfg.Cloud.AnthropicModel,
			GeminiKey:      cfg.Cloud.GeminiKey,
			GeminiModel:    cfg.Cloud.GeminiModel,
			MaxTokens:      cfg.Cloud.MaxTokens,
			TimeoutSec:     cfg.Cloud.TimeoutSec,
		}
	}

	// ── Sandbox ─────────────────────────────────────────────────────────
	var sandboxCfg *tools.SandboxConfig
	if cfg.Security.Sandbox.Enabled {
		sandboxCfg = &tools.SandboxConfig{
			Enabled:         true,
			AllowedDirs:     cfg.Security.Sandbox.AllowedDirs,
			BlockedPatterns: cfg.Security.Sandbox.BlockedPatterns,
			AllowedBinaries: cfg.Security.Sandbox.AllowedBinaries,
			TimeoutSec:      cfg.Security.Sandbox.TimeoutSec,
			MaxOutputBytes:  cfg.Security.Sandbox.MaxOutputBytes,
		}
		appLog.Info("Sandbox: enabled (timeout=%ds)", sandboxCfg.TimeoutSec)
	}

	// ── Tools ───────────────────────────────────────────────────────────
	toolReg := tools.NewRegistry(cfg.Tools, cloudCfg, sandboxCfg)
	guard := guardrail.New(cfg.Security)

	// Wire memory searcher for search_memory tool
	toolReg.SetMemory(&memorySearchAdapter{store: mem})

	// ── Mode Manager ────────────────────────────────────────────────────
	modeManager := orchestrator.NewModeManager(orchestrator.ModeManagerConfig{
		Engine:        engine,
		LocalRunner:   localRunner,
		CloudRunner:   cloudRunner,
		OnCloudToggle: toolReg.SetCloudEnabled,
	})

	// Set initial mode
	initialMode := cfg.Model.DefaultMode
	if initialMode == "" {
		initialMode = "hybrid"
	}
	if err := modeManager.SetMode(orchestrator.Mode(initialMode)); err != nil {
		appLog.Warn("Failed to set initial mode '%s': %v — falling back to hybrid", initialMode, err)
		modeManager.SetMode(orchestrator.ModeHybrid)
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
	})

	// ── Wails App ───────────────────────────────────────────────────────
	app = NewApp(orch, modeManager)

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
