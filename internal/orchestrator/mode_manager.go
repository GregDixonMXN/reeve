package orchestrator

import (
	"fmt"
	"sync"

	"axiom/internal/cognitive"
	"axiom/internal/cognitive/adapters"
)

// Mode represents Axiom's operating mode.
type Mode string

const (
	// ModeLocal uses only the local LLM (Ollama/Phi-4).
	// Cloud delegation tool is DISABLED.
	ModeLocal Mode = "local"

	// ModeHybrid uses the local LLM with cloud delegation available.
	// Phi-4 decides when to hand off to Claude/Gemini.
	ModeHybrid Mode = "hybrid"

	// ModeCloud uses Claude as the primary LLM.
	// Claude drives the entire Think-Verify-Act loop natively,
	// with full access to Axiom's tool definitions via Claude's tools array.
	ModeCloud Mode = "cloud"
)

// ModeManager handles switching between operating modes.
type ModeManager struct {
	mu            sync.RWMutex
	currentMode   Mode
	engine        *cognitive.Engine
	localRunner   cognitive.LLMRunner
	cloudRunner   cognitive.LLMRunner
	hybridRunner  cognitive.LLMRunner       // auto-routing runner
	cloudDisabler func(enabled bool)        // toggles ask_cloud_model tool visibility
}

type ModeManagerConfig struct {
	Engine        *cognitive.Engine
	LocalRunner   cognitive.LLMRunner
	CloudRunner   *adapters.AnthropicRunner // nil if no API key
	OnCloudToggle func(enabled bool)        // called to enable/disable cloud tool
}

func NewModeManager(cfg ModeManagerConfig) *ModeManager {
	mm := &ModeManager{
		currentMode:   ModeHybrid, // default
		engine:        cfg.Engine,
		localRunner:   cfg.LocalRunner,
		cloudDisabler: cfg.OnCloudToggle,
	}

	if cfg.CloudRunner != nil {
		mm.cloudRunner = cfg.CloudRunner

		// Build the hybrid router: classifies locally, dispatches to best runner
		router := cognitive.NewHybridRouter(cognitive.HybridRouterConfig{
			LocalRunner: cfg.LocalRunner,
			CloudRunner: cfg.CloudRunner,
		})
		mm.hybridRunner = cognitive.NewHybridRunner(router)
	}

	return mm
}

// SetMode switches Axiom's operating mode.
// Returns an error if the requested mode requires unavailable resources.
func (mm *ModeManager) SetMode(mode Mode) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	switch mode {
	case ModeLocal:
		if mm.localRunner == nil {
			return fmt.Errorf("local mode requires Ollama — is it running?")
		}
		mm.engine.SetRunner(mm.localRunner)
		mm.engine.SetMode("local") // tell the engine: no cloud delegation, use all local tools
		if mm.cloudDisabler != nil {
			mm.cloudDisabler(false) // disable ask_cloud_model
		}
		mm.currentMode = ModeLocal

	case ModeHybrid:
		if mm.localRunner == nil {
			return fmt.Errorf("hybrid mode requires Ollama — is it running?")
		}
		if mm.hybridRunner != nil {
			// Smart routing: classify locally, dispatch to best runner
			mm.engine.SetRunner(mm.hybridRunner)
		} else {
			// No cloud key — hybrid falls back to local with cloud tool enabled
			mm.engine.SetRunner(mm.localRunner)
		}
		mm.engine.SetMode("hybrid")
		if mm.cloudDisabler != nil {
			mm.cloudDisabler(true) // enable ask_cloud_model as fallback
		}
		mm.currentMode = ModeHybrid

	case ModeCloud:
		if mm.cloudRunner == nil {
			return fmt.Errorf("cloud mode requires an Anthropic API key (set cloud.anthropic_key in axiom.toml)")
		}
		mm.engine.SetRunner(mm.cloudRunner)
		mm.engine.SetMode("cloud")
		if mm.cloudDisabler != nil {
			mm.cloudDisabler(false) // cloud model IS the runner, no delegation needed
		}
		mm.currentMode = ModeCloud

	default:
		return fmt.Errorf("unknown mode: %s (use local, hybrid, or cloud)", mode)
	}

	return nil
}

// CurrentMode returns the active operating mode.
func (mm *ModeManager) CurrentMode() Mode {
	mm.mu.RLock()
	defer mm.mu.RUnlock()
	return mm.currentMode
}

// AvailableModes returns which modes are available based on configuration.
func (mm *ModeManager) AvailableModes() []Mode {
	modes := []Mode{}
	if mm.localRunner != nil {
		modes = append(modes, ModeLocal, ModeHybrid)
	}
	if mm.cloudRunner != nil {
		modes = append(modes, ModeCloud)
	}
	return modes
}
