package orchestrator

import (
	"fmt"
	"sync"

	"reeve/internal/cognitive"
)

// Mode represents Reeve's operating mode.
type Mode string

const (
	// ModeLocal uses only the configured local LLM.
	// Cloud delegation tool is DISABLED.
	ModeLocal Mode = "local"

	// ModeHybrid uses the local LLM with cloud delegation available.
	// Reeve routes complex work to the configured cloud provider.
	ModeHybrid Mode = "hybrid"

	// ModeCloud uses the configured cloud provider as the primary LLM.
	// The cloud model drives the Think-Verify-Act loop with native tools.
	ModeCloud Mode = "cloud"
)

// ModeManager handles switching between operating modes.
type ModeManager struct {
	mu             sync.RWMutex
	currentMode    Mode
	engine         *cognitive.Engine
	localRunner    cognitive.LLMRunner
	cloudRunner    cognitive.LLMRunner
	hybridRunner   cognitive.LLMRunner // auto-routing runner
	hybridDelegate bool                // ask_cloud_model has a configured backend
	cloudDisabler  func(enabled bool)  // toggles ask_cloud_model tool visibility
	cloudProvider  string
}

type ModeManagerConfig struct {
	Engine                    *cognitive.Engine
	LocalRunner               cognitive.LLMRunner
	CloudRunner               cognitive.LLMRunner // nil if the selected provider is not configured
	CloudProvider             string
	HybridDelegationAvailable bool
	OnCloudToggle             func(enabled bool) // called to enable/disable cloud tool
}

func NewModeManager(cfg ModeManagerConfig) *ModeManager {
	mm := &ModeManager{
		engine:         cfg.Engine,
		localRunner:    cfg.LocalRunner,
		hybridDelegate: cfg.HybridDelegationAvailable,
		cloudDisabler:  cfg.OnCloudToggle,
		cloudProvider:  cfg.CloudProvider,
	}
	if mm.cloudProvider == "" {
		mm.cloudProvider = "cloud"
	}

	if cfg.CloudRunner != nil {
		mm.cloudRunner = cfg.CloudRunner

		// Build the hybrid router: classifies locally, dispatches to best runner
		if cfg.LocalRunner != nil {
			router := cognitive.NewHybridRouter(cognitive.HybridRouterConfig{
				LocalRunner: cfg.LocalRunner,
				CloudRunner: cfg.CloudRunner,
			})
			mm.hybridRunner = cognitive.NewHybridRunner(router)
		}
	}

	return mm
}

// SetMode switches Reeve's operating mode.
// Returns an error if the requested mode requires unavailable resources.
func (mm *ModeManager) SetMode(mode Mode) error {
	mm.mu.Lock()
	defer mm.mu.Unlock()

	switch mode {
	case ModeLocal:
		if mm.localRunner == nil {
			return fmt.Errorf("local mode is disabled or has no configured runner")
		}
		mm.engine.SetRunner(mm.localRunner)
		mm.engine.SetMode("local") // tell the engine: no cloud delegation, use all local tools
		if mm.cloudDisabler != nil {
			mm.cloudDisabler(false) // disable ask_cloud_model
		}
		mm.currentMode = ModeLocal

	case ModeHybrid:
		if mm.localRunner == nil {
			return fmt.Errorf("hybrid mode requires an enabled local runner")
		}
		if mm.hybridRunner == nil && !mm.hybridDelegate {
			return fmt.Errorf("hybrid mode requires a configured cloud runner or delegation provider")
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
			return fmt.Errorf(
				"cloud mode requires configured %s credentials",
				mm.cloudProvider,
			)
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
		modes = append(modes, ModeLocal)
		if mm.hybridRunner != nil || mm.hybridDelegate {
			modes = append(modes, ModeHybrid)
		}
	}
	if mm.cloudRunner != nil {
		modes = append(modes, ModeCloud)
	}
	return modes
}
