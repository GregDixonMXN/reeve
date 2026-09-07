package orchestrator

import (
	"context"
	"slices"
	"strings"
	"testing"

	"reeve/internal/cognitive"
	"reeve/internal/config"
)

type modeTestRunner struct{}

func (*modeTestRunner) Complete(context.Context, string, int) (string, error) {
	return `{"reasoning":"","tool_call":null,"content":"ok"}`, nil
}

func (*modeTestRunner) Unload() error { return nil }

func TestModeManagerCloudOnlyAvailability(t *testing.T) {
	t.Parallel()

	engine := cognitive.NewEngine(config.ModelConfig{})
	manager := NewModeManager(ModeManagerConfig{
		Engine:        engine,
		CloudRunner:   &modeTestRunner{},
		CloudProvider: "openai",
	})

	if got := manager.AvailableModes(); !slices.Equal(got, []Mode{ModeCloud}) {
		t.Fatalf("AvailableModes() = %v, want [cloud]", got)
	}
	if err := manager.SetMode(ModeCloud); err != nil {
		t.Fatalf("SetMode(cloud): %v", err)
	}
	if manager.CurrentMode() != ModeCloud {
		t.Fatalf("CurrentMode() = %q, want cloud", manager.CurrentMode())
	}
	if err := manager.SetMode(ModeLocal); err == nil {
		t.Fatal("local mode unexpectedly available without a local runner")
	}
}

func TestModeManagerMissingOpenAICredentialsIsExplicit(t *testing.T) {
	t.Parallel()

	manager := NewModeManager(ModeManagerConfig{
		Engine:        cognitive.NewEngine(config.ModelConfig{}),
		CloudProvider: "openai",
	})

	err := manager.SetMode(ModeCloud)
	if err == nil || !strings.Contains(err.Error(), "openai credentials") {
		t.Fatalf("SetMode(cloud) error = %v, want OpenAI credentials guidance", err)
	}
	if manager.CurrentMode() != "" {
		t.Fatalf("CurrentMode() = %q, want empty mode after failed setup", manager.CurrentMode())
	}
}

func TestModeManagerDoesNotAdvertiseHybridWithoutCloudRoute(t *testing.T) {
	t.Parallel()

	manager := NewModeManager(ModeManagerConfig{
		Engine:      cognitive.NewEngine(config.ModelConfig{}),
		LocalRunner: &modeTestRunner{},
	})

	if got := manager.AvailableModes(); !slices.Equal(got, []Mode{ModeLocal}) {
		t.Fatalf("AvailableModes() = %v, want [local]", got)
	}
	if err := manager.SetMode(ModeHybrid); err == nil || !strings.Contains(err.Error(), "cloud runner or delegation") {
		t.Fatalf("SetMode(hybrid) error = %v, want unavailable cloud-route guidance", err)
	}
}
