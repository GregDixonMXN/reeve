package orchestrator

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"reeve/internal/cognitive"
	"reeve/internal/cognitive/adapters"
	"reeve/internal/config"
	"reeve/internal/guardrail"
	"reeve/internal/memory"
	"reeve/internal/tools"
	"reeve/pkg/models"
)

type liveContextMemorySearch struct{}

func (liveContextMemorySearch) Search(
	_ context.Context,
	_ string,
	_ int,
) ([]tools.MemoryResult, error) {
	return nil, nil
}

// This test reproduces the formerly failing large coding turn without
// executing the returned tool call. It is opt-in because it requires the
// dedicated local Ollama service and real model.
func TestLiveOllamaLargeCodingTurnFitsContextAndProducesAction(t *testing.T) {
	if os.Getenv("REEVE_OLLAMA_LIVE") != "1" {
		t.Skip("set REEVE_OLLAMA_LIVE=1 to run against the configured Ollama service")
	}

	projectRoot, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatalf("resolve project root: %v", err)
	}
	cfg, err := config.Load(filepath.Join(projectRoot, "reeve.toml"))
	if err != nil {
		t.Fatalf("load config: %v", err)
	}
	if cfg.Model.RunnerType != "ollama" {
		t.Skipf("configured runner is %q, not ollama", cfg.Model.RunnerType)
	}

	runner := adapters.NewRemoteRunner(adapters.RemoteRunnerConfig{
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
	engine := cognitive.NewEngine(cfg.Model)
	engine.SetRunner(runner)
	engine.SetMode("local")
	engine.SetWorkspaceDirs(cfg.Tools.AllowedDirs)

	registry := tools.NewRegistry(cfg.Tools, nil, nil)
	registry.SetMemory(liveContextMemorySearch{})
	guard := guardrail.New(cfg.Security)
	guard.SetIsolatedExecutionAvailable(true)
	registry.SetGuardrail(guard)
	registry.SetCloudEnabled(false)

	messages := make([]models.Message, 0, 22)
	messages = append(messages, models.Message{
		Role:    models.RoleUser,
		Content: "Earlier conversation anchor about Reeve development.",
	})
	for index := 0; index < 19; index++ {
		role := models.RoleAssistant
		if index%2 == 0 {
			role = models.RoleUser
		}
		messages = append(messages, models.Message{
			Role: role,
			Content: "Bounded recent context " +
				strings.Repeat("implementation detail ", 12),
		})
	}
	const prompt = "ok create a folder on desktop called reeve game and use pygame and python to write your version of the most complete steam ready game you can whatever genre you want to do"
	messages = append(messages, models.Message{
		Role: models.RoleUser,
		Content: prompt + `

[TASK PLAN — EXECUTE THIS EXACTLY]
Create /home/shki/Desktop/reeve game as a polished original Pygame project.
Start with /home/shki/Desktop/reeve game/main.py, then add its supporting
modules, generated original assets, tests, README, requirements, packaging,
and Steam release checklist. Run headless tests after every file exists.
[END TASK PLAN]

[MANDATORY EXECUTION RULE]
Execute the plan using write_file tool calls, one file per call. Do not print
file contents in chat. Start with the first file now.`,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	response, err := engine.Generate(ctx, cognitive.Request{
		ConversationID: "live-context-budget",
		Messages:       messages,
		MemoryContext: []memory.MemoryEntry{{
			Content: "The Desktop is an approved workspace for user-requested projects.",
			Score:   0.91,
		}},
		Tools: registry.Definitions(),
	})
	if err != nil {
		t.Fatalf(
			"large coding completion failed (context=%d, tools=%d): %v",
			cfg.Model.ContextSize,
			len(registry.Definitions()),
			err,
		)
	}
	if response == nil || response.ToolCall == nil {
		t.Fatalf("large coding turn returned no tool action: %#v", response)
	}
	if response.ToolCall.Name == "" {
		t.Fatalf("large coding turn returned an unnamed tool action: %#v", response.ToolCall)
	}
	t.Logf(
		"context=%d tools=%d first_action=%s",
		cfg.Model.ContextSize,
		len(registry.Definitions()),
		response.ToolCall.Name,
	)
}
