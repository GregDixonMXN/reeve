package orchestrator

import (
	"context"

	"reeve/internal/cognitive"
	"reeve/internal/memory"
	"reeve/pkg/models"
)

type toolLoopPolicy struct {
	beforeIteration func(iteration int)
	prepareMessages func(iteration int, messages []models.Message, lastToolResult string)
	intercept       func(response *cognitive.Response) (*cognitive.Response, bool)
	onToolCall      func(iteration int, response *cognitive.Response, intercepted bool)
	onToolOutcome   func(iteration int, outcome toolTurnOutcome)
}

type toolLoopResult struct {
	FinalResponse  *cognitive.Response
	ToolsUsed      []string
	Iterations     int
	HitLimit       bool
	lastToolResult string
}

type toolLoopError struct {
	Iteration int
	Err       error
}

func (e *toolLoopError) Error() string {
	return e.Err.Error()
}

func (e *toolLoopError) Unwrap() error {
	return e.Err
}

// runToolLoop is the one canonical Think→Guard→Tool→Result engine. Public
// entry points provide policy hooks for planning, recovery, interception, and
// progress events without reimplementing the protocol.
func (o *Orchestrator) runToolLoop(
	ctx context.Context,
	conversationID string,
	conversation *models.Conversation,
	memories []memory.MemoryEntry,
	maxIterations int,
	policy toolLoopPolicy,
) (toolLoopResult, error) {
	result := toolLoopResult{}
	toolDefinitions := o.tools.Definitions()

	for iteration := 1; iteration <= maxIterations; iteration++ {
		result.Iterations = iteration
		if policy.beforeIteration != nil {
			policy.beforeIteration(iteration)
		}

		rawMessages := append([]models.Message(nil), conversation.Messages...)
		llmMessages := trimContextMessages(rawMessages)
		if policy.prepareMessages != nil {
			policy.prepareMessages(iteration, llmMessages, result.lastToolResult)
		}

		response, err := o.cognitive.Generate(ctx, cognitive.Request{
			ConversationID: conversationID,
			Messages:       llmMessages,
			MemoryContext:  memories,
			Tools:          toolDefinitions,
		})
		if err != nil {
			return result, &toolLoopError{Iteration: iteration, Err: err}
		}

		intercepted := false
		if response.ToolCall == nil && policy.intercept != nil {
			if replacement, ok := policy.intercept(response); ok {
				response = replacement
				intercepted = true
			}
		}
		if response.ToolCall == nil {
			result.FinalResponse = response
			return result, nil
		}

		result.ToolsUsed = append(result.ToolsUsed, response.ToolCall.Name)
		if policy.onToolCall != nil {
			policy.onToolCall(iteration, response, intercepted)
		}
		outcome, err := o.executeToolTurn(ctx, conversationID, conversation, response)
		if err != nil {
			return result, &toolLoopError{Iteration: iteration, Err: err}
		}
		result.lastToolResult = outcome.ContextResult
		if policy.onToolOutcome != nil {
			policy.onToolOutcome(iteration, outcome)
		}
	}

	result.HitLimit = true
	return result, nil
}
