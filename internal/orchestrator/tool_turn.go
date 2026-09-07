package orchestrator

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"herald/internal/cognitive"
	"herald/pkg/models"
)

type toolTurnStatus uint8

const (
	toolTurnSucceeded toolTurnStatus = iota
	toolTurnBlocked
	toolTurnFailed
)

type toolTurnOutcome struct {
	Status        toolTurnStatus
	ToolName      string
	ContextResult string
	EventMessage  string
}

// encodeAssistantToolIntent is the single persisted representation of an
// assistant tool call. Provider adapters use this exact envelope to reconnect
// opaque native tool state (such as OpenAI call_id values) on the next turn.
func encodeAssistantToolIntent(response *cognitive.Response) (string, error) {
	if response == nil || response.ToolCall == nil {
		return "", fmt.Errorf("assistant tool intent is missing a tool call")
	}
	encoded, err := json.Marshal(models.LLMResponse{
		Reasoning: response.Reasoning,
		ToolCall:  response.ToolCall,
		Content:   response.Content,
	})
	if err != nil {
		return "", fmt.Errorf("encode assistant tool intent: %w", err)
	}
	return string(encoded), nil
}

func formatToolResult(result string) string {
	return fmt.Sprintf("[TOOL RESULT]\n%s\n[END TOOL RESULT]", result)
}

// executeToolTurn owns the safety-critical assistant→guard→tool→result
// protocol shared by both public orchestration entry points.
func (o *Orchestrator) executeToolTurn(
	ctx context.Context,
	conversationID string,
	conversation *models.Conversation,
	response *cognitive.Response,
) (toolTurnOutcome, error) {
	assistantIntent, err := encodeAssistantToolIntent(response)
	if err != nil {
		return toolTurnOutcome{}, err
	}
	call := response.ToolCall
	outcome := toolTurnOutcome{ToolName: call.Name}
	conversation.AddInternalMessage(models.RoleAssistant, assistantIntent)

	if err := o.guardrail.Check(call); err != nil {
		o.log.Warn("Guardrail blocked %s: %v", call.Name, err)
		blocked := fmt.Sprintf("[BLOCKED] %s: %s", call.Name, err)
		conversation.AddInternalMessage(models.RoleTool, blocked)
		outcome.Status = toolTurnBlocked
		outcome.ContextResult = blocked
		outcome.EventMessage = blocked
	} else {
		result, executeErr := o.tools.Execute(ctx, call)
		if executeErr != nil {
			failed := fmt.Sprintf("[ERROR] %s: %s", call.Name, executeErr)
			conversation.AddInternalMessage(models.RoleTool, failed)
			outcome.Status = toolTurnFailed
			outcome.ContextResult = failed
			outcome.EventMessage = failed
		} else {
			formatted := formatToolResult(result)
			conversation.AddInternalMessage(models.RoleTool, formatted)
			outcome.Status = toolTurnSucceeded
			outcome.ContextResult = formatted
			outcome.EventMessage = result
		}
	}

	if err := o.persistRunSnapshot(conversationID, conversation); err != nil {
		return toolTurnOutcome{}, err
	}
	return outcome, nil
}

func (o *Orchestrator) reflectFinalResponse(ctx context.Context, userMessage, content string) string {
	if !o.reflectionEnabled || len(content) <= 200 {
		return content
	}
	reflection, err := o.cognitive.Generate(ctx, cognitive.Request{
		Messages: []models.Message{
			{Role: models.RoleUser, Content: userMessage},
			{Role: models.RoleAssistant, Content: content},
			{
				Role: models.RoleUser,
				Content: `Review your response above. Is it accurate, complete, and helpful?
If yes, respond with only: LGTM
If no, provide an improved response.`,
			},
		},
		Tools: nil,
	})
	if err == nil && reflection != nil &&
		strings.TrimSpace(reflection.Content) != "" &&
		!containsApprovalToken(reflection.Content) {
		o.log.Info("Reflection improved response")
		return reflection.Content
	}
	return content
}

func containsApprovalToken(content string) bool {
	return strings.TrimSpace(content) == "LGTM"
}
