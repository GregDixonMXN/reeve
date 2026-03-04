package cognitive

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"axiom/internal/config"
	"axiom/internal/memory"
	"axiom/pkg/models"
)

// LLMRunner is the interface any inference backend must implement.
type LLMRunner interface {
	Complete(ctx context.Context, prompt string, maxTokens int) (string, error)
	Unload() error
}

// ToolAwareRunner extends LLMRunner with native tool calling support.
// Runners that implement this interface (e.g., Ollama /api/chat) will
// receive the full tool array for structured tool calling.
type ToolAwareRunner interface {
	LLMRunner
	CompleteWithTools(ctx context.Context, prompt string, maxTokens int, tools []models.ToolDefinition) (string, error)
}

// Request is a generation request to the cognitive engine.
type Request struct {
	Messages      []models.Message
	MemoryContext []memory.MemoryEntry
	Tools         []models.ToolDefinition
}

// Response is the parsed output.
type Response struct {
	Content   string
	ToolCall  *models.ToolCall
	ToolsUsed []string
	Reasoning string
}

// Engine wraps the local LLM (DeepSeek/Phi-4) with prompt construction and output parsing.
type Engine struct {
	cfg    config.ModelConfig
	runner LLMRunner
}

func NewEngine(cfg config.ModelConfig) *Engine {
	return &Engine{cfg: cfg}
}

func (e *Engine) SetRunner(r LLMRunner) {
	e.runner = r
}

// Generate executes the Think step of the Think-Verify-Act loop.
func (e *Engine) Generate(ctx context.Context, req Request) (*Response, error) {
	if e.runner == nil {
		return nil, fmt.Errorf("cognitive engine: no LLM runner configured (call SetRunner first)")
	}

	prompt := e.buildPrompt(req)

	// If the runner supports native tool calling, use it for better structured output
	var raw string
	var err error
	if tar, ok := e.runner.(ToolAwareRunner); ok && len(req.Tools) > 0 {
		raw, err = tar.CompleteWithTools(ctx, prompt, e.cfg.ContextSize, req.Tools)
	} else {
		raw, err = e.runner.Complete(ctx, prompt, e.cfg.ContextSize)
	}
	if err != nil {
		return nil, fmt.Errorf("LLM completion failed: %w", err)
	}

	// Parse the raw output into a structured response (handling XML tags and markdown quirks)
	return e.parseResponse(raw)
}

const systemPrompt = `You are Axiom — a precise, capable autonomous operator. Sharp, resourceful, and purposeful. Not a search engine with a chat interface, but an agent with judgment and opinions. You come back with answers, not questions.

CRITICAL RULES:
1. Respond ONLY with a single, valid JSON object.
2. CLOUD DELEGATION: DEFAULT TO CLAUDE for ANY task requiring deep reasoning, multi-step planning, code architecture, refactoring, analysis, creative writing, or nuanced judgment. Only handle simple factual lookups and direct file operations locally. When in doubt, delegate to claude.
3. VERIFIED KNOWLEDGE: Use "wolfram" for ALL factual data involving math, science, history, geography, or units. Never estimate dates or numbers.
4. LIVE INTELLIGENCE: Use "web_search" and "web_scrape" for any information after 2024, technical documentation, or breaking news.
5. LOCAL EXECUTION: Use "write_file" and "execute_code" for local development. NEVER skip steps (e.g., write the file before you run it).
6. TOOL INTEGRITY: If an action is required, the "tool_call" field MUST contain the payload. NEVER summarize an action in "content" without executing it first.
   - NEVER output file contents as text in "content" when write_file should be called. Writing code to "content" instead of disk is a failure.
   - When building a project with multiple files: call write_file for EACH file individually, one tool call per iteration.
   - When using ask_cloud_model for code generation: "output_path" is MANDATORY — provide the EXACT file path on disk (e.g. /home/shki/projects/myapp/main.py). Omitting output_path will cause an error. Never use a directory as output_path — always a file path with an extension.
   - Do NOT call execute_code to "run" code returned by ask_cloud_model unless the task explicitly requires execution. The goal is to WRITE the file, not execute it.
7. TASK COMPLETION: When you have fully completed the user's request and have no more tool calls to make, you MUST include the exact token <TASK_COMPLETE> at the end of your "content" field. This signals the agent loop to stop. Do NOT output <TASK_COMPLETE> if you still have pending tool calls.
8. PERSONA: You are Axiom. Be direct, precise, and resourceful. Skip filler phrases like "Great question!" or "I'd be happy to help". Have opinions. If something is wrong, say so. Come back with answers, not questions. Earn trust through competence.

RESPONSE FORMAT:
{
  "reasoning": "Internal logic for tool selection and approach",
  "tool_call": {"name": "tool_name", "args": {...}} | null,
  "content": "Message to user (only if tool_call is null or reporting a result). Append <TASK_COMPLETE> when fully done."
}`

func (e *Engine) buildPrompt(req Request) string {
	// 1. Start with the System Prompt
	prompt := "<|system|>\n" + systemPrompt + "\n"

	// 1b. Inject live repo map so the LLM knows exact file paths
	repoTree := BuildRepoTree(e.cfg.ProjectRoot)
	if repoTree != "" {
		prompt += "\nREPO STRUCTURE (use these exact paths — do NOT invent paths):\n"
		prompt += repoTree + "\n"
	}

	// 2. Inject Available Tools Schema
	if len(req.Tools) > 0 {
		prompt += "\nAVAILABLE TOOLS:\n"
		for _, t := range req.Tools {
			prompt += fmt.Sprintf("- %s: %s\n  Args: %s\n", t.Name, t.Description, t.ArgsSchema)
		}
	}

	// 3. Inject Semantic Memory Context
	if len(req.MemoryContext) > 0 {
		prompt += "\nRELEVANT MEMORY:\n"
		for _, m := range req.MemoryContext {
			prompt += fmt.Sprintf("- [%.2f] %s\n", m.Score, m.Content)
		}
	}
	prompt += "<|end|>\n"

	// 4. Append Conversation History
	for _, msg := range req.Messages {
		switch msg.Role {
		case models.RoleUser:
			prompt += fmt.Sprintf("<|user|>\n%s\n<|end|>\n", msg.Content)
		case models.RoleAssistant:
			prompt += fmt.Sprintf("<|assistant|>\n%s\n<|end|>\n", msg.Content)
		case models.RoleTool:
			prompt += fmt.Sprintf("<|tool|>\n%s\n<|end|>\n", msg.Content)
		}
	}

	// 5. Cue the Assistant
	prompt += "<|assistant|>\n"
	return prompt
}

type structuredOutput struct {
	Reasoning string          `json:"reasoning"`
	ToolCall  *toolCallOutput `json:"tool_call"`
	Content   string          `json:"content"`
}

type toolCallOutput struct {
	Name string                 `json:"name"`
	Args map[string]interface{} `json:"args"`
}

// cleanJSON is the sanitizer that fixes the markdown bug.
// It aggressively strips ```json code blocks and conversational filler.
func cleanJSON(raw string) string {
	// Strategy 1: Regex to extract content inside ```json ... ``` blocks
	re := regexp.MustCompile("(?s)```(?:json)?\\s*(.*?)\\s*```")
	match := re.FindStringSubmatch(raw)
	if len(match) > 1 {
		return match[1]
	}

	// Strategy 2: Fallback to finding the first '{' and last '}'
	start := strings.Index(raw, "{")
	end := strings.LastIndex(raw, "}")
	if start != -1 && end != -1 && start < end {
		return raw[start : end+1]
	}

	// Strategy 3: Return raw string if no JSON structure found
	return strings.TrimSpace(raw)
}

func (e *Engine) parseResponse(raw string) (*Response, error) {
	// 1. Extract DeepSeek's native <think> tags
	// We do this BEFORE JSON parsing because the think block contains characters
	// that confuse the JSON extractor (like curly braces).
	thinkRegex := regexp.MustCompile(`(?s)<think>(.*?)</think>`)
	matches := thinkRegex.FindStringSubmatch(raw)

	extractedReasoning := ""
	textForJSON := raw

	if len(matches) > 1 {
		// Capture the raw thought process for the UI
		extractedReasoning = strings.TrimSpace(matches[1])
		// Remove the entire <think> block from the string we pass to the JSON parser
		textForJSON = thinkRegex.ReplaceAllString(raw, "")
	}

	// 2. Sanitize the remaining text to find the JSON block
	cleaned := cleanJSON(textForJSON)

	var out structuredOutput
	// Attempt to unmarshal the clean JSON
	if err := json.Unmarshal([]byte(cleaned), &out); err != nil {
		// Fallback: If JSON fails completely, treat the entire response as plain text content.
		// However, we still attach the extracted reasoning so the UI isn't blank.
		return &Response{
			Content:   textForJSON, // Use the text stripped of <think> tags
			Reasoning: extractedReasoning,
		}, nil
	}

	// 3. Construct the final response
	resp := &Response{
		Content:   out.Content,
		Reasoning: out.Reasoning, // Prefer JSON reasoning if it exists
	}

	// If the JSON didn't have a "reasoning" field (common with DeepSeek),
	// inject the extracted <think> content.
	if resp.Reasoning == "" {
		resp.Reasoning = extractedReasoning
	}

	if out.ToolCall != nil {
		resp.ToolCall = &models.ToolCall{
			Name: out.ToolCall.Name,
			Args: out.ToolCall.Args,
		}
		resp.ToolsUsed = append(resp.ToolsUsed, out.ToolCall.Name)
	}

	return resp, nil
}
