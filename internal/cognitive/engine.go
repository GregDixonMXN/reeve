package cognitive

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

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
// receive the system context string and the full structured message history
// separately so the chat API gets proper multi-turn message structure
// instead of one monolithic user string.
type ToolAwareRunner interface {
	LLMRunner
	CompleteWithTools(ctx context.Context, systemContext string, messages []models.Message, maxTokens int, tools []models.ToolDefinition) (string, error)
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
	cfg           config.ModelConfig
	runner        LLMRunner
	mode          string   // "local", "hybrid", "cloud"
	workspaceDirs []string // allowed dirs for file operations (injected into system prompt)
}

func NewEngine(cfg config.ModelConfig) *Engine {
	return &Engine{cfg: cfg, mode: "hybrid"}
}

func (e *Engine) SetRunner(r LLMRunner) {
	e.runner = r
}

// SetMode updates the engine's operating mode, which changes the system prompt
// rules — specifically how the LLM is instructed to use tools vs. delegate to cloud.
func (e *Engine) SetMode(mode string) {
	e.mode = mode
}

// SetWorkspaceDirs tells the engine which directories are available for file ops.
// They're injected into the system prompt so the LLM knows exact paths without guessing.
func (e *Engine) SetWorkspaceDirs(dirs []string) {
	e.workspaceDirs = dirs
}

// Generate executes the Think step of the Think-Verify-Act loop.
func (e *Engine) Generate(ctx context.Context, req Request) (*Response, error) {
	if e.runner == nil {
		return nil, fmt.Errorf("cognitive engine: no LLM runner configured (call SetRunner first)")
	}

	prompt := e.buildPrompt(req)

	// If the runner supports native tool calling, pass structured messages so
	// Ollama's chat API gets proper multi-turn history instead of one giant string.
	var raw string
	var err error
	if tar, ok := e.runner.(ToolAwareRunner); ok && len(req.Tools) > 0 {
		raw, err = tar.CompleteWithTools(ctx, e.buildSystemSection(req), req.Messages, e.cfg.ContextSize, req.Tools)
	} else {
		raw, err = e.runner.Complete(ctx, prompt, e.cfg.ContextSize)
	}
	if err != nil {
		return nil, fmt.Errorf("LLM completion failed: %w", err)
	}

	// Parse the raw output into a structured response (handling XML tags and markdown quirks).
	resp, err := e.parseResponse(raw)
	if err != nil {
		return nil, err
	}

	// JSON retry: if we fell back to plain text (no tool call and the JSON parse
	// failed), send one corrective message asking for valid JSON.
	if resp.ToolCall == nil {
		thinkRe := regexp.MustCompile(`(?s)<think>(.*?)</think>`)
		textForJSON := thinkRe.ReplaceAllString(raw, "")
		cleaned := cleanJSON(textForJSON)
		var probe structuredOutput
		if json.Unmarshal([]byte(cleaned), &probe) != nil {
			// JSON parse failed — retry once with an explicit correction prompt.
			retryPrompt := prompt +
				fmt.Sprintf("<|assistant|>\n%s\n<|end|>\n", raw) +
				"<|user|>\nYour last response was not valid JSON. You MUST respond with ONLY " +
				"a single valid JSON object and nothing else — no markdown, no code fences, " +
				"no text outside the braces. Required format:\n" +
				`{"reasoning":"...","tool_call":null,"content":"..."}` + "\n<|end|>\n<|assistant|>\n"
			raw2, err2 := e.runner.Complete(ctx, retryPrompt, e.cfg.ContextSize)
			if err2 == nil {
				if resp2, err3 := e.parseResponse(raw2); err3 == nil &&
					(resp2.ToolCall != nil || resp2.Content != "") {
					return resp2, nil
				}
			}
		}
	}

	return resp, nil
}

// loadUserContextFile reads AXIOM.md from the project root (if it exists) and
// returns its contents for injection into the system prompt. This lets the user
// define persistent context: their name, preferred stack, project conventions,
// working directory preferences — anything Axiom should always know.
func loadUserContextFile(projectRoot string) string {
	if projectRoot == "" {
		return ""
	}
	candidates := []string{
		filepath.Join(projectRoot, "AXIOM.md"),
		filepath.Join(projectRoot, "axiom.md"),
	}
	for _, path := range candidates {
		data, err := os.ReadFile(path)
		if err == nil && len(data) > 0 {
			content := strings.TrimSpace(string(data))
			// Cap at 2000 chars to avoid bloating the system prompt
			if len(content) > 2000 {
				content = content[:2000] + "\n... [truncated]"
			}
			return content
		}
	}
	return ""
}

// modeRule2 returns the tool-use strategy instruction for rule #2, keyed by mode.
// Local mode gets full autonomous tool use; hybrid/cloud keep the delegation-first behaviour.
func modeRule2(mode string) string {
	if mode == "local" {
		return `2. AUTONOMOUS LOCAL OPERATION: You are running in LOCAL mode — there is NO cloud delegation available. Do NOT call "ask_cloud_model"; it does not exist in this mode. Instead, use ALL of your available tools autonomously to complete every task, no matter how complex:
   - Complex code generation → use write_file + execute_code iteratively
   - Research / facts → use web_search + web_scrape + wolfram
   - File work → use read_file, write_file, edit_file, list_dir
   - Git operations → use git_ops
   - Memory → use search_memory
   Handle ALL tasks — architecture, refactoring, multi-step coding, analysis, creative writing — using your tools and your own reasoning. Never refuse a task because it feels complex; break it into steps and execute them.`
	}
	if mode == "cloud" {
		return `2. CLOUD MODE — YOU ARE THE BRAIN: You are Claude running as the primary orchestrator. Do NOT call "ask_cloud_model" (that tool does not exist in cloud mode). You have direct access to all tools — use them yourself:
   - File work → read_file, write_file, edit_file, list_dir
   - Code execution → execute_code
   - Web → web_search, web_scrape
   - Facts → wolfram
   - Git → git_ops
   - Memory → search_memory
   Build projects by calling write_file for each file directly. Never output file contents as text — always write them to disk. Execute, verify, fix if needed.`
	}
	// hybrid
	return `2. CLOUD DELEGATION: DEFAULT TO CLAUDE for ANY task requiring deep reasoning, multi-step planning, code architecture, refactoring, analysis, creative writing, or nuanced judgment. Only handle simple factual lookups and direct file operations locally. When in doubt, delegate to claude.`
}

// buildSystemPrompt returns the full system prompt for the given operating mode.
// In cloud mode the JSON-output constraint is relaxed because Claude uses native
// tool-use blocks rather than embedding tool calls in a JSON content field.
func buildSystemPrompt(mode string) string {
	if mode == "cloud" {
		return `You are Axiom — a precise, capable autonomous agent. Sharp, resourceful, and purposeful.

` + modeRule2(mode) + `

RULES:
3. WORKSPACE: All file operations MUST use paths inside the allowed workspace directories listed in the prompt. NEVER invent a path — use only exact paths provided.
4. FILE MANIFEST FIRST: When building a multi-file project, your FIRST action must be to list every file you will create with its exact absolute path. Then write them ALL before running anything. Do not call execute_code until every file in your manifest exists on disk.
5. WRITE, THEN RUN: Never call execute_code before calling write_file at least once in this session. Build first, verify second.
6. SURGICAL FIXES: When execute_code returns an error, read the full stderr/traceback. Fix ONLY the specific line or import causing it — do not rewrite the entire file. One targeted write_file call, then re-run.
7. VERIFIED KNOWLEDGE: Use wolfram for math, science, history, or unit conversions. Never estimate.
8. LIVE INTELLIGENCE: Use web_search and web_scrape for anything after 2024 or technical docs.
9. TASK COMPLETION: When fully done with no more tool calls, end your response with <TASK_COMPLETE>.
10. PERSONA: Be direct, precise, and resourceful. Skip filler. Have opinions. Come back with answers, not questions.`
	}

	return `You are Axiom — a precise, capable autonomous operator. Sharp, resourceful, and purposeful. Not a search engine with a chat interface, but an agent with judgment and opinions. You come back with answers, not questions.

CRITICAL RULES:
1. Respond ONLY with a single, valid JSON object.
` + modeRule2(mode) + `
3. VERIFIED KNOWLEDGE: Use "wolfram" for ALL factual data involving math, science, history, geography, or units. Never estimate dates or numbers.
4. LIVE INTELLIGENCE: Use "web_search" and "web_scrape" for any information after 2024, technical documentation, or breaking news.
5. LOCAL EXECUTION: Use "write_file" and "execute_code" iteratively for local development. After every execute_code call, inspect the output and stderr:
   - If stderr contains an error → read the FULL error, identify the SPECIFIC file and line, fix ONLY that with write_file, then execute_code again
   - Do NOT rewrite the entire file for a targeted error — surgical edits only
   - Repeat the write → execute → read-error → fix cycle until the code runs cleanly
   - NEVER report success without having executed the code to verify it works
   - NEVER call execute_code before write_file — write the files first
6. TOOL INTEGRITY: If an action is required, the "tool_call" field MUST contain the payload. NEVER summarize an action in "content" without executing it first.
   - ❌ FORBIDDEN: Outputting file contents as markdown/code blocks in "content". This is a HARD FAILURE. No exceptions.
   - ✅ REQUIRED: Every file you create MUST be written via a write_file tool call. One file = one write_file call.
   - When building a project with multiple files: call write_file for EACH file individually, one tool call per iteration. Never batch them in content.
   - When using ask_cloud_model for code generation: "output_path" is MANDATORY — provide the EXACT file path on disk (e.g. /home/shki/projects/myapp/main.py). Omitting output_path will cause an error. Never use a directory as output_path — always a file path with an extension.
   - Do NOT call execute_code to "run" code returned by ask_cloud_model unless the task explicitly requires execution. The goal is to WRITE the file, not execute it.
   - SELF-CHECK before every response: Am I about to put code in "content"? If yes, STOP and put it in a write_file tool call instead.
7. TASK COMPLETION: When you have fully completed the user's request and have no more tool calls to make, you MUST include the exact token <TASK_COMPLETE> at the end of your "content" field. This signals the agent loop to stop. Do NOT output <TASK_COMPLETE> if you still have pending tool calls.
8. PERSONA: You are Axiom. Be direct, precise, and resourceful. Skip filler phrases like "Great question!" or "I'd be happy to help". Have opinions. If something is wrong, say so. Come back with answers, not questions. Earn trust through competence.

RESPONSE FORMAT:
{
  "reasoning": "Internal logic for tool selection and approach",
  "tool_call": {"name": "tool_name", "args": {...}} | null,
  "content": "Message to user (only if tool_call is null or reporting a result). Append <TASK_COMPLETE> when fully done."
}`
}

// buildSystemSection returns the system prompt + repo tree + tools + memory block,
// without conversation history. Used by ToolAwareRunner so the chat API can
// receive a proper system message alongside structured multi-turn history.
func (e *Engine) buildSystemSection(req Request) string {
	section := buildSystemPrompt(e.mode) + "\n"

	// Current date/time — injected so the LLM never has to guess or estimate
	section += fmt.Sprintf("\nCURRENT DATE/TIME: %s\n", time.Now().Format("Monday, January 2 2006 — 15:04 MST"))

	// User context file — AXIOM.md in the project root (if it exists)
	if userCtx := loadUserContextFile(e.cfg.ProjectRoot); userCtx != "" {
		section += "\nUSER CONTEXT:\n" + userCtx + "\n"
	}

	// Workspace listing: shallow top-level view of each allowed dir so the LLM
	// knows exact paths without us having to dump the full tree into context.
	if len(e.workspaceDirs) > 0 {
		listing := BuildShallowListing(e.workspaceDirs)
		if listing != "" {
			section += "\nWORKSPACE DIRECTORIES (use these exact paths for file operations — do NOT invent paths):\n"
			section += listing + "\n"
		}
	}

	repoTree := BuildRepoTree(e.cfg.ProjectRoot)
	if repoTree != "" {
		section += "\nREPO STRUCTURE (use these exact paths — do NOT invent paths):\n"
		section += repoTree + "\n"
	}

	if len(req.Tools) > 0 {
		section += "\nAVAILABLE TOOLS:\n"
		for _, t := range req.Tools {
			section += fmt.Sprintf("- %s: %s\n  Args: %s\n", t.Name, t.Description, t.ArgsSchema)
		}
	}

	if len(req.MemoryContext) > 0 {
		section += "\nRELEVANT MEMORY:\n"
		for _, m := range req.MemoryContext {
			section += fmt.Sprintf("- [%.2f] %s\n", m.Score, m.Content)
		}
	}

	return section
}

func (e *Engine) buildPrompt(req Request) string {
	// 1. Start with the System Prompt (mode-aware)
	prompt := "<|system|>\n" + e.buildSystemSection(req)
	prompt += "<|end|>\n"

	// 2. Append Conversation History
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

	// 3. Cue the Assistant
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
