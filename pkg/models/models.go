package models

import (
	"time"
)

// ─── Tool Definitions ───────────────────────────────────────────────────────

type ToolDefinition struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	// Parameters is the canonical recursive JSON Schema for this tool's
	// arguments. ArgsSchema remains as a compatibility input for older callers
	// and dynamic tool definitions; CanonicalSchema always prefers Parameters.
	Parameters *JSONSchema `json:"parameters,omitempty"`
	ArgsSchema string      `json:"args_schema"` // Deprecated: use Parameters.
}

type ToolCall struct {
	Name string                 `json:"name"`
	Args map[string]interface{} `json:"args"`
}

// ─── LLM Communication ─────────────────────────────────────────────────────

type LLMResponse struct {
	Reasoning string    `json:"reasoning"`
	ToolCall  *ToolCall `json:"tool_call"`
	Content   string    `json:"content"`
	ToolUseID string    `json:"tool_use_id,omitempty"` // Claude native tool use ID
}

// ─── Agent Response (returned to frontend) ──────────────────────────────────

type AgentResponse struct {
	Content        string   `json:"content"`
	Reasoning      string   `json:"reasoning,omitempty"`
	ToolsUsed      []string `json:"tools_used,omitempty"`
	MemoryRecalled int      `json:"memory_recalled"`
	LatencyMs      int64    `json:"latency_ms"`
	Mode           string   `json:"mode,omitempty"`
}

// ─── Roles ──────────────────────────────────────────────────────────────────

const (
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleTool      = "tool"
	RoleSystem    = "system"
)

// ─── Conversation ───────────────────────────────────────────────────────────

type Message struct {
	Role      string    `json:"role"`
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
	Internal  bool      `json:"internal,omitempty"`
}

type Conversation struct {
	ID           string    `json:"id"`
	Title        string    `json:"title"`
	Messages     []Message `json:"messages"`
	LastActivity time.Time `json:"last_activity"`
}

func NewConversation(id string) *Conversation {
	return &Conversation{
		ID:           id,
		Title:        "New Conversation",
		Messages:     make([]Message, 0),
		LastActivity: time.Now(),
	}
}

// Clone returns a detached copy that callers can safely serialize or inspect
// after the orchestrator releases its conversation lock.
func (c *Conversation) Clone() *Conversation {
	if c == nil {
		return nil
	}
	clone := *c
	clone.Messages = append([]Message(nil), c.Messages...)
	return &clone
}

func (c *Conversation) AddMessage(role, content string) {
	c.addMessage(role, content, false)
}

// AddInternalMessage records tool protocol needed for future model context but
// omitted from the user-visible transcript.
func (c *Conversation) AddInternalMessage(role, content string) {
	c.addMessage(role, content, true)
}

func (c *Conversation) addMessage(role, content string, internal bool) {
	c.Messages = append(c.Messages, Message{
		Role:      role,
		Content:   content,
		Timestamp: time.Now(),
		Internal:  internal,
	})
	c.LastActivity = time.Now()
}

func (c *Conversation) VisibleMessageCount() int {
	count := 0
	for _, message := range c.Messages {
		if !message.Internal {
			count++
		}
	}
	return count
}

type ConversationSummary struct {
	ID           string    `json:"id"`
	Title        string    `json:"title"`
	MessageCount int       `json:"message_count"`
	LastActivity time.Time `json:"last_activity"`
}
