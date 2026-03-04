package models

import (
	"time"
)

// ─── Tool Definitions ───────────────────────────────────────────────────────

type ToolDefinition struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	ArgsSchema  string `json:"args_schema"` // JSON string describing arguments
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

func (c *Conversation) AddMessage(role, content string) {
	c.Messages = append(c.Messages, Message{
		Role:      role,
		Content:   content,
		Timestamp: time.Now(),
	})
	c.LastActivity = time.Now()
}

type ConversationSummary struct {
	ID           string    `json:"id"`
	Title        string    `json:"title"`
	MessageCount int       `json:"message_count"`
	LastActivity time.Time `json:"last_activity"`
}
