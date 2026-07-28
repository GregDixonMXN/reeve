// The Wails-facing API is defined in backend.ts so the frontend remains
// type-safe even before generated bindings are refreshed by a desktop build.
export type {
  AgentResponse,
  ConversationSummary,
  RuntimeStatus,
  StoredConversation,
  StoredMessage,
} from "./backend";
