export type AgentResponse = {
  conversation_id: string;
  run_id: string;
  content?: string;
  reasoning?: string;
  tools_used?: string[];
  memory_recalled?: number;
  latency_ms?: number;
  mode?: string;
};

export type StoredMessage = {
  role: string;
  content: string;
  timestamp: string;
};

export type StoredConversation = {
  id: string;
  title: string;
  messages: StoredMessage[];
  last_activity?: string;
};

export type ConversationSummary = {
  id: string;
  title: string;
  message_count: number;
  last_activity?: string;
};

export type RuntimeStatus = {
  provider: string;
  model: string;
  cloud_provider: string;
  cloud_model: string;
  local_provider: string;
  local_model: string;
  cloud_configured: boolean;
  local_enabled: boolean;
  local_configured: boolean;
  network_allowed: boolean;
  active_mode: string;
  available_modes: string[];
  ready: boolean;
  setup_hint: string;
};

export const CancelAgentLoop = (conversationID: string, runID: string) =>
  cancelAgentLoop(conversationID, runID);

export const CreateConversation = async (conversationID: string) =>
  (await createConversation(conversationID)) as StoredConversation;

export const GetAvailableModes = () => getAvailableModes();

export const GetConversation = async (conversationID: string) =>
  (await getConversation(conversationID)) as StoredConversation | null;

export const GetConversations = async () =>
  (await getConversations()) as ConversationSummary[];

export const GetMode = () => getMode();

export const GetRuntimeStatus = async () =>
  (await getRuntimeStatus()) as RuntimeStatus;

export const RunAgentLoop = (
  conversationID: string,
  runID: string,
  message: string,
) => runAgentLoop(conversationID, runID, message) as Promise<AgentResponse>;

export const SetMode = (mode: string) => setMode(mode);

export const WipeMemory = () => wipeMemory();
import {
  CancelAgentLoop as cancelAgentLoop,
  CreateConversation as createConversation,
  GetAvailableModes as getAvailableModes,
  GetConversation as getConversation,
  GetConversations as getConversations,
  GetMode as getMode,
  GetRuntimeStatus as getRuntimeStatus,
  RunAgentLoop as runAgentLoop,
  SetMode as setMode,
  WipeMemory as wipeMemory,
} from "../../wailsjs/go/main/App";
