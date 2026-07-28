import { Component, createSignal, createEffect, For, Show } from "solid-js";
import {
  CreateConversation,
  GetConversations,
  type ConversationSummary,
} from "../lib/backend";

interface Props {
  activeId: string;
  onSelect: (id: string) => void;
  onToggle: () => void;
  onBusyChange: (busy: boolean) => void;
  onHistoryChanged: () => void;
  isOpen: boolean;
  disabled: boolean;
  refreshKey: number;
}

const Sidebar: Component<Props> = (props) => {
  const [convos, setConvos] = createSignal<ConversationSummary[]>([]);
  const [creating, setCreating] = createSignal(false);
  const [historyError, setHistoryError] = createSignal("");
  let fetchSequence = 0;

  const fetchConvos = async () => {
    const sequence = ++fetchSequence;
    try {
      const list = await GetConversations();
      if (sequence === fetchSequence) {
        setConvos(list || []);
        setHistoryError("");
      }
    } catch (err) {
      console.error("Failed to load conversations:", err);
      if (sequence === fetchSequence) {
        setHistoryError(`History unavailable: ${err}`);
      }
    }
  };

  // Load on mount and whenever refreshKey bumps
  createEffect(() => {
    // reactive dependency on refreshKey
    props.refreshKey;
    fetchConvos();
  });

  const newChat = async () => {
    if (props.disabled || creating()) return;

    const id = crypto.randomUUID();
    setCreating(true);
    props.onBusyChange(true);
    let released = false;
    try {
      await CreateConversation(id);
      setCreating(false);
      props.onBusyChange(false);
      released = true;
      props.onSelect(id);
      props.onHistoryChanged();
      await fetchConvos();
    } catch (err) {
      alert(`Failed to create conversation: ${err}`);
    } finally {
      if (!released) {
        setCreating(false);
        props.onBusyChange(false);
      }
    }
  };

  return (
    <Show
      when={props.isOpen}
      fallback={
        <button
          class="absolute top-4 left-4 z-20 w-8 h-8 flex items-center justify-center text-gray-400 hover:text-white bg-[#13131a] border border-[#2a2a35] rounded transition-colors"
          onClick={props.onToggle}
          title="Open sidebar"
        >
          &#9776;
        </button>
      }
    >
      <aside class="w-64 flex-shrink-0 bg-[#13131a] border-r border-[#2a2a35] flex flex-col">
        {/* Header */}
        <div class="flex items-center justify-between px-4 py-3 border-b border-[#2a2a35]">
          <span class="text-xs font-bold tracking-widest uppercase text-indigo-400">
            ⚡ Axiom
          </span>
          <button
            class="text-gray-500 hover:text-white text-lg leading-none transition-colors"
            onClick={props.onToggle}
            title="Close sidebar"
          >
            &times;
          </button>
        </div>

        {/* New Chat Button */}
        <div class="p-3">
          <button
            class="w-full py-2 px-3 text-xs font-bold tracking-widest uppercase border border-dashed border-[#3a3a4e] text-gray-400 rounded hover:border-indigo-500 hover:text-indigo-400 transition-colors"
            onClick={newChat}
            disabled={props.disabled || creating()}
            classList={{
              "opacity-40 cursor-not-allowed": props.disabled || creating(),
            }}
          >
            {creating() ? "Creating..." : "+ New Chat"}
          </button>
        </div>

        {/* Conversation List */}
        <div class="flex-1 overflow-y-auto px-2 pb-3 space-y-0.5">
          <Show when={historyError()}>
            <div class="mx-1 mb-2 rounded border border-red-500/20 bg-red-950/20 p-2 text-[10px] text-red-400">
              <p class="break-words">{historyError()}</p>
              <button
                class="mt-2 uppercase tracking-widest hover:text-red-200"
                onClick={() => void fetchConvos()}
              >
                Retry
              </button>
            </div>
          </Show>
          <For
            each={convos()}
            fallback={
              <p class="text-[10px] text-gray-600 text-center mt-6 uppercase tracking-widest">
                No history yet
              </p>
            }
          >
            {(c) => (
              <button
                class={`w-full flex items-center justify-between px-3 py-2 rounded text-left text-xs transition-colors group ${
                  c.id === props.activeId
                    ? "bg-[#1e1e2a] border-l-2 border-indigo-500 text-white"
                    : "text-gray-400 hover:bg-[#1a1a24] hover:text-white"
                }`}
                onClick={() => props.onSelect(c.id)}
                disabled={props.disabled || creating()}
                classList={{
                  "cursor-not-allowed opacity-50": props.disabled || creating(),
                }}
                title={
                  creating()
                    ? "Creating a new conversation"
                    : props.disabled
                      ? "Finish the current operation before switching chats"
                      : c.title || "Untitled"
                }
              >
                <span class="truncate flex-1 mr-2">{c.title || "Untitled"}</span>
                <span class="text-[10px] text-gray-600 flex-shrink-0">
                  {c.message_count > 0 ? c.message_count : ""}
                </span>
              </button>
            )}
          </For>
        </div>
      </aside>
    </Show>
  );
};

export default Sidebar;
