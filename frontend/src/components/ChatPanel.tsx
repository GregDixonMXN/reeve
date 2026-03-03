import {
  Component,
  createSignal,
  createEffect,
  For,
  Show,
  onMount,
  onCleanup,
} from "solid-js";
import {
  RunAgentLoop,
  WipeMemory,
  SetMode,
  GetMode,
  GetAvailableModes,
} from "../../wailsjs/go/main/App";
import { EventsOn, EventsOff } from "../../wailsjs/runtime/runtime";

// ─── Types ───────────────────────────────────────────────────────────────────

type LoopEventKind =
  | "thinking"
  | "tool_call"
  | "tool_result"
  | "blocked"
  | "done"
  | "max_iter"
  | "error";

type LoopEvent = {
  kind: LoopEventKind;
  iteration: number;
  message: string;
  tool_name?: string;
};

type LoopStep = {
  label: string;
  kind: LoopEventKind;
};

type Message = {
  role: "user" | "axiom" | "tool";
  content: string;
  reasoning?: string;
  tools_used?: string[];
  latency_ms?: number;
  memory_recalled?: number;
  timestamp: Date;
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

function kindGlyph(kind: LoopEventKind): string {
  switch (kind) {
    case "thinking":    return "⟳";
    case "tool_call":   return "⚡";
    case "tool_result": return "✓";
    case "blocked":     return "🛡";
    case "done":        return "✔";
    case "max_iter":    return "⚠";
    case "error":       return "✗";
    default:            return "·";
  }
}

function buildStatusLine(e: LoopEvent): string {
  switch (e.kind) {
    case "thinking":
      return `${kindGlyph(e.kind)}  Thinking...  (${e.iteration} / 10)`;
    case "tool_call":
      return `${kindGlyph(e.kind)}  Calling ${e.tool_name ?? "tool"}...`;
    case "tool_result":
      return `${kindGlyph(e.kind)}  ${e.tool_name ?? "Tool"} → result received`;
    case "blocked":
      return `${kindGlyph(e.kind)}  Guardrail blocked: ${e.tool_name ?? "unknown"}`;
    case "done":
      return `${kindGlyph(e.kind)}  Task complete`;
    case "max_iter":
      return `${kindGlyph(e.kind)}  Safety ceiling hit (10 / 10) — returning control`;
    case "error":
      return `${kindGlyph(e.kind)}  Error: ${e.message}`;
    default:
      return e.message;
  }
}

// ─── Component ───────────────────────────────────────────────────────────────

const ChatPanel: Component<{
  conversationId: string;
  onMessageSent?: () => void;
}> = (props) => {
  const [input, setInput] = createSignal("");
  const [conversation, setConversation] = createSignal<Message[]>([]);
  const [isThinking, setIsThinking] = createSignal(false);
  const [loopStatus, setLoopStatus] = createSignal<string>("");
  const MAX_TRACE = 5;
  const [loopTrace, setLoopTrace] = createSignal<LoopStep[]>([]);
  const [mode, setMode] = createSignal<string>("hybrid");
  const [availableModes, setAvailableModes] = createSignal<string[]>([]);

  let chatContainer: HTMLDivElement | undefined;
  let inputRef: HTMLTextAreaElement | undefined;

  // ── Reset conversation when switching chats ────────────────────────────────
  createEffect(() => {
    // Reactive on conversationId — clears state whenever it changes
    const _id = props.conversationId;
    setConversation([]);
    setLoopTrace([]);
    setLoopStatus("");
  });

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  onMount(async () => {
    try {
      const modes = await GetAvailableModes();
      setAvailableModes(modes || []);
      const current = await GetMode();
      setMode(current || "hybrid");
    } catch (err) {
      console.error("Failed to load modes:", err);
    }

    EventsOn("axiom:loop_event", (event: LoopEvent) => {
      const label = buildStatusLine(event);
      setLoopStatus(label);
      setLoopTrace((prev) => {
        const next: LoopStep[] = [...prev, { label, kind: event.kind }];
        return next.slice(-MAX_TRACE);
      });
    });
  });

  onCleanup(() => {
    EventsOff("axiom:loop_event");
  });

  // Auto-scroll to bottom on new messages
  createEffect(() => {
    conversation();
    isThinking();
    if (chatContainer) {
      // Use setTimeout to let DOM update first
      setTimeout(() => {
        if (chatContainer) {
          chatContainer.scrollTop = chatContainer.scrollHeight;
        }
      }, 0);
    }
  });

  // ── Input: auto-grow textarea ──────────────────────────────────────────────

  const handleInputKeyDown = (e: KeyboardEvent) => {
    // Enter (without shift) submits
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ── Handlers ───────────────────────────────────────────────────────────────

  const handleSend = async () => {
    const val = input().trim();
    if (!val || isThinking()) return;

    const userMsg: Message = {
      role: "user",
      content: val,
      timestamp: new Date(),
    };
    setConversation((prev) => [...prev, userMsg]);
    setInput("");

    // Reset textarea height
    if (inputRef) inputRef.style.height = "auto";

    setIsThinking(true);
    setLoopTrace([]);
    setLoopStatus("⟳  Initialising agent loop...");

    try {
      const response = (await RunAgentLoop(props.conversationId, val)) as any;

      const traceText = loopTrace()
        .map((s) => s.label)
        .join("\n");
      const reasoning =
        response.reasoning ||
        (response.tools_used?.length
          ? `Tools used: ${response.tools_used.join(" → ")}`
          : "");

      const axiomMsg: Message = {
        role: "axiom",
        content: response.content || "Action completed.",
        reasoning: [reasoning, traceText].filter(Boolean).join("\n\n──────\n\n"),
        tools_used: response.tools_used,
        latency_ms: response.latency_ms,
        memory_recalled: response.memory_recalled,
        timestamp: new Date(),
      };

      setConversation((prev) => [...prev, axiomMsg]);
      props.onMessageSent?.();
    } catch (err) {
      setConversation((prev) => [
        ...prev,
        {
          role: "tool",
          content: `[SYSTEM ERROR] ${err}`,
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsThinking(false);
      setLoopStatus("");
      inputRef?.focus();
    }
  };

  const handleWipeMemory = async () => {
    if (
      confirm(
        "⚠️ NUCLEAR OPTION: This will permanently delete all long-term vectors and short-term context. Are you sure?",
      )
    ) {
      try {
        const result = await WipeMemory();
        alert(result);
        setConversation([]);
        setLoopTrace([]);
      } catch (err) {
        alert(`Failed to wipe memory: ${err}`);
      }
    }
  };

  const handleExportChat = () => {
    const messages = conversation();
    if (messages.length === 0) {
      alert("No messages to export");
      return;
    }

    let text = `AXIOM CHAT EXPORT\nSession: ${props.conversationId}\nExported: ${new Date().toISOString()}\n\n`;
    text += "=".repeat(80) + "\n\n";

    messages.forEach((msg) => {
      const ts = new Date(msg.timestamp).toLocaleString();
      text += `[${ts}] ${msg.role.toUpperCase()}:\n${msg.content}\n`;
      if (msg.reasoning) text += `\n  > Reasoning: ${msg.reasoning}\n`;
      if (msg.tools_used?.length)
        text += `  > Tools: ${msg.tools_used.join(", ")}\n`;
      text += "\n" + "-".repeat(80) + "\n\n";
    });

    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `axiom_chat_${props.conversationId}_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleModeChange = async (
    e: Event & { currentTarget: HTMLSelectElement },
  ) => {
    const newMode = e.currentTarget.value;
    try {
      await SetMode(newMode);
      setMode(newMode);
    } catch (err) {
      alert(`Failed to change mode: ${err}`);
      e.currentTarget.value = mode();
    }
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div class="flex flex-col h-full min-h-0 bg-[#0d0d12] font-mono text-sm">

      {/* ── Header ── */}
      <header class="flex-shrink-0 h-12 border-b border-[#2a2a35] flex justify-between items-center px-5 bg-[#13131a]">
        <div class="flex items-center gap-2">
          <div class="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          <span class="text-[10px] text-gray-400 font-bold tracking-widest uppercase">
            Axiom v1.0 // {mode()} online
          </span>
        </div>

        <div class="flex items-center gap-3">
          <select
            value={mode()}
            onChange={handleModeChange}
            class="bg-[#0d0d12] text-gray-400 text-[10px] uppercase font-bold tracking-widest border border-[#2a2a35] rounded px-2 py-1 focus:outline-none focus:border-green-500 cursor-pointer hover:text-white transition-colors"
          >
            <For each={availableModes()}>
              {(m) => <option value={m}>{m} MODE</option>}
            </For>
          </select>

          <button
            onClick={handleExportChat}
            class="text-[10px] text-blue-500/60 hover:text-blue-400 font-bold tracking-widest uppercase hover:bg-blue-500/10 px-2 py-1 rounded transition-colors"
          >
            [ EXPORT ]
          </button>

          <button
            onClick={handleWipeMemory}
            class="text-[10px] text-red-500/60 hover:text-red-400 font-bold tracking-widest uppercase hover:bg-red-500/10 px-2 py-1 rounded transition-colors"
          >
            [ PURGE ]
          </button>
        </div>
      </header>

      {/* ── Chat Messages ── */}
      <div
        ref={chatContainer}
        class="flex-1 min-h-0 overflow-y-auto px-6 py-5 space-y-6"
      >
        {/* Empty state */}
        <Show when={conversation().length === 0 && !isThinking()}>
          <div class="flex flex-col items-center justify-center h-full gap-3 text-gray-600 select-none">
            <div class="text-4xl opacity-40">⚡</div>
            <p class="text-xs uppercase tracking-widest">Axiom ready</p>
            <p class="text-[10px] text-gray-700">Enter a command below</p>
          </div>
        </Show>

        <For each={conversation()}>
          {(msg) => (
            <div
              class={`flex flex-col gap-1 ${
                msg.role === "user" ? "items-end" : "items-start"
              }`}
            >
              {/* Role label */}
              <span class={`text-[10px] font-bold uppercase tracking-widest ${
                msg.role === "user"
                  ? "text-gray-500"
                  : msg.role === "tool"
                  ? "text-red-500/60"
                  : "text-green-500/70"
              }`}>
                {msg.role === "axiom" ? "▸ Axiom" : msg.role === "tool" ? "⚠ System" : "You"}
              </span>

              <div
                class={`rounded-lg transition-all ${
                  msg.role === "user"
                    ? "max-w-[75%] bg-[#1e1e2a] text-white border border-[#3f3f4e] px-4 py-3"
                    : msg.role === "tool"
                    ? "w-full max-w-[90%] bg-red-950/20 border border-red-500/20 px-4 py-3 text-red-400"
                    : "w-full max-w-[90%]"
                }`}
              >
                {/* Collapsible reasoning block */}
                <Show when={msg.role === "axiom" && msg.reasoning}>
                  <details class="mb-3 group">
                    <summary class="text-[10px] text-green-500/50 cursor-pointer hover:text-green-400 uppercase tracking-widest list-none flex items-center gap-2 select-none">
                      <span class="inline-block group-open:rotate-90 transition-transform duration-150">▶</span>
                      <span>
                        {msg.tools_used?.length ? "Tool Sequence" : "Internal Reasoning"}
                      </span>
                      <Show when={msg.tools_used?.length}>
                        <span class="text-yellow-500/50 normal-case tracking-normal">
                          [{msg.tools_used!.join(" → ")}]
                        </span>
                      </Show>
                    </summary>
                    <div class="mt-2 p-3 bg-[#13131a] border border-[#2a2a35] rounded text-[11px] text-gray-500 leading-relaxed whitespace-pre-wrap break-words">
                      {msg.reasoning}
                    </div>
                  </details>
                </Show>

                {/* Message content */}
                <p
                  class={`whitespace-pre-wrap break-words leading-relaxed ${
                    msg.role === "axiom" ? "text-gray-200" : ""
                  }`}
                >
                  {msg.content}
                </p>

                {/* Footer: latency + memory */}
                <Show when={msg.role === "axiom" && (msg.latency_ms || msg.memory_recalled)}>
                  <div class="mt-3 flex items-center gap-4 text-[9px] text-gray-600">
                    <Show when={msg.latency_ms}>
                      <span>⏱ {(msg.latency_ms! / 1000).toFixed(2)}s</span>
                    </Show>
                    <Show when={msg.memory_recalled}>
                      <span>🧠 {msg.memory_recalled} recalled</span>
                    </Show>
                  </div>
                </Show>
              </div>
            </div>
          )}
        </For>

        {/* ── Live Agent Loop Status ── */}
        <Show when={isThinking()}>
          <div class="flex flex-col gap-2 pl-1">
            <div class="flex items-center gap-3">
              <span class="text-green-500 text-xs animate-pulse font-bold tracking-widest">
                &gt; AGENT_LOOP_ACTIVE
              </span>
              <div class="flex gap-1">
                <div class="w-1 h-1 bg-green-500 rounded-full animate-bounce [animation-delay:-0.3s]" />
                <div class="w-1 h-1 bg-green-500 rounded-full animate-bounce [animation-delay:-0.15s]" />
                <div class="w-1 h-1 bg-green-500 rounded-full animate-bounce" />
              </div>
            </div>

            <Show when={loopStatus()}>
              <div class="text-[11px] text-green-400 pl-3 border-l-2 border-green-500/50 py-0.5">
                {loopStatus()}
              </div>
            </Show>

            <Show when={loopTrace().length > 1}>
              <div class="pl-3 border-l border-[#2a2a35] space-y-0.5">
                <For each={loopTrace().slice(0, -1)}>
                  {(step, idx) => {
                    const age = loopTrace().length - 1 - idx();
                    const opacity =
                      age === 1 ? "opacity-40" :
                      age === 2 ? "opacity-25" :
                                  "opacity-10";
                    return (
                      <div class={`text-[10px] text-gray-500 ${opacity}`}>
                        {step.label}
                      </div>
                    );
                  }}
                </For>
              </div>
            </Show>
          </div>
        </Show>
      </div>

      {/* ── Input Area ── */}
      <div class="flex-shrink-0 px-5 py-4 bg-[#13131a] border-t border-[#2a2a35]">
        <div class="flex items-end gap-3">
          {/* Prompt glyph */}
          <span class="text-green-500 font-bold text-sm mb-3 flex-shrink-0 select-none">
            {isThinking() ? "×" : ">"}
          </span>

          {/* Auto-grow textarea */}
          <textarea
            ref={inputRef}
            rows={1}
            value={input()}
            onInput={(e) => {
              setInput(e.currentTarget.value);
              // Auto-grow
              e.currentTarget.style.height = "auto";
              e.currentTarget.style.height =
                Math.min(e.currentTarget.scrollHeight, 160) + "px";
            }}
            onKeyDown={handleInputKeyDown}
            disabled={isThinking()}
            placeholder={
              isThinking()
                ? "Orchestrator busy — loop running..."
                : "Enter command... (Enter to send, Shift+Enter for newline)"
            }
            class="flex-1 min-w-0 bg-[#0d0d12] text-white px-4 py-3 rounded border border-[#2a2a35] focus:outline-none focus:border-green-500 transition-all font-mono text-xs placeholder-gray-700 disabled:opacity-50 resize-none overflow-hidden leading-relaxed"
            autofocus
          />

          {/* Send button */}
          <button
            onClick={handleSend}
            disabled={isThinking() || !input().trim()}
            class="flex-shrink-0 mb-0.5 px-4 py-3 bg-green-600 hover:bg-green-500 disabled:opacity-30 disabled:cursor-not-allowed text-black font-bold text-xs rounded transition-colors uppercase tracking-widest"
          >
            Send
          </button>
        </div>
        <p class="text-[9px] text-gray-700 mt-1 pl-5 select-none">
          Session: {props.conversationId.slice(0, 8).toUpperCase()}
        </p>
      </div>
    </div>
  );
};

export default ChatPanel;
