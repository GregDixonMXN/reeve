import {
  Component,
  createEffect,
  createSignal,
  For,
  onCleanup,
  onMount,
  Show,
} from "solid-js";
import { EventsOn } from "../../wailsjs/runtime/runtime";
import {
  CancelAgentLoop,
  GetAvailableModes,
  GetConversation,
  GetMode,
  GetRuntimeStatus,
  RunAgentLoop,
  SetMode,
  WipeMemory,
  type RuntimeStatus,
  type StoredMessage,
} from "../lib/backend";

type LoopEventKind =
  | "thinking"
  | "tool_call"
  | "tool_result"
  | "blocked"
  | "done"
  | "max_iter"
  | "error";

type LoopEvent = {
  conversation_id: string;
  run_id: string;
  kind: LoopEventKind;
  iteration: number;
  max_iterations: number;
  message: string;
  tool_name?: string;
};

type TokenEvent = {
  conversation_id: string;
  run_id: string;
  token: string;
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

type ActiveRun = {
  conversationID: string;
  runID: string;
};

function kindGlyph(kind: LoopEventKind): string {
  switch (kind) {
    case "thinking":
      return "⟳";
    case "tool_call":
      return "⚡";
    case "tool_result":
      return "✓";
    case "blocked":
      return "🛡";
    case "done":
      return "✔";
    case "max_iter":
      return "⚠";
    case "error":
      return "✗";
  }
}

function buildStatusLine(event: LoopEvent): string {
  const progress =
    event.iteration > 0 && event.max_iterations > 0
      ? ` (${event.iteration} / ${event.max_iterations})`
      : "";

  switch (event.kind) {
    case "thinking": {
      if (event.iteration <= 0) {
        return `${kindGlyph(event.kind)}  ${event.message || "Planning task..."}`;
      }
      const thinkingProgress =
        event.max_iterations > 0
          ? ` (${event.iteration} / ${event.max_iterations})`
          : ` (iteration ${event.iteration})`;
      return `${kindGlyph(event.kind)}  Thinking...${thinkingProgress}`;
    }
    case "tool_call":
      return `${kindGlyph(event.kind)}  Calling ${event.tool_name ?? "tool"}...${progress}`;
    case "tool_result":
      return `${kindGlyph(event.kind)}  ${event.tool_name ?? "Tool"} → result received${progress}`;
    case "blocked":
      return `${kindGlyph(event.kind)}  Guardrail blocked: ${event.tool_name ?? "unknown"}${progress}`;
    case "done":
      return `${kindGlyph(event.kind)}  Task complete${progress}`;
    case "max_iter":
      return `${kindGlyph(event.kind)}  ${event.message}`;
    case "error":
      return `${kindGlyph(event.kind)}  Error: ${event.message}`;
  }
}

function restoredMessage(message: StoredMessage): Message {
  const role =
    message.role === "user"
      ? "user"
      : message.role === "assistant"
        ? "axiom"
        : "tool";
  const timestamp = new Date(message.timestamp);

  return {
    role,
    content: message.content,
    timestamp: Number.isNaN(timestamp.valueOf()) ? new Date() : timestamp,
  };
}

const ChatPanel: Component<{
  conversationId: string;
  externalLock?: boolean;
  onHistoryChanged?: () => void;
  onInteractionLockChange?: (locked: boolean) => void;
}> = (props) => {
  const MAX_TRACE = 5;
  const [input, setInput] = createSignal("");
  const [conversation, setConversation] = createSignal<Message[]>([]);
  const [loadingConversation, setLoadingConversation] = createSignal(true);
  const [loadError, setLoadError] = createSignal("");
  const [reloadKey, setReloadKey] = createSignal(0);
  const [isThinking, setIsThinking] = createSignal(false);
  const [isCancelling, setIsCancelling] = createSignal(false);
  const [isPurging, setIsPurging] = createSignal(false);
  const [isChangingMode, setIsChangingMode] = createSignal(false);
  const [activeRun, setActiveRun] = createSignal<ActiveRun | null>(null);
  const [streamedContent, setStreamedContent] = createSignal("");
  const [streamIteration, setStreamIteration] = createSignal<number | null>(null);
  const [loopStatus, setLoopStatus] = createSignal("");
  const [loopTrace, setLoopTrace] = createSignal<LoopStep[]>([]);
  const [mode, setMode] = createSignal("");
  const [availableModes, setAvailableModes] = createSignal<string[]>([]);
  const [runtimeStatus, setRuntimeStatus] = createSignal<RuntimeStatus | null>(
    null,
  );

  let chatContainer: HTMLDivElement | undefined;
  let inputRef: HTMLTextAreaElement | undefined;
  let loadSequence = 0;
  let cancelRequestedFor = "";
  let stopLoopEvents = () => {};
  let stopTokenEvents = () => {};

  const runtimeReady = () => runtimeStatus()?.ready === true;
  const providerLabel = () => {
    const provider = runtimeStatus()?.provider;
    if (provider === "openai") return "OpenAI";
    if (provider === "anthropic") return "Anthropic";
    if (provider === "ollama") return "Ollama";
    if (provider === "hybrid") return "Hybrid";
    return provider || "No provider";
  };

  const eventBelongsToActiveRun = (event: {
    conversation_id: string;
    run_id: string;
  }) => {
    const run = activeRun();
    return (
      run !== null &&
      event.conversation_id === run.conversationID &&
      event.run_id === run.runID
    );
  };

  createEffect(() => {
    const conversationID = props.conversationId;
    reloadKey();
    const sequence = ++loadSequence;

    setLoadingConversation(true);
    setLoadError("");
    setConversation([]);
    setLoopTrace([]);
    setLoopStatus("");
    setStreamedContent("");
    setStreamIteration(null);

    void (async () => {
      try {
        const stored = await GetConversation(conversationID);
        if (sequence !== loadSequence || props.conversationId !== conversationID) {
          return;
        }
        setConversation((stored?.messages ?? []).map(restoredMessage));
      } catch (err) {
        if (sequence !== loadSequence || props.conversationId !== conversationID) {
          return;
        }
        console.error("Failed to restore conversation:", err);
        setLoadError(`Could not load this conversation: ${err}`);
      } finally {
        if (sequence === loadSequence && props.conversationId === conversationID) {
          setLoadingConversation(false);
          queueMicrotask(() => inputRef?.focus());
        }
      }
    })();
  });

  onMount(() => {
    stopLoopEvents = EventsOn("axiom:loop_event", (event: LoopEvent) => {
      if (!eventBelongsToActiveRun(event)) return;

      if (event.kind === "thinking" && streamIteration() !== event.iteration) {
        setStreamIteration(event.iteration);
        setStreamedContent("");
      }

      // A model may emit a short natural-language preamble before a native
      // tool call. Once the backend confirms this is a tool turn, retract that
      // transient text so it is never presented as the final answer.
      if (event.kind === "tool_call") {
        setStreamedContent("");
      }

      const label = buildStatusLine(event);
      setLoopStatus(label);
      setLoopTrace((previous) =>
        [...previous, { label, kind: event.kind }].slice(-MAX_TRACE),
      );
    });

    stopTokenEvents = EventsOn("axiom:token", (event: TokenEvent) => {
      if (!eventBelongsToActiveRun(event) || !event.token) return;
      setStreamedContent((content) => content + event.token);
    });

    void (async () => {
      try {
        const [modes, currentMode, status] = await Promise.all([
          GetAvailableModes(),
          GetMode(),
          GetRuntimeStatus(),
        ]);
        setAvailableModes(modes || []);
        setMode(currentMode || "");
        setRuntimeStatus(status);
      } catch (err) {
        console.error("Failed to load modes:", err);
        setRuntimeStatus({
          provider: "",
          model: "",
          cloud_provider: "",
          cloud_model: "",
          local_provider: "",
          local_model: "",
          cloud_configured: false,
          local_enabled: false,
          local_configured: false,
          network_allowed: false,
          active_mode: "",
          available_modes: [],
          ready: false,
          setup_hint: `Could not load runtime status: ${err}`,
        });
      }
    })();
  });

  onCleanup(() => {
    stopLoopEvents();
    stopTokenEvents();
    props.onInteractionLockChange?.(false);
  });

  createEffect(() => {
    conversation();
    streamedContent();
    isThinking();
    if (!chatContainer) return;

    setTimeout(() => {
      if (chatContainer) chatContainer.scrollTop = chatContainer.scrollHeight;
    }, 0);
  });

  const handleInputKeyDown = (event: KeyboardEvent) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void handleSend();
    }
  };

  const handleSend = async () => {
    const value = input().trim();
    if (
      !value ||
      props.externalLock ||
      isThinking() ||
      isChangingMode() ||
      loadingConversation() ||
      !!loadError() ||
      isPurging() ||
      !runtimeReady()
    ) {
      return;
    }

    const run: ActiveRun = {
      conversationID: props.conversationId,
      runID: crypto.randomUUID(),
    };
    cancelRequestedFor = "";
    setActiveRun(run);
    setIsThinking(true);
    setIsCancelling(false);
    setStreamedContent("");
    setStreamIteration(null);
    setLoopTrace([]);
    setLoopStatus("⟳  Initialising agent loop...");
    props.onInteractionLockChange?.(true);

    setConversation((previous) => [
      ...previous,
      { role: "user", content: value, timestamp: new Date() },
    ]);
    setInput("");
    if (inputRef) inputRef.style.height = "auto";

    try {
      const response = await RunAgentLoop(
        run.conversationID,
        run.runID,
        value,
      );
      const current = activeRun();
      if (
        current?.runID !== run.runID ||
        current.conversationID !== run.conversationID ||
        props.conversationId !== run.conversationID ||
        response.conversation_id !== run.conversationID ||
        response.run_id !== run.runID
      ) {
        throw new Error("Backend returned a response for a different run");
      }

      const traceText = loopTrace()
        .map((step) => step.label)
        .join("\n");
      const reasoning =
        response.reasoning ||
        (response.tools_used?.length
          ? `Tools used: ${response.tools_used.join(" → ")}`
          : "");

      setConversation((previous) => [
        ...previous,
        {
          role: "axiom",
          content: response.content || "Action completed.",
          reasoning: [reasoning, traceText]
            .filter(Boolean)
            .join("\n\n──────\n\n"),
          tools_used: response.tools_used,
          latency_ms: response.latency_ms,
          memory_recalled: response.memory_recalled,
          timestamp: new Date(),
        },
      ]);
      props.onHistoryChanged?.();
    } catch (err) {
      const current = activeRun();
      if (
        current?.runID !== run.runID ||
        current.conversationID !== run.conversationID ||
        props.conversationId !== run.conversationID
      ) {
        return;
      }

      const cancelled = cancelRequestedFor === run.runID;
      setConversation((previous) => [
        ...previous,
        {
          role: "tool",
          content: cancelled ? "[RUN CANCELLED]" : `[SYSTEM ERROR] ${err}`,
          timestamp: new Date(),
        },
      ]);
      props.onHistoryChanged?.();
    } finally {
      const current = activeRun();
      if (current?.runID === run.runID) {
        setActiveRun(null);
        setIsThinking(false);
        setIsCancelling(false);
        setStreamedContent("");
        setStreamIteration(null);
        setLoopStatus("");
        props.onInteractionLockChange?.(false);
        queueMicrotask(() => inputRef?.focus());
      }
    }
  };

  const handleCancel = async () => {
    const run = activeRun();
    if (!run || isCancelling()) return;

    cancelRequestedFor = run.runID;
    setIsCancelling(true);
    setLoopStatus("⚠  Cancelling active run...");
    try {
      await CancelAgentLoop(run.conversationID, run.runID);
    } catch (err) {
      if (activeRun()?.runID === run.runID) {
        cancelRequestedFor = "";
        setIsCancelling(false);
        setLoopStatus(`✗  Cancellation failed: ${err}`);
      }
    }
  };

  const handleWipeMemory = async () => {
    if (
      props.externalLock ||
      isThinking() ||
      isChangingMode() ||
      loadingConversation() ||
      isPurging()
    ) {
      return;
    }
    if (
      !confirm(
        "⚠️ NUCLEAR OPTION: This permanently deletes all stored conversations and semantic memory. Are you sure?",
      )
    ) {
      return;
    }

    setIsPurging(true);
    props.onInteractionLockChange?.(true);
    try {
      const result = await WipeMemory();
      setConversation([]);
      setLoadError("");
      setLoopTrace([]);
      setLoopStatus("");
      props.onHistoryChanged?.();
      alert(result);
    } catch (err) {
      alert(`Failed to wipe memory: ${err}`);
    } finally {
      setIsPurging(false);
      props.onInteractionLockChange?.(false);
      queueMicrotask(() => inputRef?.focus());
    }
  };

  const handleExportChat = () => {
    const messages = conversation();
    if (messages.length === 0) {
      alert("No messages to export");
      return;
    }

    let text = `AXIOM CHAT EXPORT\nSession: ${props.conversationId}\nExported: ${new Date().toISOString()}\n\n`;
    text += `${"=".repeat(80)}\n\n`;
    for (const message of messages) {
      text += `[${message.timestamp.toLocaleString()}] ${message.role.toUpperCase()}:\n${message.content}\n`;
      if (message.reasoning) text += `\n  > Reasoning: ${message.reasoning}\n`;
      if (message.tools_used?.length) {
        text += `  > Tools: ${message.tools_used.join(", ")}\n`;
      }
      text += `\n${"-".repeat(80)}\n\n`;
    }

    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `axiom_chat_${props.conversationId}_${Date.now()}.txt`;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  };

  const handleModeChange = async (
    event: Event & { currentTarget: HTMLSelectElement },
  ) => {
    const select = event.currentTarget;
    if (
      props.externalLock ||
      isThinking() ||
      isChangingMode() ||
      isPurging()
    ) {
      select.value = mode();
      return;
    }

    const newMode = select.value;
    setIsChangingMode(true);
    props.onInteractionLockChange?.(true);
    try {
      await SetMode(newMode);
      const status = await GetRuntimeStatus();
      setRuntimeStatus(status);
      setMode(status.active_mode || newMode);
    } catch (err) {
      alert(`Failed to change mode: ${err}`);
      select.value = mode();
    } finally {
      setIsChangingMode(false);
      props.onInteractionLockChange?.(false);
      queueMicrotask(() => inputRef?.focus());
    }
  };

  return (
    <div class="flex flex-col h-full min-h-0 bg-[#0d0d12] font-mono text-sm">
      <header class="flex-shrink-0 h-12 border-b border-[#2a2a35] flex justify-between items-center px-5 bg-[#13131a]">
        <div class="flex items-center gap-2">
          <div
            class="w-2 h-2 rounded-full"
            classList={{
              "bg-yellow-500 animate-pulse": isThinking(),
              "bg-green-500": !isThinking() && runtimeReady(),
              "bg-red-500": !isThinking() && !runtimeReady(),
            }}
          />
          <span class="text-[10px] text-gray-400 font-bold tracking-widest uppercase">
            Axiom v1.0 // {providerLabel()}
            <Show when={runtimeStatus()?.model}>
              {" • "}
              {runtimeStatus()!.model}
            </Show>
            {" // "}
            {mode() || "setup"} {isThinking() ? "working" : runtimeReady() ? "ready" : "offline"}
          </span>
        </div>

        <div class="flex items-center gap-3">
          <span class="text-[9px] text-gray-600 uppercase tracking-widest">
            Run mode
          </span>
          <select
            value={mode()}
            onChange={handleModeChange}
            disabled={
              props.externalLock ||
              isThinking() ||
              isChangingMode() ||
              isPurging() ||
              availableModes().length === 0
            }
            class="bg-[#0d0d12] text-gray-400 text-[10px] uppercase font-bold tracking-widest border border-[#2a2a35] rounded px-2 py-1 focus:outline-none focus:border-green-500 cursor-pointer hover:text-white transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <For each={availableModes()}>
              {(availableMode) => (
                <option value={availableMode}>{availableMode} MODE</option>
              )}
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
            disabled={
              props.externalLock ||
              isThinking() ||
              isChangingMode() ||
              loadingConversation() ||
              isPurging()
            }
            class="text-[10px] text-red-500/60 hover:text-red-400 font-bold tracking-widest uppercase hover:bg-red-500/10 px-2 py-1 rounded transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
          >
            {isPurging() ? "[ PURGING... ]" : "[ PURGE ]"}
          </button>
        </div>
      </header>

      <div
        ref={chatContainer}
        class="flex-1 min-h-0 overflow-y-auto px-6 py-5 space-y-6"
      >
        <Show when={runtimeStatus() && !runtimeReady()}>
          <div class="w-full bg-red-950/20 border border-red-500/30 px-4 py-3 rounded text-red-300 text-xs">
            <div class="font-bold uppercase tracking-widest text-[10px] mb-1">
              {providerLabel()} setup required
            </div>
            <div>{runtimeStatus()!.setup_hint}</div>
          </div>
        </Show>

        <Show when={loadingConversation()}>
          <div class="flex items-center justify-center h-full text-[10px] text-gray-600 uppercase tracking-widest animate-pulse">
            Restoring conversation...
          </div>
        </Show>

        <Show when={!loadingConversation() && loadError()}>
          <div class="w-full bg-red-950/20 border border-red-500/20 px-4 py-3 rounded text-red-400 text-xs flex items-center justify-between gap-4">
            <span>{loadError()}</span>
            <button
              onClick={() => setReloadKey((key) => key + 1)}
              class="flex-shrink-0 px-3 py-1 border border-red-500/40 rounded text-[10px] uppercase tracking-widest hover:bg-red-500/10"
            >
              Retry
            </button>
          </div>
        </Show>

        <Show
          when={
            !loadingConversation() &&
            !loadError() &&
            conversation().length === 0 &&
            !isThinking() &&
            runtimeReady()
          }
        >
          <div class="flex flex-col items-center justify-center h-full gap-3 text-gray-600 select-none">
            <div class="text-4xl opacity-40">⚡</div>
            <p class="text-xs uppercase tracking-widest">Axiom ready</p>
            <p class="text-[10px] text-gray-700">Enter a command below</p>
          </div>
        </Show>

        <For each={conversation()}>
          {(message) => (
            <div
              class={`flex flex-col gap-1 ${
                message.role === "user" ? "items-end" : "items-start"
              }`}
            >
              <span
                class={`text-[10px] font-bold uppercase tracking-widest ${
                  message.role === "user"
                    ? "text-gray-500"
                    : message.role === "tool"
                      ? "text-yellow-500/60"
                      : "text-green-500/70"
                }`}
              >
                {message.role === "axiom"
                  ? "▸ Axiom"
                  : message.role === "tool"
                    ? "⚡ Tool"
                    : "You"}
              </span>

              <div
                class={`rounded-lg transition-all ${
                  message.role === "user"
                    ? "max-w-[75%] bg-[#1e1e2a] text-white border border-[#3f3f4e] px-4 py-3"
                    : message.role === "tool"
                      ? "w-full max-w-[90%] bg-yellow-950/10 border border-yellow-500/20 px-4 py-3 text-yellow-200/70"
                      : "w-full max-w-[90%]"
                }`}
              >
                <Show when={message.role === "axiom" && message.reasoning}>
                  <details class="mb-3 group">
                    <summary class="text-[10px] text-green-500/50 cursor-pointer hover:text-green-400 uppercase tracking-widest list-none flex items-center gap-2 select-none">
                      <span class="inline-block group-open:rotate-90 transition-transform duration-150">
                        ▶
                      </span>
                      <span>
                        {message.tools_used?.length
                          ? "Tool Sequence"
                          : "Internal Reasoning"}
                      </span>
                      <Show when={message.tools_used?.length}>
                        <span class="text-yellow-500/50 normal-case tracking-normal">
                          [{message.tools_used!.join(" → ")}]
                        </span>
                      </Show>
                    </summary>
                    <div class="mt-2 p-3 bg-[#13131a] border border-[#2a2a35] rounded text-[11px] text-gray-500 leading-relaxed whitespace-pre-wrap break-words">
                      {message.reasoning}
                    </div>
                  </details>
                </Show>

                <p
                  class={`whitespace-pre-wrap break-words leading-relaxed ${
                    message.role === "axiom" ? "text-gray-200" : ""
                  }`}
                >
                  {message.content}
                </p>

                <Show
                  when={
                    message.role === "axiom" &&
                    (message.latency_ms || message.memory_recalled)
                  }
                >
                  <div class="mt-3 flex items-center gap-4 text-[9px] text-gray-600">
                    <Show when={message.latency_ms}>
                      <span>
                        ⏱ {(message.latency_ms! / 1000).toFixed(2)}s
                      </span>
                    </Show>
                    <Show when={message.memory_recalled}>
                      <span>🧠 {message.memory_recalled} recalled</span>
                    </Show>
                  </div>
                </Show>
              </div>
            </div>
          )}
        </For>

        <Show
          when={
            isThinking() && streamIteration() !== 0 && streamedContent()
          }
        >
          <div class="flex flex-col gap-1 items-start">
            <span class="text-[10px] font-bold uppercase tracking-widest text-green-500/70">
              ▸ Axiom // live
            </span>
            <div class="w-full max-w-[90%] text-gray-300 whitespace-pre-wrap break-words leading-relaxed">
              {streamedContent()}
              <span class="inline-block w-1.5 h-3 ml-1 bg-green-500 animate-pulse align-middle" />
            </div>
          </div>
        </Show>

        <Show when={isThinking()}>
          <div class="flex flex-col gap-2 pl-1">
            <div class="flex items-center gap-3">
              <span class="text-green-500 text-xs animate-pulse font-bold tracking-widest">
                &gt; {isCancelling() ? "CANCELLING_AGENT_LOOP" : "AGENT_LOOP_ACTIVE"}
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
                  {(step, index) => {
                    const age = loopTrace().length - 1 - index();
                    const opacity =
                      age === 1
                        ? "opacity-40"
                        : age === 2
                          ? "opacity-25"
                          : "opacity-10";
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

      <div class="flex-shrink-0 px-5 py-4 bg-[#13131a] border-t border-[#2a2a35]">
        <div class="flex items-end gap-3">
          <span class="text-green-500 font-bold text-sm mb-3 flex-shrink-0 select-none">
            {isThinking() ? "×" : ">"}
          </span>

          <textarea
            ref={inputRef}
            rows={1}
            value={input()}
            onInput={(event) => {
              setInput(event.currentTarget.value);
              event.currentTarget.style.height = "auto";
              event.currentTarget.style.height = `${Math.min(
                event.currentTarget.scrollHeight,
                160,
              )}px`;
            }}
            onKeyDown={handleInputKeyDown}
            disabled={
              props.externalLock ||
              isThinking() ||
              isChangingMode() ||
              loadingConversation() ||
              !!loadError() ||
              isPurging() ||
              !runtimeReady()
            }
            placeholder={
              loadingConversation()
                ? "Restoring conversation..."
                : loadError()
                  ? "Conversation unavailable — retry before sending"
                : isThinking()
                  ? "Orchestrator busy — cancel the active run to regain control"
                  : !runtimeReady()
                    ? runtimeStatus()?.setup_hint || "Configure an inference provider"
                  : "Enter command... (Enter to send, Shift+Enter for newline)"
            }
            class="flex-1 min-w-0 bg-[#0d0d12] text-white px-4 py-3 rounded border border-[#2a2a35] focus:outline-none focus:border-green-500 transition-all font-mono text-xs placeholder-gray-700 disabled:opacity-50 resize-none overflow-hidden leading-relaxed"
            autofocus
          />

          <Show
            when={isThinking()}
            fallback={
              <button
                onClick={handleSend}
                disabled={
                  props.externalLock ||
                  isChangingMode() ||
                  loadingConversation() ||
                  !!loadError() ||
                  isPurging() ||
                  !runtimeReady() ||
                  !input().trim()
                }
                class="flex-shrink-0 mb-0.5 px-4 py-3 bg-green-600 hover:bg-green-500 disabled:opacity-30 disabled:cursor-not-allowed text-black font-bold text-xs rounded transition-colors uppercase tracking-widest"
              >
                Send
              </button>
            }
          >
            <button
              onClick={handleCancel}
              disabled={isCancelling()}
              class="flex-shrink-0 mb-0.5 px-4 py-3 bg-red-600 hover:bg-red-500 disabled:opacity-40 disabled:cursor-wait text-white font-bold text-xs rounded transition-colors uppercase tracking-widest"
            >
              {isCancelling() ? "Cancelling..." : "Cancel"}
            </button>
          </Show>
        </div>
        <p class="text-[9px] text-gray-700 mt-1 pl-5 select-none">
          Session: {props.conversationId.slice(0, 8).toUpperCase()}
        </p>
      </div>
    </div>
  );
};

export default ChatPanel;
