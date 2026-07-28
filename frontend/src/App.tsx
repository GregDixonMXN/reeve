import { Component, createSignal } from "solid-js";
import ChatPanel from "./components/ChatPanel";
import Sidebar from "./components/Sidebar";

const App: Component = () => {
  const [activeConversation, setActiveConversation] =
    createSignal<string>("default");
  const [sidebarOpen, setSidebarOpen] = createSignal(true);
  const [sidebarRefreshKey, setSidebarRefreshKey] = createSignal(0);
  const [chatLocked, setChatLocked] = createSignal(false);
  const [sidebarBusy, setSidebarBusy] = createSignal(false);

  const refreshSidebar = () => setSidebarRefreshKey((k) => k + 1);
  const interactionLocked = () => chatLocked() || sidebarBusy();
  const selectConversation = (id: string) => {
    if (!interactionLocked()) setActiveConversation(id);
  };

  return (
    <div class="flex h-screen w-screen bg-[#0d0d12] text-gray-200 overflow-hidden font-mono">
      <Sidebar
        activeId={activeConversation()}
        onSelect={selectConversation}
        onToggle={() => setSidebarOpen((v) => !v)}
        onBusyChange={setSidebarBusy}
        onHistoryChanged={refreshSidebar}
        isOpen={sidebarOpen()}
        disabled={interactionLocked()}
        refreshKey={sidebarRefreshKey()}
      />

      <main class="flex-1 flex flex-col min-w-0 relative">
        <ChatPanel
          conversationId={activeConversation()}
          externalLock={sidebarBusy()}
          onHistoryChanged={refreshSidebar}
          onInteractionLockChange={setChatLocked}
        />
      </main>
    </div>
  );
};

export default App;
