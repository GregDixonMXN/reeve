import { Component, createSignal } from "solid-js";
import ChatPanel from "./components/ChatPanel";
import Sidebar from "./components/Sidebar";

const App: Component = () => {
  const [activeConversation, setActiveConversation] =
    createSignal<string>("default");
  const [sidebarOpen, setSidebarOpen] = createSignal(true);
  const [sidebarRefreshKey, setSidebarRefreshKey] = createSignal(0);

  const refreshSidebar = () => setSidebarRefreshKey((k) => k + 1);

  return (
    <div class="flex h-screen w-screen bg-[#0d0d12] text-gray-200 overflow-hidden font-mono">
      <Sidebar
        activeId={activeConversation()}
        onSelect={setActiveConversation}
        onToggle={() => setSidebarOpen((v) => !v)}
        isOpen={sidebarOpen()}
        refreshKey={sidebarRefreshKey()}
      />

      <main class="flex-1 flex flex-col min-w-0 relative">
        <ChatPanel
          conversationId={activeConversation()}
          onMessageSent={refreshSidebar}
        />
      </main>
    </div>
  );
};

export default App;
