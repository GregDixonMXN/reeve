# Axiom AI Agent

**Local-First, Performance-First AI Agentic Runtime**

```
  SolidJS (Wails Shell)
         │
    Go Orchestrator ──── Guardrail
         │
    ┌────┼────┐
    │    │    │
  Static Mojo  Python
  Tools  Bridge Bridge
         │       │
    vector_ops  Wolfram|Alpha
    embedding
         │
    SQLite + sqlite-vec
```

## Stack

| Layer    | Tech               | Purpose                      |
|----------|--------------------|------------------------------|
| Backend  | Go (Wails v2)      | Orchestration, concurrency   |
| Frontend | SolidJS + Vite     | Reactive desktop UI          |
| AI       | Mojo + llama.cpp   | Inference, vector math       |
| Memory   | SQLite + sqlite-vec| Semantic search, persistence |
| Oracle   | Wolfram Alpha API  | Verified computation         |

## Quickstart

```bash
# 1. Setup
make setup

# 2. Start Ollama with a model
ollama pull phi4

# 3. Configure
cp axiom.example.toml axiom.toml

# 4. Run
make dev
```

## Structure

```
axiom/
├── main.go                          # Wails entry point
├── app.go                           # Wails-bound methods → frontend
├── internal/
│   ├── config/config.go             # TOML configuration
│   ├── orchestrator/orchestrator.go # Hub: Think-Verify-Act loop
│   ├── cognitive/
│   │   ├── engine.go                # LLM interface + CoT prompts
│   │   └── adapters/
│   │       └── remote_runner.go     # Ollama / OpenAI HTTP adapter
│   ├── memory/store.go              # SQLite + sqlite-vec
│   ├── tools/registry.go            # Static + dynamic tools
│   ├── guardrail/guardrail.go       # Security enforcement
│   └── bridge/bridge.go             # Go↔Mojo, Go↔Python IPC
├── pkg/
│   ├── models/models.go             # Shared types
│   └── logger/logger.go             # Leveled logging
├── mojo/kernels/
│   ├── vector_ops.mojo              # SIMD cosine similarity
│   └── embedding.mojo               # Text embedding (Phase 3)
├── scripts/wolfram_query.py         # Wolfram|Alpha bridge
└── frontend/
    ├── index.html
    ├── vite.config.ts
    └── src/
        ├── index.tsx
        ├── App.tsx
        ├── components/
        │   ├── ChatPanel.tsx
        │   └── Sidebar.tsx
        ├── lib/wails.d.ts
        └── styles/global.css
```

## Roadmap

- **Phase 1 (Skeleton):** Wails shell, Go tools, Ollama adapter ← you are here
- **Phase 2 (Brain):** Fine-tune CoT prompts, streaming responses
- **Phase 3 (Memory):** Mojo vector indexer, sqlite-vec semantic search
- **Phase 4 (Oracle):** Wolfram|Alpha tool, full tool chain
