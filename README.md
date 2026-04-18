# Axiom AI Agent

**Local-first AI agent runtime with a desktop UI, tool orchestration, semantic memory, and guarded execution.**

Axiom is a real agent runtime, not just a chat wrapper. The core job is to assemble context, choose tools, execute safely, recover from errors, and persist useful memory.

## Current state

Axiom currently provides:
- Wails desktop shell with SolidJS frontend
- Go-based orchestration loop
- Local model support through Ollama
- Optional cloud model support for Anthropic and Gemini
- SQLite plus sqlite-vec semantic memory
- File, git, web, and sandboxed execution tools
- Mode switching across local, hybrid, and cloud

## Architecture

```text
SolidJS + Wails UI
        |
        v
Go App Bindings
        |
        v
Orchestrator -> Cognitive Engine -> Local or Cloud Runner
        |
        +-> Guardrail
        +-> Tool Registry
        +-> Sandbox
        +-> Semantic Memory
```

## Stack

| Layer | Tech | Purpose |
|---|---|---|
| Backend | Go + Wails | Orchestration, desktop runtime, app bindings |
| Frontend | SolidJS + Vite | Chat UI and controls |
| Inference | Ollama, Anthropic, Gemini | Local and cloud reasoning |
| Memory | SQLite + sqlite-vec | Persistence and semantic recall |
| Tooling | Go + Python | File ops, web, git, execution |

## Quickstart

```bash
# 1. Install dependencies
make setup

# 2. Copy config
cp axiom.example.toml axiom.toml

# 3. Put secrets in either:
#    - environment variables, or
#    - axiom.local.toml (gitignored)

# 4. Start Ollama models you want
ollama pull qwen2.5-coder:32b
ollama pull nomic-embed-text

# 5. Run desktop app
wails dev -tags webkit2_41
```

## Configuration model

Axiom loads configuration in this order:
1. `axiom.toml`
2. `axiom.local.toml` if present
3. environment variable overrides

Supported secret env vars:
- `AXIOM_ANTHROPIC_KEY`
- `ANTHROPIC_API_KEY`
- `AXIOM_GEMINI_KEY`
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`
- `AXIOM_WOLFRAM_APP_ID`
- `WOLFRAM_APP_ID`

## Security posture

Recent hardening changes:
- removed committed secrets from base config
- capped runaway iteration counts
- strengthened guardrail validation for paths and execution
- restricted `execute_code` to single-command execution without shell chaining
- enforced commit-safe local override config via `axiom.local.toml`

Important note: Axiom is safer now, but still a developer-focused agent runtime. Treat it as powerful software, not a toy.

## Recommended workflow

- Use **local** mode for day-to-day project work
- Use **cloud** mode when you need stronger reasoning or code generation
- Keep sandbox enabled
- Verify code changes with execution before considering a task complete
- Keep secrets out of repo-tracked files

## Project layout

```text
axiom/
├── main.go
├── app.go
├── axiom.toml
├── axiom.example.toml
├── internal/
│   ├── cognitive/
│   ├── config/
│   ├── guardrail/
│   ├── memory/
│   ├── orchestrator/
│   └── tools/
├── frontend/
├── scripts/
└── mojo/
```

## Production direction

The next product-grade steps should be:
- split the orchestrator into smaller subsystems
- add richer typed tool schemas and typed tool results
- add explicit loop drift detection and recovery policies
- add automated tests around guardrails, sandboxing, and tool execution
- add structured telemetry and user-visible audit trails

## Dev verification

```bash
go test ./...
cd frontend && npm run build
```
