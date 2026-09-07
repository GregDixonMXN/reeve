# Reeve AI Agent

**Hybrid desktop agent runtime with a private Qwen 3.6 brain, OpenAI cloud routing, persistent conversations, and guarded tool execution.**

Reeve is a real agent runtime, not just a chat wrapper. The core job is to assemble context, choose tools, execute safely, recover from errors, and persist useful memory.

## Current state

Reeve currently provides:
- Wails desktop shell with SolidJS frontend
- Go-based orchestration loop
- Local Qwen 3.6 support through Ollama with native thinking and function tools
- OpenAI support through the Responses API with native function tools
- Legacy Anthropic and Gemini integrations
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
| Inference | Qwen 3.6 through Ollama, OpenAI Responses API | Local and cloud reasoning |
| Memory | SQLite + sqlite-vec | Persistence and semantic recall |
| Tooling | Go + Python | File ops, web, git, execution |

## Prerequisites

- Go 1.24+
- Node.js 20.19.x or 22.12+
- Python 3.10+ with `venv`
- Wails v2.11.0 (installed by `make setup`)
- Ollama 0.32.5 or newer
- About 32 GiB RAM for the default 27B Q4 model
- An OpenAI Platform API key for hybrid/cloud mode (optional for local mode)

## Quickstart

```bash
# 1. Install dependencies
make setup

# 2. Copy config
cp reeve.example.toml reeve.toml

# 3. Install the pinned local agent model
ollama pull qwen3.6:27b-mtp-q4_K_M

# 4. Put secrets in either:
#    - environment variables (recommended), or
#    - a protected config outside Reeve's writable workspaces

# 5. Configure OpenAI for hybrid/cloud mode (optional)
export REEVE_OPENAI_API_KEY="your-platform-api-key"

# 6. Run desktop app
make dev
```

The default profile starts in `hybrid` mode with
`qwen3.6:27b-mtp-q4_K_M` as the local agent and `gpt-5.6-sol` through
`POST /v1/responses` as the cloud agent. Without an API key, startup falls back
to local mode. The model tag is deliberately pinned: bare `qwen3.6` currently
selects a different 35B-A3B model. The default 65,536-token context is the
practical agent/coding starting point for a 32 GiB machine.

OpenAI API billing is separate from a ChatGPT subscription. Keep the platform
key in the environment; Reeve never sends it to the frontend.

## Configuration model

Reeve finds its base config from `REEVE_CONFIG`, the current directory,
`$XDG_CONFIG_HOME/reeve`, or the executable and its parent directories. An
explicit `REEVE_CONFIG` must name a readable file; missing or unreadable paths
fail startup instead of silently loading defaults. Reeve then loads the adjacent
`reeve.local.toml` and finally applies environment overrides. Relative
database/runtime paths are resolved from the selected config file, and the
registered Python tool scripts are embedded in the desktop binary.

`make setup` installs the Python dependencies in `~/.reeve/venv`, outside the
workspace directories that agent tools may modify. Keep `tools.python_path`
outside every configured writable/allowed directory; Reeve rejects an embedded
tool runtime inside a writable workspace. Rerun `make setup` after upgrading
from a checkout that used the former project-local `.venv`.

Supported secret env vars:
- `REEVE_OPENAI_API_KEY` (preferred for Reeve)
- `OPENAI_API_KEY`
- `REEVE_ANTHROPIC_KEY`
- `ANTHROPIC_API_KEY`
- `REEVE_GEMINI_KEY`
- `GEMINI_API_KEY`
- `GOOGLE_API_KEY`
- `REEVE_WOLFRAM_APP_ID`
- `WOLFRAM_APP_ID`

The Reeve-specific OpenAI variable wins when both OpenAI variables are set.
Never commit a key to `reeve.toml`, expose it to the frontend, or paste it into
chat. The UI reports only whether a credential was resolved.

The hybrid settings are:

```toml
[model]
enabled = true
runner_model = "qwen3.6:27b-mtp-q4_K_M"
context_size = 65536
enable_thinking = true
default_mode = "hybrid"

[embedding]
enabled = false

[cloud]
provider = "openai"
openai_model = "gpt-5.6-sol"
reasoning_effort = "medium"
```

Supported reasoning efforts are `none`, `low`, `medium`, `high`, `xhigh`, and
`max`. `medium` is the balanced default. Keep `security.allow_network = true`
for cloud inference.

Modes behave as follows:

- `local`: every turn stays on Qwen/Ollama.
- `hybrid`: simple/private turns use Qwen; complex turns are automatically
  routed to OpenAI when its API key is configured.
- `cloud`: every turn uses OpenAI.

Ollama thinking and native tool calls are preserved across the full
assistant→tool→result loop. Streaming remains disabled by default so the
non-streaming protocol path is used consistently.

## Security posture

Recent hardening changes:
- removed committed secrets from base config
- capped runaway iteration counts
- strengthened guardrail validation for paths and execution
- restricted `execute_code` to single-command execution without shell chaining
- hardened `web_scrape` against private/loopback/metadata targets, DNS rebinding,
  redirect pivots, and oversized downloads
- made `allow_network = false` fail closed: web, cloud, vision, and Git push are
  unavailable; core model/embedding endpoints must be loopback-only;
  local Git actions remain available only behind enforced network isolation
- placed `execute_code` behind Linux Landlock filesystem rules plus seccomp
  syscall/network filtering; it stays unavailable if that boundary cannot be
  established (or if `require_os_isolation` is disabled while networking is off)
- applied finite per-process CPU-time, virtual-memory, process-count, file-size,
  open-file, and core-dump limits, with process-group cancellation on wall-clock
  timeout
- unified both public chat paths behind one guarded tool-loop engine and one
  canonical assistant/tool/result transcript protocol
- replaced provider-specific argument guessing with one recursive typed JSON
  Schema contract shared by OpenAI, Ollama, Anthropic, prompts, and guardrails
- isolated the trusted Python helper environment from writable workspaces and
  reject configurations that could let project code poison that environment
- restricted SQLite database/WAL files to the current user and made unsupported
  encryption settings fail closed instead of silently writing plaintext
- enforced commit-safe local override config via `reeve.local.toml`

The command boundary still does not provide a PID namespace, cgroup-wide
CPU/memory accounting, or a container image. Linux rlimits are inherited by
children but some limits are per process or per user, so Reeve remains a
developer-focused agent runtime rather than a multi-tenant execution service.

## Recommended workflow

- Use **hybrid** for normal work, **local** for private/offline work, and
  **cloud** when every turn should use OpenAI
- Keep the sandbox and `require_os_isolation` enabled
- Verify code changes with execution before considering a task complete
- Keep secrets out of repo-tracked files

## Project layout

```text
reeve/
├── main.go
├── app.go
├── reeve.toml
├── reeve.example.toml
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
- migrate persisted messages from JSON-in-content tool envelopes to structured
  tool calls/results (with a SQLite data migration)
- add explicit loop drift detection and recovery policies
- add optional cgroup/PID-namespace or container isolation for untrusted
  multi-tenant execution
- expand automated tests around provider failures and end-to-end desktop flows
- add structured telemetry and user-visible audit trails

## Dev verification

```bash
make test
```

`make test` installs the locked frontend dependencies, type-checks and builds
the frontend first, then runs the Go test suite with the race detector. Building
the frontend before Go verification ensures the embedded asset tree is valid.

The equivalent frontend-only checks are:

```bash
npm --prefix frontend ci
npm --prefix frontend run check
```

GitHub Actions runs the same frontend, Python, Go vet, and race checks on every
push and pull request. The manual `release-readiness` workflow builds and hashes
the Linux binary but deliberately cannot upload or publish it.
(Historical note: the 9 GB model blob was stripped from Git history and the
legacy Anthropic/Google credentials scrubbed and confirmed dead before the
repo went public. API keys live only in environment variables — never in
`reeve.toml`, which is git-ignored.)
