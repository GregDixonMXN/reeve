# Guard cut (from Reeve)

Product: a CLI that answers "may this process do that" — `guard check`,
`guard exec`, `guard schema`. Exits 0 allow, 2 deny, 1 broken (same
numbers as `annalist gate`). Reeve stays the dogfood host; do not merge
the binaries.

## Take (copy into the guard module at split time)

- `internal/guardrail/guardrail.go` — `Guard.Check`, schema validation,
  tool semantics (absolute paths, `execute_code` shell-operator blocklist:
  `&& || ; | $( \``), network policy. Tested (292-line test file comes too).
- `pkg/models/tool_schema.go` — `ToolCall`, `ToolDefinition`,
  `JSONSchema`. 236 lines, no Reeve imports (verify at split).
- `internal/tools/sandbox.go` — `SandboxConfig` + execution boundary.
  Stdlib only.
- `internal/tools/isolation.go`, `isolation_linux.go` — Landlock/seccomp
  boundary (`golang.org/x/sys/unix` only). Linux test comes too.

## Rewrite (do not import)

- Policy: guard's own TOML struct — `allow_paths`, `deny_globs`,
  `allow_network`, `allow_binaries`, `require_os_isolation`, plus the
  rlimit/timeout fields. `config.SecurityConfig`/`SandboxSection` already
  look like this, but importing `internal/config` drags the whole app
  config. Copy the shape, not the package. BurntSushi/toml is already a dep.

## Leave (imports cognitive, memory, Wails, or scripts — stays)

- `internal/tools/registry.go` (1288 lines: cloud, memory, image tools),
  `cloud_delegator.go`, `internal/cognitive`, `internal/memory`,
  frontend + Wails (`app.go`), Ollama/OpenAI routing, `scripts/`,
  `pkg/models/models.go` conversation/LLM types.

## Until the split

`cmd/guard` lives in this module and imports the take-set in place
(`reeve/internal/...`). Zero copy, proves the CLI surface. Split to
`GregDixonMXN/guard` only when `guard check/exec` + the five deny tests
are green with `-race`.

## Split (done 2026-09-09)

Landed as GregDixonMXN/guard `main`: guardrail + tool-schema models +
sandbox copied over, imports rewritten, `config.SecurityConfig` replaced
by a local struct (same three fields), `pathWithin` extracted from the
registry, `package tools` renamed to `sandbox`. `-race` green standalone;
stranger command verified from the split binary. `cmd/guard` remains here
as the in-repo prototype; freeze it unless Guard needs a caller fix.
