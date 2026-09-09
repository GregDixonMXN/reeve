# Calling guard (20 lines)

Put the agent CLI in the `--` slot. Guard gates argv, runs it sandboxed,
then flips a successful run to deny if it produced a denied file:

annalist run -- guard exec --policy policy.toml -- python3 src/tool.py
annalist run -- guard exec --policy policy.toml -- claude -p "fix tests"
annalist run -- guard exec --policy policy.toml -- codex exec "fix tests"
annalist run -- guard exec --policy policy.toml -- muse exec "fix tests"

Rules the slot inherits: binary must be in `allow_binaries`, argv paths
must satisfy `allow_paths`/`deny_globs`, no OS isolation means no run
(exit 1), runtime secret writes mean deny (exit 2). Flags are not paths.

Reeve's own tools should call `guard exec` instead of spawning directly;
until that surgery, the CLI above is the integration.
