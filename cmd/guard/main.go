// Package main implements the guard CLI: policy decisions over tool calls
// and sandboxed command execution.
//
// guard check --policy p.toml -- toolname '{"json":"args"}'
// guard exec  --policy p.toml -- argv...
// guard schema --tool name
//
// Exit 0 allow, 2 policy deny, 1 sandbox/setup broken (same numbers as
// annalist gate). Annalist records what happened; guard decides whether
// it may run.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"reeve/internal/config"
	"reeve/internal/guardrail"
	"reeve/internal/tools"
	"reeve/pkg/models"
)

// builtinDefs mirrors the file/code tool contracts in tools.registerBuiltins
// without importing the registry (cloud/memory entanglement stays out).
func builtinDefs() map[string]models.ToolDefinition {
	mk := func(name, desc string, params map[string]models.JSONSchema, required ...string) models.ToolDefinition {
		return models.ToolDefinition{
			Name:        name,
			Description: desc,
			Parameters:  models.ObjectSchema(params, required...),
		}
	}
	str := func(desc string) models.JSONSchema { return models.JSONSchema{Type: "string", Description: desc} }
	return map[string]models.ToolDefinition{
		"read_file": mk("read_file", "Read file contents from an allowed directory",
			map[string]models.JSONSchema{"path": str("Absolute path to the file")}, "path"),
		"write_file": mk("write_file", "Write content to a file in an allowed directory",
			map[string]models.JSONSchema{
				"path":    str("Absolute path to the file"),
				"content": str("Complete content to write"),
			}, "path", "content"),
		"edit_file": mk("edit_file", "Replace the first occurrence of old_text with new_text in a file",
			map[string]models.JSONSchema{
				"path":     str("Absolute path to the file"),
				"old_text": str("Exact text to replace"),
				"new_text": str("Replacement text"),
			}, "path", "old_text", "new_text"),
		"list_dir": mk("list_dir", "List files and directories at a path",
			map[string]models.JSONSchema{"path": str("Absolute directory path")}, "path"),
		"execute_code": mk("execute_code", "Execute a single binary command in a sandboxed directory. Shell chaining, pipes, redirects, and subshells are blocked.",
			map[string]models.JSONSchema{
				"command": str("Single command with no shell operators"),
				"dir":     str("Absolute working directory"),
			}, "command", "dir"),
	}
}

func newGuard(p *Policy) *guardrail.Guard {
	g := guardrail.New(config.SecurityConfig{
		EnableGuardrails: true,
		AllowNetwork:     p.AllowNetwork,
	})
	for name, def := range builtinDefs() {
		g.RegisterDefinition(def)
		_ = name
	}
	return g
}

func fail(msg string, args ...any) int {
	fmt.Fprintf(os.Stderr, "guard: "+msg+"\n", args...)
	return 1
}

func deny(msg string, args ...any) int {
	fmt.Fprintf(os.Stderr, "guard: deny: "+msg+"\n", args...)
	return 2
}

func runCheck(p *Policy, toolName, rawArgs string) int {
	var args map[string]any
	if err := json.Unmarshal([]byte(rawArgs), &args); err != nil {
		return fail("bad JSON args: %v", err)
	}
	if args == nil {
		args = map[string]any{}
	}
	call := &models.ToolCall{Name: toolName, Args: args}
	if err := newGuard(p).Check(call); err != nil {
		return deny("%s", err)
	}
	cwd, err := os.Getwd()
	if err != nil {
		return fail("getwd: %v", err)
	}
	for key, val := range args {
		s, ok := val.(string)
		if !ok || s == "" || strings.HasPrefix(s, "-") {
			continue // non-strings, empty, and flags are not paths
		}
		if reason := p.EvalPath(cwd, s); reason != "" {
			return deny("%s (%s)", s, reason)
		}
		_ = key
	}
	fmt.Printf("guard: allow %s\n", toolName)
	return 0
}

func runExec(p *Policy, argv []string) int {
	if len(argv) == 0 {
		return fail("usage: guard exec --policy p.toml -- argv...")
	}
	bin := filepath.Base(argv[0])
	if len(p.AllowBinaries) > 0 && !containsFold(p.AllowBinaries, bin) {
		return deny("binary '%s' not in allow_binaries", bin)
	}
	cwd, err := os.Getwd()
	if err != nil {
		return fail("getwd: %v", err)
	}
	for _, a := range argv[1:] {
		if a == "" || strings.HasPrefix(a, "-") {
			continue // flags are not paths
		}
		if reason := p.EvalPath(cwd, a); reason != "" {
			return deny("%s (%s)", a, reason)
		}
	}
	cfg := tools.SandboxConfig{
		Enabled:            true,
		RequireOSIsolation: p.RequireOSIsolation,
		AllowNetwork:       p.AllowNetwork,
		AllowedDirs:        []string{cwd},
		BlockedPatterns:    p.DenyGlobs,
		AllowedBinaries:    p.AllowBinaries,
		TimeoutSec:         intOr(p.TimeoutSec, 30),
		MaxOutputBytes:     1024 * 1024,
		CPUTimeSec:         10,
		MaxMemoryBytes:     512 * 1024 * 1024,
		MaxProcesses:       32,
		MaxFileSizeBytes:   10 * 1024 * 1024,
		MaxOpenFiles:       64,
	}
	if p.RequireOSIsolation {
		helperPath, helperErr := tools.IsolationHelperPath()
		isolationErr := tools.IsolationSupported(p.AllowNetwork)
		switch {
		case helperErr != nil:
			return fail("OS isolation unavailable (helper): %v", helperErr)
		case isolationErr != nil:
			return fail("OS isolation unavailable: %v", isolationErr)
		default:
			cfg.HelperPath = helperPath
		}
	}
	sb := tools.NewSandbox(cfg)
	if p.RequireOSIsolation && !sb.HasOSIsolation() {
		return fail("OS isolation required but not active; refusing to run")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(cfg.TimeoutSec+5)*time.Second)
	defer cancel()
	res := sb.ExecuteArgs(ctx, argv, cwd, nil)
	fmt.Print(res.String())
	if res.Error != "" {
		return fail("sandbox: %s", res.Error)
	}
	if res.TimedOut {
		return fail("command timed out")
	}
	if res.ExitCode != 0 {
		return fail("command exited %d", res.ExitCode)
	}
	return 0
}

func runSchema(tool string) int {
	def, ok := builtinDefs()[tool]
	if !ok {
		return fail("unknown tool '%s'", tool)
	}
	out, err := json.MarshalIndent(def, "", "  ")
	if err != nil {
		return fail("encode schema: %v", err)
	}
	fmt.Println(string(out))
	return 0
}

func containsFold(list []string, s string) bool {
	for _, v := range list {
		if strings.EqualFold(v, s) {
			return true
		}
	}
	return false
}

func intOr(v, def int) int {
	if v <= 0 {
		return def
	}
	return v
}

func usage() int {
	fmt.Fprintln(os.Stderr, `guard — policy gate and sandboxed exec (exits 0 allow, 2 deny, 1 broken)

  guard check --policy p.toml -- toolname '{"json":"args"}'
  guard exec  --policy p.toml -- argv...
  guard schema --tool name`)
	return 1
}

func main() {
	// Sandbox helper re-exec: restricted child, never the CLI.
	if tools.IsSandboxHelperInvocation(os.Args) {
		if err := tools.RunSandboxHelper(os.Args); err != nil {
			fmt.Fprintln(os.Stderr, "guard helper:", err)
			os.Exit(1)
		}
		return
	}
	args := os.Args[1:]
	if len(args) == 0 {
		os.Exit(usage())
	}
	cmd, args := args[0], args[1:]
	policyPath := ""
	rest := []string{}
	for i := 0; i < len(args); i++ {
		a := args[i]
		if a == "--" {
			rest = append(rest, args[i+1:]...)
			break
		}
		if a == "--policy" {
			i++
			if i >= len(args) {
				fmt.Fprintln(os.Stderr, "guard: --policy needs a file")
				os.Exit(1)
			}
			policyPath = args[i]
			continue
		}
		if strings.HasPrefix(a, "--policy=") {
			policyPath = strings.TrimPrefix(a, "--policy=")
			continue
		}
		if cmd == "schema" && a == "--tool" {
			i++
			if i >= len(args) {
				fmt.Fprintln(os.Stderr, "guard: --tool needs a name")
				os.Exit(1)
			}
			rest = append(rest, a, args[i])
			continue
		}
		rest = append(rest, a)
	}
	policy := DefaultPolicy()
	if policyPath != "" {
		var err error
		policy, err = LoadPolicy(policyPath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "guard:", err)
			os.Exit(1)
		}
	}
	switch cmd {
	case "check":
		if len(rest) < 1 {
			fmt.Fprintln(os.Stderr, "guard: usage: guard check --policy p.toml -- toolname '{\"args\"}'")
			os.Exit(1)
		}
		raw := "{}"
		if len(rest) > 1 {
			raw = rest[1]
		}
		os.Exit(runCheck(policy, rest[0], raw))
	case "exec":
		os.Exit(runExec(policy, rest))
	case "schema":
		name := ""
		for i := 0; i < len(rest); i++ {
			if rest[i] == "--tool" && i+1 < len(rest) {
				name = rest[i+1]
			}
		}
		if name == "" {
			fmt.Fprintln(os.Stderr, "guard: usage: guard schema --tool name")
			os.Exit(1)
		}
		os.Exit(runSchema(name))
	default:
		os.Exit(usage())
	}
}
