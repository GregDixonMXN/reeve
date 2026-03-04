package tools

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"time"
)

// SandboxConfig controls what the execute_code tool is allowed to do.
type SandboxConfig struct {
	Enabled         bool     `toml:"enabled"`
	AllowedDirs     []string `toml:"allowed_dirs"`
	BlockedPatterns []string `toml:"blocked_patterns"`
	AllowedBinaries []string `toml:"allowed_binaries"`
	TimeoutSec      int      `toml:"timeout_sec"`
	MaxOutputBytes  int      `toml:"max_output_bytes"`
}

// Sandbox runs commands in a controlled environment with 3 security layers.
type Sandbox struct {
	cfg SandboxConfig
}

func NewSandbox(cfg SandboxConfig) *Sandbox {
	if cfg.TimeoutSec <= 0 {
		cfg.TimeoutSec = 30
	}
	if cfg.MaxOutputBytes <= 0 {
		cfg.MaxOutputBytes = 64 * 1024 // 64KB
	}

	// Ensure core dangerous patterns are always blocked
	coreBlocked := []string{
		"rm -rf /",
		"rm -rf /*",
		"sudo ",
		"mkfs",
		"dd if=",
		":(){ :|:& };:",
		"> /dev/sd",
		"chmod 777 /",
		"chown ",
		"passwd",
		"shutdown",
		"reboot",
		"init 0",
		"kill -9 1",
		"/etc/shadow",
		"/etc/passwd",
		"curl | sh",
		"curl | bash",
		"wget -O - | sh",
		"eval ",
		"`",  // backtick command substitution
		"$(", // subshell command substitution
	}

	existing := make(map[string]bool)
	for _, p := range cfg.BlockedPatterns {
		existing[p] = true
	}
	for _, p := range coreBlocked {
		if !existing[p] {
			cfg.BlockedPatterns = append(cfg.BlockedPatterns, p)
		}
	}

	return &Sandbox{cfg: cfg}
}

// ExecuteResult holds the output of a sandboxed command.
type ExecuteResult struct {
	Stdout   string `json:"stdout"`
	Stderr   string `json:"stderr"`
	ExitCode int    `json:"exit_code"`
	TimedOut bool   `json:"timed_out"`
	Error    string `json:"error,omitempty"`
}

func (r *ExecuteResult) String() string {
	var sb strings.Builder

	if r.Error != "" {
		sb.WriteString(fmt.Sprintf("[SANDBOX ERROR] %s\n", r.Error))
		return sb.String()
	}

	if r.TimedOut {
		sb.WriteString(fmt.Sprintf("[TIMEOUT] Command killed after exceeding time limit\n"))
	}

	if r.Stdout != "" {
		sb.WriteString(fmt.Sprintf("[STDOUT]\n%s\n", r.Stdout))
	}
	if r.Stderr != "" {
		sb.WriteString(fmt.Sprintf("[STDERR]\n%s\n", r.Stderr))
	}

	if r.ExitCode == 0 && !r.TimedOut {
		sb.WriteString("[EXIT CODE] 0 (success)\n")
	} else {
		sb.WriteString(fmt.Sprintf("[EXIT CODE] %d\n", r.ExitCode))
	}

	return sb.String()
}

// Execute runs a command inside the sandbox with all 3 security layers.
func (s *Sandbox) Execute(ctx context.Context, command, dir string) *ExecuteResult {
	// ── Layer 0: Sandbox enabled check ──────────────────────────────────
	if !s.cfg.Enabled {
		return &ExecuteResult{Error: "Code execution is disabled. Set security.sandbox.enabled = true in axiom.toml"}
	}

	// ── Layer 1: Directory Jail ─────────────────────────────────────────
	if dir == "" {
		return &ExecuteResult{Error: "Working directory 'dir' is required"}
	}

	absDir, err := filepath.Abs(dir)
	if err != nil {
		return &ExecuteResult{Error: fmt.Sprintf("Invalid directory: %v", err)}
	}

	// Resolve symlinks to prevent jail escape via symlink chains
	realDir, err := filepath.EvalSymlinks(absDir)
	if err != nil {
		// Directory might not exist yet — check the parent
		parentDir := filepath.Dir(absDir)
		realParent, err2 := filepath.EvalSymlinks(parentDir)
		if err2 != nil {
			return &ExecuteResult{Error: fmt.Sprintf("Directory not accessible: %v", err)}
		}
		realDir = filepath.Join(realParent, filepath.Base(absDir))
	}

	if !s.isInsideAllowedDir(realDir) {
		return &ExecuteResult{
			Error: fmt.Sprintf("Directory '%s' is outside allowed paths. Allowed: %v", dir, s.cfg.AllowedDirs),
		}
	}

	// Verify directory exists
	info, err := os.Stat(realDir)
	if err != nil || !info.IsDir() {
		return &ExecuteResult{Error: fmt.Sprintf("Directory does not exist: %s", dir)}
	}

	// ── Layer 2: Command Blacklist ──────────────────────────────────────
	if violation := s.checkBlacklist(command); violation != "" {
		return &ExecuteResult{
			Error: fmt.Sprintf("BLOCKED: Command contains forbidden pattern '%s'", violation),
		}
	}

	// ── Layer 2b: Binary Whitelist (optional) ───────────────────────────
	if len(s.cfg.AllowedBinaries) > 0 {
		if !s.isBinaryAllowed(command) {
			return &ExecuteResult{
				Error: fmt.Sprintf("BLOCKED: Binary not in whitelist. Allowed: %v", s.cfg.AllowedBinaries),
			}
		}
	}

	// ── Layer 3: Timeout Watchdog ───────────────────────────────────────
	timeout := time.Duration(s.cfg.TimeoutSec) * time.Second
	execCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	// Build the command — use shell to support pipes and redirects
	var cmd *exec.Cmd
	if runtime.GOOS == "windows" {
		cmd = exec.CommandContext(execCtx, "cmd", "/C", command)
	} else {
		cmd = exec.CommandContext(execCtx, "sh", "-c", command)
	}

	cmd.Dir = realDir

	// Sanitize environment — only pass safe variables
	cmd.Env = s.sanitizedEnv()

	// Set process group so we can kill the entire tree on timeout
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	// Capture output
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	// Run
	err = cmd.Run()

	// Build result
	result := &ExecuteResult{
		Stdout: s.truncateOutput(stdout.String()),
		Stderr: s.truncateOutput(stderr.String()),
	}

	if execCtx.Err() == context.DeadlineExceeded {
		result.TimedOut = true
		result.ExitCode = -1

		// Kill the entire process group to prevent zombies
		if cmd.Process != nil {
			syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		}
	} else if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			result.ExitCode = exitErr.ExitCode()
		} else {
			result.ExitCode = -1
			result.Error = err.Error()
		}
	}

	return result
}

// ── Security Layer Implementations ──────────────────────────────────────────

func (s *Sandbox) isInsideAllowedDir(targetDir string) bool {
	for _, allowed := range s.cfg.AllowedDirs {
		// Expand ~ to home directory
		if strings.HasPrefix(allowed, "~/") {
			home, err := os.UserHomeDir()
			if err != nil {
				continue
			}
			allowed = filepath.Join(home, allowed[2:])
		}

		absAllowed, err := filepath.Abs(allowed)
		if err != nil {
			continue
		}

		// Resolve symlinks on the allowed dir too
		realAllowed, err := filepath.EvalSymlinks(absAllowed)
		if err != nil {
			// Allowed dir might not exist — use unresolved
			realAllowed = absAllowed
		}

		if strings.HasPrefix(targetDir, realAllowed+string(filepath.Separator)) || targetDir == realAllowed {
			return true
		}
	}
	return false
}

func (s *Sandbox) checkBlacklist(command string) string {
	normalized := strings.ToLower(strings.TrimSpace(command))
	for _, pattern := range s.cfg.BlockedPatterns {
		if strings.Contains(normalized, strings.ToLower(pattern)) {
			return pattern
		}
	}
	return ""
}

func (s *Sandbox) isBinaryAllowed(command string) bool {
	// Extract the first word (binary name) from the command
	parts := strings.Fields(strings.TrimSpace(command))
	if len(parts) == 0 {
		return false
	}
	binary := filepath.Base(parts[0])

	for _, allowed := range s.cfg.AllowedBinaries {
		if binary == allowed {
			return true
		}
	}
	return false
}

func (s *Sandbox) sanitizedEnv() []string {
	// Only pass through safe environment variables
	safeKeys := map[string]bool{
		"PATH":        true,
		"HOME":        true,
		"USER":        true,
		"LANG":        true,
		"LC_ALL":      true,
		"TERM":        true,
		"TMPDIR":      true,
		"GOPATH":      true,
		"GOROOT":      true,
		"CARGO_HOME":  true,
		"RUSTUP_HOME": true,
		"NODE_PATH":   true,
		"PYTHON":      true,
		"VIRTUAL_ENV": true,
	}

	var env []string
	for _, e := range os.Environ() {
		key := strings.SplitN(e, "=", 2)[0]
		if safeKeys[key] {
			env = append(env, e)
		}
	}
	return env
}

func (s *Sandbox) truncateOutput(output string) string {
	if len(output) > s.cfg.MaxOutputBytes {
		truncated := output[:s.cfg.MaxOutputBytes]
		truncated += fmt.Sprintf("\n\n[TRUNCATED: output exceeded %d bytes]", s.cfg.MaxOutputBytes)
		return truncated
	}
	return output
}
