package tools

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
)

// SandboxConfig controls what the execute_code tool is allowed to do.
type SandboxConfig struct {
	Enabled            bool     `toml:"enabled"`
	RequireOSIsolation bool     `toml:"require_os_isolation"`
	AllowNetwork       bool     `toml:"allow_network"`
	HelperPath         string   `toml:"-"`
	AllowedDirs        []string `toml:"allowed_dirs"`
	BlockedPatterns    []string `toml:"blocked_patterns"`
	AllowedBinaries    []string `toml:"allowed_binaries"`
	TimeoutSec         int      `toml:"timeout_sec"`
	MaxOutputBytes     int      `toml:"max_output_bytes"`
	CPUTimeSec         int      `toml:"cpu_time_sec"`
	MaxMemoryBytes     int64    `toml:"max_memory_bytes"`
	MaxProcesses       int      `toml:"max_processes"`
	MaxFileSizeBytes   int64    `toml:"max_file_size_bytes"`
	MaxOpenFiles       int      `toml:"max_open_files"`
}

// Sandbox runs commands in a controlled environment with 3 security layers.
type Sandbox struct {
	cfg SandboxConfig
}

// boundedBuffer implements io.Writer while retaining at most limit bytes.
// Write always reports the full input length so os/exec can continue draining
// the child process after the retained output reaches the configured cap.
type boundedBuffer struct {
	mu        sync.Mutex
	buf       bytes.Buffer
	limit     int
	truncated bool
}

func newBoundedBuffer(limit int) *boundedBuffer {
	return &boundedBuffer{limit: limit}
}

func (b *boundedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	written := len(p)
	remaining := b.limit - b.buf.Len()
	if remaining > 0 {
		if len(p) > remaining {
			_, _ = b.buf.Write(p[:remaining])
			b.truncated = true
		} else {
			_, _ = b.buf.Write(p)
		}
	} else if len(p) > 0 {
		b.truncated = true
	}
	return written, nil
}

func (b *boundedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()

	output := b.buf.String()
	if b.truncated {
		output += fmt.Sprintf("\n\n[TRUNCATED: output exceeded %d bytes]", b.limit)
	}
	return output
}

func (b *boundedBuffer) Len() int {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Len()
}

func NewSandbox(cfg SandboxConfig) *Sandbox {
	if cfg.TimeoutSec <= 0 {
		cfg.TimeoutSec = 30
	}
	if cfg.MaxOutputBytes <= 0 {
		cfg.MaxOutputBytes = 64 * 1024 // 64KB
	}
	if cfg.CPUTimeSec <= 0 {
		cfg.CPUTimeSec = cfg.TimeoutSec
	}
	if cfg.MaxMemoryBytes <= 0 {
		cfg.MaxMemoryBytes = defaultSandboxMaxMemoryBytes
	}
	if cfg.MaxProcesses <= 0 {
		cfg.MaxProcesses = defaultSandboxMaxProcesses
	}
	if cfg.MaxFileSizeBytes <= 0 {
		cfg.MaxFileSizeBytes = defaultSandboxMaxFileSize
	}
	if cfg.MaxOpenFiles <= 0 {
		cfg.MaxOpenFiles = defaultSandboxMaxOpenFiles
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

// Execute parses a user-supplied single command and runs it inside the sandbox.
func (s *Sandbox) Execute(ctx context.Context, command, dir string) *ExecuteResult {
	parts := strings.Fields(strings.TrimSpace(command))
	if len(parts) == 0 {
		return &ExecuteResult{Error: "Empty command"}
	}
	return s.executeParts(ctx, command, parts, dir, nil, s.cfg.AllowedDirs)
}

// ExecuteArgs runs an already-tokenized trusted command. It is used for
// embedded Herald helper scripts so source code never needs shell quoting.
func (s *Sandbox) ExecuteArgs(ctx context.Context, parts []string, dir string, stdin io.Reader) *ExecuteResult {
	if len(parts) == 0 || strings.TrimSpace(parts[0]) == "" {
		return &ExecuteResult{Error: "Empty command"}
	}
	return s.executeParts(ctx, parts[0], append([]string(nil), parts...), dir, stdin, s.cfg.AllowedDirs)
}

func (s *Sandbox) HasOSIsolation() bool {
	return s != nil && s.cfg.Enabled && s.cfg.RequireOSIsolation && s.cfg.HelperPath != "" && IsolationSupported(s.cfg.AllowNetwork) == nil
}

func (s *Sandbox) executeParts(ctx context.Context, policyCommand string, parts []string, dir string, stdin io.Reader, isolationRoots []string) *ExecuteResult {
	// ── Layer 0: Sandbox enabled check ──────────────────────────────────
	if !s.cfg.Enabled {
		return &ExecuteResult{Error: "Code execution is disabled. Set security.sandbox.enabled = true in herald.toml"}
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
	if violation := s.checkBlacklist(policyCommand); violation != "" {
		return &ExecuteResult{
			Error: fmt.Sprintf("BLOCKED: Command contains forbidden pattern '%s'", violation),
		}
	}

	// ── Layer 2b: Binary Whitelist (optional) ───────────────────────────
	if len(s.cfg.AllowedBinaries) > 0 {
		if !s.isBinaryAllowed(policyCommand) {
			return &ExecuteResult{
				Error: fmt.Sprintf("BLOCKED: Binary not in whitelist. Allowed: %v", s.cfg.AllowedBinaries),
			}
		}
	}

	// ── Layer 3: Timeout Watchdog ───────────────────────────────────────
	timeout := time.Duration(s.cfg.TimeoutSec) * time.Second
	execCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	commandEnv := s.sanitizedEnv()
	var cmd *exec.Cmd
	if s.cfg.RequireOSIsolation {
		if err := IsolationSupported(s.cfg.AllowNetwork); err != nil {
			return &ExecuteResult{Error: fmt.Sprintf("OS isolation unavailable: %v", err)}
		}
		// Give isolated tools an ephemeral home/cache. Landlock prevents the
		// command from renaming its entry in /tmp, making cleanup safe.
		tempHome, err := os.MkdirTemp("", "herald-sandbox-")
		if err != nil {
			return &ExecuteResult{Error: fmt.Sprintf("Create isolated home: %v", err)}
		}
		defer os.RemoveAll(tempHome)
		request := isolationRequest{
			AllowedDirs:  append(append([]string(nil), isolationRoots...), tempHome),
			WorkingDir:   realDir,
			Command:      parts,
			AllowNetwork: s.cfg.AllowNetwork,
			ResourceLimits: isolationResourceLimits{
				CPUTimeSec:       s.cfg.CPUTimeSec,
				MaxMemoryBytes:   s.cfg.MaxMemoryBytes,
				MaxProcesses:     s.cfg.MaxProcesses,
				MaxFileSizeBytes: s.cfg.MaxFileSizeBytes,
				MaxOpenFiles:     s.cfg.MaxOpenFiles,
			},
		}
		cmd, err = newIsolatedCommand(execCtx, s.cfg.HelperPath, request)
		if err != nil {
			return &ExecuteResult{Error: err.Error()}
		}
		commandEnv = withEnvironmentOverrides(commandEnv, map[string]string{
			"HOME":                tempHome,
			"TMPDIR":              tempHome,
			"GOCACHE":             filepath.Join(tempHome, "go-cache"),
			"CARGO_HOME":          filepath.Join(tempHome, "cargo-home"),
			"XDG_CACHE_HOME":      filepath.Join(tempHome, "xdg-cache"),
			"NPM_CONFIG_CACHE":    filepath.Join(tempHome, "npm-cache"),
			"PIP_CACHE_DIR":       filepath.Join(tempHome, "pip-cache"),
			"PYTHONPYCACHEPREFIX": filepath.Join(tempHome, "python-cache"),
		})
	} else {
		cmd = exec.CommandContext(execCtx, parts[0], parts[1:]...)
	}

	cmd.Dir = realDir
	cmd.Stdin = stdin

	// Sanitize environment — only pass safe variables.
	cmd.Env = commandEnv

	// Set process group so we can kill the entire tree on timeout
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return os.ErrProcessDone
		}
		err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		if err == syscall.ESRCH {
			return os.ErrProcessDone
		}
		return err
	}
	cmd.WaitDelay = 2 * time.Second

	// Capture output without ever retaining more than the configured cap.
	// stdout and stderr have independent caps, matching the previous result
	// truncation behavior while avoiding unbounded bytes.Buffer growth.
	stdout := newBoundedBuffer(s.cfg.MaxOutputBytes)
	stderr := newBoundedBuffer(s.cfg.MaxOutputBytes)
	cmd.Stdout = stdout
	cmd.Stderr = stderr

	// Run
	err = cmd.Run()

	// Build result
	result := &ExecuteResult{
		Stdout: stdout.String(),
		Stderr: stderr.String(),
	}

	if execCtx.Err() == context.DeadlineExceeded {
		result.TimedOut = true
		result.ExitCode = -1

		// CommandContext invokes the process-group cancellation above. Repeat it
		// after Wait as a best-effort cleanup for a child that raced the timeout.
		if cmd.Process != nil {
			_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
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

		if pathWithin(targetDir, realAllowed) {
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

func withEnvironmentOverrides(env []string, overrides map[string]string) []string {
	result := make([]string, 0, len(env)+len(overrides))
	for _, entry := range env {
		key, _, _ := strings.Cut(entry, "=")
		if _, replaced := overrides[key]; !replaced {
			result = append(result, entry)
		}
	}
	for key, value := range overrides {
		result = append(result, key+"="+value)
	}
	return result
}
