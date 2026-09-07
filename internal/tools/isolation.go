package tools

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os/exec"
)

const sandboxHelperArgument = "__reeve_sandbox_exec"

const (
	defaultSandboxCPUTimeSec     = 60
	defaultSandboxMaxMemoryBytes = int64(8 * 1024 * 1024 * 1024)
	defaultSandboxMaxProcesses   = 4096
	defaultSandboxMaxFileSize    = int64(1024 * 1024 * 1024)
	defaultSandboxMaxOpenFiles   = 1024
)

type isolationResourceLimits struct {
	CPUTimeSec       int   `json:"cpu_time_sec"`
	MaxMemoryBytes   int64 `json:"max_memory_bytes"`
	MaxProcesses     int   `json:"max_processes"`
	MaxFileSizeBytes int64 `json:"max_file_size_bytes"`
	MaxOpenFiles     int   `json:"max_open_files"`
}

type isolationRequest struct {
	AllowedDirs    []string                `json:"allowed_dirs"`
	WorkingDir     string                  `json:"working_dir"`
	Command        []string                `json:"command"`
	AllowNetwork   bool                    `json:"allow_network"`
	ResourceLimits isolationResourceLimits `json:"resource_limits"`
}

// IsolationSupported reports whether this host can enforce Reeve's filesystem
// boundary and, when requested, its no-network policy. Callers should fail
// closed when this returns an error.
func IsolationSupported(allowNetwork bool) error {
	return isolationSupported(allowNetwork)
}

// IsolationHelperPath returns a stable reference to the currently running
// Reeve executable. The path must not be replaceable from a writable workspace
// before the helper installs Landlock and seccomp.
func IsolationHelperPath() (string, error) {
	return isolationHelperPath()
}

// IsSandboxHelperInvocation reports whether the current process was started as
// Reeve's restricted exec helper. main calls this before initializing Wails.
func IsSandboxHelperInvocation(args []string) bool {
	return len(args) == 3 && args[1] == sandboxHelperArgument
}

// RunSandboxHelper decodes and executes a request under the platform's OS
// isolation boundary. On success it replaces the current process and does not
// return.
func RunSandboxHelper(args []string) error {
	if !IsSandboxHelperInvocation(args) {
		return fmt.Errorf("invalid sandbox helper invocation")
	}
	payload, err := base64.RawURLEncoding.DecodeString(args[2])
	if err != nil {
		return fmt.Errorf("decode sandbox request: %w", err)
	}
	var request isolationRequest
	if err := json.Unmarshal(payload, &request); err != nil {
		return fmt.Errorf("parse sandbox request: %w", err)
	}
	if request.WorkingDir == "" || len(request.Command) == 0 || request.Command[0] == "" {
		return fmt.Errorf("sandbox request is missing working_dir or command")
	}
	return runIsolated(request)
}

func newIsolatedCommand(ctx context.Context, helperPath string, request isolationRequest) (*exec.Cmd, error) {
	if helperPath == "" {
		return nil, fmt.Errorf("OS isolation helper path is empty")
	}
	payload, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("encode sandbox request: %w", err)
	}
	encoded := base64.RawURLEncoding.EncodeToString(payload)
	return exec.CommandContext(ctx, helperPath, sandboxHelperArgument, encoded), nil
}
