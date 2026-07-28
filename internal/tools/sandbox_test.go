package tools

import (
	"bytes"
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"
)

func TestBoundedBufferCapsRetainedBytes(t *testing.T) {
	const limit = 128
	buffer := newBoundedBuffer(limit)
	payload := bytes.Repeat([]byte("x"), 1024*1024)

	n, err := buffer.Write(payload)
	if err != nil {
		t.Fatal(err)
	}
	if n != len(payload) {
		t.Fatalf("Write reported %d bytes, want %d", n, len(payload))
	}
	if buffer.Len() != limit {
		t.Fatalf("retained %d bytes, want cap %d", buffer.Len(), limit)
	}
	if !strings.Contains(buffer.String(), "[TRUNCATED:") {
		t.Fatal("truncation marker missing")
	}
}

func TestSandboxExecuteUsesOSIsolation(t *testing.T) {
	if err := IsolationSupported(false); err != nil {
		t.Skip(err)
	}
	helperPath, err := IsolationHelperPath()
	if err != nil {
		t.Fatal(err)
	}
	cat, err := exec.LookPath("cat")
	if err != nil {
		t.Skip("cat is unavailable")
	}
	allowed := t.TempDir()
	outside := t.TempDir()
	outsideFile := filepath.Join(outside, "secret.txt")
	if err := os.WriteFile(outsideFile, []byte("outside-secret"), 0600); err != nil {
		t.Fatal(err)
	}

	sandbox := NewSandbox(SandboxConfig{
		Enabled:            true,
		RequireOSIsolation: true,
		AllowNetwork:       false,
		HelperPath:         helperPath,
		AllowedDirs:        []string{allowed},
		AllowedBinaries:    []string{"cat"},
		TimeoutSec:         5,
		MaxOutputBytes:     1024,
	})
	result := sandbox.Execute(context.Background(), cat+" "+outsideFile, allowed)
	if result.Error != "" {
		t.Fatalf("Execute returned infrastructure error: %s", result.Error)
	}
	if result.ExitCode == 0 {
		t.Fatalf("outside file read succeeded: %q", result.Stdout)
	}
	if strings.Contains(result.Stdout, "outside-secret") || !strings.Contains(strings.ToLower(result.Stderr), "permission denied") {
		t.Fatalf("outside file was not denied: stdout=%q stderr=%q", result.Stdout, result.Stderr)
	}
}

func TestSandboxExecuteRunsGoWithEphemeralHome(t *testing.T) {
	if err := IsolationSupported(false); err != nil {
		t.Skip(err)
	}
	helperPath, err := IsolationHelperPath()
	if err != nil {
		t.Fatal(err)
	}
	goBinary, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go is unavailable")
	}
	allowed := t.TempDir()
	sandbox := NewSandbox(SandboxConfig{
		Enabled:            true,
		RequireOSIsolation: true,
		AllowNetwork:       false,
		HelperPath:         helperPath,
		AllowedDirs:        []string{allowed},
		AllowedBinaries:    []string{"go"},
		TimeoutSec:         10,
		MaxOutputBytes:     4096,
	})
	result := sandbox.Execute(context.Background(), goBinary+" version", allowed)
	if result.Error != "" || result.ExitCode != 0 {
		t.Fatalf("isolated Go failed: %s", result.String())
	}
	if !strings.HasPrefix(result.Stdout, "go version ") {
		t.Fatalf("unexpected Go output: %q", result.Stdout)
	}
}

func TestSandboxExecuteBuildsGoProject(t *testing.T) {
	if err := IsolationSupported(false); err != nil {
		t.Skip(err)
	}
	helperPath, err := IsolationHelperPath()
	if err != nil {
		t.Fatal(err)
	}
	goBinary, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go is unavailable")
	}
	project := t.TempDir()
	if err := os.WriteFile(filepath.Join(project, "go.mod"), []byte("module sandboxprobe\n\ngo 1.24\n"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(project, "main.go"), []byte("package main\nfunc main() {}\n"), 0600); err != nil {
		t.Fatal(err)
	}
	sandbox := NewSandbox(SandboxConfig{
		Enabled:            true,
		RequireOSIsolation: true,
		AllowNetwork:       false,
		HelperPath:         helperPath,
		AllowedDirs:        []string{project},
		AllowedBinaries:    []string{"go"},
		TimeoutSec:         20,
		MaxOutputBytes:     4096,
	})
	result := sandbox.Execute(context.Background(), goBinary+" build ./...", project)
	if result.Error != "" || result.ExitCode != 0 {
		t.Fatalf("isolated Go build failed: %s", result.String())
	}
	if _, err := os.Stat(filepath.Join(project, "sandboxprobe")); err != nil {
		t.Fatalf("build output missing: %v", err)
	}
}

func TestSandboxExecuteCapsOutputDuringCapture(t *testing.T) {
	head, err := exec.LookPath("head")
	if err != nil {
		t.Skip("head is unavailable")
	}
	const limit = 256
	sandbox := NewSandbox(SandboxConfig{
		Enabled:         true,
		AllowedDirs:     []string{t.TempDir()},
		AllowedBinaries: []string{"head"},
		TimeoutSec:      5,
		MaxOutputBytes:  limit,
	})

	result := sandbox.Execute(context.Background(), head+" -c 65536 /dev/zero", sandbox.cfg.AllowedDirs[0])
	if result.Error != "" {
		t.Fatalf("Execute: %s", result.Error)
	}
	if result.ExitCode != 0 {
		t.Fatalf("exit code = %d, stderr = %q", result.ExitCode, result.Stderr)
	}
	marker := strings.Index(result.Stdout, "\n\n[TRUNCATED:")
	if marker != limit {
		t.Fatalf("retained output before marker = %d bytes, want %d", marker, limit)
	}
}

func TestSandboxEnvironmentDoesNotTrustVirtualEnv(t *testing.T) {
	t.Setenv("VIRTUAL_ENV", filepath.Join(t.TempDir(), "workspace-venv"))
	sandbox := NewSandbox(SandboxConfig{})
	for _, entry := range sandbox.sanitizedEnv() {
		if strings.HasPrefix(entry, "VIRTUAL_ENV=") {
			t.Fatalf("sandbox environment retained untrusted virtualenv marker: %q", entry)
		}
	}
}

func TestSandboxTimeoutKillsEntireProcessGroup(t *testing.T) {
	sh, err := exec.LookPath("sh")
	if err != nil {
		t.Skip("sh is unavailable")
	}
	allowed := t.TempDir()
	pidPath := filepath.Join(allowed, "child.pid")
	sandbox := NewSandbox(SandboxConfig{
		Enabled:         true,
		AllowedDirs:     []string{allowed},
		AllowedBinaries: []string{"sh"},
		TimeoutSec:      1,
		MaxOutputBytes:  1024,
	})
	result := sandbox.ExecuteArgs(
		context.Background(),
		[]string{sh, "-c", `sleep 30 & child=$!; echo "$child" > "$1"; wait`, "sh", pidPath},
		allowed,
		nil,
	)
	if !result.TimedOut {
		t.Fatalf("command did not time out: %s", result.String())
	}
	rawPID, err := os.ReadFile(pidPath)
	if err != nil {
		t.Fatal(err)
	}
	childPID, err := strconv.Atoi(strings.TrimSpace(string(rawPID)))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = syscall.Kill(childPID, syscall.SIGKILL) }()

	deadline := time.Now().Add(2 * time.Second)
	for {
		err = syscall.Kill(childPID, 0)
		if err == syscall.ESRCH {
			break
		}
		if err != nil {
			t.Fatalf("probe child process %d: %v", childPID, err)
		}
		if time.Now().After(deadline) {
			t.Fatalf("child process %d survived process-group timeout", childPID)
		}
		time.Sleep(20 * time.Millisecond)
	}
}
