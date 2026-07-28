//go:build linux

package tools

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

const isolationTestRequestEnv = "AXIOM_ISOLATION_TEST_REQUEST"

func TestMain(m *testing.M) {
	if IsSandboxHelperInvocation(os.Args) {
		if err := RunSandboxHelper(os.Args); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(125)
		}
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func TestIsolationHelperProcess(t *testing.T) {
	encoded := os.Getenv(isolationTestRequestEnv)
	if encoded == "" {
		return
	}
	payload, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(125)
	}
	var request isolationRequest
	if err := json.Unmarshal(payload, &request); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(125)
	}
	if err := runIsolated(request); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(125)
	}
}

func TestIsolationHelperUsesStableSelfExecutable(t *testing.T) {
	path, err := IsolationHelperPath()
	if err != nil {
		t.Fatal(err)
	}
	if path != "/proc/self/exe" {
		t.Fatalf("isolation helper path = %q, want /proc/self/exe", path)
	}
}

func TestNormalizeResourceLimitsFailsClosedToFiniteDefaults(t *testing.T) {
	limits := normalizeResourceLimits(isolationResourceLimits{})
	if limits.CPUTimeSec != defaultSandboxCPUTimeSec ||
		limits.MaxMemoryBytes != defaultSandboxMaxMemoryBytes ||
		limits.MaxProcesses != defaultSandboxMaxProcesses ||
		limits.MaxFileSizeBytes != defaultSandboxMaxFileSize ||
		limits.MaxOpenFiles != defaultSandboxMaxOpenFiles {
		t.Fatalf("unexpected normalized resource limits: %+v", limits)
	}
}

func runIsolationTest(t *testing.T, request isolationRequest) (string, error) {
	t.Helper()
	payload, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestIsolationHelperProcess$")
	cmd.Dir = request.WorkingDir
	cmd.Env = append(os.Environ(), isolationTestRequestEnv+"="+base64.RawURLEncoding.EncodeToString(payload))
	output, err := cmd.CombinedOutput()
	return string(output), err
}

func TestIsolationEnforcesFileSizeLimit(t *testing.T) {
	if err := isolationSupported(false); err != nil {
		t.Skip(err)
	}
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 unavailable")
	}
	allowed := t.TempDir()
	outputPath := filepath.Join(allowed, "oversized.bin")
	output, err := runIsolationTest(t, isolationRequest{
		AllowedDirs:  []string{allowed},
		WorkingDir:   allowed,
		Command:      []string{python, "-c", `import pathlib, sys; pathlib.Path(sys.argv[1]).write_bytes(b"x" * 8192)`, outputPath},
		AllowNetwork: false,
		ResourceLimits: isolationResourceLimits{
			CPUTimeSec:       10,
			MaxMemoryBytes:   512 * 1024 * 1024,
			MaxProcesses:     2048,
			MaxFileSizeBytes: 1024,
			MaxOpenFiles:     256,
		},
	})
	if err == nil {
		t.Fatalf("oversized file write succeeded: %q", output)
	}
	info, statErr := os.Stat(outputPath)
	if statErr != nil && !errors.Is(statErr, os.ErrNotExist) {
		t.Fatal(statErr)
	}
	if statErr == nil && info.Size() > 1024 {
		t.Fatalf("file size = %d, want at most 1024", info.Size())
	}
}

func TestLandlockRestrictsFilesystemToAllowedDirectories(t *testing.T) {
	if err := isolationSupported(false); err != nil {
		t.Skip(err)
	}
	allowed := t.TempDir()
	outside := t.TempDir()
	allowedFile := filepath.Join(allowed, "allowed.txt")
	outsideFile := filepath.Join(outside, "outside.txt")
	if err := os.WriteFile(allowedFile, []byte("allowed-content"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(outsideFile, []byte("outside-secret"), 0600); err != nil {
		t.Fatal(err)
	}

	output, err := runIsolationTest(t, isolationRequest{
		AllowedDirs: []string{allowed},
		WorkingDir:  allowed,
		Command:     []string{"sh", "-c", `cat "$1"; cat "$2"`, "sh", allowedFile, outsideFile},
	})
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
		t.Fatalf("isolated command error = %v, output = %q; want exit code 1", err, output)
	}
	if !strings.Contains(output, "allowed-content") {
		t.Fatalf("allowed file was unreadable: %q", output)
	}
	if strings.Contains(output, "outside-secret") || !strings.Contains(strings.ToLower(output), "permission denied") {
		t.Fatalf("outside file was not denied: %q", output)
	}
}

func TestIsolationRejectsWorkingDirectoryOutsideAllowedRoots(t *testing.T) {
	if err := isolationSupported(false); err != nil {
		t.Skip(err)
	}
	allowed := t.TempDir()
	outside := t.TempDir()
	output, err := runIsolationTest(t, isolationRequest{
		AllowedDirs:  []string{allowed},
		WorkingDir:   outside,
		Command:      []string{"pwd"},
		AllowNetwork: false,
	})
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 125 {
		t.Fatalf("isolation helper error = %v, output = %q; want exit code 125", err, output)
	}
	if !strings.Contains(output, "outside the allowed roots") {
		t.Fatalf("unexpected isolation error: %q", output)
	}
}

func TestSeccompDeniesTCPAndUDPWhenNetworkDisabled(t *testing.T) {
	if err := isolationSupported(false); err != nil {
		t.Skip(err)
	}
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 unavailable")
	}
	allowed := t.TempDir()
	script := `import socket
for kind in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
    try:
        socket.socket(socket.AF_INET, kind)
        print("created")
    except OSError as exc:
        print("denied", exc.errno)
`
	output, err := runIsolationTest(t, isolationRequest{
		AllowedDirs:  []string{allowed},
		WorkingDir:   allowed,
		Command:      []string{python, "-c", script},
		AllowNetwork: false,
	})
	if err != nil {
		t.Fatalf("isolated Python failed: %v\n%s", err, output)
	}
	if strings.Count(output, "denied 1") != 2 || strings.Contains(output, "created") {
		t.Fatalf("TCP/UDP socket creation was not denied: %q", output)
	}
}

func TestSeccompDeniesMetadataMutationOutsideLandlockBoundary(t *testing.T) {
	if err := isolationSupported(false); err != nil {
		t.Skip(err)
	}
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 unavailable")
	}
	allowed := t.TempDir()
	outside := filepath.Join(t.TempDir(), "outside.txt")
	if err := os.WriteFile(outside, []byte("secret"), 0600); err != nil {
		t.Fatal(err)
	}
	output, err := runIsolationTest(t, isolationRequest{
		AllowedDirs: []string{allowed},
		WorkingDir:  allowed,
		Command: []string{python, "-c", `import os, sys
try:
    os.chmod(sys.argv[1], 0o644)
    print("changed")
except OSError as exc:
    print("denied", exc.errno)
`, outside},
		AllowNetwork: false,
	})
	if err != nil {
		t.Fatalf("isolated metadata probe failed: %v\n%s", err, output)
	}
	if !strings.Contains(output, "denied 1") || strings.Contains(output, "changed") {
		t.Fatalf("outside chmod was not denied: %q", output)
	}
	info, err := os.Stat(outside)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0600 {
		t.Fatalf("outside mode changed to %o", info.Mode().Perm())
	}
}

func TestIsolationCanExplicitlyAllowNetwork(t *testing.T) {
	if err := isolationSupported(true); err != nil {
		t.Skip(err)
	}
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 unavailable")
	}
	allowed := t.TempDir()
	output, err := runIsolationTest(t, isolationRequest{
		AllowedDirs:  []string{allowed},
		WorkingDir:   allowed,
		Command:      []string{python, "-c", `import socket; socket.socket(); print("created")`},
		AllowNetwork: true,
	})
	if err != nil {
		t.Fatalf("isolated Python failed: %v\n%s", err, output)
	}
	if !strings.Contains(output, "created") {
		t.Fatalf("network opt-in did not permit socket creation: %q", output)
	}
}

func TestIsolationRunsApprovedDevelopmentRuntime(t *testing.T) {
	if err := isolationSupported(false); err != nil {
		t.Skip(err)
	}
	goBinary, err := exec.LookPath("go")
	if err != nil {
		t.Skip("go is unavailable")
	}
	allowed := t.TempDir()
	output, err := runIsolationTest(t, isolationRequest{
		AllowedDirs:  []string{allowed},
		WorkingDir:   allowed,
		Command:      []string{goBinary, "version"},
		AllowNetwork: false,
	})
	if err != nil {
		t.Fatalf("isolated Go runtime failed: %v\n%s", err, output)
	}
	if !strings.HasPrefix(output, "go version ") {
		t.Fatalf("unexpected Go runtime output: %q", output)
	}
}

func TestIsolationPreservesVirtualEnvironmentInterpreter(t *testing.T) {
	if err := isolationSupported(false); err != nil {
		t.Skip(err)
	}
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 unavailable")
	}
	allowed := t.TempDir()
	venv := filepath.Join(allowed, "venv")
	if output, err := exec.Command(python, "-m", "venv", venv).CombinedOutput(); err != nil {
		t.Skipf("python venv unavailable: %v (%s)", err, output)
	}
	venvPython := filepath.Join(venv, "bin", "python")
	output, err := runIsolationTest(t, isolationRequest{
		AllowedDirs:  []string{allowed},
		WorkingDir:   allowed,
		Command:      []string{venvPython, "-c", `import sys; print(sys.prefix)`},
		AllowNetwork: false,
	})
	if err != nil {
		t.Fatalf("isolated venv Python failed: %v\n%s", err, output)
	}
	if strings.TrimSpace(output) != venv {
		t.Fatalf("isolated interpreter prefix = %q, want %q", strings.TrimSpace(output), venv)
	}
}

func TestReadonlyRuntimePathsIgnoreVirtualEnvEnvironmentVariable(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 unavailable")
	}
	untrustedVenv := t.TempDir()
	t.Setenv("VIRTUAL_ENV", untrustedVenv)
	for _, path := range readonlyRuntimePaths(python, python) {
		if path == untrustedVenv {
			t.Fatalf("read-only runtime paths trusted VIRTUAL_ENV=%q", untrustedVenv)
		}
	}
}
