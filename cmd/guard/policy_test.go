package main

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"reeve/internal/tools"
)

// TestMain gives the test binary the same sandbox-helper entry point as
// the guard binary: under OS isolation the helper re-exec is /proc/self/exe,
// which during tests is this test binary.
func TestMain(m *testing.M) {
	if tools.IsSandboxHelperInvocation(os.Args) {
		if err := tools.RunSandboxHelper(os.Args); err != nil {
			fmt.Fprintln(os.Stderr, "guard helper:", err)
			os.Exit(1)
		}
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func testPolicy() *Policy {
	return &Policy{
		AllowPaths:         []string{"src/"},
		DenyGlobs:          []string{".env", ".env.*", "*.pem", "**/secrets/**"},
		AllowBinaries:      []string{"ls", "cat", "python3", "git", "sh", "curl"},
		RequireOSIsolation: true,
		TimeoutSec:         30,
	}
}

func TestGlobMatch(t *testing.T) {
	yes := []struct{ pat, path string }{
		{".env", ".env"},
		{".env.*", ".env.local"},
		{"*.pem", "tls/key.pem"},
		{"**/secrets/**", "a/secrets/dump.sql"},
		{"**/secrets/**", "secrets/dump.sql"},
		{"src/**", "src/a/b.go"},
		{"src/**", "src"},
		{".env", "a/.env"},
	}
	for _, tc := range yes {
		if !globMatch(tc.pat, tc.path) {
			t.Errorf("globMatch(%q, %q) = false, want true", tc.pat, tc.path)
		}
	}
	no := []struct{ pat, path string }{
		{"*.pem", "src/main.go"},
		{"**/secrets/**", "src/main.go"},
		{"src/**", "docs/a.md"},
	}
	for _, tc := range no {
		if globMatch(tc.pat, tc.path) {
			t.Errorf("globMatch(%q, %q) = true, want false", tc.pat, tc.path)
		}
	}
}

func TestEvalPath(t *testing.T) {
	p := testPolicy()
	cwd := "/work/proj"
	// NOTE: EvalPath does not skip "-" strings; callers (check/exec) skip
	// flags before calling.
	if got := p.EvalPath(cwd, "src/a.txt"); got != "" {
		t.Errorf("src/a.txt = %q, want allow", got)
	}
	for _, s := range []string{".env", "a/.env.local", "tls/key.pem", "a/secrets/dump.sql"} {
		if got := p.EvalPath(cwd, s); got == "" {
			t.Errorf("%s = allow, want deny", s)
		}
	}
	if got := p.EvalPath(cwd, "notes.txt"); got != "outside allow_paths" {
		t.Errorf("notes.txt = %q, want outside allow_paths", got)
	}
	if got := p.EvalPath(cwd, "/etc/passwd"); got != "outside workspace" {
		t.Errorf("/etc/passwd = %q, want outside workspace", got)
	}
}

func TestLoadPolicyUnknownKey(t *testing.T) {
	dir := t.TempDir()
	bad := filepath.Join(dir, "bad.toml")
	if err := os.WriteFile(bad, []byte("allow_paths = []\nbogus = 1\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadPolicy(bad); err == nil {
		t.Error("LoadPolicy(bogus key) = nil, want error")
	}
	if _, err := LoadPolicy(filepath.Join(dir, "missing.toml")); err == nil {
		t.Error("LoadPolicy(missing) = nil, want error")
	}
	ok := filepath.Join(dir, "ok.toml")
	body := "allow_paths = [\"src/\"]\ndeny_globs = [\".env\"]\nallow_network = false\nallow_binaries = [\"ls\"]\nrequire_os_isolation = true\ntimeout_sec = 30\n"
	if err := os.WriteFile(ok, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	p, err := LoadPolicy(ok)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.AllowPaths) != 1 || p.TimeoutSec != 30 || p.AllowNetwork {
		t.Errorf("policy parsed wrong: %+v", p)
	}
}

// chdirTemp moves the test into a fixture dir (restored afterwards) so
// runCheck/runExec see a controlled workspace.
func chdirTemp(t *testing.T, files map[string]string) {
	t.Helper()
	dir := t.TempDir()
	for name, content := range files {
		full := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	old, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chdir(old) })
}

func TestCheckProofs(t *testing.T) {
	chdirTemp(t, map[string]string{"src/a.txt": "hi", ".env": "SECRET=x"})
	p := testPolicy()
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	// Reeve file tools require absolute paths; the allow list sees the
	// workspace-relative form.
	abs := filepath.Join(cwd, "src", "a.txt")
	if got := runCheck(p, "read_file", `{"path": "`+abs+`"}`); got != 0 {
		t.Errorf("check read_file src = %d, want 0", got)
	}
	if got := runCheck(p, "read_file", `{"path": ".env"}`); got != 2 {
		t.Errorf("check read_file .env = %d, want 2", got)
	}
	if got := runCheck(p, "execute_code", `{"command": "go test ./...", "dir": "/work"}`); got == 0 {
		t.Errorf("check execute_code with chaining-ish command unexpectedly allowed")
	}
}

func TestExecProofs(t *testing.T) {
	chdirTemp(t, map[string]string{"src/a.txt": "hi"})
	p := testPolicy()
	// P1: allowed binary + allowed path.
	if got := runExec(p, []string{"ls", "src"}); got != 0 {
		t.Errorf("exec ls src = %d, want 0", got)
	}
	// P2: secret denied without running.
	if err := os.WriteFile(".env", []byte("SECRET=x"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := runExec(p, []string{"cat", ".env"}); got != 2 {
		t.Errorf("exec cat .env = %d, want 2", got)
	}
	// P3: chained payload cannot escape the sandbox (canary must not exist).
	if got := runExec(p, []string{"sh", "-c", "touch canary-outside"}); got == 0 {
		t.Errorf("exec chained write = 0, want nonzero")
	}
	if _, err := os.Stat("canary-outside"); !os.IsNotExist(err) {
		t.Errorf("chained payload escaped: canary exists")
	}
	// P4: runtime secret writes flip a successful run to deny.
	if err := os.WriteFile("evil.py", []byte("from pathlib import Path\nPath('.env').write_text('x')\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if got := runExec(p, []string{"python3", "evil.py"}); got != 2 {
		t.Errorf("exec runtime secret = %d, want 2", got)
	}
	_ = os.Remove(".env")
	_ = os.Remove("evil.py")
}
