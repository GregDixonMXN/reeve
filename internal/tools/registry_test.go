package tools

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"axiom/internal/config"
	"axiom/internal/guardrail"
	"axiom/pkg/models"
)

type registryRoundTripFunc func(*http.Request) (*http.Response, error)

func (f registryRoundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

type registryMemoryRecorder struct {
	limit   int
	results []MemoryResult
}

func (m *registryMemoryRecorder) Search(_ context.Context, _ string, limit int) ([]MemoryResult, error) {
	m.limit = limit
	return m.results, nil
}

func testRegistry(allowedDirs ...string) *Registry {
	return NewRegistry(config.ToolsConfig{
		AllowedDirs:    allowedDirs,
		MaxExecTimeSec: 5,
		MaxFileSize:    1024 * 1024,
		MaxOutputBytes: 1024 * 1024,
	}, nil, nil)
}

func TestResolveAllowedPath(t *testing.T) {
	allowed := t.TempDir()
	outside := t.TempDir()

	existing := filepath.Join(allowed, "existing.txt")
	if err := os.WriteFile(existing, []byte("inside"), 0600); err != nil {
		t.Fatal(err)
	}
	insideDir := filepath.Join(allowed, "inside")
	if err := os.Mkdir(insideDir, 0700); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(insideDir, filepath.Join(allowed, "inside-link")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(allowed, "escape")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Join(outside, "missing.txt"), filepath.Join(allowed, "dangling")); err != nil {
		t.Fatal(err)
	}

	r := testRegistry(allowed)
	tests := []struct {
		name    string
		path    string
		want    string
		wantErr bool
	}{
		{name: "existing file", path: existing, want: existing},
		{name: "nonexistent nested write target", path: filepath.Join(allowed, "new", "nested", "file.txt"), want: filepath.Join(allowed, "new", "nested", "file.txt")},
		{name: "symlink within root", path: filepath.Join(allowed, "inside-link", "new.txt"), want: filepath.Join(insideDir, "new.txt")},
		{name: "existing symlink escape", path: filepath.Join(allowed, "escape", "secret.txt"), wantErr: true},
		{name: "dangling symlink escape", path: filepath.Join(allowed, "dangling"), wantErr: true},
		{name: "lexical sibling", path: allowed + "-sibling/file.txt", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := r.resolveAllowedPath(tt.path)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("resolveAllowedPath(%q) unexpectedly returned %q", tt.path, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("resolveAllowedPath(%q): %v", tt.path, err)
			}
			if got != tt.want {
				t.Fatalf("resolveAllowedPath(%q) = %q, want %q", tt.path, got, tt.want)
			}
		})
	}
}

func TestFileAndCloudToolsRejectSymlinkEscape(t *testing.T) {
	allowed := t.TempDir()
	outside := t.TempDir()
	if err := os.WriteFile(filepath.Join(outside, "secret.txt"), []byte("secret"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(outside, "image.png"), []byte("not important"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, filepath.Join(allowed, "escape")); err != nil {
		t.Fatal(err)
	}

	r := testRegistry(allowed)
	escape := filepath.Join(allowed, "escape")
	tests := []struct {
		name string
		call func() error
	}{
		{
			name: "read_file",
			call: func() error {
				_, err := r.readFile(context.Background(), map[string]interface{}{"path": filepath.Join(escape, "secret.txt")})
				return err
			},
		},
		{
			name: "write_file",
			call: func() error {
				_, err := r.writeFile(context.Background(), map[string]interface{}{"path": filepath.Join(escape, "new.txt"), "content": "escaped"})
				return err
			},
		},
		{
			name: "edit_file",
			call: func() error {
				_, err := r.editFile(context.Background(), map[string]interface{}{"path": filepath.Join(escape, "secret.txt"), "old_text": "secret", "new_text": "changed"})
				return err
			},
		},
		{
			name: "list_dir",
			call: func() error {
				_, err := r.listDir(context.Background(), map[string]interface{}{"path": escape})
				return err
			},
		},
		{
			name: "analyze_image",
			call: func() error {
				_, err := r.analyzeImage(context.Background(), map[string]interface{}{"image_path": filepath.Join(escape, "image.png")})
				return err
			},
		},
		{
			name: "ask_cloud_model output_path",
			call: func() error {
				_, err := r.askCloudModel(context.Background(), map[string]interface{}{
					"provider":    "claude",
					"prompt":      "write code",
					"output_path": filepath.Join(escape, "generated.go"),
				})
				return err
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.call(); err == nil {
				t.Fatal("symlink escape was not rejected")
			}
		})
	}

	if _, err := os.Stat(filepath.Join(outside, "new.txt")); !os.IsNotExist(err) {
		t.Fatalf("escaped write created outside file: %v", err)
	}
	secret, err := os.ReadFile(filepath.Join(outside, "secret.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if string(secret) != "secret" {
		t.Fatalf("escaped edit changed outside file to %q", secret)
	}
}

func TestWriteFileAllowsNonexistentInRootTarget(t *testing.T) {
	allowed := t.TempDir()
	r := testRegistry(allowed)
	target := filepath.Join(allowed, "new", "nested", "file.txt")

	if _, err := r.writeFile(context.Background(), map[string]interface{}{"path": target, "content": "safe"}); err != nil {
		t.Fatalf("writeFile: %v", err)
	}
	data, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "safe" {
		t.Fatalf("written content = %q, want safe", data)
	}
}

func TestRootedWriteRejectsSymlinkSwapAfterAuthorization(t *testing.T) {
	allowed := t.TempDir()
	outside := t.TempDir()
	inside := filepath.Join(allowed, "inside")
	if err := os.Mkdir(inside, 0700); err != nil {
		t.Fatal(err)
	}
	r := testRegistry(allowed)
	root, relative, _, err := r.openAllowedRootPath(filepath.Join(inside, "result.txt"))
	if err != nil {
		t.Fatal(err)
	}
	defer root.Close()

	if err := os.Rename(inside, inside+"-old"); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(outside, inside); err != nil {
		t.Fatal(err)
	}
	if err := writeFileToRoot(root, relative, []byte("escaped"), 0600); err == nil {
		t.Fatal("rooted write followed a swapped symlink outside the allowed root")
	}
	if _, err := os.Stat(filepath.Join(outside, "result.txt")); !os.IsNotExist(err) {
		t.Fatalf("outside file was created: %v", err)
	}
}

func TestWriteFileAllowsEmptyContent(t *testing.T) {
	allowed := t.TempDir()
	r := testRegistry(allowed)
	target := filepath.Join(allowed, "empty.txt")
	if _, err := r.writeFile(context.Background(), map[string]interface{}{"path": target, "content": ""}); err != nil {
		t.Fatalf("writeFile empty content: %v", err)
	}
	info, err := os.Stat(target)
	if err != nil {
		t.Fatal(err)
	}
	if info.Size() != 0 {
		t.Fatalf("empty file size = %d", info.Size())
	}
}

func TestListDirOutputIsBounded(t *testing.T) {
	allowed := t.TempDir()
	for i := 0; i < 30; i++ {
		name := filepath.Join(allowed, fmt.Sprintf("entry-%02d-with-a-long-name.txt", i))
		if err := os.WriteFile(name, nil, 0600); err != nil {
			t.Fatal(err)
		}
	}
	r := NewRegistry(config.ToolsConfig{
		AllowedDirs:    []string{allowed},
		MaxExecTimeSec: 5,
		MaxFileSize:    1024,
		MaxOutputBytes: 64,
	}, nil, nil)
	result, err := r.listDir(context.Background(), map[string]interface{}{"path": allowed})
	if err != nil {
		t.Fatal(err)
	}
	if marker := strings.Index(result, "\n\n[TRUNCATED:"); marker != 64 {
		t.Fatalf("retained directory output = %d bytes, want 64: %q", marker, result)
	}
}

func TestSearchMemoryCapsLimitAndOutput(t *testing.T) {
	results := make([]MemoryResult, 25)
	for i := range results {
		results[i] = MemoryResult{Content: strings.Repeat("x", 40), Score: 0.9}
	}
	recorder := &registryMemoryRecorder{results: results}
	r := NewRegistry(config.ToolsConfig{MaxExecTimeSec: 5, MaxOutputBytes: 80}, nil, nil)
	r.SetMemory(recorder)
	result, err := r.searchMemory(context.Background(), map[string]interface{}{
		"query": "axiom",
		"limit": float64(10_000),
	})
	if err != nil {
		t.Fatal(err)
	}
	if recorder.limit != 20 {
		t.Fatalf("memory limit = %d, want 20", recorder.limit)
	}
	if marker := strings.Index(result, "\n\n[TRUNCATED:"); marker != 80 {
		t.Fatalf("retained memory output = %d bytes, want 80: %q", marker, result)
	}
}

func TestPrepareDynamicArgsConstrainsGitCWD(t *testing.T) {
	allowed := t.TempDir()
	outside := t.TempDir()
	if err := os.Symlink(outside, filepath.Join(allowed, "escape")); err != nil {
		t.Fatal(err)
	}
	r := testRegistry(allowed)
	tool := dynamicTool{def: models.ToolDefinition{Name: "git_ops"}}

	normalized, err := r.prepareDynamicArgs(tool, map[string]interface{}{"cwd": allowed, "action": "status"})
	if err != nil {
		t.Fatalf("allowed cwd rejected: %v", err)
	}
	if normalized["cwd"] != allowed {
		t.Fatalf("normalized cwd = %v, want %s", normalized["cwd"], allowed)
	}
	if _, err := r.prepareDynamicArgs(tool, map[string]interface{}{"cwd": filepath.Join(allowed, "escape"), "action": "status"}); err == nil {
		t.Fatal("git_ops accepted a symlink escape")
	}
}

func TestGitToolRequiresAndUsesOSIsolation(t *testing.T) {
	if err := IsolationSupported(false); err != nil {
		t.Skip(err)
	}
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 is unavailable")
	}
	helperPath, err := IsolationHelperPath()
	if err != nil {
		t.Fatal(err)
	}
	repository := t.TempDir()
	initCommand := exec.Command("git", "init", "--quiet", repository)
	if output, err := initCommand.CombinedOutput(); err != nil {
		t.Skipf("git init unavailable: %v (%s)", err, output)
	}

	withoutIsolation := NewRegistry(config.ToolsConfig{AllowedDirs: []string{repository}}, nil, nil)
	if _, present := withoutIsolation.dynamic["git_ops"]; present {
		t.Fatal("git_ops was exposed without OS isolation")
	}

	sandboxConfig := &SandboxConfig{
		Enabled:            true,
		RequireOSIsolation: true,
		AllowNetwork:       false,
		HelperPath:         helperPath,
		AllowedDirs:        []string{repository},
		AllowedBinaries:    []string{"python3"},
		TimeoutSec:         10,
		MaxOutputBytes:     4096,
	}
	r := NewRegistry(config.ToolsConfig{
		PythonPath:     python,
		AllowedDirs:    []string{repository},
		MaxExecTimeSec: 10,
		MaxOutputBytes: 4096,
	}, nil, sandboxConfig)
	tool, present := r.dynamic["git_ops"]
	if !present {
		t.Fatal("git_ops was not exposed with OS isolation")
	}
	result, err := r.executeDynamic(context.Background(), tool, map[string]interface{}{
		"action": "status",
		"cwd":    repository,
	})
	if err != nil {
		t.Fatalf("isolated git status: %v", err)
	}
	if !strings.Contains(result, "No commits yet") {
		t.Fatalf("unexpected git status: %q", result)
	}
}

func TestPrepareDynamicArgsInjectsConfiguredWolframCredential(t *testing.T) {
	r := NewRegistry(config.ToolsConfig{WolframAppID: "configured-id"}, nil, nil)
	normalized, err := r.prepareDynamicArgs(dynamicTool{def: models.ToolDefinition{Name: "wolfram"}}, map[string]interface{}{"query": "1+1"})
	if err != nil {
		t.Fatal(err)
	}
	if normalized["app_id"] != "configured-id" {
		t.Fatalf("app_id = %v", normalized["app_id"])
	}
}

func TestCloudOutputPathIsConditionallyOptional(t *testing.T) {
	r := NewRegistry(config.ToolsConfig{}, &CloudConfig{AnthropicKey: "configured"}, nil)
	tool, ok := r.static["ask_cloud_model"]
	if !ok {
		t.Fatal("ask_cloud_model was not registered")
	}
	if !strings.Contains(strings.ToLower(tool.def.ArgsSchema), `"output_path": "string (optional`) {
		t.Fatalf("output_path schema is not optional: %s", tool.def.ArgsSchema)
	}

	g := guardrail.New(config.SecurityConfig{EnableGuardrails: true, AllowNetwork: true})
	g.RegisterSchema(tool.def.Name, tool.def.ArgsSchema)
	call := &models.ToolCall{
		Name: tool.def.Name,
		Args: map[string]interface{}{"provider": "claude", "prompt": "Explain what happened"},
	}
	if err := g.Check(call); err != nil {
		t.Fatalf("non-code delegation without output_path rejected: %v", err)
	}
}

func TestCloudOutputPathIsRecheckedAfterNetworkCall(t *testing.T) {
	allowed := t.TempDir()
	outside := t.TempDir()
	outputDir := filepath.Join(allowed, "output")
	if err := os.Mkdir(outputDir, 0700); err != nil {
		t.Fatal(err)
	}
	outputPath := filepath.Join(outputDir, "generated.go")

	r := NewRegistry(config.ToolsConfig{
		AllowedDirs:    []string{allowed},
		MaxExecTimeSec: 5,
		MaxFileSize:    1024 * 1024,
		MaxOutputBytes: 1024 * 1024,
	}, &CloudConfig{AnthropicKey: "test", AnthropicModel: "test"}, nil)
	r.cloud.client.Transport = registryRoundTripFunc(func(_ *http.Request) (*http.Response, error) {
		if err := os.Rename(outputDir, outputDir+"-original"); err != nil {
			return nil, err
		}
		if err := os.Symlink(outside, outputDir); err != nil {
			return nil, err
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader(`{"content":[{"type":"text","text":"package main"}]}`)),
		}, nil
	})

	_, err := r.askCloudModel(context.Background(), map[string]interface{}{
		"provider":    "claude",
		"prompt":      "write code",
		"output_path": outputPath,
	})
	if err == nil {
		t.Fatal("cloud output accepted a path changed to an escaping symlink during inference")
	}
	if _, err := os.Stat(filepath.Join(outside, "generated.go")); !os.IsNotExist(err) {
		t.Fatalf("cloud response escaped to outside path: %v", err)
	}
}

func TestPromptLooksLikeCodeTaskUsesWholeWords(t *testing.T) {
	if promptLooksLikeCodeTask("Explain what happened yesterday") {
		t.Fatal("'happened' incorrectly matched the 'app' code keyword")
	}
	if promptLooksLikeCodeTask("Explain this code") {
		t.Fatal("non-generative code explanation was treated as file generation")
	}
	if !promptLooksLikeCodeTask("Build an app") {
		t.Fatal("explicit app-building request was not detected")
	}
	if !promptLooksLikeCodeTask("Please refactor this module") {
		t.Fatal("explicit refactoring request was not detected")
	}
	if !promptLooksLikeCodeTask("Implementing authentication") {
		t.Fatal("inflected implementation request was not detected")
	}
	if !promptLooksLikeCodeTask("Refactoring the parser") {
		t.Fatal("inflected refactoring request was not detected")
	}
}

func TestDynamicToolOutputIsBounded(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 is unavailable")
	}
	dir := t.TempDir()
	script := filepath.Join(dir, "large_output.py")
	if err := os.WriteFile(script, []byte("print('x' * 65536)\n"), 0600); err != nil {
		t.Fatal(err)
	}

	r := NewRegistry(config.ToolsConfig{
		PythonPath:     python,
		MaxExecTimeSec: 5,
		MaxOutputBytes: 256,
	}, nil, nil)
	result, err := r.executeDynamic(context.Background(), dynamicTool{
		def:     models.ToolDefinition{Name: "large_output"},
		runtime: "python",
		script:  script,
	}, map[string]interface{}{})
	if err != nil {
		t.Fatalf("executeDynamic: %v", err)
	}
	marker := strings.Index(result, "\n\n[TRUNCATED:")
	if marker != 256 {
		t.Fatalf("retained output before marker = %d, want 256", marker)
	}
}

func TestEmbeddedPythonToolsIgnoreWorkspaceModules(t *testing.T) {
	python, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("python3 is unavailable")
	}
	workspace := t.TempDir()
	// A trusted embedded tool imports json at startup. This workspace module
	// would hijack that import unless Python is launched in isolated mode.
	if err := os.WriteFile(filepath.Join(workspace, "json.py"), []byte("raise RuntimeError('workspace module loaded')\n"), 0600); err != nil {
		t.Fatal(err)
	}
	r := NewRegistry(config.ToolsConfig{
		PythonPath:     python,
		AllowedDirs:    []string{workspace},
		MaxExecTimeSec: 5,
		MaxOutputBytes: 4096,
	}, nil, nil)
	tool := r.dynamic["web_scrape"]
	result, err := r.executeDynamic(context.Background(), tool, map[string]interface{}{})
	if err != nil {
		t.Fatalf("embedded tool loaded a workspace module: %v", err)
	}
	if !strings.Contains(result, "No URL provided") {
		t.Fatalf("unexpected embedded tool result: %q", result)
	}
}

func TestEmbeddedPythonToolRejectsWorkspaceInterpreter(t *testing.T) {
	workspace := t.TempDir()
	python := filepath.Join(workspace, "python")
	if err := os.WriteFile(python, []byte("#!/bin/sh\nexit 99\n"), 0700); err != nil {
		t.Fatal(err)
	}
	r := NewRegistry(config.ToolsConfig{
		PythonPath:     python,
		AllowedDirs:    []string{workspace},
		MaxExecTimeSec: 5,
		MaxOutputBytes: 4096,
	}, nil, nil)

	_, err := r.executeDynamic(context.Background(), r.dynamic["web_scrape"], map[string]interface{}{})
	if err == nil {
		t.Fatal("embedded helper accepted a Python interpreter from its writable workspace")
	}
	if !strings.Contains(err.Error(), "trusted Python runtime") || !strings.Contains(err.Error(), "writable workspace root") {
		t.Fatalf("unexpected rejection: %v", err)
	}
}

func TestTrustedPythonRuntimeRejectsSymlinkIntoWorkspace(t *testing.T) {
	workspace := t.TempDir()
	python := filepath.Join(workspace, "python")
	if err := os.WriteFile(python, []byte("#!/bin/sh\nexit 99\n"), 0700); err != nil {
		t.Fatal(err)
	}
	trustedDir := t.TempDir()
	alias := filepath.Join(trustedDir, "python")
	if err := os.Symlink(python, alias); err != nil {
		t.Fatal(err)
	}
	r := NewRegistry(config.ToolsConfig{AllowedDirs: []string{workspace}}, nil, nil)

	_, err := r.resolveTrustedPythonRuntime(alias, workspace)
	if err == nil {
		t.Fatal("trusted runtime validation accepted an external alias targeting the writable workspace")
	}
	if !strings.Contains(err.Error(), "writable workspace root") {
		t.Fatalf("unexpected rejection: %v", err)
	}
}

func TestTrustedPythonRuntimeAllowsExternalVirtualEnvironment(t *testing.T) {
	workspace := t.TempDir()
	venv := t.TempDir()
	if err := os.WriteFile(filepath.Join(venv, "pyvenv.cfg"), []byte("home = /usr/bin\n"), 0600); err != nil {
		t.Fatal(err)
	}
	binDir := filepath.Join(venv, "bin")
	if err := os.Mkdir(binDir, 0700); err != nil {
		t.Fatal(err)
	}
	python := filepath.Join(binDir, "python")
	if err := os.WriteFile(python, []byte("#!/bin/sh\nexit 0\n"), 0700); err != nil {
		t.Fatal(err)
	}
	r := NewRegistry(config.ToolsConfig{AllowedDirs: []string{workspace}}, nil, nil)

	got, err := r.resolveTrustedPythonRuntime(python, workspace)
	if err != nil {
		t.Fatalf("external virtual environment rejected: %v", err)
	}
	if got != python {
		t.Fatalf("resolved interpreter = %q, want %q", got, python)
	}
}

func TestRegistryGuardRunsBeforeCacheLookup(t *testing.T) {
	r := NewRegistry(config.ToolsConfig{MaxExecTimeSec: 5}, nil, nil)
	invocations := 0
	r.static["analyze_image"] = staticTool{
		def: models.ToolDefinition{
			Name:       "analyze_image",
			ArgsSchema: `{"image_path":"string"}`,
		},
		fn: func(context.Context, map[string]interface{}) (string, error) {
			invocations++
			return "vision result", nil
		},
	}
	call := &models.ToolCall{
		Name: "analyze_image",
		Args: map[string]interface{}{"image_path": "/tmp/image.png"},
	}

	r.SetGuardrail(guardrail.New(config.SecurityConfig{
		EnableGuardrails: true,
		AllowNetwork:     true,
	}))
	if _, err := r.Execute(context.Background(), call); err != nil {
		t.Fatalf("first Execute: %v", err)
	}
	if result, err := r.Execute(context.Background(), call); err != nil {
		t.Fatalf("cached Execute: %v", err)
	} else if !strings.Contains(result, "[cached]") {
		t.Fatalf("second Execute result = %q, want cached marker", result)
	}
	if invocations != 1 {
		t.Fatalf("tool invoked %d times, want 1 before policy change", invocations)
	}

	// Tighten the policy after the result has been cached. The guard must run
	// before cache lookup, and network policy must apply even with heuristic
	// guardrails disabled.
	r.SetGuardrail(guardrail.New(config.SecurityConfig{
		EnableGuardrails: false,
		AllowNetwork:     false,
	}))
	if result, err := r.Execute(context.Background(), call); err == nil {
		t.Fatalf("cached network result bypassed policy: %q", result)
	}
	if invocations != 1 {
		t.Fatalf("blocked call invoked tool; invocations = %d", invocations)
	}
}

func TestRegistryBlocksNetworkCapableExecutionPaths(t *testing.T) {
	r := NewRegistry(config.ToolsConfig{MaxExecTimeSec: 5}, nil, nil)
	invoked := make(map[string]int)
	for _, name := range []string{"ask_cloud_model", "analyze_image", "execute_code", "git_ops"} {
		toolName := name
		r.static[name] = staticTool{
			def: models.ToolDefinition{Name: name, ArgsSchema: `{}`},
			fn: func(context.Context, map[string]interface{}) (string, error) {
				invoked[toolName]++
				return "ran", nil
			},
		}
	}
	r.SetGuardrail(guardrail.New(config.SecurityConfig{
		EnableGuardrails: false,
		AllowNetwork:     false,
	}))

	blocked := []*models.ToolCall{
		{Name: "ask_cloud_model", Args: map[string]interface{}{}},
		{Name: "analyze_image", Args: map[string]interface{}{}},
		{Name: "execute_code", Args: map[string]interface{}{}},
		{Name: "git_ops", Args: map[string]interface{}{"action": "push"}},
	}
	for _, call := range blocked {
		if result, err := r.Execute(context.Background(), call); err == nil {
			t.Errorf("%s was allowed: %q", call.Name, result)
		}
	}
	for name, count := range invoked {
		if count != 0 {
			t.Errorf("blocked tool %s invoked %d times", name, count)
		}
	}

	if result, err := r.Execute(context.Background(), &models.ToolCall{
		Name: "git_ops",
		Args: map[string]interface{}{"action": "status"},
	}); err == nil {
		t.Fatalf("local git action bypassed network isolation policy: %q", result)
	}
}

func TestSetGuardrailRegistersDynamicSchemas(t *testing.T) {
	r := NewRegistry(config.ToolsConfig{}, nil, nil)
	g := guardrail.New(config.SecurityConfig{EnableGuardrails: true, AllowNetwork: true})
	r.SetGuardrail(g)

	err := g.Check(&models.ToolCall{Name: "web_search", Args: map[string]interface{}{}})
	if err == nil || !strings.Contains(err.Error(), "missing required argument 'query'") {
		t.Fatalf("dynamic tool schema was not enforced: %v", err)
	}
}

func TestDefinitionsAreDeterministicByName(t *testing.T) {
	t.Parallel()

	r := NewRegistry(config.ToolsConfig{}, nil, nil)
	first := r.Definitions()
	second := r.Definitions()
	if len(first) != len(second) {
		t.Fatalf("definition counts differ: %d != %d", len(first), len(second))
	}
	for i := range first {
		if first[i].Name != second[i].Name {
			t.Fatalf("definitions changed order at %d: %q != %q", i, first[i].Name, second[i].Name)
		}
		if i > 0 && first[i-1].Name > first[i].Name {
			t.Fatalf("definitions not sorted: %q precedes %q", first[i-1].Name, first[i].Name)
		}
	}
}

func TestSetGuardrailPrefersTypedToolSchema(t *testing.T) {
	t.Parallel()

	r := NewRegistry(config.ToolsConfig{}, nil, nil)
	r.dynamic["typed_dynamic"] = dynamicTool{def: models.ToolDefinition{
		Name:       "typed_dynamic",
		ArgsSchema: `{"query":"string (optional)"}`,
		Parameters: models.ObjectSchema(map[string]models.JSONSchema{
			"query": {Type: "string"},
		}, "query"),
	}}
	g := guardrail.New(config.SecurityConfig{EnableGuardrails: true, AllowNetwork: true})
	r.SetGuardrail(g)

	err := g.Check(&models.ToolCall{Name: "typed_dynamic", Args: map[string]interface{}{}})
	if err == nil || !strings.Contains(err.Error(), "missing required argument 'query'") {
		t.Fatalf("typed schema was not enforced: %v", err)
	}
}

func TestDefinitionsHideNetworkCapableToolsWhenNetworkDisabled(t *testing.T) {
	r := NewRegistry(config.ToolsConfig{}, nil, &SandboxConfig{Enabled: true})
	r.static["ask_cloud_model"] = staticTool{def: models.ToolDefinition{Name: "ask_cloud_model"}}
	r.static["analyze_image"] = staticTool{def: models.ToolDefinition{Name: "analyze_image"}}
	r.dynamic["wolfram"] = dynamicTool{def: models.ToolDefinition{Name: "wolfram"}}
	r.SetGuardrail(guardrail.New(config.SecurityConfig{AllowNetwork: false}))

	for _, definition := range r.Definitions() {
		switch definition.Name {
		case "web_search", "web_scrape", "wolfram", "ask_cloud_model",
			"analyze_image", "execute_code", "git_ops":
			t.Errorf("network-capable tool %q remained visible", definition.Name)
		}
	}
}
