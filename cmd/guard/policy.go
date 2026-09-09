package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/BurntSushi/toml"
)

// Policy is guard's own allow/deny model. Shape mirrors the enforcement
// fields of config.SecurityConfig without importing the app config package
// (split-time: this file moves to the guard module unchanged).
type Policy struct {
	AllowPaths         []string `toml:"allow_paths"`
	DenyGlobs          []string `toml:"deny_globs"`
	AllowNetwork       bool     `toml:"allow_network"`
	AllowBinaries      []string `toml:"allow_binaries"`
	RequireOSIsolation bool     `toml:"require_os_isolation"`
	TimeoutSec         int      `toml:"timeout_sec"`
}

// DefaultSecretGlobs always apply: the recorder-style guarantee that
// secrets are denied even when no policy file is given.
var DefaultSecretGlobs = []string{".env", ".env.*", "*.pem", "**/secrets/**"}

func DefaultPolicy() *Policy {
	return &Policy{RequireOSIsolation: true, TimeoutSec: 30}
}

var knownPolicyKeys = map[string]bool{
	"allow_paths": true, "deny_globs": true, "allow_network": true,
	"allow_binaries": true, "require_os_isolation": true, "timeout_sec": true,
}

func LoadPolicy(path string) (*Policy, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("policy file not found: %s", path)
	}
	// Unknown keys are an error (same rule as annalist gate).
	var raw map[string]any
	if err := toml.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("policy malformed: %w", err)
	}
	for k := range raw {
		if !knownPolicyKeys[k] {
			return nil, fmt.Errorf("policy unknown key: %s", k)
		}
	}
	p := DefaultPolicy()
	if err := toml.Unmarshal(data, p); err != nil {
		return nil, fmt.Errorf("policy malformed: %w", err)
	}
	return p, nil
}

// segMatch is * / ? within one path segment (no slashes).
func segMatch(pattern, name string) bool {
	px, nx := 0, 0
	star, starN := -1, 0
	for nx < len(name) {
		if px < len(pattern) && (pattern[px] == '?' || pattern[px] == name[nx]) {
			px++
			nx++
		} else if px < len(pattern) && pattern[px] == '*' {
			star, starN = px, nx
			px++
		} else if star != -1 {
			starN++
			nx = starN
			px = star + 1
		} else {
			return false
		}
	}
	for px < len(pattern) && pattern[px] == '*' {
		px++
	}
	return px == len(pattern)
}

// globMatch reports whether pattern matches rel (slash-separated, relative).
// Subset: exact paths, * / ? per segment, trailing /**, leading **/.
func globMatch(pattern, rel string) bool {
	if pattern == "**" {
		return true
	}
	if strings.HasSuffix(pattern, "/**") {
		prefix := pattern[:len(pattern)-3]
		if strings.HasPrefix(prefix, "**/") {
			needle := prefix[3:]
			if needle == "" {
				return true
			}
			for off := 0; off <= len(rel); {
				rest := rel[off:]
				if rest == needle {
					return true
				}
				if len(rest) > len(needle) && strings.HasPrefix(rest, needle) && rest[len(needle)] == '/' {
					return true
				}
				i := strings.IndexByte(rest, '/')
				if i < 0 {
					break
				}
				off += i + 1
			}
			return false
		}
		if len(rel) < len(prefix) || !strings.HasPrefix(rel, prefix) {
			return false
		}
		return len(rel) == len(prefix) || rel[len(prefix)] == '/'
	}
	ps := strings.Split(pattern, "/")
	ns := strings.Split(rel, "/")
	if len(ps) != len(ns) {
		// Try suffix alignment so "a" patterns match "x/a" (secret names
		// count at any depth, same rule as the recorder ignores).
		if len(ps) >= len(ns) {
			return false
		}
		ns = ns[len(ns)-len(ps):]
	}
	for i := range ps {
		if ps[i] == "**" {
			return true
		}
		if !segMatch(ps[i], ns[i]) {
			return false
		}
	}
	return true
}

func underDir(path, dir string) bool {
	clean := strings.TrimRight(dir, "/")
	if clean == "" || clean == "." {
		return true
	}
	if path == clean {
		return true
	}
	return strings.HasPrefix(path, clean+"/")
}

// EvalPath returns the deny reason for one path-ish string, or "" when
// allowed. Absolute paths must sit inside cwd; the relative form is then
// checked against deny globs (plus secret defaults) and allow_paths.
func (p *Policy) EvalPath(cwd, s string) string {
	rel := s
	if filepath.IsAbs(s) {
		clean := filepath.Clean(s)
		if clean != cwd && !strings.HasPrefix(clean, cwd+"/") {
			return "outside workspace"
		}
		rel, _ = filepath.Rel(cwd, clean)
	} else {
		rel = filepath.Clean(s)
	}
	rel = filepath.ToSlash(rel)
	for _, g := range p.DenyGlobs {
		if globMatch(g, rel) {
			return "policy deny_glob"
		}
	}
	for _, g := range DefaultSecretGlobs {
		if globMatch(g, rel) {
			return "secret"
		}
	}
	if len(p.AllowPaths) > 0 {
		for _, d := range p.AllowPaths {
			if underDir(rel, filepath.ToSlash(filepath.Clean(d))) {
				return ""
			}
		}
		return "outside allow_paths"
	}
	return ""
}
