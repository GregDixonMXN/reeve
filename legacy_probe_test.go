package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestLegacyAxiomProbe(t *testing.T) {
	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	t.Setenv("HOME", dir)
	if err := os.Chdir(dir); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = os.Chdir(cwd) }()

	if notes := legacyAxiomProbe(); len(notes) != 0 {
		t.Fatalf("expected silence with no legacy files, got %v", notes)
	}

	if err := os.WriteFile(filepath.Join(dir, "axiom.toml"), []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	notes := legacyAxiomProbe()
	if len(notes) != 1 || !strings.Contains(notes[0], "mv axiom.toml reeve.toml") {
		t.Fatalf("expected axiom.toml notice, got %v", notes)
	}

	if err := os.WriteFile(filepath.Join(dir, "reeve.toml"), []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	if notes := legacyAxiomProbe(); len(notes) != 0 {
		t.Fatalf("expected silence once migrated, got %v", notes)
	}
}
