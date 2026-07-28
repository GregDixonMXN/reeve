package orchestrator

import "testing"

func TestLooksLikeFilePath(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		path string
		want bool
	}{
		{name: "slash path without extension", path: "/foo", want: false},
		{name: "nested path without extension", path: "/tmp/archive", want: false},
		{name: "supported extension", path: "/tmp/main.go", want: true},
		{name: "unsupported extension", path: "/tmp/archive.bin", want: false},
		{name: "relative path", path: "tmp/main.go", want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			if got := looksLikeFilePath(tt.path); got != tt.want {
				t.Fatalf("looksLikeFilePath(%q) = %v, want %v", tt.path, got, tt.want)
			}
		})
	}
}

func TestExtractCodeBlockIgnoresSlashPathWithoutExtension(t *testing.T) {
	t.Parallel()

	path, code := extractCodeBlock("/foo\n```text\nnot a file annotation\n```")
	if path != "" || code != "" {
		t.Fatalf("extractCodeBlock returned (%q, %q), want empty result", path, code)
	}
}
