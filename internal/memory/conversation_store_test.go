package memory

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"axiom/internal/config"
)

func openConversationTestStore(t *testing.T, path string) *Store {
	t.Helper()
	store, err := NewStore(config.DatabaseConfig{Path: path, VecDim: 4}, nil)
	if err != nil {
		t.Fatalf("NewStore: %v", err)
	}
	t.Cleanup(func() {
		if err := store.Close(); err != nil {
			t.Errorf("Close: %v", err)
		}
	})
	return store
}

func TestConversationRecordSurvivesRestart(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "memory.db")
	store := openConversationTestStore(t, dbPath)
	wantTime := time.Date(2026, 7, 18, 12, 30, 0, 0, time.UTC)
	want := ConversationRecord{
		ID:           "conv-1",
		Title:        "Repair Axiom persistence",
		LastActivity: wantTime,
		Messages: []ConversationMessage{
			{Role: "user", Content: "Repair persistence", Timestamp: wantTime.Add(-time.Minute)},
			{Role: "tool", Content: "internal protocol", Timestamp: wantTime.Add(-time.Second), Internal: true},
			{Role: "assistant", Content: "Done", Timestamp: wantTime},
		},
	}
	if err := store.SaveConversationRecord(context.Background(), want); err != nil {
		t.Fatalf("SaveConversationRecord: %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close before restart: %v", err)
	}

	reopened, err := NewStore(config.DatabaseConfig{Path: dbPath, VecDim: 4}, nil)
	if err != nil {
		t.Fatalf("NewStore after restart: %v", err)
	}
	defer reopened.Close()

	summaries, err := reopened.ListConversations(context.Background())
	if err != nil {
		t.Fatalf("ListConversations: %v", err)
	}
	if len(summaries) != 1 {
		t.Fatalf("summaries len = %d, want 1", len(summaries))
	}
	if summaries[0].ID != want.ID || summaries[0].Title != want.Title || summaries[0].MessageCount != 2 {
		t.Fatalf("summary = %#v", summaries[0])
	}

	got, err := reopened.LoadConversationRecord(context.Background(), want.ID)
	if err != nil {
		t.Fatalf("LoadConversationRecord: %v", err)
	}
	if got == nil {
		t.Fatal("LoadConversationRecord returned nil")
	}
	if got.Title != want.Title || len(got.Messages) != len(want.Messages) {
		t.Fatalf("record = %#v", got)
	}
	for i := range want.Messages {
		if got.Messages[i].Role != want.Messages[i].Role || got.Messages[i].Content != want.Messages[i].Content || got.Messages[i].Internal != want.Messages[i].Internal {
			t.Fatalf("message %d = %#v", i, got.Messages[i])
		}
	}
}

func TestWipeAllRemovesConversationRecords(t *testing.T) {
	store := openConversationTestStore(t, filepath.Join(t.TempDir(), "memory.db"))
	if err := store.SaveConversationRecord(context.Background(), ConversationRecord{
		ID:           "conv-1",
		Title:        "Temporary",
		LastActivity: time.Now(),
	}); err != nil {
		t.Fatalf("SaveConversationRecord: %v", err)
	}
	if err := store.WipeAll(); err != nil {
		t.Fatalf("WipeAll: %v", err)
	}
	summaries, err := store.ListConversations(context.Background())
	if err != nil {
		t.Fatalf("ListConversations: %v", err)
	}
	if len(summaries) != 0 {
		t.Fatalf("summaries after wipe = %#v", summaries)
	}
}

func TestStoreRestrictsDatabasePermissions(t *testing.T) {
	path := filepath.Join(t.TempDir(), "memory.db")
	if err := os.WriteFile(path, []byte{}, 0644); err != nil {
		t.Fatal(err)
	}
	store := openConversationTestStore(t, path)
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode().Perm() != 0600 {
		t.Fatalf("database permissions = %o, want 600", info.Mode().Perm())
	}
	if err := store.SaveConversationRecord(context.Background(), ConversationRecord{ID: "mode-check"}); err != nil {
		t.Fatal(err)
	}
	for _, artifact := range []string{path + "-wal", path + "-shm"} {
		if info, err := os.Stat(artifact); err == nil && info.Mode().Perm() != 0600 {
			t.Fatalf("%s permissions = %o, want 600", filepath.Base(artifact), info.Mode().Perm())
		}
	}
}

func TestStoreRejectsUnsupportedEncryptionConfiguration(t *testing.T) {
	path := filepath.Join(t.TempDir(), "memory.db")
	if _, err := NewStore(config.DatabaseConfig{Path: path, Encrypted: true, Passphrase: "secret", VecDim: 4}, nil); err == nil {
		t.Fatal("encrypted configuration silently created a plaintext database")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("database was created despite rejected encryption: %v", err)
	}
}
