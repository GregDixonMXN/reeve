package memory

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
	"sync"
	"time"

	sqlite_vec "github.com/asg017/sqlite-vec-go-bindings/cgo"
	_ "github.com/mattn/go-sqlite3"

	"axiom/internal/config"
)

// ─── Chunking Constants ──────────────────────────────────────────────────────
//
// ChunkSize is the target character length for each memory chunk.
// 1200 chars ≈ ~300 tokens — safely under every common embedding model's limit
// (nomic-embed-text: 8192 tokens, all-minilm: 512 tokens, mxbai-embed-large: 512 tokens).
//
// ChunkOverlap ensures context isn't lost at chunk boundaries — the tail of
// one chunk is prepended to the next.

const (
	ChunkSize    = 1200
	ChunkOverlap = 150
)

// MemoryEntry is a single recalled memory with its similarity score.
type MemoryEntry struct {
	ID        int64
	Content   string
	Metadata  string
	Score     float64
	CreatedAt time.Time
}

// ConversationMessage represents a single message in a conversation.
type ConversationMessage struct {
	Role      string
	Content   string
	Timestamp time.Time
}

// ConversationSummary provides a brief overview of a conversation.
type ConversationSummary struct {
	ID           string
	MessageCount int
	LastActivity time.Time
}

// Store manages Axiom's semantic memory:
//   - Short-term: in-process conversation state (owned by Orchestrator)
//   - Long-term: SQLite + sqlite-vec for vector KNN search
type Store struct {
	mu       sync.RWMutex // Added Mutex for thread-safety during wipes
	db       *sql.DB
	embedder *Embedder
	vecDim   int
}

func init() {
	sqlite_vec.Auto()
}

func NewStore(cfg config.DatabaseConfig, embedder *Embedder) (*Store, error) {
	dsn := cfg.Path
	if dsn == "" {
		dsn = ":memory:"
	}

	db, err := sql.Open("sqlite3", dsn)
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}

	// Verify sqlite-vec loaded
	var vecVersion string
	if err := db.QueryRow("SELECT vec_version()").Scan(&vecVersion); err != nil {
		return nil, fmt.Errorf("sqlite-vec not loaded: %w", err)
	}
	fmt.Printf("[MEMORY] sqlite-vec %s loaded\n", vecVersion)

	// Enable WAL for concurrent read performance
	if _, err := db.Exec("PRAGMA journal_mode=WAL"); err != nil {
		return nil, fmt.Errorf("enable WAL: %w", err)
	}

	dim := cfg.VecDim
	if embedder != nil {
		dim = embedder.Dim()
	}

	s := &Store{db: db, embedder: embedder, vecDim: dim}

	if err := s.migrate(); err != nil {
		return nil, err
	}

	return s, nil
}

func (s *Store) Close() error {
	return s.db.Close()
}

// WipeAll permanently deletes ALL vector data and reclaims disk space.
// This is the "Nuclear Option" for fixing context locks.
func (s *Store) WipeAll() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	// 1. Clear the virtual vector table
	if _, err := tx.Exec("DELETE FROM memory_vec"); err != nil {
		return fmt.Errorf("failed to clear vectors: %w", err)
	}

	// 2. Clear the main content table
	if _, err := tx.Exec("DELETE FROM memories"); err != nil {
		return fmt.Errorf("failed to clear memories: %w", err)
	}

	// 3. Clear conversations history if desired (optional, keeping it clean here)
	if _, err := tx.Exec("DELETE FROM conversations"); err != nil {
		return fmt.Errorf("failed to clear conversations: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return err
	}

	// 4. Run VACUUM to physically shrink the .db file and remove artifacts
	if _, err := s.db.Exec("VACUUM"); err != nil {
		return fmt.Errorf("failed to vacuum database: %w", err)
	}

	return nil
}

func (s *Store) migrate() error {
	schema := `
	CREATE TABLE IF NOT EXISTS memories (
		id          INTEGER PRIMARY KEY AUTOINCREMENT,
		content     TEXT NOT NULL,
		metadata    TEXT DEFAULT '{}',
		source      TEXT DEFAULT 'conversation',
		created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
	);

	CREATE TABLE IF NOT EXISTS conversations (
		id             TEXT PRIMARY KEY,
		last_activity  DATETIME DEFAULT CURRENT_TIMESTAMP
	);

	CREATE TABLE IF NOT EXISTS conversation_messages (
		id         INTEGER PRIMARY KEY AUTOINCREMENT,
		conv_id    TEXT NOT NULL,
		role       TEXT NOT NULL,
		content    TEXT NOT NULL,
		timestamp  DATETIME DEFAULT CURRENT_TIMESTAMP
	);

	CREATE TABLE IF NOT EXISTS projects (
		id          INTEGER PRIMARY KEY AUTOINCREMENT,
		name        TEXT NOT NULL,
		root_path   TEXT NOT NULL,
		description TEXT DEFAULT '',
		created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
	);
	`

	if _, err := s.db.Exec(schema); err != nil {
		return fmt.Errorf("schema migration: %w", err)
	}

	// Ensure conversations table has last_activity column (migration for existing DBs)
	if err := s.ensureColumn("conversations", "last_activity", "DATETIME DEFAULT CURRENT_TIMESTAMP"); err != nil {
		return fmt.Errorf("column migration (last_activity): %w", err)
	}

	// ── Column migrations ─────────────────────────────────────────────────────
	// SQLite's CREATE TABLE IF NOT EXISTS won't add columns to an existing table.
	// We check PRAGMA table_info and ALTER TABLE for any columns added after the
	// initial schema was deployed.
	if err := s.ensureColumn("memories", "source", "TEXT DEFAULT 'conversation'"); err != nil {
		return fmt.Errorf("column migration (source): %w", err)
	}
	if err := s.ensureColumn("memories", "metadata", "TEXT DEFAULT '{}'"); err != nil {
		return fmt.Errorf("column migration (metadata): %w", err)
	}

	// Vector table using sqlite-vec's vec0 virtual table
	vecTable := fmt.Sprintf(`
	CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
		memory_id INTEGER PRIMARY KEY,
		embedding float[%d]
	);`, s.vecDim)

	if _, err := s.db.Exec(vecTable); err != nil {
		return fmt.Errorf("vector table: %w", err)
	}

	return nil
}

// ensureColumn adds a column to a table if it doesn't already exist.
// SQLite doesn't support ALTER TABLE ... ADD COLUMN IF NOT EXISTS until 3.35,
// so we check PRAGMA table_info manually.
func (s *Store) ensureColumn(table, column, definition string) error {
	rows, err := s.db.Query(fmt.Sprintf("PRAGMA table_info(%s)", table))
	if err != nil {
		return err
	}
	defer rows.Close()

	for rows.Next() {
		var cid int
		var name, colType string
		var notNull int
		var dfltVal interface{}
		var pk int
		if err := rows.Scan(&cid, &name, &colType, &notNull, &dfltVal, &pk); err != nil {
			continue
		}
		if name == column {
			return nil // already exists
		}
	}

	// Column missing — add it
	_, err = s.db.Exec(fmt.Sprintf("ALTER TABLE %s ADD COLUMN %s %s", table, column, definition))
	if err != nil {
		return fmt.Errorf("ALTER TABLE %s ADD COLUMN %s: %w", table, column, err)
	}
	fmt.Printf("[MEMORY] Migrated: added column %s.%s\n", table, column)
	return nil
}

// ─── Write Path ─────────────────────────────────────────────────────────────

// Store persists a user/assistant exchange. Large responses are automatically
// chunked so the full content is indexed — no silent truncation by the
// embedding model.
func (s *Store) Store(ctx context.Context, userMsg, assistantMsg string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Build the full exchange string
	full := fmt.Sprintf("User: %s\nAssistant: %s", userMsg, assistantMsg)

	// If small enough, store as a single entry (fast path)
	if len(full) <= ChunkSize {
		if s.embedder != nil {
			emb, err := s.embedder.Embed(ctx, full)
			if err != nil {
				fmt.Printf("[MEMORY] Embedding failed, storing without vector: %v\n", err)
				return s.storeTextOnly(full)
			}
			return s.storeWithVector(full, emb)
		}
		return s.storeTextOnly(full)
	}

	// Large content — chunk and store each piece
	return s.storeChunked(ctx, userMsg, assistantMsg, full)
}

// storeChunked splits a large exchange into overlapping chunks and stores each
// one. The user message is prepended to every chunk as context anchor so each
// chunk is independently searchable without losing the question it answered.
func (s *Store) storeChunked(ctx context.Context, userMsg, assistantMsg, full string) error {
	// Build a compact prefix from the user message.
	// Cap it so the prefix never eats into the chunk budget — if the user
	// message itself is huge, truncate it with an ellipsis.
	const maxPrefixUserLen = 300
	userSummary := userMsg
	if len(userSummary) > maxPrefixUserLen {
		userSummary = userSummary[:maxPrefixUserLen] + "…"
	}
	prefix := fmt.Sprintf("User: %s\nAssistant (excerpt): ", userSummary)

	chunkBudget := ChunkSize - len(prefix)
	if chunkBudget < 200 {
		// Prefix is pathologically large — use full ChunkSize with no prefix
		prefix = "Assistant (excerpt): "
		chunkBudget = ChunkSize - len(prefix)
	}

	chunks := chunkText(assistantMsg, chunkBudget, ChunkOverlap)

	// If chunking produced nothing (very short after all), fall back to full
	if len(chunks) == 0 {
		chunks = []string{full}
	}

	total := len(chunks)
	fmt.Printf("[MEMORY] Large response — storing as %d chunk(s) (prefix=%d chars, overlap=%d chars)\n",
		total, len(prefix), ChunkOverlap)

	if s.embedder == nil {
		// No embedder — store chunks as plain text
		for i, chunk := range chunks {
			content := fmt.Sprintf("%s%s", prefix, chunk)
			meta := fmt.Sprintf(`{"chunk":%d,"total":%d}`, i+1, total)
			if err := s.storeTextOnlyWithMeta(content, "conversation", meta); err != nil {
				return fmt.Errorf("chunk %d/%d store: %w", i+1, total, err)
			}
		}
		return nil
	}

	// Build content strings for all chunks
	contents := make([]string, total)
	for i, chunk := range chunks {
		contents[i] = fmt.Sprintf("%s%s", prefix, chunk)
	}

	// Embed all chunks in a single batch call — one round-trip regardless of count
	embeddings, err := s.embedder.EmbedBatch(ctx, contents)
	if err != nil {
		fmt.Printf("[MEMORY] Batch embedding failed, storing chunks as text-only: %v\n", err)
		for i, content := range contents {
			meta := fmt.Sprintf(`{"chunk":%d,"total":%d}`, i+1, total)
			if err := s.storeTextOnlyWithMeta(content, "conversation", meta); err != nil {
				return fmt.Errorf("chunk %d/%d text store: %w", i+1, total, err)
			}
		}
		return nil
	}

	// Store each chunk with its embedding
	for i, content := range contents {
		meta := fmt.Sprintf(`{"chunk":%d,"total":%d}`, i+1, total)
		if err := s.storeWithVectorAndMeta(content, "conversation", meta, embeddings[i]); err != nil {
			return fmt.Errorf("chunk %d/%d vector store: %w", i+1, total, err)
		}
	}

	return nil
}

// chunkText splits text into overlapping chunks of at most maxSize characters.
// It tries to split at paragraph boundaries (double newline), then sentence
// boundaries (". "), then falls back to hard character splits.
func chunkText(text string, maxSize, overlap int) []string {
	if maxSize <= 0 {
		maxSize = ChunkSize
	}
	if overlap < 0 {
		overlap = 0
	}
	if len(text) <= maxSize {
		return []string{text}
	}

	var chunks []string
	start := 0

	for start < len(text) {
		end := start + maxSize
		if end >= len(text) {
			chunks = append(chunks, text[start:])
			break
		}

		// Try to break at a paragraph boundary within the last 20% of the chunk
		breakAt := end
		searchFrom := start + (maxSize * 4 / 5) // search in last 20%

		if idx := strings.LastIndex(text[searchFrom:end], "\n\n"); idx >= 0 {
			breakAt = searchFrom + idx + 2 // include the newlines
		} else if idx := strings.LastIndex(text[searchFrom:end], ". "); idx >= 0 {
			breakAt = searchFrom + idx + 2 // include ". "
		} else if idx := strings.LastIndex(text[searchFrom:end], "\n"); idx >= 0 {
			breakAt = searchFrom + idx + 1
		}
		// If none found, hard-cut at maxSize

		chunks = append(chunks, text[start:breakAt])

		// Next chunk starts overlap chars before the break
		next := breakAt - overlap
		if next <= start {
			next = breakAt // safety: always advance
		}
		start = next
	}

	return chunks
}

func (s *Store) StoreDocument(ctx context.Context, content, source, metadata string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.embedder == nil {
		return s.storeTextOnlyWithMeta(content, source, metadata)
	}

	embedding, err := s.embedder.Embed(ctx, content)
	if err != nil {
		return s.storeTextOnlyWithMeta(content, source, metadata)
	}

	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	res, err := tx.Exec(
		"INSERT INTO memories (content, source, metadata) VALUES (?, ?, ?)",
		content, source, metadata,
	)
	if err != nil {
		return err
	}

	memID, err := res.LastInsertId()
	if err != nil {
		return err
	}

	serialized, err := sqlite_vec.SerializeFloat32(embedding)
	if err != nil {
		return fmt.Errorf("serialize vector: %w", err)
	}

	_, err = tx.Exec(
		"INSERT INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
		memID, serialized,
	)
	if err != nil {
		return fmt.Errorf("vector insert: %w", err)
	}

	return tx.Commit()
}

func (s *Store) storeWithVector(content string, embedding []float32) error {
	return s.storeWithVectorAndMeta(content, "conversation", "{}", embedding)
}

func (s *Store) storeWithVectorAndMeta(content, source, metadata string, embedding []float32) error {
	tx, err := s.db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	res, err := tx.Exec(
		"INSERT INTO memories (content, source, metadata) VALUES (?, ?, ?)",
		content, source, metadata,
	)
	if err != nil {
		return err
	}

	memID, err := res.LastInsertId()
	if err != nil {
		return err
	}

	serialized, err := sqlite_vec.SerializeFloat32(embedding)
	if err != nil {
		return fmt.Errorf("serialize vector: %w", err)
	}

	_, err = tx.Exec(
		"INSERT INTO memory_vec (memory_id, embedding) VALUES (?, ?)",
		memID, serialized,
	)
	if err != nil {
		return fmt.Errorf("vector insert: %w", err)
	}

	return tx.Commit()
}

func (s *Store) storeTextOnly(content string) error {
	_, err := s.db.Exec("INSERT INTO memories (content) VALUES (?)", content)
	return err
}

func (s *Store) storeTextOnlyWithMeta(content, source, metadata string) error {
	_, err := s.db.Exec(
		"INSERT INTO memories (content, source, metadata) VALUES (?, ?, ?)",
		content, source, metadata,
	)
	return err
}

// ─── Read Path ─────────────────────────────────────────────────────────────

func (s *Store) Search(ctx context.Context, query string, topK int) ([]MemoryEntry, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.embedder == nil {
		return s.textSearch(query, topK)
	}

	queryVec, err := s.embedder.Embed(ctx, query)
	if err != nil {
		fmt.Printf("[MEMORY] Query embedding failed, falling back to text search: %v\n", err)
		return s.textSearch(query, topK)
	}

	return s.vectorSearch(queryVec, topK)
}

func (s *Store) vectorSearch(queryVec []float32, topK int) ([]MemoryEntry, error) {
	serialized, err := sqlite_vec.SerializeFloat32(queryVec)
	if err != nil {
		return nil, fmt.Errorf("serialize query: %w", err)
	}

	rows, err := s.db.Query(`
		SELECT
			m.id,
			m.content,
			m.metadata,
			m.created_at,
			v.distance
		FROM memory_vec v
		JOIN memories m ON m.id = v.memory_id
		WHERE v.embedding MATCH ? AND k = ?
		ORDER BY v.distance
	`, serialized, topK)
	if err != nil {
		return nil, fmt.Errorf("vector search: %w", err)
	}
	defer rows.Close()

	var results []MemoryEntry
	for rows.Next() {
		var e MemoryEntry
		var distance float64
		if err := rows.Scan(&e.ID, &e.Content, &e.Metadata, &e.CreatedAt, &distance); err != nil {
			continue
		}
		e.Score = 1.0 / (1.0 + distance)
		results = append(results, e)
	}

	return results, nil
}

func (s *Store) textSearch(query string, topK int) ([]MemoryEntry, error) {
	rows, err := s.db.Query(`
		SELECT id, content, metadata, created_at
		FROM memories
		WHERE content LIKE ?
		ORDER BY created_at DESC
		LIMIT ?
	`, "%"+query+"%", topK)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []MemoryEntry
	for rows.Next() {
		var e MemoryEntry
		if err := rows.Scan(&e.ID, &e.Content, &e.Metadata, &e.CreatedAt); err != nil {
			continue
		}
		e.Score = 0.5
		results = append(results, e)
	}
	return results, nil
}

// ─── Stats ──────────────────────────────────────────────────────────────────

func (s *Store) MemoryCount() (int, error) {
	var count int
	err := s.db.QueryRow("SELECT COUNT(*) FROM memories").Scan(&count)
	return count, err
}

func (s *Store) VectorCount() (int, error) {
	var count int
	err := s.db.QueryRow("SELECT COUNT(*) FROM memory_vec").Scan(&count)
	return count, err
}

// ─── Conversation Persistence ───────────────────────────────────────────────

// SaveConversation persists a conversation's messages to SQLite.
func (s *Store) SaveConversation(ctx context.Context, id string, messages []ConversationMessage) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	// Upsert conversation record
	_, err = tx.ExecContext(ctx, `
		INSERT INTO conversations (id, last_activity) VALUES (?, CURRENT_TIMESTAMP)
		ON CONFLICT(id) DO UPDATE SET last_activity = CURRENT_TIMESTAMP
	`, id)
	if err != nil {
		return fmt.Errorf("upsert conversation: %w", err)
	}

	// Clear existing messages and insert fresh
	_, err = tx.ExecContext(ctx, "DELETE FROM conversation_messages WHERE conv_id = ?", id)
	if err != nil {
		return fmt.Errorf("clear messages: %w", err)
	}

	for _, msg := range messages {
		_, err = tx.ExecContext(ctx,
			"INSERT INTO conversation_messages (conv_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
			id, msg.Role, msg.Content, msg.Timestamp,
		)
		if err != nil {
			return fmt.Errorf("insert message: %w", err)
		}
	}

	return tx.Commit()
}

// LoadConversation retrieves a conversation's messages from SQLite.
func (s *Store) LoadConversation(ctx context.Context, id string) ([]ConversationMessage, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.QueryContext(ctx, `
		SELECT role, content, timestamp
		FROM conversation_messages
		WHERE conv_id = ?
		ORDER BY id ASC
	`, id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var messages []ConversationMessage
	for rows.Next() {
		var msg ConversationMessage
		if err := rows.Scan(&msg.Role, &msg.Content, &msg.Timestamp); err != nil {
			continue
		}
		messages = append(messages, msg)
	}
	return messages, nil
}

// ListConversations returns summaries of all stored conversations.
func (s *Store) ListConversations(ctx context.Context) ([]ConversationSummary, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	rows, err := s.db.QueryContext(ctx, `
		SELECT c.id, c.last_activity, COUNT(m.id) as msg_count
		FROM conversations c
		LEFT JOIN conversation_messages m ON c.id = m.conv_id
		GROUP BY c.id
		ORDER BY c.last_activity DESC
	`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var summaries []ConversationSummary
	for rows.Next() {
		var s ConversationSummary
		if err := rows.Scan(&s.ID, &s.LastActivity, &s.MessageCount); err != nil {
			continue
		}
		summaries = append(summaries, s)
	}
	return summaries, nil
}
