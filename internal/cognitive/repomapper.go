package cognitive

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// skipNames are directory names excluded at any depth.
var skipNames = map[string]bool{
	"node_modules":  true,
	".git":          true,
	".venv":         true,
	"venv":          true,
	".cache":        true,
	"build":         true,
	"dist":          true,
	"__pycache__":   true,
	"site-packages": true,
}

// repoTreeTTL is how long a cached repo tree is considered fresh.
const repoTreeTTL = 60 * time.Second

// repoTreeCache holds the last-built tree and when it was built.
var repoTreeCache struct {
	sync.RWMutex
	root  string
	tree  string
	built time.Time
}

// BuildRepoTree returns a formatted text tree of the directory structure
// rooted at root, skipping noise directories at any depth.
// Results are cached for repoTreeTTL to avoid repeated filesystem walks
// on every LLM Generate call.
func BuildRepoTree(root string) string {
	if root == "" {
		return ""
	}

	// Fast path: serve from cache if still fresh
	repoTreeCache.RLock()
	if repoTreeCache.root == root && time.Since(repoTreeCache.built) < repoTreeTTL {
		cached := repoTreeCache.tree
		repoTreeCache.RUnlock()
		return cached
	}
	repoTreeCache.RUnlock()

	// Slow path: rebuild the tree
	repoTreeCache.Lock()
	defer repoTreeCache.Unlock()

	// Double-check under write lock (another goroutine may have rebuilt already)
	if repoTreeCache.root == root && time.Since(repoTreeCache.built) < repoTreeTTL {
		return repoTreeCache.tree
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("PROJECT ROOT: %s\n", root))
	buildTree(&sb, root, "")

	repoTreeCache.root = root
	repoTreeCache.tree = sb.String()
	repoTreeCache.built = time.Now()

	return repoTreeCache.tree
}

// BuildShallowListing returns a compact listing of the top-level contents of
// each dir in the provided list. This gives the LLM exact project paths without
// flooding the context window with a deep recursive tree.
func BuildShallowListing(dirs []string) string {
	if len(dirs) == 0 {
		return ""
	}
	var sb strings.Builder
	for _, dir := range dirs {
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		sb.WriteString(fmt.Sprintf("%s/\n", dir))
		// Sort: dirs first, then files
		sort.Slice(entries, func(i, j int) bool {
			di, dj := entries[i].IsDir(), entries[j].IsDir()
			if di != dj {
				return di
			}
			return entries[i].Name() < entries[j].Name()
		})
		for _, e := range entries {
			if skipNames[e.Name()] {
				continue
			}
			if e.IsDir() {
				sb.WriteString(fmt.Sprintf("  %s/\n", e.Name()))
			} else {
				sb.WriteString(fmt.Sprintf("  %s\n", e.Name()))
			}
		}
	}
	return strings.TrimRight(sb.String(), "\n")
}

func buildTree(sb *strings.Builder, dir, prefix string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return
	}

	// Filter skipped dirs by name
	var visible []os.DirEntry
	for _, e := range entries {
		if e.IsDir() && skipNames[e.Name()] {
			continue
		}
		visible = append(visible, e)
	}

	// Sort: directories first, then files, alphabetical within each
	sort.Slice(visible, func(i, j int) bool {
		di, dj := visible[i].IsDir(), visible[j].IsDir()
		if di != dj {
			return di
		}
		return visible[i].Name() < visible[j].Name()
	})

	for i, e := range visible {
		isLast := i == len(visible)-1
		connector := "├── "
		childPrefix := "│   "
		if isLast {
			connector = "└── "
			childPrefix = "    "
		}

		if e.IsDir() {
			sb.WriteString(fmt.Sprintf("%s%s%s/\n", prefix, connector, e.Name()))
			buildTree(sb, dir+"/"+e.Name(), prefix+childPrefix)
		} else {
			sb.WriteString(fmt.Sprintf("%s%s%s\n", prefix, connector, e.Name()))
		}
	}
}
