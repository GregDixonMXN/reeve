// Package scripts embeds Axiom's trusted Python tool entrypoints so packaged
// desktop builds do not depend on the process working directory.
package scripts

import "embed"

// ToolFiles contains only the Python entrypoints registered by the tool
// registry; development/export helpers are intentionally excluded.
//
//go:embed web_search.py web_scrape.py git_ops.py wolfram_query.py
var ToolFiles embed.FS
