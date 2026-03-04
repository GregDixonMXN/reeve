.PHONY: dev build clean test setup

dev:
	wails dev

build:
	wails build -clean

test:
	go test ./... -v -race

clean:
	rm -rf build/ frontend/dist/ frontend/node_modules/

setup:
	@echo "── Installing Wails CLI ──"
	go install github.com/wailsapp/wails/v2/cmd/wails@latest
	@echo "── Installing frontend dependencies ──"
	cd frontend && npm install
	@echo "── Creating data directory ──"
	mkdir -p ~/.axiom/models
	@echo ""
	@echo "── Setup complete! ──"
	@echo "Next steps:"
	@echo "  1. Install & start Ollama: curl -fsSL https://ollama.com/install.sh | sh"
	@echo "  2. Pull a model: ollama pull phi4"
	@echo "  3. Copy config: cp axiom.example.toml axiom.toml"
	@echo "  4. Run: make dev"
