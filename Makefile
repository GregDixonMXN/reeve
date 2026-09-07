WAILS_VERSION ?= v2.11.0
WAILS_TAGS ?= webkit2_41
NPM := npm --prefix frontend
GO_BIN := $(shell go env GOBIN)
ifeq ($(strip $(GO_BIN)),)
GO_BIN := $(shell go env GOPATH)/bin
endif
WAILS ?= $(GO_BIN)/wails
PYTHON ?= python3
HERALD_DATA_DIR ?= $(HOME)/.herald
VENV ?= $(HERALD_DATA_DIR)/venv
VENV_PYTHON := $(VENV)/bin/python

.PHONY: dev build clean test setup

dev:
	$(WAILS) dev -tags $(WAILS_TAGS)

build:
	$(WAILS) build -clean -tags $(WAILS_TAGS)

test:
	$(NPM) ci
	$(NPM) run typecheck
	$(NPM) run build
	$(PYTHON) scripts/test_web_scrape.py
	go test ./... -v -race

clean:
	$(NPM) run clean:dist
	rm -rf build/ frontend/node_modules/

setup:
	@echo "── Installing Wails CLI ──"
	go install github.com/wailsapp/wails/v2/cmd/wails@$(WAILS_VERSION)
	@echo "── Installing frontend dependencies ──"
	$(NPM) ci
	@echo "── Installing pinned Python tool dependencies ──"
	$(PYTHON) -m venv "$(VENV)"
	"$(VENV_PYTHON)" -m pip install --requirement requirements.txt
	@echo "── Creating data directory ──"
	mkdir -p "$(HERALD_DATA_DIR)/models"
	@echo ""
	@echo "── Setup complete! ──"
	@echo "Next steps:"
	@echo "  1. Copy config: cp herald.example.toml herald.toml"
	@echo "  2. Export HERALD_OPENAI_API_KEY (or OPENAI_API_KEY)"
	@echo "  3. Run: make dev"
