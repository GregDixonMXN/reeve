# AXIOM User Context

## Who I Am
- **Name:** Gregori (The Watcher / Centurion)
- **Timezone:** CST (Oklahoma)
- **Occupation:** Tattoo shop owner & artist, pivoting to game dev & AI development

## Running Axiom
- Dev: `wails dev -tags webkit2_41` (from `/home/shki/projects/axiom`)
- Build: `wails build -clean -tags webkit2_41`
- The `-tags webkit2_41` flag is required — don't omit it

## Working Directories
- All projects live in `/home/shki/projects/`
- Documents: `/home/shki/Documents/`
- Desktop: `/home/shki/Desktop/`
- **Always use absolute paths starting from the above directories**

## Preferred Stack
- **AI/Agents:** Go (orchestration), Python (scripts/tools), Axiom itself
- **Game dev:** Python + Pygame for prototypes, Lua for Playdate (LÖVE/Playdate SDK)
- **Frontend:** SolidJS / React + Vite
- **Backend:** Go, Node.js
- **Default Python:** Use `/home/shki/.axiom/venv/bin/python` for Axiom's embedded tools, otherwise `python3`

## Conventions
- Commit changes with git after completing a task
- Prefer modular file structure over monolithic files
- Write a README.md for every new project
- Use `requirements.txt` for Python projects
- When building games: include a working main entry point that launches without errors

## Current Focus
- Axiom: hybrid AI agent runtime using local Qwen 3.6 and OpenAI cloud routing (this project)
- Totembra: Balatro-like dice roguelike in Pygame (`/home/shki/projects/totembra`)
- Exploring: AI consulting for local service businesses

## Preferences
- Direct, no-filler responses
- Build things that actually run — verify with execute_code before reporting done
- When creating multi-file projects: write ALL files before running anything
