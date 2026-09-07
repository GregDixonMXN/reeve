#!/usr/bin/env python3
"""
Reeve Git Operations Tool
═════════════════════════
Executes git commands in a project directory.
Called by the Go tool registry via stdin JSON.

Input:  {"action": "status|diff|log|add|commit|push|branch|checkout", "args": "...", "cwd": "/path/to/repo"}
Output: Combined stdout+stderr from git command

Safety: The Go registry anchors cwd to a configured workspace and runs this
wrapper plus Git children under Landlock/seccomp isolation.
"""

import json
import os
import subprocess
import sys


ALLOWED_ACTIONS = {
    "status": ["git", "status"],
    "diff": ["git", "diff", "HEAD"],
    "log": ["git", "log", "--oneline", "-10"],
    "add": ["git", "add"],
    "commit": ["git", "-c", "commit.gpgsign=false", "commit", "-m"],
    "push": ["git", "push"],
    "branch": ["git", "branch"],
    "checkout": ["git", "checkout"],
}


def is_safe_cwd(cwd: str) -> bool:
    """Require an existing absolute canonical directory."""
    real_cwd = os.path.realpath(cwd)
    return os.path.isabs(cwd) and cwd == real_cwd and os.path.isdir(real_cwd)


def run_git(action: str, args: str, cwd: str) -> str:
    if action not in ALLOWED_ACTIONS:
        return json.dumps({"error": f"Unknown action: {action}. Supported: {list(ALLOWED_ACTIONS.keys())}"})

    if not is_safe_cwd(cwd):
        return json.dumps({"error": f"cwd '{cwd}' is not an absolute canonical directory"})

    if not os.path.isdir(cwd):
        return json.dumps({"error": f"Directory does not exist: {cwd}"})

    cmd = ALLOWED_ACTIONS[action].copy()

    # Handle actions that need args
    if action in ("add", "commit", "checkout", "branch") and args:
        if action == "commit":
            cmd.append(args)  # commit message
        elif action == "branch" and args.startswith("-"):
            cmd.append(args)  # branch flags like -a, -d
        else:
            cmd.extend(args.split())  # files or branch names
    elif action == "diff" and args:
        # Override default diff target
        cmd = ["git", "diff"] + args.split()
    elif action == "log" and args:
        # Override default log args
        cmd = ["git", "log", "--oneline"] + args.split()

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout + result.stderr
        return output.strip() if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Command timed out after 60 seconds"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def main():
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"error": "No input"}))
        return

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"Invalid JSON: {e}"}))
        return

    action = data.get("action", "")
    args = data.get("args", "")
    cwd = data.get("cwd", "")

    if not action:
        print(json.dumps({"error": "'action' required"}))
        return

    if not cwd:
        print(json.dumps({"error": "'cwd' required"}))
        return

    result = run_git(action, args, cwd)
    print(result)


if __name__ == "__main__":
    main()
