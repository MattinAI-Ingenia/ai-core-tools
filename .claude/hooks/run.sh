#!/bin/bash
# Lanza un hook .ps1 en WSL (powershell.exe + ruta Windows) o en Linux/macOS (pwsh).
f="$CLAUDE_PROJECT_DIR/.claude/hooks/$1"
if command -v wslpath >/dev/null; then
  exec powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$(wslpath -w "$f")"
else
  exec pwsh -NoProfile -File "$f"
fi
