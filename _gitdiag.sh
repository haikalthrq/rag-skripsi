#!/usr/bin/env bash
# Diagnostic script - writes clean output to a file (avoids tmux log flooding)
OUT=/workspace/rag-skripsi/_gitdiag.txt
{
  echo "=== gh version ==="
  if command -v gh >/dev/null 2>&1; then gh --version; else echo "GH_NOT_INSTALLED"; fi
  echo ""
  echo "=== gh auth status ==="
  gh auth status 2>&1 || echo "GH_AUTH_FAIL_OR_MISSING"
  echo ""
  echo "=== git remote -v ==="
  git -C /workspace/rag-skripsi remote -v 2>&1 || echo "NO_REMOTE"
  echo ""
  echo "=== git branch --show-current ==="
  git -C /workspace/rag-skripsi branch --show-current 2>&1
  echo ""
  echo "=== git config user ==="
  echo "name: $(git -C /workspace/rag-skripsi config user.name)"
  echo "email: $(git -C /workspace/rag-skripsi config user.email)"
  echo ""
  echo "=== git log -5 ==="
  git -C /workspace/rag-skripsi log --oneline -5 2>&1 || echo "NO_LOG"
  echo ""
  echo "=== git status --short (count) ==="
  git -C /workspace/rag-skripsi status --short 2>&1 | wc -l
  echo ""
  echo "=== git status --short (first 60 lines) ==="
  git -C /workspace/rag-skripsi status --short 2>&1 | head -60
} > "$OUT" 2>&1
echo "DIAG_DONE"
