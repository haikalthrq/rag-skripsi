#!/bin/bash
set -e

cd /workspace/rag-skripsi

# Setup git config
git config user.email "dev@rag-skripsi.local"
git config user.name "RAG Skripsi Dev"

# Check status
echo "=== Git Status ==="
git status

# Add all changes
echo -e "\n=== Adding all files ==="
git add -A
git status --short

# Commit
echo -e "\n=== Committing ==="
COMMIT_MSG="feat: update rag_chat.py with dual metrics, sidebar history, and formatted output

- Display strict + lenient retrieval metrics simultaneously
- Move chat history to sidebar expander (remove history tab)
- Format output with clear Question/Ground Truth/Generated Answer sections
- Add _compute_both_retrieval_metrics() and _render_retrieval_metrics_both()
- Update _render_history_turn() for new display format
- Remove single-mode chat_relevance_mode radio from sidebar"

git commit -m "$COMMIT_MSG"

# Push to main
echo -e "\n=== Pushing to main ==="
git push origin main

echo -e "\n✅ Push completed!"
