#!/usr/bin/env python3
"""Run git operations: add all, commit, push to main."""
import subprocess
import sys
import os

os.chdir("/workspace/rag-skripsi")

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(f"$ {cmd}")
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.stderr.strip():
        print("[stderr]", r.stderr.strip())
    print(f"[exit {r.returncode}]")
    return r

# 1. Check remote
run("git remote -v")

# 2. Check if remote is correct, set if not
r = run("git remote get-url origin")
if "haikalthrq/rag-skripsi" not in r.stdout:
    run("git remote set-url origin https://github.com/haikalthrq/rag-skripsi.git")

# 3. Status
run("git status --short")

# 4. Stage all
run("git add -A")

# 5. Commit
run('git commit -m "feat: strict+lenient metrics, riwayat chat ke sidebar, format Q/GT/Answer"')

# 6. Push to main
run("git push origin main")

print("\n=== DONE ===")
