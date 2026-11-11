#!/usr/bin/env bash
# Runs the two LLM configurations on the 60-transcript eval subset (resumable).
# Expect ~2.5 h + ~1.5 h on an Apple M2 (16 GB) with qwen2.5:7b via Ollama.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
.venv/bin/python -m scribe_eval.cli extract --system segmented --rag --subset llm
.venv/bin/python -m scribe_eval.cli extract --system llm --subset llm
