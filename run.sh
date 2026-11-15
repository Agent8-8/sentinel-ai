#!/usr/bin/env bash
# One-command reproduce + launch. Optional: export GEMINI_API_KEY=... first.
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -m pytest -q tests
[ -f reports/experiments.json ] || .venv/bin/python -m sentinel.experiments
[ -f artifacts/sentinel_model.pkl ] || .venv/bin/python -m sentinel.evaluate
echo "Dashboard: http://localhost:8000"
exec .venv/bin/uvicorn sentinel.api:app --port 8000
