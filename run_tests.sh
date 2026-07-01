#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest tests/ -v --tb=short --cov=src --cov-report=html:logs/coverage_report
