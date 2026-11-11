#!/usr/bin/env bash
# Download the public CMS source files used to build data/reference/.
# Requires curl, unzip and poppler (pdftotext).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/cms/pdgm
cd data/raw/cms
UA="Mozilla/5.0 (oasis-scribe-eval research script)"
curl -sSL -A "$UA" -o oasis_e2_manual.pdf \
  https://www.cms.gov/files/document/final-oasis-e2-manual-04-01-2026.pdf
curl -sSL -A "$UA" -o pdgm_cy2026.zip \
  https://www.cms.gov/files/zip/cy-2026-final-hh-pdgm-case-mix-weights-lupa-thresholds.zip
unzip -oq pdgm_cy2026.zip -d pdgm
pdftotext -layout oasis_e2_manual.pdf manual.txt
echo "Fetched CMS files into data/raw/cms/"
