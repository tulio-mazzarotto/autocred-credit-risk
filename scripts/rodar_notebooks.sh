#!/usr/bin/env bash
# Regenera os .ipynb a partir das fontes .py e executa os quatro notebooks em ordem.
set -euo pipefail
cd "$(dirname "$0")/.."
BIN=.venv/Scripts            # Windows
[ -d "$BIN" ] || BIN=.venv/bin   # Linux e macOS
for nb in 01_diagnostico 02_torneio_modelos 03_modelo_final 04_politica; do
  "$BIN/jupytext" --to ipynb "notebooks/$nb.py"
  "$BIN/jupyter" nbconvert --to notebook --execute --inplace "notebooks/$nb.ipynb" --ExecutePreprocessor.timeout=3600
done
echo "Notebooks executados. Saídas em saidas/"
