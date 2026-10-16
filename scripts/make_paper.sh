#!/bin/sh
# Rebuilds statistics, figures and both papers from the result files (needs pdflatex, bibtex, IEEEtran and llncs classes).
cd "$(dirname "$0")/.." || exit 1
python scripts/final_stats.py > /dev/null
python visualization/final_figures.py
python visualization/ieee6_figures.py
[ -f data/results/causalgame.csv ] && python external/causalgame/analyze.py > /dev/null
python paper/build_final.py
cd paper && for f in main main_ieee6; do pdflatex -interaction=nonstopmode $f.tex >/dev/null; bibtex $f >/dev/null; pdflatex -interaction=nonstopmode $f.tex >/dev/null; pdflatex -interaction=nonstopmode $f.tex >/dev/null; done
