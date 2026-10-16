.PHONY: test nonrl train eval paper all
test:   ; pytest -q
nonrl:  ; scripts/run_stages.sh nonrl
train:  ; scripts/train_macro_seeds.sh && scripts/train_generic_all.sh && scripts/train_vault_all.sh
eval:   ; AGENT_SET=rl scripts/run_stages.sh rl && python experiments/run_vault.py
paper:  ; python scripts/analyze.py && python visualization/plots.py && python paper/build_paper.py && cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
all:    ; python scripts/run_all.py
