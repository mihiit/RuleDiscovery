**Rule-permutation re-evaluation over 30 independently trained learned models/evaluation seeds.**

| Agent        |   N seeds |     Mean |       SD |   95% CI low |   95% CI high |
|:-------------|----------:|---------:|---------:|-------------:|--------------:|
| random       |        30 | 0        | 0        |     0        |      0        |
| reactive     |        30 | 0.115556 | 0.062962 |     0.094444 |      0.137778 |
| psrl         |        30 | 0.862222 | 0.063568 |     0.838889 |      0.883333 |
| hypothesis   |        30 | 0.98     | 0.025671 |     0.97     |      0.987778 |
| hyp_adaptive |        30 | 0.98     | 0.025671 |     0.97     |      0.987778 |
| hyp_transfer |        30 | 1        | 0        |     1        |      1        |
| macro_ff     |        30 | 0.554444 | 0.115298 |     0.513333 |      0.594444 |
| macro_gru    |        30 | 0.611111 | 0.095224 |     0.576667 |      0.644444 |
| macro_gruB   |        30 | 0.98     | 0.024132 |     0.971111 |      0.987778 |
