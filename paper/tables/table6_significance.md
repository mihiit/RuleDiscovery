**Pre-declared significance tests (diff = first minus second; Holm-corrected over the original test family).**

| Experiment     | Comparison                                               |     diff |   welch_p |   mwu_p |   cohens_d |   welch_p_holm |   mwu_p_holm |
|:---------------|:---------------------------------------------------------|---------:|----------:|--------:|-----------:|---------------:|-------------:|
| scaling        | hypothesis vs reactive, d=1, success                     |   0      |  nan      |  1      |     0      |         1      |       1      |
| scaling        | hypothesis vs psrl, d=1, success                         |   0      |  nan      |  1      |     0      |         1      |       1      |
| scaling        | hypothesis vs macro_gru, d=1, success                    |   0      |  nan      |  1      |     0      |         1      |       1      |
| scaling        | hypothesis vs macro_ff, d=1, success                     |   0      |  nan      |  1      |     0      |         1      |       1      |
| scaling        | hypothesis vs generic_gru, d=1, success                  |   0      |  nan      |  1      |     0      |         1      |       1      |
| scaling        | hypothesis vs generic_ff, d=1, success                   |   0      |  nan      |  1      |     0      |         1      |       1      |
| scaling        | hypothesis vs ppo_gru, d=1, success                      |   0.8333 |    0      |  0      |    12.0761 |         0      |       0      |
| scaling        | hypothesis vs ppo_ff, d=1, success                       |   0.68   |    0      |  0      |     9.4821 |         0      |       0      |
| scaling        | macro_gru vs generic_gru, d=1, success                   |   0      |  nan      |  1      |     0      |         1      |       1      |
| scaling        | macro_gru vs generic_ff, d=1, success                    |   0      |  nan      |  1      |     0      |         1      |       1      |
| scaling        | hypothesis vs reactive, d=1, steps-to-success            |  -2.1033 |    0      |  0      |    -2.7153 |         0      |       0      |
| scaling        | hypothesis vs reactive, d=2, success                     |   0.0133 |    0.0014 |  0.0006 |     0.9103 |         0.0841 |       0.0396 |
| scaling        | hypothesis vs psrl, d=2, success                         |   0      |  nan      |  1      |     0      |         1      |       1      |
| scaling        | hypothesis vs macro_gru, d=2, success                    |   0      |  nan      |  1      |     0      |         1      |       1      |
| scaling        | hypothesis vs macro_ff, d=2, success                     |   0      |  nan      |  1      |     0      |         1      |       1      |
| scaling        | hypothesis vs generic_gru, d=2, success                  |   0      |  nan      |  1      |     0      |         1      |       1      |
| scaling        | hypothesis vs generic_ff, d=2, success                   |   0      |  nan      |  1      |     0      |         1      |       1      |
| scaling        | hypothesis vs ppo_gru, d=2, success                      |   0.9967 |    0      |  0      |   109.179  |         0      |       0      |
| scaling        | hypothesis vs ppo_ff, d=2, success                       |   0.99   |    0      |  0      |    67.6299 |         0      |       0      |
| scaling        | macro_gru vs generic_gru, d=2, success                   |   0      |  nan      |  1      |     0      |         1      |       1      |
| scaling        | macro_gru vs generic_ff, d=2, success                    |   0      |  nan      |  1      |     0      |         1      |       1      |
| scaling        | hypothesis vs reactive, d=2, steps-to-success            |  -2.9039 |    0      |  0      |    -2.8155 |         0      |       0      |
| scaling        | hypothesis vs reactive, d=3, success                     |   0.5844 |    0      |  0      |     8.6908 |         0      |       0      |
| scaling        | hypothesis vs psrl, d=3, success                         |  -0.0011 |    0.8601 |  0.8209 |    -0.0457 |         1      |       1      |
| scaling        | hypothesis vs macro_gru, d=3, success                    |   0.0022 |    0.7384 |  0.8094 |     0.0866 |         1      |       1      |
| scaling        | hypothesis vs macro_ff, d=3, success                     |   0.0011 |    0.8678 |  0.9933 |     0.0432 |         1      |       1      |
| scaling        | hypothesis vs generic_gru, d=3, success                  |   0.0411 |    0.0001 |  0.0001 |     1.1547 |         0.0035 |       0.0069 |
| scaling        | hypothesis vs generic_ff, d=3, success                   |   0.0233 |    0.0013 |  0.0012 |     0.8737 |         0.0785 |       0.0711 |
| scaling        | hypothesis vs ppo_gru, d=3, success                      |   0.9811 |    0      |  0      |    57.1826 |         0      |       0      |
| scaling        | hypothesis vs ppo_ff, d=3, success                       |   0.9811 |    0      |  0      |    57.1826 |         0      |       0      |
| scaling        | macro_gru vs generic_gru, d=3, success                   |   0.0389 |    0.0001 |  0.0002 |     1.0637 |         0.0097 |       0.0155 |
| scaling        | macro_gru vs generic_ff, d=3, success                    |   0.0211 |    0.0049 |  0.0027 |     0.7548 |         0.2865 |       0.1555 |
| scaling        | hypothesis vs reactive, d=3, steps-to-success            | -13.2442 |    0      |  0      |    -1.8175 |         0      |       0      |
| scaling        | hypothesis vs reactive, d=4, success                     |   0.85   |    0      |  0      |    18.6958 |         0      |       0      |
| scaling        | hypothesis vs psrl, d=4, success                         |   0.0511 |    0.0001 |  0.0001 |     1.1254 |         0.0038 |       0.0072 |
| scaling        | hypothesis vs macro_gru, d=4, success                    |   0.1522 |    0      |  0      |     2.3809 |         0      |       0      |
| scaling        | hypothesis vs macro_ff, d=4, success                     |   0.1733 |    0      |  0      |     2.7393 |         0      |       0      |
| scaling        | hypothesis vs generic_gru, d=4, success                  |   0.3689 |    0      |  0      |     5.2182 |         0      |       0      |
| scaling        | hypothesis vs generic_ff, d=4, success                   |   0.2689 |    0      |  0      |     4.0086 |         0      |       0      |
| scaling        | hypothesis vs ppo_gru, d=4, success                      |   0.93   |    0      |  0      |    31.7808 |         0      |       0      |
| scaling        | hypothesis vs ppo_ff, d=4, success                       |   0.93   |    0      |  0      |    31.7808 |         0      |       0      |
| scaling        | macro_gru vs generic_gru, d=4, success                   |   0.2167 |    0      |  0      |     2.5234 |         0      |       0      |
| scaling        | macro_gru vs generic_ff, d=4, success                    |   0.1167 |    0      |  0      |     1.4071 |         0.0001 |       0.0005 |
| scaling        | hypothesis vs reactive, d=4, steps-to-success            |  -8.6475 |    0.1444 |  0.5184 |    -0.4007 |         1      |       1      |
| scaling        | hypothesis vs reactive, d=5, success                     |   0.6611 |    0      |  0      |    11.0981 |         0      |       0      |
| scaling        | hypothesis vs psrl, d=5, success                         |   0.08   |    0.0002 |  0.0005 |     1.0474 |         0.0108 |       0.0329 |
| scaling        | hypothesis vs macro_gru, d=5, success                    |   0.1856 |    0      |  0      |     2.7176 |         0      |       0      |
| scaling        | hypothesis vs macro_ff, d=5, success                     |   0.1656 |    0      |  0      |     2.4387 |         0      |       0      |
| scaling        | hypothesis vs generic_gru, d=5, success                  |   0.3833 |    0      |  0      |     5.3981 |         0      |       0      |
| scaling        | hypothesis vs generic_ff, d=5, success                   |   0.2733 |    0      |  0      |     3.3129 |         0      |       0      |
| scaling        | hypothesis vs ppo_gru, d=5, success                      |   0.7356 |    0      |  0      |    16.9877 |         0      |       0      |
| scaling        | hypothesis vs ppo_ff, d=5, success                       |   0.7356 |    0      |  0      |    16.9877 |         0      |       0      |
| scaling        | macro_gru vs generic_gru, d=5, success                   |   0.1978 |    0      |  0      |     2.5629 |         0      |       0      |
| scaling        | macro_gru vs generic_ff, d=5, success                    |   0.0878 |    0.0003 |  0.0006 |     0.999  |         0.0184 |       0.0396 |
| scaling        | hypothesis vs reactive, d=5, steps-to-success            |   2.9392 |    0.5397 |  0.729  |     0.1738 |         1      |       1      |
| generalization | A_interpolation: hypothesis vs reactive, test SR         |   0.1722 |    0      |  0      |     4.462  |         0      |       0      |
| generalization | A_interpolation: hypothesis vs psrl, test SR             |   0      |    1      |  1      |     0      |         1      |       1      |
| generalization | A_interpolation: hypothesis vs macro_gru, test SR        |   0.0089 |    0.025  |  0.0233 |     0.605  |         1      |       1      |
| generalization | A_interpolation: hypothesis vs macro_ff, test SR         |   0.0078 |    0.0439 |  0.0451 |     0.5399 |         1      |       1      |
| generalization | A_interpolation: hypothesis vs reactive, gap             |  -0.1556 |    0      |  0      |    -3.9919 |         0      |       0      |
| generalization | A_interpolation: hyp_transfer vs hypothesis, test SR     |  -0.0022 |    0.3097 |  0.3128 |    -0.2651 |         1      |       1      |
| generalization | B_compositional: hypothesis vs reactive, test SR         |   0.5689 |    0      |  0      |     9.7458 |         0      |       0      |
| generalization | B_compositional: hypothesis vs psrl, test SR             |  -0.0022 |    0.796  |  0.8226 |    -0.0671 |         1      |       1      |
| generalization | B_compositional: hypothesis vs macro_gru, test SR        |   0.0156 |    0.0884 |  0.0561 |     0.4475 |         1      |       1      |
| generalization | B_compositional: hypothesis vs macro_ff, test SR         |   0.0089 |    0.3321 |  0.3015 |     0.2525 |         1      |       1      |
| generalization | B_compositional: hypothesis vs reactive, gap             |  -0.3933 |    0      |  0      |    -5.5477 |         0      |       0      |
| generalization | B_compositional: hyp_transfer vs hypothesis, test SR     |  -0.0011 |    0.9001 |  0.9048 |    -0.0325 |         1      |       1      |
| generalization | C_chain_length: hypothesis vs reactive, test SR          |   0.4511 |    0      |  0      |     6.1145 |         0      |       0      |
| generalization | C_chain_length: hypothesis vs psrl, test SR              |   0.0067 |    0.1235 |  0.177  |     0.4052 |         1      |       1      |
| generalization | C_chain_length: hypothesis vs macro_gru, test SR         |   0.0711 |    0      |  0      |     2.3615 |         0      |       0      |
| generalization | C_chain_length: hypothesis vs macro_ff, test SR          |   0.0956 |    0      |  0      |     2.3897 |         0      |       0      |
| generalization | C_chain_length: hypothesis vs reactive, gap              |  -0.4156 |    0      |  0      |    -5.237  |         0      |       0      |
| generalization | C_chain_length: hyp_transfer vs hypothesis, test SR      |  -0.0033 |    0.3744 |  0.4713 |    -0.2314 |         1      |       1      |
| generalization | C2_size_only: hypothesis vs reactive, test SR            |   0.4089 |    0      |  0      |     6.1528 |         0      |       0      |
| generalization | C2_size_only: hypothesis vs psrl, test SR                |   0      |    1      |  1      |     0      |         1      |       1      |
| generalization | C2_size_only: hypothesis vs macro_gru, test SR           |   0.0056 |    0.0739 |  0.075  |     0.4719 |         1      |       1      |
| generalization | C2_size_only: hypothesis vs macro_ff, test SR            |   0.0044 |    0.1342 |  0.1352 |     0.3933 |         1      |       1      |
| generalization | C2_size_only: hypothesis vs reactive, gap                |   0.1833 |    0      |  0      |     1.7083 |         0      |       0      |
| generalization | C2_size_only: hyp_transfer vs hypothesis, test SR        |   0      |    1      |  1      |     0      |         1      |       1      |
| generalization | E_novel_combination: hypothesis vs reactive, test SR     |   0.5567 |    0      |  0      |     9.2553 |         0      |       0      |
| generalization | E_novel_combination: hypothesis vs psrl, test SR         |  -0.0022 |    0.8026 |  0.7411 |    -0.0648 |         1      |       1      |
| generalization | E_novel_combination: hypothesis vs macro_gru, test SR    |   0.0122 |    0.1955 |  0.1679 |     0.3382 |         1      |       1      |
| generalization | E_novel_combination: hypothesis vs macro_ff, test SR     |   0.0067 |    0.4685 |  0.4781 |     0.1884 |         1      |       1      |
| generalization | E_novel_combination: hypothesis vs reactive, gap         |  -0.52   |    0      |  0      |    -7.9678 |         0      |       0      |
| generalization | E_novel_combination: hyp_transfer vs hypothesis, test SR |  -0.0011 |    0.9001 |  0.9048 |    -0.0325 |         1      |       1      |
| ablation       | full_memory vs no_memory, success                        |   0.9278 |    0      |  0      |    27.7934 |         0      |       0      |
| ablation       | full_memory vs fixed_memory_3, success                   |   0.0211 |    0.0868 |  0.1    |     0.4498 |         1      |       1      |
| ablation       | info_driven vs random_explore, success                   |   0.1133 |    0      |  0      |     2.1464 |         0      |       0      |
| ablation       | revision_on vs revision_off, success                     |   0.7    |    0      |  0      |    10.584  |         0      |       0      |
| ablation       | lambda_0.25 vs lambda_0.0, success (d4)                  |   0.0122 |    0.3366 |  0.3842 |     0.2502 |         1      |       1      |
| ablation       | lambda_0.25_d5 vs lambda_0.0_d5, success (d5)            |   0.03   |    0.1409 |  0.127  |     0.3856 |         1      |       1      |
| ablation       | lambda_0.5 vs lambda_0.0, success (d4)                   |   0.0144 |    0.2507 |  0.3029 |     0.2996 |         1      |       1      |
| ablation       | lambda_0.5_d5 vs lambda_0.0_d5, success (d5)             |   0.0589 |    0.0061 |  0.0079 |     0.7353 |         0.3474 |       0.4521 |
| ablation       | lambda_1.0 vs lambda_0.0, success (d4)                   |   0.0222 |    0.0696 |  0.1155 |     0.4775 |         1      |       1      |
| ablation       | lambda_1.0_d5 vs lambda_0.0_d5, success (d5)             |   0.0778 |    0.0002 |  0.0003 |     1.0416 |         0.0108 |       0.02   |
| ablation       | lambda_2.0 vs lambda_0.0, success (d4)                   |   0.0167 |    0.1797 |  0.2328 |     0.3507 |         1      |       1      |
| ablation       | lambda_2.0_d5 vs lambda_0.0_d5, success (d5)             |   0.1022 |    0      |  0      |     1.3127 |         0.0003 |       0.0012 |
| ablation       | lambda_4.0 vs lambda_0.0, success (d4)                   |   0.0311 |    0.0111 |  0.0173 |     0.6781 |         0.6202 |       0.9685 |
| ablation       | lambda_4.0_d5 vs lambda_0.0_d5, success (d5)             |   0.1144 |    0      |  0      |     1.5245 |         0      |       0.0002 |
| ablation       | lambda_8.0 vs lambda_0.0, success (d4)                   |   0.0633 |    0      |  0      |     1.6229 |         0      |       0      |
| ablation       | lambda_8.0_d5 vs lambda_0.0_d5, success (d5)             |   0.1533 |    0      |  0      |     2.0739 |         0      |       0      |
| ablation       | mu_0.0 vs mu_0.05, success                               |  -0.02   |    0.1616 |  0.2382 |    -0.3664 |         1      |       1      |
| ablation       | mu_0.2 vs mu_0.05, success                               |  -0.3944 |    0      |  0      |    -5.1844 |         0      |       0      |
| rule_types     | typed_SD_d3: hyp_open_vocab vs hyp_closed_vocab          |   0.5311 |    0      |  0      |     8.715  |         0      |       0      |
| rule_types     | typed_SD_d3: hyp_open_vocab vs reactive                  |   0.0378 |    0      |  0      |     1.59   |         0.0001 |       0      |
| rule_types     | typed_SD_d4: hyp_open_vocab vs hyp_closed_vocab          |   0.7444 |    0      |  0      |    13.7988 |         0      |       0      |
| rule_types     | typed_SD_d4: hyp_open_vocab vs reactive                  |   0.0478 |    0      |  0      |     1.5371 |         0.0001 |       0      |
| rule_types     | standard_d3: hyp_open_vocab vs hyp_closed_vocab          |   0      |    1      |  0.9168 |     0      |         1      |       1      |
| rule_types     | standard_d3: hyp_open_vocab vs reactive                  |   0.57   |    0      |  0      |     7.2797 |         0      |       0      |
| recovery       | hyp_with_revision vs hyp_no_revision, success            |   0.6011 |    0      |  0      |     7.7731 |         0      |       0      |
| recovery       | hyp_with_revision vs hyp_no_revision, recovery           |   0.6993 |    0      |  0      |    10.4105 |         0      |       0      |
| recovery       | hyp_with_revision vs reactive, success                   |   0.1133 |    0      |  0      |     1.355  |         0.0002 |       0.0003 |
| recovery       | hyp_with_revision vs reactive, recovery                  |   0.3072 |    0      |  0      |     6.2703 |         0      |       0      |
| recovery       | hyp_no_prior vs hyp_with_revision, success               |   0.22   |    0      |  0      |     3.734  |         0      |       0      |
| recovery       | hyp_no_prior vs hyp_with_revision, recovery              |   0.0817 |    0      |  0      |     2.5902 |         0      |       0      |
| recovery       | hyp_with_revision vs psrl, success                       |  -0.2178 |    0      |  0      |    -3.6774 |         0      |       0      |
| recovery       | hyp_with_revision vs psrl, recovery                      |  -0.0817 |    0      |  0      |    -2.5902 |         0      |       0      |
| factorial      | d4: hyp_EV_l8 vs hyp_EV_l0                               |   0.0589 |    0      |  0      |     1.5039 |         0.0001 |       0      |
| factorial      | d4: psrl_S_l8 vs psrl_S_l0                               |   0.0522 |    0.0001 |  0.0003 |     1.1013 |         0.0055 |       0.0195 |
| factorial      | d4: hyp_EV_l0 vs psrl_S_l0                               |   0.0244 |    0.0752 |  0.0707 |     0.4678 |         1      |       1      |
| factorial      | d4: hyp_EV_l8 vs psrl_S_l8                               |   0.0311 |    0.0005 |  0.0008 |     0.9596 |         0.0318 |       0.0485 |
| factorial      | d5: hyp_EV_l8 vs hyp_EV_l0                               |   0.1911 |    0      |  0      |     2.8287 |         0      |       0      |
| factorial      | d5: psrl_S_l8 vs psrl_S_l0                               |   0.1344 |    0      |  0      |     1.6096 |         0      |       0.0001 |
| factorial      | d5: hyp_EV_l0 vs psrl_S_l0                               |   0.0256 |    0.2183 |  0.242  |     0.3215 |         1      |       1      |
| factorial      | d5: hyp_EV_l8 vs psrl_S_l8                               |   0.0822 |    0      |  0.0001 |     1.1376 |         0.0033 |       0.01   |
| master_key     | master_key_d3: hyp_nonexcl vs hypothesis                 |   0.77   |    0      |  0      |    11.9508 |         0      |       0      |
| master_key     | master_key_d3: psrl_nonexcl vs psrl                      |   0      |  nan      |  1      |     0      |         1      |       1      |
| master_key     | master_key_d3: hypothesis vs psrl                        |  -0.77   |    0      |  0      |   -11.9508 |         0      |       0      |
| master_key     | master_key_d4: hyp_nonexcl vs hypothesis                 |   0.8178 |    0      |  0      |    15.4913 |         0      |       0      |
| master_key     | master_key_d4: psrl_nonexcl vs psrl                      |   0.0144 |    0.0992 |  0.0494 |     0.4327 |         1      |       1      |
| master_key     | master_key_d4: hypothesis vs psrl                        |  -0.7833 |    0      |  0      |   -13.3981 |         0      |       0      |
| master_key     | exclusive_control_d3: hyp_nonexcl vs hypothesis          |   0      |  nan      |  1      |     0      |         1      |       1      |
| master_key     | exclusive_control_d3: psrl_nonexcl vs psrl               |  -0.0011 |    0.3256 |  0.3337 |    -0.2582 |         1      |       1      |
| master_key     | exclusive_control_d3: hypothesis vs psrl                 |   0      |  nan      |  1      |     0      |         1      |       1      |
| master_key     | exclusive_control_d4: hyp_nonexcl vs hypothesis          |  -0.12   |    0      |  0      |    -2.4529 |         0      |       0      |
| master_key     | exclusive_control_d4: psrl_nonexcl vs psrl               |  -0.0067 |    0.6834 |  0.6309 |    -0.1058 |         1      |       1      |
| master_key     | exclusive_control_d4: hypothesis vs psrl                 |   0.11   |    0      |  0      |     2.3172 |         0      |       0      |
| vault          | in_dist: hypothesis vs vault_gru, success                |   0      |  nan      |  1      |     0      |         1      |       1      |
| vault          | in_dist: hypothesis vs vault_ff, success                 |   0.05   |    0      |  0      |     1.5334 |         0.0001 |       0      |
| vault          | in_dist: hypothesis vs reactive, success                 |   0.0511 |    0      |  0      |     1.568  |         0.0001 |       0      |
| vault          | larger: hypothesis vs vault_gru, success                 |   0.8578 |    0      |  0      |    21.9294 |         0      |       0      |
| vault          | larger: hypothesis vs vault_ff, success                  |   0.93   |    0      |  0      |    24.9202 |         0      |       0      |
| vault          | larger: hypothesis vs reactive, success                  |   0.9256 |    0      |  0      |    24.0415 |         0      |       0      |
