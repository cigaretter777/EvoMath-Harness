# Round-1 budget diagnostics tables

Aggregation: table A accuracy is per-question macro (primary) and per-episode micro; shares are episode-level descriptive. Tables B/C are per (budget, task) -- never pooled across budgets, and four same-question samples are not four independent questions.
Table B recovery has two operationalizations: recovered_n = a failed tool result followed later by a successful one; recovered_invalid_n = an episode with invalid-action steps that still ends correct.
Any GRPO-style advantage recomputed offline (here or in the runner's groups.jsonl) is a diagnostic value (重算诊断值), not the trainer's actual advantage. Table C variances are population variances (ddof=0), matching the runner's group_advantages.

## Table A -- budget effect

| budget | questions | episodes | accuracy_macro_by_question | accuracy_micro_by_episode | mean_tool_calls | share_calls_0 | share_calls_1 | share_calls_2 | share_calls_3 | share_calls_4 | mean_generated_tokens | mean_python_seconds | mean_invalid_actions | term_final | term_max_steps | term_max_tool_calls | term_python_time_budget | term_model_error | term_infrastructure_error | term_cancelled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b0 | 20 | 80 | 0.25 | 0.25 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 721.2 | 0.0 | 0.425 | 0.9625 | 0.0375 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| b1 | 20 | 80 | 0.2625 | 0.2625 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 893.3 | 0.0 | 0.662 | 0.95 | 0.05 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| b2 | 20 | 80 | 0.2125 | 0.2125 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1011.3 | 0.0 | 0.812 | 0.9125 | 0.0875 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| b4 | 20 | 80 | 0.275 | 0.275 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 815.5 | 0.0 | 0.537 | 0.95 | 0.05 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Table B -- within-question room (per budget, task)

| budget | task_id | samples | correct_n | call_range_correct | call_range_wrong | recovered_n | recovered_invalid_n |
|---|---|---|---|---|---|---|---|
| b0 | omni_math:2e55f04edc10a9c42f9d | 4 | 2 | 0-0 | 0-0 | 0 | 1 |
| b0 | omni_math:31379d4ab3c591a63f08 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:3e81816fbdb983295ded | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:40e067a90f6a0432e579 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:46e39a2cac5a7070682a | 4 | 4 | 0-0 |  | 0 | 0 |
| b0 | omni_math:48a0e06e36e421cba955 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:5aab3eb4acd50196d860 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:5c95294cc128be4d87ec | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:6317a0595c2f92ff6874 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:842e8b1f3ecd8c82ac00 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:8924663d0e48f9815d9d | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:8cbe0c48c29daa190277 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:8cc783279bbf4cd36bd0 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:bc59bae1f8d5e2511c53 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:bc8aca553a71b9139a79 | 4 | 4 | 0-0 |  | 0 | 0 |
| b0 | omni_math:cdaaad37badd1126c60d | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:cfd1da8604a325e749e2 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:e7ce0a36c776a990ac85 | 4 | 4 | 0-0 |  | 0 | 0 |
| b0 | omni_math:f1069a0605746e77f02b | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:ffc5a7d0d1b7309e726f | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:2e55f04edc10a9c42f9d | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:31379d4ab3c591a63f08 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b1 | omni_math:3e81816fbdb983295ded | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:40e067a90f6a0432e579 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b1 | omni_math:46e39a2cac5a7070682a | 4 | 4 | 0-0 |  | 0 | 0 |
| b1 | omni_math:48a0e06e36e421cba955 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:5aab3eb4acd50196d860 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b1 | omni_math:5c95294cc128be4d87ec | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:6317a0595c2f92ff6874 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b1 | omni_math:842e8b1f3ecd8c82ac00 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b1 | omni_math:8924663d0e48f9815d9d | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b1 | omni_math:8cbe0c48c29daa190277 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:8cc783279bbf4cd36bd0 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b1 | omni_math:bc59bae1f8d5e2511c53 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:bc8aca553a71b9139a79 | 4 | 4 | 0-0 |  | 0 | 0 |
| b1 | omni_math:cdaaad37badd1126c60d | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:cfd1da8604a325e749e2 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:e7ce0a36c776a990ac85 | 4 | 4 | 0-0 |  | 0 | 0 |
| b1 | omni_math:f1069a0605746e77f02b | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:ffc5a7d0d1b7309e726f | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:2e55f04edc10a9c42f9d | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:31379d4ab3c591a63f08 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:3e81816fbdb983295ded | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:40e067a90f6a0432e579 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:46e39a2cac5a7070682a | 4 | 4 | 0-0 |  | 0 | 0 |
| b2 | omni_math:48a0e06e36e421cba955 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:5aab3eb4acd50196d860 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:5c95294cc128be4d87ec | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:6317a0595c2f92ff6874 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:842e8b1f3ecd8c82ac00 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:8924663d0e48f9815d9d | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:8cbe0c48c29daa190277 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:8cc783279bbf4cd36bd0 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:bc59bae1f8d5e2511c53 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:bc8aca553a71b9139a79 | 4 | 4 | 0-0 |  | 0 | 0 |
| b2 | omni_math:cdaaad37badd1126c60d | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:cfd1da8604a325e749e2 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:e7ce0a36c776a990ac85 | 4 | 4 | 0-0 |  | 0 | 0 |
| b2 | omni_math:f1069a0605746e77f02b | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:ffc5a7d0d1b7309e726f | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:2e55f04edc10a9c42f9d | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:31379d4ab3c591a63f08 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:3e81816fbdb983295ded | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:40e067a90f6a0432e579 | 4 | 4 | 0-0 |  | 0 | 0 |
| b4 | omni_math:46e39a2cac5a7070682a | 4 | 4 | 0-0 |  | 0 | 0 |
| b4 | omni_math:48a0e06e36e421cba955 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:5aab3eb4acd50196d860 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:5c95294cc128be4d87ec | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:6317a0595c2f92ff6874 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:842e8b1f3ecd8c82ac00 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:8924663d0e48f9815d9d | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:8cbe0c48c29daa190277 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:8cc783279bbf4cd36bd0 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:bc59bae1f8d5e2511c53 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:bc8aca553a71b9139a79 | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:cdaaad37badd1126c60d | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:cfd1da8604a325e749e2 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:e7ce0a36c776a990ac85 | 4 | 4 | 0-0 |  | 0 | 0 |
| b4 | omni_math:f1069a0605746e77f02b | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:ffc5a7d0d1b7309e726f | 4 | 0 |  | 0-0 | 0 | 0 |

## Table C -- reward visibility (per budget, task)

| budget | task_id | answer_reward_var | total_reward_var | call_cost_var | all_equal_reward | correct_but_diff_cost |
|---|---|---|---|---|---|---|
| b0 | omni_math:2e55f04edc10a9c42f9d | 0.25 | 0.25 | 0.0 | False | False |
| b0 | omni_math:31379d4ab3c591a63f08 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:3e81816fbdb983295ded | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:40e067a90f6a0432e579 | 0.1875 | 0.1875 | 0.0 | False | False |
| b0 | omni_math:46e39a2cac5a7070682a | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:48a0e06e36e421cba955 | 0.1875 | 0.1875 | 0.0 | False | False |
| b0 | omni_math:5aab3eb4acd50196d860 | 0.25 | 0.25 | 0.0 | False | False |
| b0 | omni_math:5c95294cc128be4d87ec | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:6317a0595c2f92ff6874 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:842e8b1f3ecd8c82ac00 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:8924663d0e48f9815d9d | 0.1875 | 0.1875 | 0.0 | False | False |
| b0 | omni_math:8cbe0c48c29daa190277 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:8cc783279bbf4cd36bd0 | 0.1875 | 0.1875 | 0.0 | False | False |
| b0 | omni_math:bc59bae1f8d5e2511c53 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:bc8aca553a71b9139a79 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:cdaaad37badd1126c60d | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:cfd1da8604a325e749e2 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:e7ce0a36c776a990ac85 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:f1069a0605746e77f02b | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:ffc5a7d0d1b7309e726f | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:2e55f04edc10a9c42f9d | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:31379d4ab3c591a63f08 | 0.1875 | 0.1875 | 0.0 | False | False |
| b1 | omni_math:3e81816fbdb983295ded | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:40e067a90f6a0432e579 | 0.25 | 0.25 | 0.0 | False | False |
| b1 | omni_math:46e39a2cac5a7070682a | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:48a0e06e36e421cba955 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:5aab3eb4acd50196d860 | 0.25 | 0.25 | 0.0 | False | False |
| b1 | omni_math:5c95294cc128be4d87ec | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:6317a0595c2f92ff6874 | 0.1875 | 0.1875 | 0.0 | False | False |
| b1 | omni_math:842e8b1f3ecd8c82ac00 | 0.1875 | 0.1875 | 0.0 | False | False |
| b1 | omni_math:8924663d0e48f9815d9d | 0.1875 | 0.1875 | 0.0 | False | False |
| b1 | omni_math:8cbe0c48c29daa190277 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:8cc783279bbf4cd36bd0 | 0.1875 | 0.1875 | 0.0 | False | False |
| b1 | omni_math:bc59bae1f8d5e2511c53 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:bc8aca553a71b9139a79 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:cdaaad37badd1126c60d | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:cfd1da8604a325e749e2 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:e7ce0a36c776a990ac85 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:f1069a0605746e77f02b | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:ffc5a7d0d1b7309e726f | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:2e55f04edc10a9c42f9d | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:31379d4ab3c591a63f08 | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:3e81816fbdb983295ded | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:40e067a90f6a0432e579 | 0.25 | 0.25 | 0.0 | False | False |
| b2 | omni_math:46e39a2cac5a7070682a | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:48a0e06e36e421cba955 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:5aab3eb4acd50196d860 | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:5c95294cc128be4d87ec | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:6317a0595c2f92ff6874 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:842e8b1f3ecd8c82ac00 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:8924663d0e48f9815d9d | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:8cbe0c48c29daa190277 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:8cc783279bbf4cd36bd0 | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:bc59bae1f8d5e2511c53 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:bc8aca553a71b9139a79 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:cdaaad37badd1126c60d | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:cfd1da8604a325e749e2 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:e7ce0a36c776a990ac85 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:f1069a0605746e77f02b | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:ffc5a7d0d1b7309e726f | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:2e55f04edc10a9c42f9d | 0.1875 | 0.1875 | 0.0 | False | False |
| b4 | omni_math:31379d4ab3c591a63f08 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:3e81816fbdb983295ded | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:40e067a90f6a0432e579 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:46e39a2cac5a7070682a | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:48a0e06e36e421cba955 | 0.1875 | 0.1875 | 0.0 | False | False |
| b4 | omni_math:5aab3eb4acd50196d860 | 0.25 | 0.25 | 0.0 | False | False |
| b4 | omni_math:5c95294cc128be4d87ec | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:6317a0595c2f92ff6874 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:842e8b1f3ecd8c82ac00 | 0.25 | 0.25 | 0.0 | False | False |
| b4 | omni_math:8924663d0e48f9815d9d | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:8cbe0c48c29daa190277 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:8cc783279bbf4cd36bd0 | 0.1875 | 0.1875 | 0.0 | False | False |
| b4 | omni_math:bc59bae1f8d5e2511c53 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:bc8aca553a71b9139a79 | 0.1875 | 0.1875 | 0.0 | False | False |
| b4 | omni_math:cdaaad37badd1126c60d | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:cfd1da8604a325e749e2 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:e7ce0a36c776a990ac85 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:f1069a0605746e77f02b | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:ffc5a7d0d1b7309e726f | 0.0 | 0.0 | 0.0 | True | False |
