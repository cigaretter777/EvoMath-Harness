# Round-1 budget diagnostics tables

Aggregation: table A accuracy is per-question macro (primary) and per-episode micro; shares are episode-level descriptive. Tables B/C are per (budget, task) -- never pooled across budgets, and four same-question samples are not four independent questions.
Table B recovery has two operationalizations: recovered_n = a failed tool result followed later by a successful one; recovered_invalid_n = an episode with invalid-action steps that still ends correct.
Any GRPO-style advantage recomputed offline (here or in the runner's groups.jsonl) is a diagnostic value (重算诊断值), not the trainer's actual advantage. Table C variances are population variances (ddof=0), matching the runner's group_advantages.

## Table A -- budget effect

| budget | questions | episodes | accuracy_macro_by_question | accuracy_micro_by_episode | mean_tool_calls | share_calls_0 | share_calls_1 | share_calls_2 | share_calls_3 | share_calls_4 | mean_generated_tokens | mean_python_seconds | mean_invalid_actions | term_final | term_max_steps | term_max_tool_calls | term_python_time_budget | term_model_error | term_infrastructure_error | term_cancelled |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b0 | 100 | 400 | 0.1175 | 0.1175 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 715.6 | 0.0 | 0.435 | 0.96 | 0.04 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| b1 | 100 | 400 | 0.14 | 0.14 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 760.4 | 0.0 | 0.532 | 0.9475 | 0.0525 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| b2 | 100 | 400 | 0.13 | 0.13 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 781.1 | 0.0 | 0.46 | 0.95 | 0.05 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| b4 | 100 | 400 | 0.1375 | 0.1375 | 0.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 812.4 | 0.0 | 0.555 | 0.94 | 0.06 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Table B -- within-question room (per budget, task)

| budget | task_id | samples | correct_n | call_range_correct | call_range_wrong | recovered_n | recovered_invalid_n |
|---|---|---|---|---|---|---|---|
| b0 | omni_math:13d6d0a2b4ffdc031fe0 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:17a7b9205f9601657433 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:1806f5474d6604668c8a | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:188bf878dbe2547a5a45 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:1ba74ec2fc0ace980a21 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:1e68298480ae1711e4bb | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:1ea9b2603a964b844abd | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:2a9f69bb19e9cedee0ce | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:2e55f04edc10a9c42f9d | 4 | 1 | 0-0 | 0-0 | 0 | 1 |
| b0 | omni_math:2e8d18406c6f7bedc812 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:31379d4ab3c591a63f08 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:36155002d87dcdeb45f6 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:36274a9a20054259af19 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:36460647c8efd7884e2c | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:3bb367b81dac35a5b595 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:3e81816fbdb983295ded | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:3fdbc6e51becad821409 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:40e067a90f6a0432e579 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:440805f38fe688d92463 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:4542318efb9f13abf5a4 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:4650eb4adb331de38940 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:46e39a2cac5a7070682a | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:48a0e06e36e421cba955 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:49477579d8ea176944a0 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:4a1aefda552febdd7ae2 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:4fc701caa56c100a1e65 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:521f74f37c6690eedac6 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:5610cdc27b941c22c02f | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:5aab3eb4acd50196d860 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:5ad36e29df97c9fc5f27 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:5c95294cc128be4d87ec | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:5d50ee9ea9d8e9ee2a8a | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:5e51ea6902200cdd3d18 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:6317a0595c2f92ff6874 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:6a41507301ee454a1eba | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:6b0e7ba7018d60cf3cb1 | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:6e8d30af7f4895f14380 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:702c923b908f4dd81336 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:730e703e9f7bfc17b403 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:7393134ba2e5cbe0f33f | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:7464a4958e412012db6f | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:74ad2002447761200b5d | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:75a27a35658b0acbe7dd | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:76b3ea8d366c2f38a330 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:7782e5a4b8566e5e44da | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:7850b7403769a872ac21 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:7c1e5ed8c3bce096d5d1 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:7d3d8b0f0158f566491d | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:842e8b1f3ecd8c82ac00 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:85f7cf50f56fd30b11a0 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:8924663d0e48f9815d9d | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:8cbe0c48c29daa190277 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:8cc783279bbf4cd36bd0 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:90f99c010d1bc49a1a2e | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:93f9c7cf0f56cd62bca6 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:9a2de48dad192786426e | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:a061e37809c94090334f | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:a23fdf11a060f9b15b4c | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:a3c15882576e19d381c0 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:a59a2acac11cf3c2bc0f | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:b06dd6f0e8140ab15e5f | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:b0cc3c51bf0918a50674 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:b2c87c609c5c81c6fd86 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:b3f533d75f5ac73a7eec | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:b3f872d6e654851a919e | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:ba16b0153a22069b60fa | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:ba25c09740c767adeea2 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:bb3a33713d807a7c207f | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:bc59bae1f8d5e2511c53 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:bc8aca553a71b9139a79 | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:bfd266d75f332f5b6720 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:c7cd7b23fc1ab1205171 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:c893e9669d9ce73010f5 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:caf128f8eabeca7f5497 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:cdaaad37badd1126c60d | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:ceaa906e9dd8113b0b6e | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:cfd1da8604a325e749e2 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:d0e6044cea3bbf96e7f0 | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:d1ea5c078fa8ad618098 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:d6dabfda17500fff2a13 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:da41daf7f13f02ff9c45 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:de362aad6d3a5e145c65 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:df5c4d142c90dc4ad606 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:e09424195c4914dac979 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:e208d2bcc4f3f9cf7c08 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:e314dea85e5dbc870c78 | 4 | 4 | 0-0 |  | 0 | 0 |
| b0 | omni_math:e702f49d33d7c654b256 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:e78524746d751e620871 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:e7ce0a36c776a990ac85 | 4 | 4 | 0-0 |  | 0 | 0 |
| b0 | omni_math:e807cc4dc4e348f9be56 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:e92ffc08bf5806e339aa | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:e9efcba41e0f997de523 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:f0f7a7e0c7a5ebd7113b | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b0 | omni_math:f1069a0605746e77f02b | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:f117cb7d7aaf2a3f6f33 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:f6fbadc7366df3ed2a7a | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:fdb774ad28314a2f7d66 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:fe3ec27e8b23b5a6ec7c | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:fec65acff1d1410cc980 | 4 | 0 |  | 0-0 | 0 | 0 |
| b0 | omni_math:ffc5a7d0d1b7309e726f | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:13d6d0a2b4ffdc031fe0 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:17a7b9205f9601657433 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b1 | omni_math:1806f5474d6604668c8a | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:188bf878dbe2547a5a45 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:1ba74ec2fc0ace980a21 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:1e68298480ae1711e4bb | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:1ea9b2603a964b844abd | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:2a9f69bb19e9cedee0ce | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b1 | omni_math:2e55f04edc10a9c42f9d | 4 | 1 | 0-0 | 0-0 | 0 | 1 |
| b1 | omni_math:2e8d18406c6f7bedc812 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:31379d4ab3c591a63f08 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b1 | omni_math:36155002d87dcdeb45f6 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:36274a9a20054259af19 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:36460647c8efd7884e2c | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:3bb367b81dac35a5b595 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:3e81816fbdb983295ded | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:3fdbc6e51becad821409 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:40e067a90f6a0432e579 | 4 | 4 | 0-0 |  | 0 | 0 |
| b1 | omni_math:440805f38fe688d92463 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:4542318efb9f13abf5a4 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:4650eb4adb331de38940 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:46e39a2cac5a7070682a | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b1 | omni_math:48a0e06e36e421cba955 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:49477579d8ea176944a0 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:4a1aefda552febdd7ae2 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:4fc701caa56c100a1e65 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:521f74f37c6690eedac6 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:5610cdc27b941c22c02f | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:5aab3eb4acd50196d860 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b1 | omni_math:5ad36e29df97c9fc5f27 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:5c95294cc128be4d87ec | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:5d50ee9ea9d8e9ee2a8a | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:5e51ea6902200cdd3d18 | 4 | 4 | 0-0 |  | 0 | 0 |
| b1 | omni_math:6317a0595c2f92ff6874 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:6a41507301ee454a1eba | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:6b0e7ba7018d60cf3cb1 | 4 | 4 | 0-0 |  | 0 | 0 |
| b1 | omni_math:6e8d30af7f4895f14380 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:702c923b908f4dd81336 | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b1 | omni_math:730e703e9f7bfc17b403 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b1 | omni_math:7393134ba2e5cbe0f33f | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:7464a4958e412012db6f | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:74ad2002447761200b5d | 4 | 4 | 0-0 |  | 0 | 0 |
| b1 | omni_math:75a27a35658b0acbe7dd | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b1 | omni_math:76b3ea8d366c2f38a330 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:7782e5a4b8566e5e44da | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:7850b7403769a872ac21 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:7c1e5ed8c3bce096d5d1 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:7d3d8b0f0158f566491d | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:842e8b1f3ecd8c82ac00 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b1 | omni_math:85f7cf50f56fd30b11a0 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:8924663d0e48f9815d9d | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:8cbe0c48c29daa190277 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:8cc783279bbf4cd36bd0 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:90f99c010d1bc49a1a2e | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:93f9c7cf0f56cd62bca6 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:9a2de48dad192786426e | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:a061e37809c94090334f | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:a23fdf11a060f9b15b4c | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:a3c15882576e19d381c0 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:a59a2acac11cf3c2bc0f | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:b06dd6f0e8140ab15e5f | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:b0cc3c51bf0918a50674 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:b2c87c609c5c81c6fd86 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:b3f533d75f5ac73a7eec | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:b3f872d6e654851a919e | 4 | 4 | 0-0 |  | 0 | 0 |
| b1 | omni_math:ba16b0153a22069b60fa | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:ba25c09740c767adeea2 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:bb3a33713d807a7c207f | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:bc59bae1f8d5e2511c53 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:bc8aca553a71b9139a79 | 4 | 4 | 0-0 |  | 0 | 0 |
| b1 | omni_math:bfd266d75f332f5b6720 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:c7cd7b23fc1ab1205171 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:c893e9669d9ce73010f5 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:caf128f8eabeca7f5497 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:cdaaad37badd1126c60d | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:ceaa906e9dd8113b0b6e | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b1 | omni_math:cfd1da8604a325e749e2 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:d0e6044cea3bbf96e7f0 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b1 | omni_math:d1ea5c078fa8ad618098 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:d6dabfda17500fff2a13 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:da41daf7f13f02ff9c45 | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b1 | omni_math:de362aad6d3a5e145c65 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:df5c4d142c90dc4ad606 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b1 | omni_math:e09424195c4914dac979 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:e208d2bcc4f3f9cf7c08 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:e314dea85e5dbc870c78 | 4 | 4 | 0-0 |  | 0 | 0 |
| b1 | omni_math:e702f49d33d7c654b256 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:e78524746d751e620871 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:e7ce0a36c776a990ac85 | 4 | 4 | 0-0 |  | 0 | 0 |
| b1 | omni_math:e807cc4dc4e348f9be56 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:e92ffc08bf5806e339aa | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:e9efcba41e0f997de523 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:f0f7a7e0c7a5ebd7113b | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:f1069a0605746e77f02b | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:f117cb7d7aaf2a3f6f33 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:f6fbadc7366df3ed2a7a | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:fdb774ad28314a2f7d66 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:fe3ec27e8b23b5a6ec7c | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:fec65acff1d1410cc980 | 4 | 0 |  | 0-0 | 0 | 0 |
| b1 | omni_math:ffc5a7d0d1b7309e726f | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:13d6d0a2b4ffdc031fe0 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:17a7b9205f9601657433 | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:1806f5474d6604668c8a | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:188bf878dbe2547a5a45 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:1ba74ec2fc0ace980a21 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:1e68298480ae1711e4bb | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:1ea9b2603a964b844abd | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:2a9f69bb19e9cedee0ce | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:2e55f04edc10a9c42f9d | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:2e8d18406c6f7bedc812 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:31379d4ab3c591a63f08 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:36155002d87dcdeb45f6 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:36274a9a20054259af19 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:36460647c8efd7884e2c | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:3bb367b81dac35a5b595 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:3e81816fbdb983295ded | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:3fdbc6e51becad821409 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:40e067a90f6a0432e579 | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:440805f38fe688d92463 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:4542318efb9f13abf5a4 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:4650eb4adb331de38940 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:46e39a2cac5a7070682a | 4 | 4 | 0-0 |  | 0 | 0 |
| b2 | omni_math:48a0e06e36e421cba955 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:49477579d8ea176944a0 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:4a1aefda552febdd7ae2 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:4fc701caa56c100a1e65 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:521f74f37c6690eedac6 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:5610cdc27b941c22c02f | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:5aab3eb4acd50196d860 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:5ad36e29df97c9fc5f27 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:5c95294cc128be4d87ec | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:5d50ee9ea9d8e9ee2a8a | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:5e51ea6902200cdd3d18 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:6317a0595c2f92ff6874 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:6a41507301ee454a1eba | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:6b0e7ba7018d60cf3cb1 | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:6e8d30af7f4895f14380 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:702c923b908f4dd81336 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:730e703e9f7bfc17b403 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:7393134ba2e5cbe0f33f | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:7464a4958e412012db6f | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:74ad2002447761200b5d | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:75a27a35658b0acbe7dd | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:76b3ea8d366c2f38a330 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:7782e5a4b8566e5e44da | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:7850b7403769a872ac21 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:7c1e5ed8c3bce096d5d1 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:7d3d8b0f0158f566491d | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:842e8b1f3ecd8c82ac00 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:85f7cf50f56fd30b11a0 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:8924663d0e48f9815d9d | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:8cbe0c48c29daa190277 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:8cc783279bbf4cd36bd0 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:90f99c010d1bc49a1a2e | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:93f9c7cf0f56cd62bca6 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:9a2de48dad192786426e | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:a061e37809c94090334f | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:a23fdf11a060f9b15b4c | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:a3c15882576e19d381c0 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:a59a2acac11cf3c2bc0f | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:b06dd6f0e8140ab15e5f | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:b0cc3c51bf0918a50674 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:b2c87c609c5c81c6fd86 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:b3f533d75f5ac73a7eec | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:b3f872d6e654851a919e | 4 | 4 | 0-0 |  | 0 | 0 |
| b2 | omni_math:ba16b0153a22069b60fa | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:ba25c09740c767adeea2 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:bb3a33713d807a7c207f | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:bc59bae1f8d5e2511c53 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:bc8aca553a71b9139a79 | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:bfd266d75f332f5b6720 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:c7cd7b23fc1ab1205171 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:c893e9669d9ce73010f5 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:caf128f8eabeca7f5497 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:cdaaad37badd1126c60d | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:ceaa906e9dd8113b0b6e | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:cfd1da8604a325e749e2 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:d0e6044cea3bbf96e7f0 | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:d1ea5c078fa8ad618098 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:d6dabfda17500fff2a13 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:da41daf7f13f02ff9c45 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:de362aad6d3a5e145c65 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:df5c4d142c90dc4ad606 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:e09424195c4914dac979 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:e208d2bcc4f3f9cf7c08 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:e314dea85e5dbc870c78 | 4 | 4 | 0-0 |  | 0 | 0 |
| b2 | omni_math:e702f49d33d7c654b256 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:e78524746d751e620871 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:e7ce0a36c776a990ac85 | 4 | 4 | 0-0 |  | 0 | 0 |
| b2 | omni_math:e807cc4dc4e348f9be56 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:e92ffc08bf5806e339aa | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:e9efcba41e0f997de523 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:f0f7a7e0c7a5ebd7113b | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:f1069a0605746e77f02b | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:f117cb7d7aaf2a3f6f33 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b2 | omni_math:f6fbadc7366df3ed2a7a | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:fdb774ad28314a2f7d66 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:fe3ec27e8b23b5a6ec7c | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:fec65acff1d1410cc980 | 4 | 0 |  | 0-0 | 0 | 0 |
| b2 | omni_math:ffc5a7d0d1b7309e726f | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:13d6d0a2b4ffdc031fe0 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:17a7b9205f9601657433 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:1806f5474d6604668c8a | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:188bf878dbe2547a5a45 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:1ba74ec2fc0ace980a21 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:1e68298480ae1711e4bb | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:1ea9b2603a964b844abd | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:2a9f69bb19e9cedee0ce | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:2e55f04edc10a9c42f9d | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:2e8d18406c6f7bedc812 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:31379d4ab3c591a63f08 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:36155002d87dcdeb45f6 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:36274a9a20054259af19 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:36460647c8efd7884e2c | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:3bb367b81dac35a5b595 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:3e81816fbdb983295ded | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:3fdbc6e51becad821409 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:40e067a90f6a0432e579 | 4 | 2 | 0-0 | 0-0 | 0 | 1 |
| b4 | omni_math:440805f38fe688d92463 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:4542318efb9f13abf5a4 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:4650eb4adb331de38940 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:46e39a2cac5a7070682a | 4 | 4 | 0-0 |  | 0 | 0 |
| b4 | omni_math:48a0e06e36e421cba955 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:49477579d8ea176944a0 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:4a1aefda552febdd7ae2 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:4fc701caa56c100a1e65 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:521f74f37c6690eedac6 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:5610cdc27b941c22c02f | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:5aab3eb4acd50196d860 | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:5ad36e29df97c9fc5f27 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:5c95294cc128be4d87ec | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:5d50ee9ea9d8e9ee2a8a | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:5e51ea6902200cdd3d18 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:6317a0595c2f92ff6874 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:6a41507301ee454a1eba | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:6b0e7ba7018d60cf3cb1 | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:6e8d30af7f4895f14380 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:702c923b908f4dd81336 | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:730e703e9f7bfc17b403 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:7393134ba2e5cbe0f33f | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:7464a4958e412012db6f | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:74ad2002447761200b5d | 4 | 4 | 0-0 |  | 0 | 0 |
| b4 | omni_math:75a27a35658b0acbe7dd | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:76b3ea8d366c2f38a330 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:7782e5a4b8566e5e44da | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:7850b7403769a872ac21 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:7c1e5ed8c3bce096d5d1 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:7d3d8b0f0158f566491d | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:842e8b1f3ecd8c82ac00 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:85f7cf50f56fd30b11a0 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:8924663d0e48f9815d9d | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:8cbe0c48c29daa190277 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:8cc783279bbf4cd36bd0 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:90f99c010d1bc49a1a2e | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:93f9c7cf0f56cd62bca6 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:9a2de48dad192786426e | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:a061e37809c94090334f | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:a23fdf11a060f9b15b4c | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:a3c15882576e19d381c0 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:a59a2acac11cf3c2bc0f | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:b06dd6f0e8140ab15e5f | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:b0cc3c51bf0918a50674 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:b2c87c609c5c81c6fd86 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:b3f533d75f5ac73a7eec | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:b3f872d6e654851a919e | 4 | 4 | 0-0 |  | 0 | 0 |
| b4 | omni_math:ba16b0153a22069b60fa | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:ba25c09740c767adeea2 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:bb3a33713d807a7c207f | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:bc59bae1f8d5e2511c53 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:bc8aca553a71b9139a79 | 4 | 4 | 0-0 |  | 0 | 0 |
| b4 | omni_math:bfd266d75f332f5b6720 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:c7cd7b23fc1ab1205171 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:c893e9669d9ce73010f5 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:caf128f8eabeca7f5497 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:cdaaad37badd1126c60d | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:ceaa906e9dd8113b0b6e | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:cfd1da8604a325e749e2 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:d0e6044cea3bbf96e7f0 | 4 | 2 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:d1ea5c078fa8ad618098 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:d6dabfda17500fff2a13 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:da41daf7f13f02ff9c45 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:de362aad6d3a5e145c65 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:df5c4d142c90dc4ad606 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:e09424195c4914dac979 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:e208d2bcc4f3f9cf7c08 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:e314dea85e5dbc870c78 | 4 | 3 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:e702f49d33d7c654b256 | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:e78524746d751e620871 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:e7ce0a36c776a990ac85 | 4 | 4 | 0-0 |  | 0 | 0 |
| b4 | omni_math:e807cc4dc4e348f9be56 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:e92ffc08bf5806e339aa | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:e9efcba41e0f997de523 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:f0f7a7e0c7a5ebd7113b | 4 | 1 | 0-0 | 0-0 | 0 | 0 |
| b4 | omni_math:f1069a0605746e77f02b | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:f117cb7d7aaf2a3f6f33 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:f6fbadc7366df3ed2a7a | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:fdb774ad28314a2f7d66 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:fe3ec27e8b23b5a6ec7c | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:fec65acff1d1410cc980 | 4 | 0 |  | 0-0 | 0 | 0 |
| b4 | omni_math:ffc5a7d0d1b7309e726f | 4 | 0 |  | 0-0 | 0 | 0 |

## Table C -- reward visibility (per budget, task)

| budget | task_id | answer_reward_var | total_reward_var | call_cost_var | all_equal_reward | correct_but_diff_cost |
|---|---|---|---|---|---|---|
| b0 | omni_math:13d6d0a2b4ffdc031fe0 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:17a7b9205f9601657433 | 0.25 | 0.25 | 0.0 | False | False |
| b0 | omni_math:1806f5474d6604668c8a | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:188bf878dbe2547a5a45 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:1ba74ec2fc0ace980a21 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:1e68298480ae1711e4bb | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:1ea9b2603a964b844abd | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:2a9f69bb19e9cedee0ce | 0.1875 | 0.1875 | 0.0 | False | False |
| b0 | omni_math:2e55f04edc10a9c42f9d | 0.1875 | 0.1875 | 0.0 | False | False |
| b0 | omni_math:2e8d18406c6f7bedc812 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:31379d4ab3c591a63f08 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:36155002d87dcdeb45f6 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:36274a9a20054259af19 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:36460647c8efd7884e2c | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:3bb367b81dac35a5b595 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:3e81816fbdb983295ded | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:3fdbc6e51becad821409 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:40e067a90f6a0432e579 | 0.25 | 0.25 | 0.0 | False | False |
| b0 | omni_math:440805f38fe688d92463 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:4542318efb9f13abf5a4 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:4650eb4adb331de38940 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:46e39a2cac5a7070682a | 0.1875 | 0.1875 | 0.0 | False | False |
| b0 | omni_math:48a0e06e36e421cba955 | 0.1875 | 0.1875 | 0.0 | False | False |
| b0 | omni_math:49477579d8ea176944a0 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:4a1aefda552febdd7ae2 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:4fc701caa56c100a1e65 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:521f74f37c6690eedac6 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:5610cdc27b941c22c02f | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:5aab3eb4acd50196d860 | 0.25 | 0.25 | 0.0 | False | False |
| b0 | omni_math:5ad36e29df97c9fc5f27 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:5c95294cc128be4d87ec | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:5d50ee9ea9d8e9ee2a8a | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:5e51ea6902200cdd3d18 | 0.25 | 0.25 | 0.0 | False | False |
| b0 | omni_math:6317a0595c2f92ff6874 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:6a41507301ee454a1eba | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:6b0e7ba7018d60cf3cb1 | 0.1875 | 0.1875 | 0.0 | False | False |
| b0 | omni_math:6e8d30af7f4895f14380 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:702c923b908f4dd81336 | 0.25 | 0.25 | 0.0 | False | False |
| b0 | omni_math:730e703e9f7bfc17b403 | 0.1875 | 0.1875 | 0.0 | False | False |
| b0 | omni_math:7393134ba2e5cbe0f33f | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:7464a4958e412012db6f | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:74ad2002447761200b5d | 0.1875 | 0.1875 | 0.0 | False | False |
| b0 | omni_math:75a27a35658b0acbe7dd | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:76b3ea8d366c2f38a330 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:7782e5a4b8566e5e44da | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:7850b7403769a872ac21 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:7c1e5ed8c3bce096d5d1 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:7d3d8b0f0158f566491d | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:842e8b1f3ecd8c82ac00 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:85f7cf50f56fd30b11a0 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:8924663d0e48f9815d9d | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:8cbe0c48c29daa190277 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:8cc783279bbf4cd36bd0 | 0.1875 | 0.1875 | 0.0 | False | False |
| b0 | omni_math:90f99c010d1bc49a1a2e | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:93f9c7cf0f56cd62bca6 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:9a2de48dad192786426e | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:a061e37809c94090334f | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:a23fdf11a060f9b15b4c | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:a3c15882576e19d381c0 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:a59a2acac11cf3c2bc0f | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:b06dd6f0e8140ab15e5f | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:b0cc3c51bf0918a50674 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:b2c87c609c5c81c6fd86 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:b3f533d75f5ac73a7eec | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:b3f872d6e654851a919e | 0.1875 | 0.1875 | 0.0 | False | False |
| b0 | omni_math:ba16b0153a22069b60fa | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:ba25c09740c767adeea2 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:bb3a33713d807a7c207f | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:bc59bae1f8d5e2511c53 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:bc8aca553a71b9139a79 | 0.1875 | 0.1875 | 0.0 | False | False |
| b0 | omni_math:bfd266d75f332f5b6720 | 0.1875 | 0.1875 | 0.0 | False | False |
| b0 | omni_math:c7cd7b23fc1ab1205171 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:c893e9669d9ce73010f5 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:caf128f8eabeca7f5497 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:cdaaad37badd1126c60d | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:ceaa906e9dd8113b0b6e | 0.1875 | 0.1875 | 0.0 | False | False |
| b0 | omni_math:cfd1da8604a325e749e2 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:d0e6044cea3bbf96e7f0 | 0.1875 | 0.1875 | 0.0 | False | False |
| b0 | omni_math:d1ea5c078fa8ad618098 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:d6dabfda17500fff2a13 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:da41daf7f13f02ff9c45 | 0.1875 | 0.1875 | 0.0 | False | False |
| b0 | omni_math:de362aad6d3a5e145c65 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:df5c4d142c90dc4ad606 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:e09424195c4914dac979 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:e208d2bcc4f3f9cf7c08 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:e314dea85e5dbc870c78 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:e702f49d33d7c654b256 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:e78524746d751e620871 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:e7ce0a36c776a990ac85 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:e807cc4dc4e348f9be56 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:e92ffc08bf5806e339aa | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:e9efcba41e0f997de523 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:f0f7a7e0c7a5ebd7113b | 0.1875 | 0.1875 | 0.0 | False | False |
| b0 | omni_math:f1069a0605746e77f02b | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:f117cb7d7aaf2a3f6f33 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:f6fbadc7366df3ed2a7a | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:fdb774ad28314a2f7d66 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:fe3ec27e8b23b5a6ec7c | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:fec65acff1d1410cc980 | 0.0 | 0.0 | 0.0 | True | False |
| b0 | omni_math:ffc5a7d0d1b7309e726f | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:13d6d0a2b4ffdc031fe0 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:17a7b9205f9601657433 | 0.25 | 0.25 | 0.0 | False | False |
| b1 | omni_math:1806f5474d6604668c8a | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:188bf878dbe2547a5a45 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:1ba74ec2fc0ace980a21 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:1e68298480ae1711e4bb | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:1ea9b2603a964b844abd | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:2a9f69bb19e9cedee0ce | 0.1875 | 0.1875 | 0.0 | False | False |
| b1 | omni_math:2e55f04edc10a9c42f9d | 0.1875 | 0.1875 | 0.0 | False | False |
| b1 | omni_math:2e8d18406c6f7bedc812 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:31379d4ab3c591a63f08 | 0.25 | 0.25 | 0.0 | False | False |
| b1 | omni_math:36155002d87dcdeb45f6 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:36274a9a20054259af19 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:36460647c8efd7884e2c | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:3bb367b81dac35a5b595 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:3e81816fbdb983295ded | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:3fdbc6e51becad821409 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:40e067a90f6a0432e579 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:440805f38fe688d92463 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:4542318efb9f13abf5a4 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:4650eb4adb331de38940 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:46e39a2cac5a7070682a | 0.1875 | 0.1875 | 0.0 | False | False |
| b1 | omni_math:48a0e06e36e421cba955 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:49477579d8ea176944a0 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:4a1aefda552febdd7ae2 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:4fc701caa56c100a1e65 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:521f74f37c6690eedac6 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:5610cdc27b941c22c02f | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:5aab3eb4acd50196d860 | 0.1875 | 0.1875 | 0.0 | False | False |
| b1 | omni_math:5ad36e29df97c9fc5f27 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:5c95294cc128be4d87ec | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:5d50ee9ea9d8e9ee2a8a | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:5e51ea6902200cdd3d18 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:6317a0595c2f92ff6874 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:6a41507301ee454a1eba | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:6b0e7ba7018d60cf3cb1 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:6e8d30af7f4895f14380 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:702c923b908f4dd81336 | 0.1875 | 0.1875 | 0.0 | False | False |
| b1 | omni_math:730e703e9f7bfc17b403 | 0.25 | 0.25 | 0.0 | False | False |
| b1 | omni_math:7393134ba2e5cbe0f33f | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:7464a4958e412012db6f | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:74ad2002447761200b5d | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:75a27a35658b0acbe7dd | 0.1875 | 0.1875 | 0.0 | False | False |
| b1 | omni_math:76b3ea8d366c2f38a330 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:7782e5a4b8566e5e44da | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:7850b7403769a872ac21 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:7c1e5ed8c3bce096d5d1 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:7d3d8b0f0158f566491d | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:842e8b1f3ecd8c82ac00 | 0.25 | 0.25 | 0.0 | False | False |
| b1 | omni_math:85f7cf50f56fd30b11a0 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:8924663d0e48f9815d9d | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:8cbe0c48c29daa190277 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:8cc783279bbf4cd36bd0 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:90f99c010d1bc49a1a2e | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:93f9c7cf0f56cd62bca6 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:9a2de48dad192786426e | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:a061e37809c94090334f | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:a23fdf11a060f9b15b4c | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:a3c15882576e19d381c0 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:a59a2acac11cf3c2bc0f | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:b06dd6f0e8140ab15e5f | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:b0cc3c51bf0918a50674 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:b2c87c609c5c81c6fd86 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:b3f533d75f5ac73a7eec | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:b3f872d6e654851a919e | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:ba16b0153a22069b60fa | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:ba25c09740c767adeea2 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:bb3a33713d807a7c207f | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:bc59bae1f8d5e2511c53 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:bc8aca553a71b9139a79 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:bfd266d75f332f5b6720 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:c7cd7b23fc1ab1205171 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:c893e9669d9ce73010f5 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:caf128f8eabeca7f5497 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:cdaaad37badd1126c60d | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:ceaa906e9dd8113b0b6e | 0.1875 | 0.1875 | 0.0 | False | False |
| b1 | omni_math:cfd1da8604a325e749e2 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:d0e6044cea3bbf96e7f0 | 0.1875 | 0.1875 | 0.0 | False | False |
| b1 | omni_math:d1ea5c078fa8ad618098 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:d6dabfda17500fff2a13 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:da41daf7f13f02ff9c45 | 0.1875 | 0.1875 | 0.0 | False | False |
| b1 | omni_math:de362aad6d3a5e145c65 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:df5c4d142c90dc4ad606 | 0.1875 | 0.1875 | 0.0 | False | False |
| b1 | omni_math:e09424195c4914dac979 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:e208d2bcc4f3f9cf7c08 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:e314dea85e5dbc870c78 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:e702f49d33d7c654b256 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:e78524746d751e620871 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:e7ce0a36c776a990ac85 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:e807cc4dc4e348f9be56 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:e92ffc08bf5806e339aa | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:e9efcba41e0f997de523 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:f0f7a7e0c7a5ebd7113b | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:f1069a0605746e77f02b | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:f117cb7d7aaf2a3f6f33 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:f6fbadc7366df3ed2a7a | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:fdb774ad28314a2f7d66 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:fe3ec27e8b23b5a6ec7c | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:fec65acff1d1410cc980 | 0.0 | 0.0 | 0.0 | True | False |
| b1 | omni_math:ffc5a7d0d1b7309e726f | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:13d6d0a2b4ffdc031fe0 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:17a7b9205f9601657433 | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:1806f5474d6604668c8a | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:188bf878dbe2547a5a45 | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:1ba74ec2fc0ace980a21 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:1e68298480ae1711e4bb | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:1ea9b2603a964b844abd | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:2a9f69bb19e9cedee0ce | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:2e55f04edc10a9c42f9d | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:2e8d18406c6f7bedc812 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:31379d4ab3c591a63f08 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:36155002d87dcdeb45f6 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:36274a9a20054259af19 | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:36460647c8efd7884e2c | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:3bb367b81dac35a5b595 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:3e81816fbdb983295ded | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:3fdbc6e51becad821409 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:40e067a90f6a0432e579 | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:440805f38fe688d92463 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:4542318efb9f13abf5a4 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:4650eb4adb331de38940 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:46e39a2cac5a7070682a | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:48a0e06e36e421cba955 | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:49477579d8ea176944a0 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:4a1aefda552febdd7ae2 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:4fc701caa56c100a1e65 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:521f74f37c6690eedac6 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:5610cdc27b941c22c02f | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:5aab3eb4acd50196d860 | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:5ad36e29df97c9fc5f27 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:5c95294cc128be4d87ec | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:5d50ee9ea9d8e9ee2a8a | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:5e51ea6902200cdd3d18 | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:6317a0595c2f92ff6874 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:6a41507301ee454a1eba | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:6b0e7ba7018d60cf3cb1 | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:6e8d30af7f4895f14380 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:702c923b908f4dd81336 | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:730e703e9f7bfc17b403 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:7393134ba2e5cbe0f33f | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:7464a4958e412012db6f | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:74ad2002447761200b5d | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:75a27a35658b0acbe7dd | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:76b3ea8d366c2f38a330 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:7782e5a4b8566e5e44da | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:7850b7403769a872ac21 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:7c1e5ed8c3bce096d5d1 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:7d3d8b0f0158f566491d | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:842e8b1f3ecd8c82ac00 | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:85f7cf50f56fd30b11a0 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:8924663d0e48f9815d9d | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:8cbe0c48c29daa190277 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:8cc783279bbf4cd36bd0 | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:90f99c010d1bc49a1a2e | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:93f9c7cf0f56cd62bca6 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:9a2de48dad192786426e | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:a061e37809c94090334f | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:a23fdf11a060f9b15b4c | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:a3c15882576e19d381c0 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:a59a2acac11cf3c2bc0f | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:b06dd6f0e8140ab15e5f | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:b0cc3c51bf0918a50674 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:b2c87c609c5c81c6fd86 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:b3f533d75f5ac73a7eec | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:b3f872d6e654851a919e | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:ba16b0153a22069b60fa | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:ba25c09740c767adeea2 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:bb3a33713d807a7c207f | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:bc59bae1f8d5e2511c53 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:bc8aca553a71b9139a79 | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:bfd266d75f332f5b6720 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:c7cd7b23fc1ab1205171 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:c893e9669d9ce73010f5 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:caf128f8eabeca7f5497 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:cdaaad37badd1126c60d | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:ceaa906e9dd8113b0b6e | 0.25 | 0.25 | 0.0 | False | False |
| b2 | omni_math:cfd1da8604a325e749e2 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:d0e6044cea3bbf96e7f0 | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:d1ea5c078fa8ad618098 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:d6dabfda17500fff2a13 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:da41daf7f13f02ff9c45 | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:de362aad6d3a5e145c65 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:df5c4d142c90dc4ad606 | 0.25 | 0.25 | 0.0 | False | False |
| b2 | omni_math:e09424195c4914dac979 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:e208d2bcc4f3f9cf7c08 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:e314dea85e5dbc870c78 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:e702f49d33d7c654b256 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:e78524746d751e620871 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:e7ce0a36c776a990ac85 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:e807cc4dc4e348f9be56 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:e92ffc08bf5806e339aa | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:e9efcba41e0f997de523 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:f0f7a7e0c7a5ebd7113b | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:f1069a0605746e77f02b | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:f117cb7d7aaf2a3f6f33 | 0.1875 | 0.1875 | 0.0 | False | False |
| b2 | omni_math:f6fbadc7366df3ed2a7a | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:fdb774ad28314a2f7d66 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:fe3ec27e8b23b5a6ec7c | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:fec65acff1d1410cc980 | 0.0 | 0.0 | 0.0 | True | False |
| b2 | omni_math:ffc5a7d0d1b7309e726f | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:13d6d0a2b4ffdc031fe0 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:17a7b9205f9601657433 | 0.25 | 0.25 | 0.0 | False | False |
| b4 | omni_math:1806f5474d6604668c8a | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:188bf878dbe2547a5a45 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:1ba74ec2fc0ace980a21 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:1e68298480ae1711e4bb | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:1ea9b2603a964b844abd | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:2a9f69bb19e9cedee0ce | 0.1875 | 0.1875 | 0.0 | False | False |
| b4 | omni_math:2e55f04edc10a9c42f9d | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:2e8d18406c6f7bedc812 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:31379d4ab3c591a63f08 | 0.25 | 0.25 | 0.0 | False | False |
| b4 | omni_math:36155002d87dcdeb45f6 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:36274a9a20054259af19 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:36460647c8efd7884e2c | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:3bb367b81dac35a5b595 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:3e81816fbdb983295ded | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:3fdbc6e51becad821409 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:40e067a90f6a0432e579 | 0.25 | 0.25 | 0.0 | False | False |
| b4 | omni_math:440805f38fe688d92463 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:4542318efb9f13abf5a4 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:4650eb4adb331de38940 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:46e39a2cac5a7070682a | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:48a0e06e36e421cba955 | 0.1875 | 0.1875 | 0.0 | False | False |
| b4 | omni_math:49477579d8ea176944a0 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:4a1aefda552febdd7ae2 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:4fc701caa56c100a1e65 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:521f74f37c6690eedac6 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:5610cdc27b941c22c02f | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:5aab3eb4acd50196d860 | 0.1875 | 0.1875 | 0.0 | False | False |
| b4 | omni_math:5ad36e29df97c9fc5f27 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:5c95294cc128be4d87ec | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:5d50ee9ea9d8e9ee2a8a | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:5e51ea6902200cdd3d18 | 0.25 | 0.25 | 0.0 | False | False |
| b4 | omni_math:6317a0595c2f92ff6874 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:6a41507301ee454a1eba | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:6b0e7ba7018d60cf3cb1 | 0.1875 | 0.1875 | 0.0 | False | False |
| b4 | omni_math:6e8d30af7f4895f14380 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:702c923b908f4dd81336 | 0.1875 | 0.1875 | 0.0 | False | False |
| b4 | omni_math:730e703e9f7bfc17b403 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:7393134ba2e5cbe0f33f | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:7464a4958e412012db6f | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:74ad2002447761200b5d | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:75a27a35658b0acbe7dd | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:76b3ea8d366c2f38a330 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:7782e5a4b8566e5e44da | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:7850b7403769a872ac21 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:7c1e5ed8c3bce096d5d1 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:7d3d8b0f0158f566491d | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:842e8b1f3ecd8c82ac00 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:85f7cf50f56fd30b11a0 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:8924663d0e48f9815d9d | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:8cbe0c48c29daa190277 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:8cc783279bbf4cd36bd0 | 0.25 | 0.25 | 0.0 | False | False |
| b4 | omni_math:90f99c010d1bc49a1a2e | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:93f9c7cf0f56cd62bca6 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:9a2de48dad192786426e | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:a061e37809c94090334f | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:a23fdf11a060f9b15b4c | 0.25 | 0.25 | 0.0 | False | False |
| b4 | omni_math:a3c15882576e19d381c0 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:a59a2acac11cf3c2bc0f | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:b06dd6f0e8140ab15e5f | 0.1875 | 0.1875 | 0.0 | False | False |
| b4 | omni_math:b0cc3c51bf0918a50674 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:b2c87c609c5c81c6fd86 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:b3f533d75f5ac73a7eec | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:b3f872d6e654851a919e | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:ba16b0153a22069b60fa | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:ba25c09740c767adeea2 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:bb3a33713d807a7c207f | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:bc59bae1f8d5e2511c53 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:bc8aca553a71b9139a79 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:bfd266d75f332f5b6720 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:c7cd7b23fc1ab1205171 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:c893e9669d9ce73010f5 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:caf128f8eabeca7f5497 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:cdaaad37badd1126c60d | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:ceaa906e9dd8113b0b6e | 0.25 | 0.25 | 0.0 | False | False |
| b4 | omni_math:cfd1da8604a325e749e2 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:d0e6044cea3bbf96e7f0 | 0.25 | 0.25 | 0.0 | False | False |
| b4 | omni_math:d1ea5c078fa8ad618098 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:d6dabfda17500fff2a13 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:da41daf7f13f02ff9c45 | 0.1875 | 0.1875 | 0.0 | False | False |
| b4 | omni_math:de362aad6d3a5e145c65 | 0.1875 | 0.1875 | 0.0 | False | False |
| b4 | omni_math:df5c4d142c90dc4ad606 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:e09424195c4914dac979 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:e208d2bcc4f3f9cf7c08 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:e314dea85e5dbc870c78 | 0.1875 | 0.1875 | 0.0 | False | False |
| b4 | omni_math:e702f49d33d7c654b256 | 0.1875 | 0.1875 | 0.0 | False | False |
| b4 | omni_math:e78524746d751e620871 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:e7ce0a36c776a990ac85 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:e807cc4dc4e348f9be56 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:e92ffc08bf5806e339aa | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:e9efcba41e0f997de523 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:f0f7a7e0c7a5ebd7113b | 0.1875 | 0.1875 | 0.0 | False | False |
| b4 | omni_math:f1069a0605746e77f02b | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:f117cb7d7aaf2a3f6f33 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:f6fbadc7366df3ed2a7a | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:fdb774ad28314a2f7d66 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:fe3ec27e8b23b5a6ec7c | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:fec65acff1d1410cc980 | 0.0 | 0.0 | 0.0 | True | False |
| b4 | omni_math:ffc5a7d0d1b7309e726f | 0.0 | 0.0 | 0.0 | True | False |
