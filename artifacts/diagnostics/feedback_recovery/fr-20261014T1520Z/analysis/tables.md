# 反馈恢复诊断：结果表（自动生成）

生成脚本：`scripts/analysis/analyze_feedback_recovery.py`；配对 bootstrap 次数=10000，seed=156994340。

## A 门槛

| 检查 | 值 | 阈值 | 通过 |
|---|---|---|---|
| executed_tool_call_trajectories | 20 | 5 | True |
| infrastructure_failures | 0 | 0 | True |
| parse_error_step_rate | 0.000 | 0.100 | True |
| length_truncated_without_action_rate | 0.024 | 0.100 | True |
| hidden_label_leakage | [] | n/a | True |

## B 自然运行（§6）

| 指标 | 值 |
|---|---|
| accuracy | 0.100 |
| conditional_accuracy_after_exec_error | 0.093 |
| error_counts | {'E_EXEC': 56, 'E_INFRA': 1, 'E_TIMEOUT': 15, 'call_form_errors': 11, 'executed_calls': 172, 'length_truncated_turns': 7} |
| exec_error_trajectory_rate | 0.338 |
| executed_tool_call_rate | 0.975 |
| infrastructure_trajectory_rate | 0.006 |
| no_tool_call_rate | 0.025 |
| tasks_with_any_exec_error | 40 |
| termination_counts | {'final': 145, 'max_steps': 1, 'python_time_budget': 14} |
| timeout_trajectory_rate | 0.094 |
| trajectories | 160 |
| trajectories_with_exec_error | 54 |

## C 锚点后正确率（仅完整锚点组）

完整锚点组=27，不完整=0，基础设施缺失分支=0；重试样本=0，重试后仍缺失=0，协议违规=0。

| 臂 | 锚点后正确率 |
|---|---|
| F | 0.083 |
| P | 0.074 |
| M | 0.083 |

## 配对差异（锚点级配对 bootstrap）

| 对比 | 差值 | 95% 区间下 | 95% 区间上 | 锚点数 | 退化 |
|---|---|---|---|---|---|
| F - M | 0.000 | -0.037 | 0.037 | 27 | False |
| P - F | -0.009 | -0.065 | 0.046 | 27 | False |

主对比 F−M，次对比 P−F，均在结果出来前指定，不按结果交换主次。

## 缺失敏感性

| 场景 | 对比 | 差值 | 95% 区间下 | 95% 区间上 |
|---|---|---|---|---|
| missing_as_correct | F - M | 0.000 | -0.037 | 0.037 |
| missing_as_correct | P - F | -0.009 | -0.065 | 0.046 |
| missing_as_wrong | F - M | 0.000 | -0.037 | 0.037 |
| missing_as_wrong | P - F | -0.009 | -0.065 | 0.046 |

## 辅助指标（§8.2，分母见列名）

| 指标 | F | P | M |
|---|---|---|---|
| accuracy | 0.083 | 0.074 | 0.083 |
| branches | 108 | 108 | 108 |
| budget_terminated_rate | 0.056 | 0.065 | 0.056 |
| called_again_rate | 0.056 | 0.074 | 0.009 |
| correct | 9 | 8 | 9 |
| correct_only_suffix_tokens_mean | 74.000 | 1551.000 | 15.000 |
| direct_final_without_calling_rate | 0.907 | 0.898 | 0.944 |
| invalid_actions_total | 36 | 40 | 50 |
| next_call_executed_and_code_changed_PROXY_only | 0.000 | 0.009 | 0.000 |
| next_call_success_all_branches | 0.000 | 0.009 | 0.000 |
| next_call_success_conditional | 0.000 | 0.125 | 0.000 |
| repeated_original_code_rate | 0.028 | 0.009 | 0.009 |
| repeated_same_error_class_rate | 0.028 | 0.009 | 0.009 |
| suffix_generated_tokens_mean | 734.231 | 776.528 | 947.250 |
| suffix_inference_ms_estimate_mean | 15835.222 | 17096.713 | 19177.083 |
| suffix_inference_ms_measured_turns | 57 | 49 | 45 |
| suffix_python_seconds_mean | 0.120 | 0.447 | 0.003 |
| suffix_tool_calls_mean | 0.250 | 0.148 | 0.009 |

## 干预文本（冻结）

- P 提示：`The tool execution failed. Inspect the error feedback and your previous code, correct the cause before retrying, and continue solving the original problem.`
- M 标记：`Tool execution failed.`
