# Tool Calling Evaluation Report v2

## 1. New Metrics Summary Table

| Method | Structural Validity Rate | First Tool Name Accuracy | **Overall F1** | **Exact Match Rate** | Tool Name F1 | Argument F1 | Avg Latency |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MiniMind SFT (Dense) | 56.00% | 51.33% | **46.26%** | 39.00% | 51.78% | 45.26% | 2.35s |
| MiniMind SFT + MoE | 95.00% | 91.00% | **85.31%** | 73.00% | 89.57% | 84.68% | 1.89s |
| MiniMind Agent RL (Dense) | 99.33% | 94.67% | **81.20%** | 56.33% | 92.05% | 77.70% | 5.77s |
| MiniMind Agent + MoE | 99.00% | 93.00% | **79.13%** | 54.00% | 89.43% | 76.08% | 2.02s |
| HF Qwen-0.5B SFT | 83.33% | 80.67% | **78.29%** | 74.00% | 80.83% | 77.80% | 2.85s |
| API qwen-turbo | 91.67% | 81.00% | **68.45%** | 52.00% | 79.98% | 66.58% | 0.84s |

## 2. Metric Descriptions

- **Overall F1**: A comprehensive F1 score evaluating both tool name recognition and argument matching. It replaces the previous `first_tool_name_accuracy`.
- **Exact Match Rate**: The strictest metric, requiring the tool name(s), arguments, and execution order to exactly match the gold standard.
- **Tool Name F1**: Calculates precision, recall, and F1 based on unordered sets of tool names; robust to multi-tool scenarios and ordering.
- **Argument F1**: Measures the correctness of argument key-value pairs.

## 3. Bad Case Statistical Analysis

### MiniMind SFT (Dense)

- Total Errors: 183 / 300
- Error Distribution:
  - `no_tool_call_in_output`: 132 cases (44.0%)
  - `wrong_args_only`: 23 cases (7.7%)
  - `missing_required_tools`: 14 cases (4.7%)
  - `wrong_first_and_missing`: 8 cases (2.7%)
  - `all_tools_wrong`: 3 cases (1.0%)
  - `wrong_first_but_others_ok`: 3 cases (1.0%)

### MiniMind SFT + MoE

- Total Errors: 27 / 100
- Error Distribution:
  - `wrong_args_only`: 13 cases (13.0%)
  - `no_tool_call_in_output`: 5 cases (5.0%)
  - `missing_required_tools`: 5 cases (5.0%)
  - `all_tools_wrong`: 2 cases (2.0%)
  - `wrong_first_but_others_ok`: 1 case (1.0%)
  - `wrong_first_and_missing`: 1 case (1.0%)

### MiniMind Agent RL (Dense)

- Total Errors: 131 / 300
- Error Distribution:
  - `wrong_args_only`: 107 cases (35.7%)
  - `missing_required_tools`: 8 cases (2.7%)
  - `wrong_first_but_others_ok`: 6 cases (2.0%)
  - `all_tools_wrong`: 5 cases (1.7%)
  - `wrong_first_and_missing`: 3 cases (1.0%)
  - `no_tool_call_in_output`: 2 cases (0.7%)
- Specific Error Patterns:
  - Most common tool name confusions (Predicted → Expected):
    - `(none)` → `get_exchange_rate`: 3 occurrences
    - `(none)` → `text_length`: 3 occurrences
    - `(none)` → `translate_text`: 3 occurrences
    - `(none)` → `calculate_math`: 2 occurrences
    - `(none)` → `unit_converter`: 2 occurrences
  - Most frequently missing tools:
    - `get_exchange_rate`: 4 occurrences
    - `text_length`: 4 occurrences
    - `calculate_math`: 3 occurrences
    - `translate_text`: 3 occurrences
    - `get_current_weather`: 2 occurrences

### MiniMind Agent + MoE

- Total Errors: 46 / 100
- Error Distribution:
  - `wrong_args_only`: 30 cases (30.0%)
  - `missing_required_tools`: 9 cases (9.0%)
  - `wrong_first_but_others_ok`: 4 cases (4.0%)
  - `wrong_first_and_missing`: 1 case (1.0%)
  - `no_tool_call_in_output`: 1 case (1.0%)
  - `all_tools_wrong`: 1 case (1.0%)

### HF Qwen-0.5B SFT

- Total Errors: 78 / 300
- Error Distribution:
  - `no_tool_call_in_output`: 50 cases (16.7%)
  - `wrong_args_only`: 16 cases (5.3%)
  - `wrong_first_and_missing`: 5 cases (1.7%)
  - `missing_required_tools`: 4 cases (1.3%)
  - `all_tools_wrong`: 3 cases (1.0%)
- Specific Error Patterns:
  - Most common tool name confusions (Predicted → Expected):
    - `(none)` → `get_current_time`: 3 occurrences
    - `(none)` → `get_current_weather`: 2 occurrences
    - `(none)` → `get_exchange_rate`: 1 occurrence
    - `(none)` → `calculate_math`: 1 occurrence
    - `(none)` → `unit_converter`: 1 occurrence
  - Most frequently missing tools:
    - `get_current_weather`: 30 occurrences
    - `get_exchange_rate`: 11 occurrences
    - `calculate_math`: 10 occurrences
    - `get_current_time`: 10 occurrences
    - `text_length`: 5 occurrences

### API qwen-turbo

- Total Errors: 144 / 300
- Error Distribution:
  - `wrong_args_only`: 81 cases (27.0%)
  - `no_tool_call_in_output`: 25 cases (8.3%)
  - `all_tools_wrong`: 21 cases (7.0%)
  - `wrong_first_and_missing`: 8 cases (2.7%)
  - `missing_required_tools`: 6 cases (2.0%)
  - `wrong_first_but_others_ok`: 3 cases (1.0%)
- Specific Error Patterns:
  - Most common tool name confusions (Predicted → Expected):
    - `web_search` → `get_current_weather`: 14 occurrences
    - `get_exchange_rate` → `get_current_weather`: 3 occurrences
    - `unit_converter` → `get_current_weather`: 3 occurrences
    - `calculate_math` → `get_current_weather`: 3 occurrences
    - `text_length` → `get_current_weather`: 2 occurrences
  - Most frequently missing tools:
    - `get_current_weather`: 39 occurrences
    - `translate_text`: 10 occurrences
    - `get_exchange_rate`: 8 occurrences
    - `calculate_math`: 3 occurrences
    - `unit_converter`: 3 occurrences
  - Most frequent extraneous tool calls:
    - `web_search`: 16 occurrences
    - `unit_converter`: 5 occurrences
    - `translate_text`: 4 occurrences
    - `random_number`: 4 occurrences
    - `text_length`: 3 occurrences
