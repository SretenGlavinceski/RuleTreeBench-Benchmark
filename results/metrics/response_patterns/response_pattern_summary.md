# RuleTreeBench exploratory response-pattern analysis

These analyses use the existing `combined_results.json`; no new model runs are needed.

## Objective metrics

- Qwen3 4B made 44 Experiment 1 errors.
- In 31/44 of them (70.5%), the selected rule was the last structurally applicable rule in prompt order.
- Uniform choice among the available general rules would give an expected count of 21.25 such selections.
- Qwen3 8B Experiment 3 (answer_only): 320/343 correct, 23 errors (93.3% accuracy).
- Qwen3 8B Experiment 3 (explanation): 257/257 correct, 0 errors (100.0% accuracy).

### Median output tokens in Experiments 1 and 2

- Qwen3 4B correct: median 438.0 tokens (n=746).
- Qwen3 4B incorrect: median 453.0 tokens (n=54).
- Gemma 3 4B correct: median 542.0 tokens (n=357).
- Gemma 3 4B incorrect: median 544.0 tokens (n=443).

### Gemma 3 4B error rate by winner position

- Positions 1-3: 47.2% error rate (n=284; 134 errors).
- Positions 4-7: 56.2% error rate (n=276; 155 errors).
- Positions 8-10: 64.2% error rate (n=240; 154 errors).

## Manual text review

Two claims depend on interpreting free-form explanation text and should be checked manually:

- `qwen3_4b_exp1_error_review.csv`: fill `manual_winner_identified_as_applicable` with yes/no.
- `gemma3_4b_explicit_none_review.csv`: fill the two manual columns with yes/no.

The heuristic columns are only search aids; do not report them as final paper metrics without review.
