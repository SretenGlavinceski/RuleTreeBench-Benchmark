# RuleTreeBench paper metrics

This file is generated from `combined_results.json`.

## Run integrity and overall scoring
- **Qwen3 4B**: 1640/1700 (96.5%) normalized accuracy; 12 recovered correct answers; 6 normalized but incorrect answers; 3 explicit-none responses; 0 parse failures; max output 790 tokens; max prompt+output 1124 tokens.
- **Qwen3 8B**: 1673/1700 (98.4%) normalized accuracy; 243 recovered correct answers; 2 normalized but incorrect answers; 0 explicit-none responses; 1 parse failures; max output 912 tokens; max prompt+output 1246 tokens.
- **Gemma 3 4B**: 1163/1700 (68.4%) normalized accuracy; 0 recovered correct answers; 0 normalized but incorrect answers; 303 explicit-none responses; 1 parse failures; max output 902 tokens; max prompt+output 1278 tokens.
- **Gemma 4 12B**: 1700/1700 (100.0%) normalized accuracy; 0 recovered correct answers; 0 normalized but incorrect answers; 0 explicit-none responses; 0 parse failures; max output 850 tokens; max prompt+output 1205 tokens.

## Full rule-selection tasks (Experiments 1, 2, and 4)
- **Qwen3 4B**: 1044/1100 correct (94.9%), 56 errors.
- **Qwen3 8B**: 1096/1100 correct (99.6%), 4 errors.
- **Gemma 3 4B**: 619/1100 correct (56.3%), 481 errors.
- **Gemma 4 12B**: 1100/1100 correct (100.0%), 0 errors.

## Experiment 1: general-rule selections
- **Qwen3 4B**: 44 general-rule errors, 0 distractors, 0 explicit-none, 0 parse failures.
- **Qwen3 8B**: 3 general-rule errors, 0 distractors, 0 explicit-none, 0 parse failures.
- **Gemma 3 4B**: 76 general-rule errors, 11 distractors, 111 explicit-none, 0 parse failures.
- **Gemma 4 12B**: 0 general-rule errors, 0 distractors, 0 explicit-none, 0 parse failures.

## Experiment 2: near-match selections
- **Qwen3 4B**: 5 near-match errors, 2 distractors, 3 explicit-none, 0 parse failures.
- **Qwen3 8B**: 0 near-match errors, 0 distractors, 0 explicit-none, 1 parse failures.
- **Gemma 3 4B**: 34 near-match errors, 27 distractors, 183 explicit-none, 1 parse failures.
- **Gemma 4 12B**: 0 near-match errors, 0 distractors, 0 explicit-none, 0 parse failures.

## Experiment 3
- **Qwen3 4B**: 4/600 diagnostic errors.
- **Qwen3 8B**: 23/600 diagnostic errors.
- **Gemma 3 4B**: 56/600 diagnostic errors.
- **Gemma 4 12B**: 0/600 diagnostic errors.

## Linked-context diagnostic
- **Qwen3 4B**: 1 full N1 failures while the linked near match was correct in isolation; 1 full N1 failures while both linked probes were correct.
- **Qwen3 8B**: 1 full N1 failures while the linked near match was correct in isolation; 1 full N1 failures while both linked probes were correct.
- **Gemma 3 4B**: 64 full N1 failures while the linked near match was correct in isolation; 46 full N1 failures while both linked probes were correct.
- **Gemma 4 12B**: 0 full N1 failures while the linked near match was correct in isolation; 0 full N1 failures while both linked probes were correct.

## Experiment 4 rescue
- **Qwen3 4B**: 44/44 rescued (100.0%); 0 persistent failures.
- **Qwen3 8B**: 3/3 rescued (100.0%); 0 persistent failures.
- **Gemma 3 4B**: 111/138 rescued (80.4%); 27 persistent failures.
- **Gemma 4 12B**: no eligible original errors; rescue rate undefined.

## Within-family descriptive comparison
- Qwen: Qwen3 4B had 60 total benchmark errors; Qwen3 8B had 27.
- Gemma: Gemma 3 4B had 537 total benchmark errors; Gemma 4 12B had 0.

These are descriptive comparisons of the evaluated model variants; they do not isolate parameter count as the only cause of the differences.
