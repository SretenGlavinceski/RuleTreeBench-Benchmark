# RuleArena Airline external evaluation

This directory contains the RuleArena Airline external evaluation used as a
separate validation experiment for RuleTreeBench.

The original RuleTreeBench benchmark remains unchanged at 1,700 prompts per
model. RuleArena Airline is reported separately and must not be merged into a
single 2,000-prompt accuracy.

## Evaluation set

- Benchmark: RuleArena
- Domain: Airline
- Examples: 300 total
- Difficulty levels: 100 Level 1, 100 Level 2, 100 Level 3
- Prompting: zero-shot
- Rule representation: original tabular rules
- Gold answers: recomputed with RuleArena's upstream `compute_answer.py`
- Scoring: deterministic exact match on the final total cost

The frozen generated dataset is:

`generated/rulearena_airline_300.jsonl`

Expected SHA256:

`e4cae947a8fad80eb47cada63805a5a0144a394ea08d6030ba4acc346673779e`

## Models

- `qwen3:4b-q4_K_M`
- `qwen3:8b-q4_K_M`
- `gemma3:4b`
- `gemma4:12b-it-q4_K_M`

## Inference settings

- `num_ctx = 8192`
- `num_predict = 4096`
- `temperature = 0`
- `seed = 42`
- `repeat_penalty = 1.0`
- `stream = false`
- thinking disabled where the Ollama model supports the `think` option

## Prepare the frozen dataset

From the repository root:

```powershell
python external_eval\rulearena\prepare_airline_dataset.py
```

## Run an engineering check

```powershell
python external_eval\rulearena\run_airline_ollama.py --engineering-test --model qwen3_4b
```

## Run the official 300-example evaluation

```powershell
python external_eval\rulearena\run_airline_ollama.py --full --model qwen3_4b
```

If an official run is interrupted, resume it with `--resume` rather than
`--overwrite`.
