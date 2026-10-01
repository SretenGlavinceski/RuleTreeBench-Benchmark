# RuleTreeBench

RuleTreeBench is a synthetic benchmark for studying context-neutral rule reasoning in large language models (LLMs). It uses abstract attributes and values rather than natural-language facts, so the task is determined by the rules given in each prompt.

This repository contains the benchmark generators, frozen benchmark data, local Ollama inference code, raw model responses, analysis code, and the derived metrics used in the accompanying research paper. It also includes an external evaluation on the RuleArena Airline domain, with the adapted evaluation dataset, local Ollama runner, final model responses, and run manifests.

## Task definition

A query assigns one value to each of six attributes:

- attributes: `a1` to `a6`
- values: `v1`, `v2`, `v3`

A rule contains a set of attribute-value conditions. A rule applies only when all of its conditions match the query. If several rules apply, the applicable rule with the largest number of conditions is the correct winner.

The main experiments use a four-condition winner and three main competitor types:

- **General rule:** a valid three-condition subset of the winner. It applies, but is less specific than the winner.
- **Near match:** a four-condition rule with exactly one failed condition. It has the same specificity as the winner but does not apply.
- **Distractor:** an invalid rule with at least two failed conditions.

The benchmark uses these operational error categories:

- **Overgeneralization:** selecting a valid but less-specific general rule instead of the winner.
- **Over-application:** selecting an invalid one-error near match.
- **Under-application:** producing an explicit no-rule response when a valid winner is present. In the normalized analysis, an empty final `ANSWER:` field is also grouped into the operational `explicit_none` category.

## Benchmark composition

The frozen benchmark contains **1,700 prompts per model**.

| Part | Families / groups | Prompts | Purpose | Answer format |
|---|---:|---:|---|---|
| Experiment 1 | 100 families | 400 | General-rule pressure | `R1`-`R10` |
| Experiment 2 | 100 families | 400 | Near-match pressure | `R1`-`R10` |
| Experiment 3A | 100 groups | 400 | Independent single-rule applicability | `YES` / `NO` |
| Experiment 3B | 100 linked families | 200 | Linked winner and near-match probes | `YES` / `NO` |
| Experiment 4 | 100 families | 300 | Applicable-only filtering | `R1`-`R10` |
| **Total** |  | **1,700** |  |  |

The benchmark generation seed is **42**.

### Experiment 1: general-rule pressure

Each family has four versions: `L0`, `L1`, `L2`, and `L4`. The query, winner, rule labels, and rule positions remain fixed within a family while the number of valid three-condition general rules changes.

Each prompt contains 10 rules. The correct winner has four conditions. The number of valid general rules is 0, 1, 2, or 4 depending on the level.

### Experiment 2: near-match pressure

Each family has four versions: `N0`, `N1`, `N2`, and `N4`. The query and winner remain fixed while selected invalid four-condition rules are replaced by one-error near matches.

Each prompt contains 10 rules. The `first_near_match` field stores the exact near match introduced in the `N1` version so that it can be reused in Experiment 3B.

### Experiment 3A: independent applicability checks

Each of 100 groups contains four single-rule probes:

- `valid_3`: valid three-condition rule
- `valid_4`: valid four-condition rule
- `one_error`: invalid rule with exactly one failed condition
- `two_error`: invalid rule with exactly two failed conditions

This gives 400 prompts: 200 with gold answer `YES` and 200 with gold answer `NO`. The invalid-rule lengths alternate across groups so that three- and four-condition invalid rules are balanced.

### Experiment 3B: linked applicability checks

For every Experiment 2 family, the winner and exact near match from the `N1` prompt are tested separately against the same query.

Each family contributes one winner probe with gold answer `YES` and one near-match probe with gold answer `NO`, giving 200 linked prompts.

### Experiment 4: applicable-only filtering

Experiment 4 is derived from Experiment 1 `L1`, `L2`, and `L4`. All inapplicable rules are removed while the original query, winner, and rule labels are preserved.

The filtered prompts contain:

- `L1`: 2 rules
- `L2`: 3 rules
- `L4`: 5 rules

There are 300 Experiment 4 prompts.

## Repository structure

```text
RuleTreeBench-Benchmark/
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│   ├── benchmark.yaml
│   ├── inference.yaml
│   └── models.yaml
│
├── src/
│   ├── models.py
│   ├── solver.py
│   ├── render.py
│   ├── parser.py
│   ├── generator_exp1.py
│   ├── generator_exp2.py
│   ├── generator_exp3.py
│   ├── build_exp1_dataset.py
│   ├── build_exp2_dataset.py
│   ├── build_exp3_independent_dataset.py
│   ├── build_exp3_linked_dataset.py
│   ├── build_exp4_dataset.py
│   ├── run_model.py
│   └── validate_all.py
│
├── tests/
│   ├── test_solver.py
│   ├── test_parser.py
│   ├── test_exp1.py
│   ├── test_exp2.py
│   └── test_exp3.py
│
├── data/
│   └── generated/
│       ├── experiment_1_general.jsonl
│       ├── experiment_2_nearmatch.jsonl
│       ├── experiment_3_independent.jsonl
│       ├── experiment_3_linked.jsonl
│       └── experiment_4_filtering.jsonl
│
├── results/
│   ├── raw/
│   │   ├── qwen3_4b/
│   │   ├── qwen3_8b/
│   │   ├── gemma3_4b/
│   │   └── gemma4_12b/
│   ├── combined/
│   │   └── combined_results.json
│   ├── metrics/
│   │   ├── condition_accuracy.csv
│   │   ├── exp3_diagnostics.csv
│   │   ├── exp4_rescue.csv
│   │   ├── integrity_summary.csv
│   │   ├── linked_context_summary.csv
│   │   ├── outcome_breakdown.csv
│   │   ├── overall_accuracy.csv
│   │   ├── paired_transitions.csv
│   │   ├── parser_accounting.csv
│   │   ├── selection_task_summary.csv
│   │   ├── paper_metrics.json
│   │   ├── paper_summary.md
│   │   └── response_patterns/
│   └── external/
│       └── rulearena_airline/
│           ├── qwen3_4b/
│           ├── qwen3_8b/
│           ├── gemma3_4b/
│           └── gemma4_12b/
│
├── external_eval/
│   └── rulearena/
│       ├── README.md
│       ├── UPSTREAM.md
│       ├── prepare_airline_dataset.py
│       ├── run_airline_ollama.py
│       ├── generated/
│       │   ├── rulearena_airline_300.jsonl
│       │   └── rulearena_airline_300.manifest.json
│       └── upstream/
│
└── analysis/
    ├── build_combined_results.py
    ├── analyze_combined_results.py
    └── analyze_response_patterns.py
```

## File descriptions

### Configuration

`config/benchmark.yaml` stores the benchmark seed, six attributes, three values, and expected dataset sizes.

`config/inference.yaml` stores the common inference settings used by the runner: temperature, seed, repetition penalty, default context size, default generation cap, thinking setting, streaming setting, and keep-alive duration.

`config/models.yaml` maps repository model names to Ollama model names, expected model digests, and thinking support. Gemma 3 4B also has model-specific context and generation overrides matching its official run.

### Core source files

`src/models.py` defines the `Query` and `Rule` data containers.

`src/solver.py` implements rule applicability, failed-condition counting, specificity, and winner selection.

`src/render.py` converts structured queries and rules into the exact model-facing prompts. Full rule-selection prompts require `ANSWER: R#`; Experiment 3 prompts require `ANSWER: YES` or `ANSWER: NO`.

`src/parser.py` contains the original strict response parsers used during inference.

### Dataset generation

`src/generator_exp1.py` generates Experiment 1 families, including the query, four-condition winner, possible three-condition general rules, and invalid distractors.

`src/generator_exp2.py` generates Experiment 2 families and possible one-error near matches.

`src/generator_exp3.py` generates the independent Experiment 3A probes.

The five `src/build_*_dataset.py` files convert generated families into the JSONL benchmark files stored in `data/generated/`. Each record contains its ID, structured benchmark data, deterministic gold answer, and fully rendered prompt.

`src/validate_all.py` checks the structure of all five datasets, including prompt counts, unique IDs, rule counts, unique winners, applicability, specificity, near-match counts, linked-family relationships, and Experiment 4 filtering constraints.

### Inference

`src/run_model.py` runs one configured model over all 1,700 prompts through the local Ollama `/api/chat` endpoint. It verifies the configured model digest, applies the model's inference settings, skips IDs already present in the output file, saves every raw response immediately, and creates a `run_manifest.json` for the run.

New inference output is stored under:

```text
results/raw/<model_name>/
```

### Analysis

`analysis/build_combined_results.py` joins the frozen benchmark files with the four raw response files, validates IDs and gold answers, and applies the formatting-only answer normalization used for the paper. It writes `results/combined/combined_results.json`.

`analysis/analyze_combined_results.py` reads the combined results and generates the main quantitative outputs used in the paper: accuracy, Wilson intervals, outcome categories, Experiment 3 diagnostics, paired-family changes, Experiment 4 rescue results, linked-context results, parser accounting, and integrity summaries.

`analysis/analyze_response_patterns.py` generates the secondary exploratory analyses used in the discussion, including response length, winner position, last-applicable-rule selections, and answer-only versus explanation behavior. It also writes CSV files for manual review of claims that depend on free-form explanation text.

### External RuleArena evaluation

`external_eval/rulearena/prepare_airline_dataset.py` adapts the frozen RuleArena Airline problems into the 300-example external evaluation file stored under `external_eval/rulearena/generated/`.

`external_eval/rulearena/run_airline_ollama.py` runs the adapted Airline evaluation through local Ollama and writes each model's responses and run manifest.

`external_eval/rulearena/UPSTREAM.md` records the provenance of the vendored RuleArena files. The corresponding upstream snapshot, including its license, Airline rules, fee tables, and synthesized problems, is preserved under `external_eval/rulearena/upstream/`.

Completed external runs are stored under `results/external/rulearena_airline/<model_name>/`. Each final model directory contains `responses.jsonl` and `manifest.json`.

### Tests

`tests/test_solver.py` tests applicability, failed-condition counting, specificity, no-rule cases, unique winners, and ties.

`tests/test_parser.py` tests the original rule-answer and YES/NO parsers.

`tests/test_exp1.py`, `tests/test_exp2.py`, and `tests/test_exp3.py` test the structural properties and reproducibility of the experiment generators.

## RuleTreeBench data format

All benchmark files are stored as **JSON Lines (`.jsonl`)**, with one complete JSON object per line.

### Experiments 1 and 2

Typical fields are:

```json
{
  "id": "exp1_f001_L0",
  "experiment": 1,
  "family_id": "exp1_f001",
  "level": 0,
  "query": {
    "a1": "v3",
    "a2": "v1",
    "a3": "v1",
    "a4": "v3",
    "a5": "v2",
    "a6": "v1"
  },
  "rules": [
    {
      "label": "R1",
      "conditions": {
        "a2": "v1",
        "a6": "v1",
        "a1": "v3",
        "a3": "v1"
      }
    }
  ],
  "gold_answer": "R1",
  "prompt": "...exact model-facing prompt..."
}
```

Experiment 2 also stores `first_near_match`, the exact one-error rule used for the linked Experiment 3B probe.

### Experiment 3

Experiment 3 stores one rule per prompt. Important fields are:

- `part`: `independent` or `linked`
- `probe_type`: `valid_3`, `valid_4`, `one_error`, `two_error`, `winner`, or `near_match`
- `specificity`: number of rule conditions
- `failed_conditions`: number of conditions that disagree with the query
- `gold_answer`: `YES` or `NO`

Linked probes additionally store `source_family_id` and `source_exp2_id` so they can be joined to the corresponding full `N1` prompt.

### Experiment 4

Experiment 4 stores `source_exp1_id`, which points to the original Experiment 1 prompt before filtering. Its `rules` list contains only rules that apply to the query.

## RuleTreeBench raw response format

Each model has a response file at:

```text
results/raw/<model_name>/responses.jsonl
```

Each line stores the benchmark ID, model output, parser result, correctness, and runtime metadata. Fields include:

- `id`, `experiment`, `model_name`, `ollama_model`
- `gold_answer`, `parsed_answer`, `correct`
- `unparsable`, `fallback_used`, `truncated`, `done_reason`
- `raw_response`
- `prompt_eval_count`, `eval_count`
- Ollama timing fields
- `wall_time_seconds`

Depending on the run version, `thinking_field` and `model_digest` may also be present.

Each model directory also contains `run_manifest.json`, which records the model identifier, Ollama version, dataset freeze commit, run start time, platform information, and inference settings used for that run.

## RuleTreeBench combined and normalized results

`results/combined/combined_results.json` joins all 1,700 benchmark records with the results of all four models.

The paper normalization is limited to the final answer field:

1. use the last `ANSWER:` marker in the response;
2. remove surrounding Markdown formatting characters (`*`, `_`, `` ` ``, `~`);
3. remove a duplicated leading `ANSWER:` prefix when present;
4. accept only `R1`-`R10` for rule-selection prompts;
5. treat `None`, `0`, or an empty final `ANSWER:` field as `explicit_none`;
6. leave invalid values such as `R0` as `parse_failure`;
7. accept only `YES` or `NO` for Experiment 3;
8. never infer an answer from the free-form explanation.

For full rule-selection prompts, normalized outcomes are classified as `correct`, `general`, `near_match`, `distractor`, `explicit_none`, or `parse_failure`.

## RuleTreeBench derived metric files

The main analysis writes these files to `results/metrics/`:

| File | Contents |
|---|---|
| `integrity_summary.csv` | Response counts, unique IDs, truncation checks, thinking-trace checks, and maximum token counts |
| `parser_accounting.csv` | Accounting for responses rejected by the original strict parser and their normalized outcomes |
| `overall_accuracy.csv` | Stored and normalized overall accuracy across all 1,700 prompts |
| `selection_task_summary.csv` | Accuracy on the 1,100 full rule-selection prompts from Experiments 1, 2, and 4 |
| `condition_accuracy.csv` | Accuracy and Wilson 95% intervals for each `L` and `N` condition |
| `outcome_breakdown.csv` | Counts by outcome type |
| `exp3_diagnostics.csv` | Experiment 3 false-NO and false-YES diagnostic error rates |
| `paired_transitions.csv` | Correct-to-wrong and wrong-to-correct changes within prompt families |
| `exp4_rescue.csv` | Eligible original errors, rescued errors, rescue rate, and persistent failures |
| `linked_context_summary.csv` | Linked Experiment 3 versus full-`N1` diagnostic counts |
| `paper_metrics.json` | Consolidated machine-readable paper metrics |
| `paper_summary.md` | Human-readable summary generated from the main metrics |

`results/metrics/response_patterns/` contains the exploratory response-pattern summaries and review CSV files.

## RuleTreeBench models and inference settings

The official RuleTreeBench runs used **Ollama 0.32.6**, temperature **0**, seed **42**, repetition penalty **1.0**, streaming disabled, thinking disabled, and `keep_alive: 30m`.

| Repository name | Ollama model | Model digest | Context | Generation cap |
|---|---|---|---:|---:|
| `qwen3_4b` | `qwen3:4b-q4_K_M` | `2bfd38a7daaf` | 8192 | 4096 |
| `qwen3_8b` | `qwen3:8b-q4_K_M` | `500a1f067a9f` | 8192 | 4096 |
| `gemma3_4b` | `gemma3:4b` | `a2af6cc3eb7f` | 4096 | 1024 |
| `gemma4_12b` | `gemma4:12b-it-q4_K_M` | `4eb23ef187e2` | 8192 | 4096 |

The default values in `config/inference.yaml` are 8192 context tokens and a 4096-token generation cap. `config/models.yaml` automatically overrides these to 4096 / 1024 for Gemma 3 4B.

The exact settings of the completed RuleTreeBench runs are preserved in `results/raw/<model_name>/run_manifest.json`.

## Installation

Python **3.10 or newer** is recommended.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Ollama must be installed separately to rerun model inference. The reported RuleTreeBench runs used Ollama 0.32.6; the external RuleArena run metadata is preserved separately in the manifests under `results/external/rulearena_airline/`.

## Tests and dataset validation

Run the unit tests from the repository root:

```bash
python -m pytest -q
```

The current test suite contains **53 tests**.

Validate the frozen datasets with:

```bash
python -m src.validate_all
```

A successful validation checks all five benchmark parts and all 1,700 prompt IDs.

## Rebuilding the benchmark datasets

The generators are deterministic with seed 42. Run the build scripts in this order because Experiment 3B depends on Experiment 2 and Experiment 4 depends on Experiment 1:

```bash
python -m src.build_exp1_dataset
python -m src.build_exp2_dataset
python -m src.build_exp3_independent_dataset
python -m src.build_exp3_linked_dataset
python -m src.build_exp4_dataset
python -m src.validate_all
```

The generated files are written directly to `data/generated/`.

## Rerunning RuleTreeBench model inference

Run one configured model with:

```bash
python -m src.run_model <model_name>
```

where `<model_name>` is one of:

```text
qwen3_4b
qwen3_8b
gemma3_4b
gemma4_12b
```

Before starting, the runner checks the local Ollama model digest against `config/models.yaml`. It reads the five files in `data/generated/` and writes results to:

```text
results/raw/<model_name>/
```

The runner is resumable: benchmark IDs already present in `responses.jsonl` are skipped. To perform a completely new run, preserve the official result directory first and run from an empty output directory for that model.

## Reproducing the RuleTreeBench analysis from the included artifacts

The repository already contains the frozen benchmark files and the four official raw response files, so the paper metrics can be regenerated without rerunning the models.

### 1. Rebuild the combined result file

```bash
python analysis/build_combined_results.py \
  --dataset \
    data/generated/experiment_1_general.jsonl \
    data/generated/experiment_2_nearmatch.jsonl \
    data/generated/experiment_3_independent.jsonl \
    data/generated/experiment_3_linked.jsonl \
    data/generated/experiment_4_filtering.jsonl \
  --responses \
    results/raw/qwen3_4b/responses.jsonl \
    results/raw/qwen3_8b/responses.jsonl \
    results/raw/gemma3_4b/responses.jsonl \
    results/raw/gemma4_12b/responses.jsonl \
  --output results/combined/combined_results.json
```

Expected validation output includes:

```text
Validated 1700 benchmark prompts.
Validated 4 model response files.
Wrote 1700 joined benchmark records to results/combined/combined_results.json
```

### 2. Regenerate the main metrics

```bash
python analysis/analyze_combined_results.py \
  --input results/combined/combined_results.json \
  --out-dir results/metrics
```

This analyzes **6,800 model-response records**: 1,700 prompts × 4 models.

### 3. Regenerate the exploratory response-pattern analysis

```bash
python analysis/analyze_response_patterns.py \
  --input results/combined/combined_results.json \
  --out-dir results/metrics/response_patterns
```

This analysis uses only the stored combined results and does not run a model.

## RuleTreeBench end-to-end workflow

```text
Generate benchmark families
        ↓
Render prompts and compute deterministic gold answers
        ↓
Validate the five benchmark files
        ↓
Run each model on the same 1,700 prompts
        ↓
Store raw responses and run manifests
        ↓
Join benchmark data with model responses
        ↓
Normalize only the final answer field
        ↓
Classify outcomes from the frozen rule structure
        ↓
Generate main metrics and diagnostics
        ↓
Generate exploratory response-pattern summaries
```

## Frozen RuleTreeBench benchmark

The benchmark was frozen before the official model runs. The dataset freeze commit recorded by the run manifests is:

```text
652d697fceeb2d9b3132fd48efc2e108a640b793
```

SHA-256 hashes of the five frozen datasets:

| Dataset | SHA-256 |
|---|---|
| `experiment_1_general.jsonl` | `76835c24ffbff7891dc3a7acd75479bea142e8f7fcde72ca2d2d4b58a2c145c8` |
| `experiment_2_nearmatch.jsonl` | `01156c9a2382ba80c252905f983473960decff2c4415572c7adcd9a9456581eb` |
| `experiment_3_independent.jsonl` | `3d8fd31a67c264c23eda897c257ed4f6e5d91731fe188cae39c82ea554f05821` |
| `experiment_3_linked.jsonl` | `8c3a634527dfd90d3a3c4f767df805203bc2f8fed2afbebeaf08b8dda81fa069` |
| `experiment_4_filtering.jsonl` | `d459e8f846ca817fc09e57d7cd8064f4d1f683573574620c24c7ccd0de1c43e9` |

## Reported RuleTreeBench normalized results

The included analysis files report the following normalized accuracy across all 1,700 prompts:

| Model | Correct | Total | Accuracy |
|---|---:|---:|---:|
| Qwen3 4B | 1,640 | 1,700 | 96.5% |
| Qwen3 8B | 1,673 | 1,700 | 98.4% |
| Gemma 3 4B | 1,163 | 1,700 | 68.4% |
| Gemma 4 12B | 1,700 | 1,700 | 100.0% |

These pooled values combine experiments with different task structures. Detailed condition-level and diagnostic results are stored in `results/metrics/`.

## External evaluation: RuleArena Airline

To test whether the observed behavior extends beyond the synthetic RuleTreeBench setting, the repository includes a separate external evaluation on the **Airline** domain of [RuleArena](https://github.com/SkyRiver-2000/RuleArena), a benchmark for rule-guided reasoning in real-world scenarios.

This evaluation is intentionally kept separate from the 1,700-prompt RuleTreeBench benchmark. The Airline task uses realistic reference rules and generated passenger scenarios, so its scores should not be interpreted as directly comparable to the pooled RuleTreeBench accuracy above.

### External dataset

The adapted external dataset contains **300 Airline examples**, with 100 examples from each of the three upstream complexity groups (`comp_0`, `comp_1`, and `comp_2`).

```text
external_eval/rulearena/generated/rulearena_airline_300.jsonl
```

Its SHA-256 hash is:

```text
e4cae947a8fad80eb47cada63805a5a0144a394ea08d6030ba4acc346673779e
```

The dataset-preparation manifest is stored at:

```text
external_eval/rulearena/generated/rulearena_airline_300.manifest.json
```

The adapted evaluation procedure is documented in `external_eval/rulearena/README.md`, while `external_eval/rulearena/UPSTREAM.md` records the provenance of the included RuleArena snapshot.

### External inference settings

All four external runs used temperature **0**, seed **42**, repetition penalty **1.0**, an **8192-token context**, a **4096-token generation cap**, streaming disabled, and thinking disabled. Exact per-run model identifiers, digests, runtime metadata, timestamps, dataset hash, and response-file hash are preserved in each `manifest.json`.

| Repository name | Ollama model | Model digest prefix |
|---|---|---|
| `qwen3_4b` | `qwen3:4b-q4_K_M` | `2bfd38a7daaf` |
| `qwen3_8b` | `qwen3:8b-q4_K_M` | `500a1f067a9f` |
| `gemma3_4b` | `gemma3:4b` | `a2af6cc3eb7f` |
| `gemma4_12b` | `gemma4:12b-it-q4_K_M` | `4eb23ef187e2` |

### External results

The completed RuleArena Airline runs contain 300 responses per model. The external evaluation produced the following correct-answer counts:

| Model | `comp_0` | `comp_1` | `comp_2` | Correct | Total | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3 4B | 1 | 0 | 0 | 1 | 300 | 0.33% |
| Qwen3 8B | 0 | 0 | 0 | 0 | 300 | 0.00% |
| Gemma 3 4B | 0 | 0 | 0 | 0 | 300 | 0.00% |
| Gemma 4 12B | 11 | 2 | 1 | 14 | 300 | 4.67% |

The final external artifacts are stored at:

```text
results/external/rulearena_airline/<model_name>/
```

Each directory contains the preserved `responses.jsonl` file and its `manifest.json`. No separate external-analysis script is required to reproduce the stored raw run artifacts; the preparation and inference procedure is documented under `external_eval/rulearena/`.

## Reproducibility notes

- RuleTreeBench raw responses are preserved unchanged in `results/raw/`.
- RuleArena Airline raw responses and manifests are preserved under `results/external/rulearena_airline/`.
- Incorrect responses were not regenerated during the official runs.
- Interrupted runs resumed from the remaining benchmark IDs.
- The analysis does not infer answers from model explanations.
- Main proportion estimates use Wilson 95% confidence intervals.
- The reported analysis is descriptive; significance tests are not used as the basis for the conclusions.
- Temperature 0 and a fixed seed reduce decoding variability but do not guarantee bit-identical output across different hardware, runtime versions, or model builds.

## External benchmark provenance

The external Airline evaluation builds on the open-source [RuleArena](https://github.com/SkyRiver-2000/RuleArena) benchmark by Ruiwen Zhou, Wenyue Hua, Liangming Pan, Sitao Cheng, Xiaobao Wu, En Yu, and William Yang Wang. The vendored upstream files retain their original license and are accompanied by provenance notes in `external_eval/rulearena/UPSTREAM.md`.
