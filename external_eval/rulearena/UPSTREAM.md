# Upstream RuleArena files

The files under `upstream/` are a minimal frozen copy of the files required for
the RuleArena Airline evaluation.

Upstream project:

- RuleArena: A Benchmark for Rule-Guided Reasoning with LLMs in Real-World Scenarios
- Repository: https://github.com/SkyRiver-2000/RuleArena
- License: MIT; the upstream `LICENSE` file is retained here.

Copied files:

- `LICENSE`
- `airline/auto_test.py`
- `airline/compute_answer.py`
- `airline/reference_rules.txt`
- `airline/fee_tables/bag_1/{0,1}.csv`
- `airline/fee_tables/bag_2/{0,1}.csv`
- `airline/fee_tables/bag_3/{0,1}.csv`
- `airline/fee_tables/bag_4/{0,1}.csv`
- `airline/synthesized_problems/comp_0.jsonl`
- `airline/synthesized_problems/comp_1.jsonl`
- `airline/synthesized_problems/comp_2.jsonl`

The external evaluation uses the original tabular rules and the 300 supplied
Airline problems. It does not run RuleArena's question generator or its
LLM-based fine-grained evaluation.

No upstream commit hash is asserted in this reconstructed copy because the
provided source archive did not contain `.git` metadata.
