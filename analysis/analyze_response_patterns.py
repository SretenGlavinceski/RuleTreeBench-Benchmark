#!/usr/bin/env python3
"""Generate exploratory response-pattern summaries and review files."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


def load_data(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "records" not in data:
        raise ValueError("Input does not look like combined_results.json")

    return data


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct(part: int | float, whole: int | float) -> float:
    if whole == 0:
        return 0.0
    return round(100.0 * part / whole, 3)


def rule_applies(rule: dict, query: dict) -> bool:
    """Return True when every condition in the rule matches the query."""
    return all(query[attr] == value for attr, value in rule["conditions"].items())


def applicable_labels(record: dict) -> list[str]:
    """Return applicable rule labels in their original prompt order."""
    query = record["benchmark"]["query"]
    rules = record["benchmark"]["rules"]
    return [rule["label"] for rule in rules if rule_applies(rule, query)]


def winner_position(record: dict) -> int:
    """Return the 1-based position of the gold winner in the rule list."""
    winner = record["gold_answer"]

    for index, rule in enumerate(record["benchmark"]["rules"], start=1):
        if rule["label"] == winner:
            return index

    raise ValueError(f"Winner {winner} not found in {record['id']}")


def is_answer_only(raw_response: str) -> bool:
    """True if the response contains only the final YES/NO answer line."""
    text = raw_response.strip()
    text = re.sub(r"[*_`~]", "", text).strip()
    return bool(re.fullmatch(r"ANSWER:\s*(YES|NO)\s*", text, flags=re.IGNORECASE))


def find_rule(record: dict, label: str) -> dict:
    for rule in record["benchmark"]["rules"]:
        if rule["label"] == label:
            return rule
    raise ValueError(f"Rule {label} not found in {record['id']}")


def clean_line(line: str) -> str:
    return re.sub(r"[*_`~]", "", line).strip()


def winner_positive_mention_heuristic(record: dict, raw_response: str) -> bool:
    """Simple helper for manual review, not a final semantic judgment.

    Returns True when a line containing the gold winner also contains
    obvious positive applicability language such as 'applies' or
    'all conditions match', while excluding common negative forms.
    """
    winner = record["gold_answer"]

    negative_phrases = (
        "does not apply",
        "doesn't apply",
        "does not match",
        "doesn't match",
        "not applicable",
    )

    for raw_line in raw_response.splitlines():
        line = clean_line(raw_line)
        lower = line.lower()

        if not re.search(rf"\b{re.escape(winner)}\b", line, flags=re.IGNORECASE):
            continue

        if any(phrase in lower for phrase in negative_phrases):
            continue

        if (
            "applies" in lower
            or "applicable" in lower
            or "all conditions match" in lower
            or "all conditions are met" in lower
            or "matches all conditions" in lower
        ):
            return True

    return False


def winner_negative_mention_heuristic(record: dict, raw_response: str) -> bool:
    """Simple helper for manual review of Gemma explicit-none responses."""
    winner = record["gold_answer"]

    negative_phrases = (
        "does not apply",
        "doesn't apply",
        "does not match",
        "doesn't match",
        "this fails",
        "fails because",
    )

    for raw_line in raw_response.splitlines():
        line = clean_line(raw_line)
        lower = line.lower()

        if not re.search(rf"\b{re.escape(winner)}\b", line, flags=re.IGNORECASE):
            continue

        if any(phrase in lower for phrase in negative_phrases):
            return True

    return False


def format_conditions(rule: dict) -> str:
    return " AND ".join(
        f"{attr}={value}" for attr, value in rule["conditions"].items()
    )


def analyze(input_path: Path, out_dir: Path) -> None:
    data = load_data(input_path)
    records = data["records"]
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []

    # Qwen3 4B Experiment 1 errors and prompt position
    qwen_errors = []

    for record in records:
        if record["experiment"] != 1:
            continue

        result = record["models"]["qwen3_4b"]
        if result["normalized_correct"]:
            continue

        applicable = applicable_labels(record)
        winner = record["gold_answer"]
        selected = result["normalized_answer"]
        last_applicable = applicable[-1]
        general_rules = [label for label in applicable if label != winner]
        selected_is_last = selected == last_applicable

        # If the model chose uniformly among the available general rules,
        # this is the probability that it would choose the last applicable
        # rule in this particular prompt.
        expected_probability = 0.0
        if general_rules and last_applicable in general_rules:
            expected_probability = 1.0 / len(general_rules)

        qwen_errors.append(
            {
                "id": record["id"],
                "condition": record["condition"],
                "winner": winner,
                "selected": selected,
                "outcome": result["outcome"],
                "applicable_rules": ", ".join(applicable),
                "general_rules": ", ".join(general_rules),
                "last_applicable": last_applicable,
                "selected_is_last_applicable": selected_is_last,
                "uniform_expected_probability": round(expected_probability, 4),
                "winner_positive_mention_heuristic": winner_positive_mention_heuristic(
                    record, result["raw_response"]
                ),
                "manual_winner_identified_as_applicable": "",
                "raw_response": result["raw_response"],
            }
        )

    qwen_last_count = sum(row["selected_is_last_applicable"] for row in qwen_errors)
    qwen_expected_count = sum(row["uniform_expected_probability"] for row in qwen_errors)

    summary_rows.extend(
        [
            {
                "analysis": "qwen3_4b_exp1_errors",
                "model": "Qwen3 4B",
                "group": "Experiment 1 errors",
                "n": len(qwen_errors),
                "value": len(qwen_errors),
                "unit": "responses",
                "note": "Total normalized Experiment 1 errors.",
            },
            {
                "analysis": "qwen3_4b_last_applicable",
                "model": "Qwen3 4B",
                "group": "Experiment 1 errors",
                "n": len(qwen_errors),
                "value": qwen_last_count,
                "unit": "responses",
                "note": f"{pct(qwen_last_count, len(qwen_errors)):.1f}% selected the last structurally applicable rule.",
            },
            {
                "analysis": "qwen3_4b_last_applicable_uniform_expectation",
                "model": "Qwen3 4B",
                "group": "Experiment 1 errors",
                "n": len(qwen_errors),
                "value": round(qwen_expected_count, 3),
                "unit": "expected responses",
                "note": "Expected count if each error chose uniformly among the available general rules.",
            },
        ]
    )

    write_csv(out_dir / "qwen3_4b_exp1_error_review.csv", qwen_errors)

    # Gemma 3 4B explicit-none responses for manual review
    gemma_none_rows = []

    for record in records:
        if record["experiment"] not in {1, 2}:
            continue

        result = record["models"]["gemma3_4b"]
        if result["parse_status"] != "explicit_none":
            continue

        winner_rule = find_rule(record, record["gold_answer"])

        gemma_none_rows.append(
            {
                "id": record["id"],
                "experiment": record["experiment"],
                "condition": record["condition"],
                "winner": record["gold_answer"],
                "winner_conditions": format_conditions(winner_rule),
                "winner_negative_mention_heuristic": winner_negative_mention_heuristic(
                    record, result["raw_response"]
                ),
                "manual_winner_said_not_to_match": "",
                "manual_verdict_evidence_mismatch": "",
                "raw_response": result["raw_response"],
            }
        )

    summary_rows.append(
        {
            "analysis": "gemma3_4b_explicit_none_review",
            "model": "Gemma 3 4B",
            "group": "Experiments 1 and 2",
            "n": len(gemma_none_rows),
            "value": len(gemma_none_rows),
            "unit": "responses",
            "note": "Rows exported for manual verdict-evidence review.",
        }
    )

    write_csv(out_dir / "gemma3_4b_explicit_none_review.csv", gemma_none_rows)

    # Qwen3 8B answer-only vs. explanation responses in Experiment 3
    qwen8_groups = {
        "answer_only": [],
        "explanation": [],
    }

    for record in records:
        if record["experiment"] != 3:
            continue

        result = record["models"]["qwen3_8b"]
        group_name = "answer_only" if is_answer_only(result["raw_response"]) else "explanation"
        qwen8_groups[group_name].append(result)

    for group_name, group in qwen8_groups.items():
        correct = sum(result["normalized_correct"] for result in group)
        errors = len(group) - correct

        summary_rows.append(
            {
                "analysis": "qwen3_8b_exp3_response_style",
                "model": "Qwen3 8B",
                "group": group_name,
                "n": len(group),
                "value": round(pct(correct, len(group)), 3),
                "unit": "accuracy_pct",
                "note": f"{correct} correct, {errors} errors.",
            }
        )

    # Output length for correct vs. incorrect responses in Experiments 1 and 2
    for model_key, display_name in [
        ("qwen3_4b", "Qwen3 4B"),
        ("gemma3_4b", "Gemma 3 4B"),
    ]:
        correct_lengths = []
        incorrect_lengths = []

        for record in records:
            if record["experiment"] not in {1, 2}:
                continue

            result = record["models"][model_key]
            length = result["eval_count"]

            if result["normalized_correct"]:
                correct_lengths.append(length)
            else:
                incorrect_lengths.append(length)

        for group_name, values in [
            ("correct", correct_lengths),
            ("incorrect", incorrect_lengths),
        ]:
            summary_rows.append(
                {
                    "analysis": "exp1_exp2_output_length",
                    "model": display_name,
                    "group": group_name,
                    "n": len(values),
                    "value": float(statistics.median(values)),
                    "unit": "median_output_tokens",
                    "note": "Uses eval_count from Experiments 1 and 2 only.",
                }
            )

    # Gemma 3 4B winner position vs. error rate in Experiments 1 and 2
    position_groups = {
        "1-3": [],
        "4-7": [],
        "8-10": [],
    }

    for record in records:
        if record["experiment"] not in {1, 2}:
            continue

        position = winner_position(record)

        if position <= 3:
            group_name = "1-3"
        elif position <= 7:
            group_name = "4-7"
        else:
            group_name = "8-10"

        result = record["models"]["gemma3_4b"]
        position_groups[group_name].append(not result["normalized_correct"])

    for group_name, error_flags in position_groups.items():
        errors = sum(error_flags)

        summary_rows.append(
            {
                "analysis": "gemma3_4b_winner_position",
                "model": "Gemma 3 4B",
                "group": group_name,
                "n": len(error_flags),
                "value": round(pct(errors, len(error_flags)), 3),
                "unit": "error_rate_pct",
                "note": f"{errors} errors among {len(error_flags)} Experiment 1+2 prompts.",
            }
        )

    # Write summary files
    write_csv(out_dir / "response_pattern_summary.csv", summary_rows)

    summary_lines = [
        "# RuleTreeBench exploratory response-pattern analysis",
        "",
        "These analyses use the existing `combined_results.json`; no new model runs are needed.",
        "",
        "## Objective metrics",
        "",
        f"- Qwen3 4B made {len(qwen_errors)} Experiment 1 errors.",
        f"- In {qwen_last_count}/{len(qwen_errors)} of them ({pct(qwen_last_count, len(qwen_errors)):.1f}%), the selected rule was the last structurally applicable rule in prompt order.",
        f"- Uniform choice among the available general rules would give an expected count of {qwen_expected_count:.2f} such selections.",
    ]

    for group_name in ["answer_only", "explanation"]:
        group = qwen8_groups[group_name]
        correct = sum(result["normalized_correct"] for result in group)
        errors = len(group) - correct
        summary_lines.append(
            f"- Qwen3 8B Experiment 3 ({group_name}): {correct}/{len(group)} correct, {errors} errors ({pct(correct, len(group)):.1f}% accuracy)."
        )

    summary_lines.extend(["", "### Median output tokens in Experiments 1 and 2", ""])

    for row in summary_rows:
        if row["analysis"] == "exp1_exp2_output_length":
            summary_lines.append(
                f"- {row['model']} {row['group']}: median {row['value']:.1f} tokens (n={row['n']})."
            )

    summary_lines.extend(["", "### Gemma 3 4B error rate by winner position", ""])

    for row in summary_rows:
        if row["analysis"] == "gemma3_4b_winner_position":
            summary_lines.append(
                f"- Positions {row['group']}: {row['value']:.1f}% error rate (n={row['n']}; {row['note'].split()[0]} errors)."
            )

    summary_lines.extend(
        [
            "",
            "## Manual text review",
            "",
            "Two claims depend on interpreting free-form explanation text and should be checked manually:",
            "",
            "- `qwen3_4b_exp1_error_review.csv`: fill `manual_winner_identified_as_applicable` with yes/no.",
            "- `gemma3_4b_explicit_none_review.csv`: fill the two manual columns with yes/no.",
            "",
            "The heuristic columns are only search aids; do not report them as final paper metrics without review.",
        ]
    )

    (out_dir / "response_pattern_summary.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print(f"Read {len(records)} benchmark records.")
    print(f"Wrote exploratory analysis to {out_dir}")
    print()
    print(f"Qwen3 4B Exp1 errors: {len(qwen_errors)}")
    print(f"Selected last applicable rule: {qwen_last_count}/{len(qwen_errors)}")
    print(f"Uniform-choice expected count: {qwen_expected_count:.2f}")
    print(f"Gemma 3 4B explicit-none review rows: {len(gemma_none_rows)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate exploratory RuleTreeBench Discussion metrics."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to results/combined/combined_results.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/metrics/response_patterns"),
        help="Directory for the generated analysis files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze(args.input, args.out_dir)


if __name__ == "__main__":
    main()
