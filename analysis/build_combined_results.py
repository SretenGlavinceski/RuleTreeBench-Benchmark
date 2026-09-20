#!/usr/bin/env python3
"""Join benchmark prompts and raw model responses for analysis."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ANSWER_RE = re.compile(r"ANSWER\s*:\s*([^\r\n]*)", re.IGNORECASE)
RULE_ANSWER_RE = re.compile(r"R(?:10|[1-9])", re.IGNORECASE)
MARKDOWN_CHARS = "*_`~"


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc
    return rows


def final_answer_payload(raw_response: str | None) -> str | None:
    if raw_response is None:
        return None
    matches = list(ANSWER_RE.finditer(raw_response))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def normalize_answer(
    raw_response: str | None,
    gold_answer: str,
) -> tuple[str | None, str, str | None]:
    """Return (normalized_answer, parse_status, final_payload).

    parse_status:
      parsed
      explicit_none
      parse_failure
    """
    payload = final_answer_payload(raw_response)
    if payload is None:
        return None, "parse_failure", None

    # Formatting-only normalization.  This handles forms such as:
    # **ANSWER: R2**
    # ANSWER: ANSWER: R2
    # `ANSWER: YES`
    for _ in range(3):
        payload = payload.strip().strip(MARKDOWN_CHARS).strip()
        duplicate = re.fullmatch(r"ANSWER\s*:\s*(.*)", payload, re.IGNORECASE)
        if duplicate:
            payload = duplicate.group(1)
        else:
            break

    payload = payload.strip().strip(MARKDOWN_CHARS).strip()

    if gold_answer in {"YES", "NO"}:
        value = payload.upper()
        if value in {"YES", "NO"}:
            return value, "parsed", payload
        return None, "parse_failure", payload

    if payload == "" or payload.lower() == "none" or payload == "0":
        return None, "explicit_none", payload

    value = payload.upper()
    if RULE_ANSWER_RE.fullmatch(value):
        return value, "parsed", payload

    return None, "parse_failure", payload


def failed_conditions(rule: dict, query: dict) -> int:
    return sum(
        query.get(attribute) != value
        for attribute, value in rule["conditions"].items()
    )


def classify_rule_roles(example: dict) -> dict[str, str]:
    """Classify each full-prompt rule from the frozen prompt itself."""
    rules = {rule["label"]: rule for rule in example["rules"]}
    winner = rules[example["gold_answer"]]
    winner_specificity = len(winner["conditions"])

    roles = {}
    for label, rule in rules.items():
        failures = failed_conditions(rule, example["query"])
        specificity = len(rule["conditions"])

        if label == example["gold_answer"]:
            role = "winner"
        elif failures == 0 and specificity < winner_specificity:
            role = "general"
        elif failures == 1 and specificity == winner_specificity:
            role = "near_match"
        elif failures == 0:
            role = "other_valid"
        else:
            role = "distractor"

        roles[label] = role

    return roles


def classify_outcome(
    example: dict,
    normalized_answer: str | None,
    parse_status: str,
) -> str:
    if parse_status == "parse_failure":
        return "parse_failure"

    if parse_status == "explicit_none":
        return "explicit_none"

    if normalized_answer == example["gold_answer"]:
        return "correct"

    if example["experiment"] == 3:
        return "false_no" if example["gold_answer"] == "YES" else "false_yes"

    roles = classify_rule_roles(example)
    if normalized_answer not in roles:
        return "out_of_set"

    role = roles[normalized_answer]
    if role == "winner":
        return "correct"
    return role


def family_id(example: dict) -> str:
    return (
        example.get("family_id")
        or example.get("source_family_id")
        or example.get("group_id")
        or ""
    )


def condition_label(example: dict) -> str:
    experiment = example["experiment"]
    if experiment in {1, 4}:
        return f"L{example['level']}"
    if experiment == 2:
        return f"N{example['level']}"
    if experiment == 3:
        return f"{example.get('part', '')}:{example.get('probe_type', '')}"
    return ""


def validate_dataset(examples: list[dict]) -> None:
    ids = [row["id"] for row in examples]
    if len(ids) != len(set(ids)):
        duplicates = [x for x, n in Counter(ids).items() if n > 1]
        raise ValueError(f"Duplicate benchmark IDs: {duplicates[:10]}")

    expected = {1: 400, 2: 400, 3: 600, 4: 300}
    counts = Counter(row["experiment"] for row in examples)
    if counts != expected:
        raise ValueError(
            f"Unexpected benchmark counts: {dict(counts)}; expected {expected}"
        )

    # Validate the structural assumptions used by the error classifier.
    for row in examples:
        if row["experiment"] not in {1, 2, 4}:
            continue

        roles = Counter(classify_rule_roles(row).values())

        if roles["winner"] != 1 or roles["other_valid"] != 0:
            raise ValueError(
                f"Unexpected winner/valid structure in {row['id']}: {dict(roles)}"
            )

        level = row["level"]

        if row["experiment"] == 1:
            if roles["general"] != level or roles["near_match"] != 0:
                raise ValueError(
                    f"Unexpected Exp1 structure in {row['id']}: {dict(roles)}"
                )

        elif row["experiment"] == 2:
            if roles["near_match"] != level or roles["general"] != 0:
                raise ValueError(
                    f"Unexpected Exp2 structure in {row['id']}: {dict(roles)}"
                )

        elif row["experiment"] == 4:
            if roles["general"] != level or roles["distractor"] != 0:
                raise ValueError(
                    f"Unexpected Exp4 structure in {row['id']}: {dict(roles)}"
                )


def build_combined(
    dataset_paths: list[Path],
    response_paths: list[Path],
    output_path: Path,
) -> None:
    dataset_parts = [read_jsonl(path) for path in dataset_paths]
    examples = [row for part in dataset_parts for row in part]
    validate_dataset(examples)

    example_by_id = {row["id"]: row for row in examples}
    dataset_ids = set(example_by_id)

    model_rows = {}
    model_metadata = {}

    for response_path in response_paths:
        responses = read_jsonl(response_path)
        if not responses:
            raise ValueError(f"Empty response file: {response_path}")

        model_name = responses[0].get("model_name") or response_path.stem

        if any(row.get("model_name") != model_name for row in responses):
            raise ValueError(f"Mixed model names in {response_path}")

        if model_name in model_rows:
            raise ValueError(f"Duplicate response file for model {model_name}")

        ids = [row["id"] for row in responses]
        response_ids = set(ids)

        missing = dataset_ids - response_ids
        extra = response_ids - dataset_ids
        duplicate_count = len(ids) - len(response_ids)

        gold_mismatches = [
            row["id"]
            for row in responses
            if row["id"] in example_by_id
            and row.get("gold_answer") != example_by_id[row["id"]]["gold_answer"]
        ]

        if missing or extra or duplicate_count or gold_mismatches:
            raise ValueError(
                f"Integrity failure for {model_name}: "
                f"missing={len(missing)}, extra={len(extra)}, "
                f"duplicates={duplicate_count}, "
                f"gold_mismatches={len(gold_mismatches)}"
            )

        by_id = {row["id"]: row for row in responses}
        model_rows[model_name] = by_id

        model_metadata[model_name] = {
            "source_file": response_path.as_posix(),
            "ollama_model": responses[0].get("ollama_model", ""),
            "model_digest": responses[0].get("model_digest", ""),
            "responses": len(responses),
            "unique_ids": len(response_ids),
            "truncated": sum(bool(row.get("truncated")) for row in responses),
            "done_reason_not_stop": sum(
                row.get("done_reason") != "stop" for row in responses
            ),
            "max_prompt_tokens": max(
                (row.get("prompt_eval_count") or 0) for row in responses
            ),
            "max_output_tokens": max(
                (row.get("eval_count") or 0) for row in responses
            ),
            "max_prompt_plus_output_tokens": max(
                (row.get("prompt_eval_count") or 0)
                + (row.get("eval_count") or 0)
                for row in responses
            ),
            "thinking_trace_nonempty": sum(
                bool(str(row.get("thinking_field") or "").strip())
                for row in responses
            ),
        }

    records = []

    for example in examples:
        item = {
            "id": example["id"],
            "experiment": example["experiment"],
            "condition": condition_label(example),
            "family_id": family_id(example),
            "gold_answer": example["gold_answer"],
            "benchmark": example,
            "models": {},
        }

        for model_name, response_lookup in model_rows.items():
            response = response_lookup[example["id"]]
            normalized_answer, parse_status, payload = normalize_answer(
                response.get("raw_response"),
                example["gold_answer"],
            )
            outcome = classify_outcome(
                example,
                normalized_answer,
                parse_status,
            )

            item["models"][model_name] = {
                "stored_parsed_answer": response.get("parsed_answer"),
                "stored_correct": bool(response.get("correct")),
                "stored_unparsable": bool(response.get("unparsable")),
                "normalized_answer": normalized_answer,
                "parse_status": parse_status,
                "answer_payload": payload,
                "normalized_correct": outcome == "correct",
                "outcome": outcome,
                "raw_response": response.get("raw_response", ""),
                "prompt_eval_count": response.get("prompt_eval_count"),
                "eval_count": response.get("eval_count"),
                "truncated": bool(response.get("truncated")),
                "done_reason": response.get("done_reason"),
                "thinking_field": response.get("thinking_field", ""),
            }

        records.append(item)

    combined = {
        "schema_version": 1,
        "description": (
            "RuleTreeBench benchmark prompts joined with normalized responses "
            "for every evaluated model."
        ),
        "normalization_policy": {
            "answer_source": "last ANSWER: marker only",
            "formatting_removed": ["*", "_", "`", "~"],
            "duplicated_answer_prefix_removed": True,
            "semantic_inference_from_explanation": False,
            "explicit_none_values": ["None", "0", "empty ANSWER field"],
            "invalid_values_such_as_R0": "parse_failure",
        },
        "benchmark_counts": dict(
            sorted(Counter(row["experiment"] for row in examples).items())
        ),
        "models": model_metadata,
        "records": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Validated {len(examples)} benchmark prompts.")
    print(f"Validated {len(model_rows)} model response files.")
    print(f"Wrote {len(records)} joined benchmark records to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one normalized RuleTreeBench JSON file."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        nargs="+",
        required=True,
        help="The five frozen benchmark JSONL files.",
    )
    parser.add_argument(
        "--responses",
        type=Path,
        nargs="+",
        required=True,
        help="One response JSONL file per model.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("combined_results.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_combined(args.dataset, args.responses, args.output)


if __name__ == "__main__":
    main()
