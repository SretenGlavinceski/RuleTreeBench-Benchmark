from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
UPSTREAM_DIR = HERE / "upstream"
AIRLINE_DIR = UPSTREAM_DIR / "airline"
GENERATED_DIR = HERE / "generated"

AUTO_TEST_PATH = AIRLINE_DIR / "auto_test.py"
COMPUTE_ANSWER_PATH = AIRLINE_DIR / "compute_answer.py"
REFERENCE_RULES_PATH = AIRLINE_DIR / "reference_rules.txt"
OUTPUT_PATH = GENERATED_DIR / "rulearena_airline_300.jsonl"
MANIFEST_PATH = GENERATED_DIR / "rulearena_airline_300.manifest.json"

EXPECTED_ROWS_PER_LEVEL = 100
EXPECTED_TOTAL_ROWS = 300


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_string_assignments(path: Path, names: set[str]) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: dict[str, str] = {}

    for node in tree.body:
        targets: list[ast.expr] = []
        value_node: ast.expr | None = None

        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value_node = node.value

        if value_node is None:
            continue

        for target in targets:
            if not isinstance(target, ast.Name) or target.id not in names:
                continue
            try:
                value = ast.literal_eval(value_node)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not safely read {target.id!r} from {path}."
                ) from exc
            if not isinstance(value, str):
                raise TypeError(f"Expected {target.id!r} in {path} to be a string.")
            found[target.id] = value

    missing = sorted(names - set(found))
    if missing:
        raise RuntimeError(
            f"Missing required string assignment(s) in {path}: {', '.join(missing)}"
        )
    return found


def required_upstream_files() -> list[Path]:
    files = [
        AUTO_TEST_PATH,
        COMPUTE_ANSWER_PATH,
        REFERENCE_RULES_PATH,
    ]

    for complexity in range(3):
        files.append(
            AIRLINE_DIR
            / "synthesized_problems"
            / f"comp_{complexity}.jsonl"
        )

    for bag_number in range(1, 5):
        for direction in range(2):
            files.append(
                AIRLINE_DIR
                / "fee_tables"
                / f"bag_{bag_number}"
                / f"{direction}.csv"
            )

    return files


def validate_upstream_files() -> None:
    missing = [path for path in required_upstream_files() if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required RuleArena file(s):\n{formatted}")


def load_compute_module() -> Any:
    # RuleArena's compute_answer.py loads CSV tables using paths relative to
    # the airline directory at import time, so temporarily change cwd.
    old_cwd = Path.cwd()
    os.chdir(AIRLINE_DIR)
    try:
        spec = importlib.util.spec_from_file_location(
            "rulearena_airline_compute_answer",
            COMPUTE_ANSWER_PATH,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not import {COMPUTE_ANSWER_PATH}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        os.chdir(old_cwd)
    return module


def to_builtin_number(value: Any) -> int | float:
    if hasattr(value, "item"):
        value = value.item()

    if isinstance(value, bool):
        raise TypeError("Gold total unexpectedly became a boolean.")

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value) if value.is_integer() else value

    raise TypeError(f"Gold total is not numeric: {value!r}")


def build_zero_shot_prompt(
    prompt_template: str,
    reference_rules: str,
    question_prompt: str,
) -> str:
    # Match RuleArena's own construction order. Plain replace is intentional:
    # the literal "$xxx" in the requested answer format must remain unchanged.
    prompt = prompt_template.replace("$reference_rules", reference_rules)
    prompt = prompt.replace("$question_prompt", question_prompt)
    prompt = prompt.replace("$example_prompt", "")
    return prompt


def load_source_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"Expected object in {path} line {line_number}")
            if "prompt" not in row or "info" not in row:
                raise KeyError(
                    f"Expected 'prompt' and 'info' in {path} line {line_number}"
                )
            rows.append(row)
    return rows


def write_dataset(rows: list[dict[str, Any]]) -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            json.dump(
                row,
                f,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            f.write("\n")


def write_manifest(rows: list[dict[str, Any]]) -> None:
    source_hashes: dict[str, str] = {}
    for path in required_upstream_files():
        rel = path.relative_to(UPSTREAM_DIR).as_posix()
        source_hashes[rel] = sha256_file(path)

    level_counts = Counter(row["difficulty_level"] for row in rows)

    manifest = {
        "benchmark": "RuleArena",
        "domain": "airline",
        "prompting": "zero-shot",
        "rule_representation": "original tabular rules",
        "num_examples": len(rows),
        "difficulty_counts": {
            str(level): level_counts[level]
            for level in sorted(level_counts)
        },
        "generated_file": OUTPUT_PATH.name,
        "generated_sha256": sha256_file(OUTPUT_PATH),
        "upstream_files": source_hashes,
    }

    with MANIFEST_PATH.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")


def validate_generated_rows(rows: list[dict[str, Any]], reference_rules: str) -> None:
    if len(rows) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_ROWS} examples, found {len(rows)}."
        )

    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Generated example IDs are not unique.")

    counts = Counter(row["difficulty_level"] for row in rows)
    expected_counts = {1: 100, 2: 100, 3: 100}
    if dict(counts) != expected_counts:
        raise RuntimeError(
            f"Unexpected difficulty counts: {dict(counts)}; "
            f"expected {expected_counts}."
        )

    for row in rows:
        if reference_rules not in row["user_prompt"]:
            raise RuntimeError(f"Reference rules missing from prompt {row['id']}")
        if row["question_prompt"] not in row["user_prompt"]:
            raise RuntimeError(f"Question missing from prompt {row['id']}")
        if not isinstance(row["gold_total_cost"], (int, float)):
            raise TypeError(f"Non-numeric gold value in {row['id']}")


def main() -> None:
    validate_upstream_files()

    assignments = extract_string_assignments(
        AUTO_TEST_PATH,
        {"system_prompt", "prompt_template"},
    )
    system_prompt = assignments["system_prompt"]
    prompt_template = assignments["prompt_template"]
    reference_rules = REFERENCE_RULES_PATH.read_text(encoding="utf-8")

    compute_module = load_compute_module()
    check_base_tables = compute_module.check_base_tables

    generated_rows: list[dict[str, Any]] = []

    for source_complexity in range(3):
        difficulty_level = source_complexity + 1
        source_path = (
            AIRLINE_DIR
            / "synthesized_problems"
            / f"comp_{source_complexity}.jsonl"
        )
        source_rows = load_source_rows(source_path)

        if len(source_rows) != EXPECTED_ROWS_PER_LEVEL:
            raise RuntimeError(
                f"Expected {EXPECTED_ROWS_PER_LEVEL} rows in {source_path}, "
                f"found {len(source_rows)}."
            )

        for source_index, source_row in enumerate(source_rows):
            question_prompt = source_row["prompt"]
            source_info = source_row["info"]

            total_cost, _ = compute_module.compute_answer(
                **source_info,
                check_base_tables=check_base_tables,
            )
            gold_total_cost = to_builtin_number(total_cost)

            user_prompt = build_zero_shot_prompt(
                prompt_template=prompt_template,
                reference_rules=reference_rules,
                question_prompt=question_prompt,
            )

            generated_rows.append(
                {
                    "id": (
                        f"rulearena_airline_L{difficulty_level}_"
                        f"{source_index:03d}"
                    ),
                    "benchmark": "RuleArena",
                    "domain": "airline",
                    "source_complexity": source_complexity,
                    "difficulty_level": difficulty_level,
                    "source_index": source_index,
                    "source_file": f"comp_{source_complexity}.jsonl",
                    "question_prompt": question_prompt,
                    "source_info": source_info,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "gold_total_cost": gold_total_cost,
                    "gold_final_answer": (
                        f"The total cost is ${gold_total_cost}."
                    ),
                }
            )

    validate_generated_rows(generated_rows, reference_rules)
    write_dataset(generated_rows)
    write_manifest(generated_rows)

    level_counts = Counter(row["difficulty_level"] for row in generated_rows)
    print("RuleArena Airline dataset prepared successfully.")
    print(f"Output:   {OUTPUT_PATH}")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Rows:     {len(generated_rows)}")
    print(f"Levels:   {dict(sorted(level_counts.items()))}")
    print(f"First ID: {generated_rows[0]['id']}")
    print(f"Last ID:  {generated_rows[-1]['id']}")
    print(f"SHA256:   {sha256_file(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
