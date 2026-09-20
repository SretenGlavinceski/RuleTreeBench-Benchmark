#!/usr/bin/env python3
"""Generate the main RuleTreeBench metrics from combined_results.json."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


MODEL_ORDER = [
    "qwen3_4b",
    "qwen3_8b",
    "gemma3_4b",
    "gemma4_12b",
]

DISPLAY_NAMES = {
    "qwen3_4b": "Qwen3 4B",
    "qwen3_8b": "Qwen3 8B",
    "gemma3_4b": "Gemma 3 4B",
    "gemma4_12b": "Gemma 4 12B",
}


def load_combined(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "records" not in data or "models" not in data:
        raise ValueError("Input does not look like combined_results.json")

    return data


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def wilson_interval(
    successes: int,
    n: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")

    p = successes / n
    z2 = z * z

    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator

    half = (
        z
        * math.sqrt(
            (p * (1 - p) / n)
            + z2 / (4 * n * n)
        )
        / denominator
    )

    return centre - half, centre + half


def model_order(models: list[str]) -> list[str]:
    known = [m for m in MODEL_ORDER if m in models]
    extras = sorted(m for m in models if m not in MODEL_ORDER)
    return known + extras


def pct(x: int, n: int) -> float:
    if n == 0:
        return float("nan")

    return round(100 * x / n, 3)


def canonical_answer(value):
    """Normalize an already-parsed answer for equality checks only."""
    if value is None:
        return None

    return str(value).strip().upper()


def flatten_records(data: dict) -> list[dict]:
    rows = []

    for item in data["records"]:
        benchmark = item["benchmark"]

        for model, result in item["models"].items():
            rows.append(
                {
                    "model": model,
                    "id": item["id"],
                    "experiment": item["experiment"],
                    "condition": item["condition"],
                    "family_id": item["family_id"],
                    "level": benchmark.get("level"),
                    "part": benchmark.get("part", ""),
                    "probe_type": benchmark.get("probe_type", ""),
                    "source_exp1_id": benchmark.get("source_exp1_id", ""),
                    "source_exp2_id": benchmark.get("source_exp2_id", ""),
                    "gold_answer": item["gold_answer"],
                    **result,
                }
            )

    return rows


def analyze(input_path: Path, out_dir: Path) -> None:
    data = load_combined(input_path)

    rows = flatten_records(data)

    models = model_order(
        list(data["models"].keys())
    )

    by_model = {
        model: [
            row
            for row in rows
            if row["model"] == model
        ]
        for model in models
    }

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Run integrity

    integrity_rows = []

    for model in models:
        meta = data["models"][model]

        integrity_rows.append(
            {
                "model": model,
                "display_name": DISPLAY_NAMES.get(
                    model,
                    model,
                ),
                "responses": meta["responses"],
                "unique_ids": meta["unique_ids"],
                "truncated": meta["truncated"],
                "done_reason_not_stop": meta[
                    "done_reason_not_stop"
                ],
                "thinking_trace_leaks": meta[
                    "thinking_trace_nonempty"
                ],
                "max_prompt_tokens": meta[
                    "max_prompt_tokens"
                ],
                "max_output_tokens": meta[
                    "max_output_tokens"
                ],
                "max_prompt_plus_output_tokens": meta[
                    "max_prompt_plus_output_tokens"
                ],
            }
        )

    write_csv(
        out_dir / "integrity_summary.csv",
        integrity_rows,
    )

    # Parser accounting and overall accuracy

    parser_rows = []
    overall_rows = []

    for model in models:
        group = by_model[model]

        n = len(group)

        stored_correct = sum(
            bool(r["stored_correct"])
            for r in group
        )

        normalized_correct = sum(
            bool(r["normalized_correct"])
            for r in group
        )

        # Partition responses rejected by the original parser.

        strict_unparsed_rows = [
            r
            for r in group
            if r["stored_unparsable"]
        ]

        strict_unparsed = len(
            strict_unparsed_rows
        )

        recovered_correct = sum(
            r["normalized_correct"]
            for r in strict_unparsed_rows
        )

        normalized_but_incorrect = sum(
            r["parse_status"] == "parsed"
            and not r["normalized_correct"]
            for r in strict_unparsed_rows
        )

        explicit_none = sum(
            r["parse_status"] == "explicit_none"
            for r in strict_unparsed_rows
        )

        parse_failures = sum(
            r["parse_status"] == "parse_failure"
            for r in strict_unparsed_rows
        )

        parser_accounted = (
            recovered_correct
            + normalized_but_incorrect
            + explicit_none
            + parse_failures
        )

        assert parser_accounted == strict_unparsed, (
            f"Parser accounting mismatch for {model}: "
            f"strict_unparsed={strict_unparsed}, "
            f"accounted={parser_accounted}"
        )

        # Accepted answers must not change during normalization.

        changed_accepted_answers = sum(
            not r["stored_unparsable"]
            and canonical_answer(
                r["stored_parsed_answer"]
            )
            != canonical_answer(
                r["normalized_answer"]
            )
            for r in group
        )

        assert changed_accepted_answers == 0, (
            "Normalization changed an already accepted answer "
            f"for {model}: "
            f"{changed_accepted_answers} changed responses"
        )

        parser_rows.append(
            {
                "model": model,
                "display_name": DISPLAY_NAMES.get(
                    model,
                    model,
                ),
                "strict_unparsed": strict_unparsed,
                "recovered_correct": recovered_correct,
                "normalized_but_incorrect": normalized_but_incorrect,
                "explicit_none": explicit_none,
                "remaining_parse_failures": parse_failures,
                "accepted_answers_changed": changed_accepted_answers,
            }
        )

        overall_rows.append(
            {
                "model": model,
                "display_name": DISPLAY_NAMES.get(
                    model,
                    model,
                ),
                "n": n,
                "stored_correct": stored_correct,
                "stored_accuracy_pct": pct(
                    stored_correct,
                    n,
                ),
                "normalized_correct": normalized_correct,
                "normalized_accuracy_pct": pct(
                    normalized_correct,
                    n,
                ),
                "total_errors": (
                    n - normalized_correct
                ),
            }
        )

    write_csv(
        out_dir / "parser_accounting.csv",
        parser_rows,
    )

    write_csv(
        out_dir / "overall_accuracy.csv",
        overall_rows,
    )

    # Full selection tasks (Experiments 1, 2, and 4)

    selection_rows = []

    for model in models:
        group = [
            r
            for r in by_model[model]
            if r["experiment"] in {1, 2, 4}
        ]

        n = len(group)

        correct = sum(
            r["normalized_correct"]
            for r in group
        )

        selection_rows.append(
            {
                "model": model,
                "display_name": DISPLAY_NAMES.get(
                    model,
                    model,
                ),
                "n": n,
                "correct": correct,
                "errors": n - correct,
                "accuracy_pct": pct(
                    correct,
                    n,
                ),
            }
        )

    write_csv(
        out_dir / "selection_task_summary.csv",
        selection_rows,
    )

    # Condition-level accuracy

    condition_rows = []

    for model in models:
        groups = defaultdict(list)

        for row in by_model[model]:
            if row["experiment"] in {1, 2, 4}:
                groups[
                    (
                        row["experiment"],
                        row["condition"],
                    )
                ].append(row)

        def cond_key(item):
            (
                experiment,
                condition,
            ), _ = item

            prefix = condition[0]
            level = int(condition[1:])

            return (
                experiment,
                prefix,
                level,
            )

        for (
            experiment,
            condition,
        ), group in sorted(
            groups.items(),
            key=cond_key,
        ):
            n = len(group)

            correct = sum(
                r["normalized_correct"]
                for r in group
            )

            low, high = wilson_interval(
                correct,
                n,
            )

            condition_rows.append(
                {
                    "model": model,
                    "display_name": DISPLAY_NAMES.get(
                        model,
                        model,
                    ),
                    "experiment": experiment,
                    "condition": condition,
                    "n": n,
                    "correct": correct,
                    "errors": n - correct,
                    "accuracy_pct": pct(
                        correct,
                        n,
                    ),
                    "wilson95_low_pct": round(
                        100 * low,
                        3,
                    ),
                    "wilson95_high_pct": round(
                        100 * high,
                        3,
                    ),
                }
            )

    write_csv(
        out_dir / "condition_accuracy.csv",
        condition_rows,
    )

    # Outcome breakdown

    outcome_order = [
        "correct",
        "general",
        "near_match",
        "distractor",
        "explicit_none",
        "parse_failure",
        "out_of_set",
        "other_valid",
    ]

    outcome_rows = []

    for model in models:
        groups = defaultdict(list)

        for row in by_model[model]:
            if row["experiment"] in {1, 2, 4}:
                groups[
                    (
                        row["experiment"],
                        row["condition"],
                    )
                ].append(row)

        for (
            experiment,
            condition,
        ), group in sorted(
            groups.items()
        ):
            counts = Counter(
                r["outcome"]
                for r in group
            )

            record = {
                "model": model,
                "display_name": DISPLAY_NAMES.get(
                    model,
                    model,
                ),
                "experiment": experiment,
                "condition": condition,
                "n": len(group),
            }

            for outcome in outcome_order:
                record[outcome] = counts.get(
                    outcome,
                    0,
                )

            outcome_rows.append(
                record
            )

    write_csv(
        out_dir / "outcome_breakdown.csv",
        outcome_rows,
    )

    # Experiment 3 diagnostics

    probe_order = [
        ("independent", "valid_3"),
        ("independent", "valid_4"),
        ("independent", "one_error"),
        ("independent", "two_error"),
        ("linked", "winner"),
        ("linked", "near_match"),
    ]

    exp3_rows = []

    for model in models:
        exp3 = [
            r
            for r in by_model[model]
            if r["experiment"] == 3
        ]

        for part, probe_type in probe_order:
            group = [
                r
                for r in exp3
                if r["part"] == part
                and r["probe_type"] == probe_type
            ]

            n = len(group)

            errors = sum(
                not r["normalized_correct"]
                for r in group
            )

            low, high = wilson_interval(
                errors,
                n,
            )

            error_kind = (
                "false_no"
                if probe_type
                in {
                    "valid_3",
                    "valid_4",
                    "winner",
                }
                else "false_yes"
            )

            exp3_rows.append(
                {
                    "model": model,
                    "display_name": DISPLAY_NAMES.get(
                        model,
                        model,
                    ),
                    "part": part,
                    "probe_type": probe_type,
                    "n": n,
                    "error_type": error_kind,
                    "errors": errors,
                    "error_rate_pct": pct(
                        errors,
                        n,
                    ),
                    "wilson95_low_pct": round(
                        100 * low,
                        3,
                    ),
                    "wilson95_high_pct": round(
                        100 * high,
                        3,
                    ),
                }
            )

    write_csv(
        out_dir / "exp3_diagnostics.csv",
        exp3_rows,
    )

    # Paired family transitions

    paired_rows = []

    for model in models:
        for experiment, targets in [
            (1, [1, 2, 4]),
            (2, [1, 2, 4]),
        ]:
            exp_rows = [
                r
                for r in by_model[model]
                if r["experiment"] == experiment
            ]

            lookup = {
                (
                    r["family_id"],
                    int(r["level"]),
                ): r
                for r in exp_rows
            }

            families = sorted(
                {
                    r["family_id"]
                    for r in exp_rows
                }
            )

            for target in targets:
                transitions = Counter()

                for family in families:
                    before = lookup[
                        (
                            family,
                            0,
                        )
                    ]["normalized_correct"]

                    after = lookup[
                        (
                            family,
                            target,
                        )
                    ]["normalized_correct"]

                    if before and after:
                        key = "correct_to_correct"

                    elif before and not after:
                        key = "correct_to_wrong"

                    elif not before and after:
                        key = "wrong_to_correct"

                    else:
                        key = "wrong_to_wrong"

                    transitions[key] += 1

                prefix = (
                    "L"
                    if experiment == 1
                    else "N"
                )

                paired_rows.append(
                    {
                        "model": model,
                        "display_name": DISPLAY_NAMES.get(
                            model,
                            model,
                        ),
                        "experiment": experiment,
                        "comparison": (
                            f"{prefix}0"
                            f"->{prefix}{target}"
                        ),
                        "families": len(
                            families
                        ),
                        "correct_to_correct": transitions[
                            "correct_to_correct"
                        ],
                        "correct_to_wrong": transitions[
                            "correct_to_wrong"
                        ],
                        "wrong_to_correct": transitions[
                            "wrong_to_correct"
                        ],
                        "wrong_to_wrong": transitions[
                            "wrong_to_wrong"
                        ],
                    }
                )

    write_csv(
        out_dir / "paired_transitions.csv",
        paired_rows,
    )

    # Experiment 4 rescue

    rescue_rows = []

    for model in models:
        exp1_lookup = {
            r["id"]: r
            for r in by_model[model]
            if r["experiment"] == 1
        }

        exp4_rows = [
            r
            for r in by_model[model]
            if r["experiment"] == 4
        ]

        for level in [
            1,
            2,
            4,
            None,
        ]:
            selected = [
                r
                for r in exp4_rows
                if level is None
                or int(r["level"]) == level
            ]

            eligible = 0
            rescued = 0
            persistent = 0

            for filtered in selected:
                source_id = filtered[
                    "source_exp1_id"
                ]

                if source_id not in exp1_lookup:
                    raise ValueError(
                        "Experiment 4 source prompt "
                        f"not found: {source_id}"
                    )

                original = exp1_lookup[
                    source_id
                ]

                if not original[
                    "normalized_correct"
                ]:
                    eligible += 1

                    if filtered[
                        "normalized_correct"
                    ]:
                        rescued += 1
                    else:
                        persistent += 1

            rescue_rows.append(
                {
                    "model": model,
                    "display_name": DISPLAY_NAMES.get(
                        model,
                        model,
                    ),
                    "condition": (
                        "all"
                        if level is None
                        else f"L{level}"
                    ),
                    "eligible_original_errors": eligible,
                    "rescued": rescued,
                    "rescue_rate_pct": (
                        ""
                        if eligible == 0
                        else pct(
                            rescued,
                            eligible,
                        )
                    ),
                    "persistent_failures": persistent,
                }
            )

    write_csv(
        out_dir / "exp4_rescue.csv",
        rescue_rows,
    )

    # Linked-context diagnostic

    linked_rows = []

    for model in models:
        n1 = {
            r["family_id"]: r
            for r in by_model[model]
            if r["experiment"] == 2
            and r["condition"] == "N1"
        }

        linked_winner = {
            r["family_id"]: r
            for r in by_model[model]
            if r["experiment"] == 3
            and r["part"] == "linked"
            and r["probe_type"] == "winner"
        }

        linked_near = {
            r["family_id"]: r
            for r in by_model[model]
            if r["experiment"] == 3
            and r["part"] == "linked"
            and r["probe_type"] == "near_match"
        }

        families = sorted(
            set(n1)
            & set(linked_winner)
            & set(linked_near)
        )

        split = Counter()
        strict_split = Counter()

        near_correct_full_wrong = 0
        both_correct_full_wrong = 0

        for family in families:
            full = n1[family]

            winner_ok = linked_winner[
                family
            ]["normalized_correct"]

            near_ok = linked_near[
                family
            ]["normalized_correct"]

            if (
                not full["normalized_correct"]
                and near_ok
            ):
                near_correct_full_wrong += 1

                split[
                    full["outcome"]
                ] += 1

            if (
                not full["normalized_correct"]
                and winner_ok
                and near_ok
            ):
                both_correct_full_wrong += 1

                strict_split[
                    full["outcome"]
                ] += 1

        linked_rows.append(
            {
                "model": model,
                "display_name": DISPLAY_NAMES.get(
                    model,
                    model,
                ),
                "families": len(
                    families
                ),
                "near_correct_full_wrong": (
                    near_correct_full_wrong
                ),
                "near_correct_full_wrong_near_match": (
                    split["near_match"]
                ),
                "near_correct_full_wrong_distractor": (
                    split["distractor"]
                ),
                "near_correct_full_wrong_explicit_none": (
                    split["explicit_none"]
                ),
                "near_correct_full_wrong_parse_failure": (
                    split["parse_failure"]
                ),
                "both_probes_correct_full_wrong": (
                    both_correct_full_wrong
                ),
                "strict_near_match": (
                    strict_split[
                        "near_match"
                    ]
                ),
                "strict_distractor": (
                    strict_split[
                        "distractor"
                    ]
                ),
                "strict_explicit_none": (
                    strict_split[
                        "explicit_none"
                    ]
                ),
                "strict_parse_failure": (
                    strict_split[
                        "parse_failure"
                    ]
                ),
            }
        )

    write_csv(
        out_dir / "linked_context_summary.csv",
        linked_rows,
    )

    # Consolidated paper metrics

    paper_metrics = {
        "baselines": {
            "experiment_1": {
                "random_pct": 10.0,
                "longest_rule_pct": 25.0,
            },
            "experiment_2": {
                "random_pct": 10.0,
                "longest_rule_pct": 12.5,
            },
            "experiment_4_random_pct": {
                "L1": 50.0,
                "L2": 33.333,
                "L4": 20.0,
            },
        },
        "integrity": integrity_rows,
        "parser_accounting": parser_rows,
        "overall_accuracy": overall_rows,
        "selection_task_summary": selection_rows,
        "condition_accuracy": condition_rows,
        "outcome_breakdown": outcome_rows,
        "experiment_3_diagnostics": exp3_rows,
        "paired_transitions": paired_rows,
        "experiment_4_rescue": rescue_rows,
        "linked_context": linked_rows,
    }

    (
        out_dir
        / "paper_metrics.json"
    ).write_text(
        json.dumps(
            paper_metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Human-readable summary

    overall_by_model = {
        r["model"]: r
        for r in overall_rows
    }

    selection_by_model = {
        r["model"]: r
        for r in selection_rows
    }

    parser_by_model = {
        r["model"]: r
        for r in parser_rows
    }

    linked_by_model = {
        r["model"]: r
        for r in linked_rows
    }

    lines = [
        "# RuleTreeBench paper metrics",
        "",
        (
            "This file is generated from "
            "`combined_results.json`."
        ),
        "",
        "## Run integrity and overall scoring",
    ]

    for model in models:
        o = overall_by_model[model]
        p = parser_by_model[model]
        meta = data["models"][model]

        lines.append(
            f"- **{DISPLAY_NAMES.get(model, model)}**: "
            f"{o['normalized_correct']}/{o['n']} "
            f"({o['normalized_accuracy_pct']:.1f}%) "
            f"normalized accuracy; "
            f"{p['recovered_correct']} recovered correct answers; "
            f"{p['normalized_but_incorrect']} normalized but incorrect answers; "
            f"{p['explicit_none']} explicit-none responses; "
            f"{p['remaining_parse_failures']} parse failures; "
            f"max output {meta['max_output_tokens']} tokens; "
            f"max prompt+output "
            f"{meta['max_prompt_plus_output_tokens']} tokens."
        )

    lines.extend(
        [
            "",
            (
                "## Full rule-selection tasks "
                "(Experiments 1, 2, and 4)"
            ),
        ]
    )

    for model in models:
        s = selection_by_model[model]

        lines.append(
            f"- **{DISPLAY_NAMES.get(model, model)}**: "
            f"{s['correct']}/{s['n']} correct "
            f"({s['accuracy_pct']:.1f}%), "
            f"{s['errors']} errors."
        )

    lines.extend(
        [
            "",
            (
                "## Experiment 1: "
                "general-rule selections"
            ),
        ]
    )

    for model in models:
        rows_m = [
            r
            for r in outcome_rows
            if r["model"] == model
            and r["experiment"] == 1
        ]

        general = sum(
            r["general"]
            for r in rows_m
        )

        distractor = sum(
            r["distractor"]
            for r in rows_m
        )

        explicit_none_total = sum(
            r["explicit_none"]
            for r in rows_m
        )

        parse_failure_total = sum(
            r["parse_failure"]
            for r in rows_m
        )

        lines.append(
            f"- **{DISPLAY_NAMES.get(model, model)}**: "
            f"{general} general-rule errors, "
            f"{distractor} distractors, "
            f"{explicit_none_total} explicit-none, "
            f"{parse_failure_total} parse failures."
        )

    lines.extend(
        [
            "",
            (
                "## Experiment 2: "
                "near-match selections"
            ),
        ]
    )

    for model in models:
        rows_m = [
            r
            for r in outcome_rows
            if r["model"] == model
            and r["experiment"] == 2
        ]

        near = sum(
            r["near_match"]
            for r in rows_m
        )

        distractor = sum(
            r["distractor"]
            for r in rows_m
        )

        explicit_none_total = sum(
            r["explicit_none"]
            for r in rows_m
        )

        parse_failure_total = sum(
            r["parse_failure"]
            for r in rows_m
        )

        lines.append(
            f"- **{DISPLAY_NAMES.get(model, model)}**: "
            f"{near} near-match errors, "
            f"{distractor} distractors, "
            f"{explicit_none_total} explicit-none, "
            f"{parse_failure_total} parse failures."
        )

    lines.extend(
        [
            "",
            "## Experiment 3",
        ]
    )

    for model in models:
        exp3_model = [
            r
            for r in exp3_rows
            if r["model"] == model
        ]

        total_errors = sum(
            r["errors"]
            for r in exp3_model
        )

        lines.append(
            f"- **{DISPLAY_NAMES.get(model, model)}**: "
            f"{total_errors}/600 diagnostic errors."
        )

    lines.extend(
        [
            "",
            "## Linked-context diagnostic",
        ]
    )

    for model in models:
        r = linked_by_model[model]

        lines.append(
            f"- **{DISPLAY_NAMES.get(model, model)}**: "
            f"{r['near_correct_full_wrong']} full N1 failures "
            f"while the linked near match was correct in isolation; "
            f"{r['both_probes_correct_full_wrong']} full N1 failures "
            f"while both linked probes were correct."
        )

    lines.extend(
        [
            "",
            "## Experiment 4 rescue",
        ]
    )

    for model in models:
        r = next(
            x
            for x in rescue_rows
            if x["model"] == model
            and x["condition"] == "all"
        )

        if (
            r["eligible_original_errors"]
            == 0
        ):
            text = (
                "no eligible original errors; "
                "rescue rate undefined."
            )

        else:
            text = (
                f"{r['rescued']}/"
                f"{r['eligible_original_errors']} "
                f"rescued "
                f"({float(r['rescue_rate_pct']):.1f}%); "
                f"{r['persistent_failures']} "
                f"persistent failures."
            )

        lines.append(
            f"- **{DISPLAY_NAMES.get(model, model)}**: "
            f"{text}"
        )

    lines.extend(
        [
            "",
            (
                "## Within-family "
                "descriptive comparison"
            ),
            (
                f"- Qwen: Qwen3 4B had "
                f"{overall_by_model['qwen3_4b']['total_errors']} "
                f"total benchmark errors; "
                f"Qwen3 8B had "
                f"{overall_by_model['qwen3_8b']['total_errors']}."
            ),
            (
                f"- Gemma: Gemma 3 4B had "
                f"{overall_by_model['gemma3_4b']['total_errors']} "
                f"total benchmark errors; "
                f"Gemma 4 12B had "
                f"{overall_by_model['gemma4_12b']['total_errors']}."
            ),
            "",
            (
                "These are descriptive comparisons of the evaluated "
                "model variants; they do not isolate parameter count "
                "as the only cause of the differences."
            ),
        ]
    )

    (
        out_dir
        / "paper_summary.md"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(
        f"Analyzed {len(rows)} "
        "model-response records."
    )

    print(
        f"Wrote paper metrics to "
        f"{out_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate RuleTreeBench metrics "
            "from combined_results.json."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="combined_results.json",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper_metrics"),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    analyze(
        args.input,
        args.out_dir,
    )


if __name__ == "__main__":
    main()