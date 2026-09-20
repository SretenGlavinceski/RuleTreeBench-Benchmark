import json
import random

from src.generator_exp3 import generate_exp3_independent_group
from src.render import render_applicability_task
from src.solver import applies, failed_conditions, specificity


def rule_to_dict(rule):
    return {
        "label": rule.label,
        "conditions": rule.conditions
    }


def build_exp3_independent_dataset():
    random.seed(42)

    dataset = []

    for group_number in range(1, 101):
        group = generate_exp3_independent_group(group_number)

        query = group["query"]

        probes = [
            ("valid_3", group["valid_3"]),
            ("valid_4", group["valid_4"]),
            ("one_error", group["one_error"]),
            ("two_error", group["two_error"])
        ]

        for probe_type, rule in probes:

            if applies(rule, query):
                gold_answer = "YES"
            else:
                gold_answer = "NO"

            group_id = "exp3_i" + str(group_number).zfill(3)

            prompt_id = (
                group_id
                + "_"
                + probe_type
            )

            item = {
                "id": prompt_id,
                "experiment": 3,
                "part": "independent",
                "group_id": group_id,
                "probe_type": probe_type,
                "query": query.values,
                "rule": rule_to_dict(rule),
                "specificity": specificity(rule),
                "failed_conditions": failed_conditions(rule, query),
                "gold_answer": gold_answer,
                "prompt": render_applicability_task(query, rule)
            }

            dataset.append(item)

    return dataset


def save_dataset(dataset, filename):
    with open(filename, "w") as file:
        for item in dataset:
            file.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    dataset = build_exp3_independent_dataset()

    save_dataset(
        dataset,
        "data/generated/experiment_3_independent.jsonl"
    )

    print("Saved", len(dataset), "Experiment 3 independent prompts")
