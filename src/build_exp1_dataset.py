import json
import random

from src.generator_exp1 import generate_exp1_family, generate_exp1_variants
from src.render import render_rule_task
from src.solver import solve


def rule_to_dict(rule):
    return {
        "label": rule.label,
        "conditions": rule.conditions
    }


def query_to_dict(query):
    return query.values


def build_exp1_dataset():
    random.seed(42)

    dataset = []

    for family_number in range(1, 101):
        family = generate_exp1_family()
        variants = generate_exp1_variants(family)

        query = family["query"]

        for level in [0, 1, 2, 4]:
            rules = variants[level]

            result = solve(query, rules)
            gold_answer = result[0].label

            family_id = "exp1_f" + str(family_number).zfill(3)
            prompt_id = family_id + "_L" + str(level)

            item = {
                "id": prompt_id,
                "experiment": 1,
                "family_id": family_id,
                "level": level,
                "query": query_to_dict(query),
                "rules": [],
                "gold_answer": gold_answer,
                "prompt": render_rule_task(query, rules)
            }

            for rule in rules:
                item["rules"].append(rule_to_dict(rule))

            dataset.append(item)

    return dataset


def save_dataset(dataset, filename):
    with open(filename, "w") as file:
        for item in dataset:
            json_line = json.dumps(item)
            file.write(json_line + "\n")


if __name__ == "__main__":
    dataset = build_exp1_dataset()

    save_dataset(
        dataset,
        "data/experiment_1_general.jsonl"
    )

    print("Saved", len(dataset), "Experiment 1 prompts")
