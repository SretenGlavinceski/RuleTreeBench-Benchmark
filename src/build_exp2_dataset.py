import json
import random

from src.generator_exp2 import (
    generate_exp2_family,
    generate_exp2_variants
)

from src.render import render_rule_task
from src.solver import solve, failed_conditions


def rule_to_dict(rule):
    return {
        "label": rule.label,
        "conditions": rule.conditions
    }


def query_to_dict(query):
    return query.values


def build_exp2_dataset():
    random.seed(42)

    dataset = []

    for family_number in range(1, 101):
        family = generate_exp2_family()
        variants = generate_exp2_variants(family)

        query = family["query"]

        # Find the exact near match introduced at N1.
        first_near_match = None

        for rule in variants[1]:
            if failed_conditions(rule, query) == 1:
                first_near_match = rule
                break

        if first_near_match is None:
            raise Exception("Could not find N1 near match")

        for level in [0, 1, 2, 4]:
            rules = variants[level]

            result = solve(query, rules)

            if len(result) != 1:
                raise Exception("Experiment 2 item does not have one winner")

            gold_answer = result[0].label

            family_id = "exp2_f" + str(family_number).zfill(3)
            prompt_id = family_id + "_N" + str(level)

            item = {
                "id": prompt_id,
                "experiment": 2,
                "family_id": family_id,
                "level": level,
                "query": query_to_dict(query),
                "rules": [],
                "gold_answer": gold_answer,

                # Useful later for Experiment 3 linked probes.
                "first_near_match": rule_to_dict(first_near_match),

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
    dataset = build_exp2_dataset()

    save_dataset(
        dataset,
        "data/generated/experiment_2_nearmatch.jsonl"
    )

    print("Saved", len(dataset), "Experiment 2 prompts")
