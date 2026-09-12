import json

from src.models import Query, Rule
from src.render import render_rule_task
from src.solver import applies, solve


def dict_to_query(query_data):
    return Query(query_data)


def dict_to_rule(rule_data):
    return Rule(
        rule_data["label"],
        rule_data["conditions"]
    )


def rule_to_dict(rule):
    return {
        "label": rule.label,
        "conditions": rule.conditions
    }


def build_exp4_dataset():
    dataset = []

    with open("data/experiment_1_general.jsonl") as file:
        for line in file:
            item = json.loads(line)

            # Experiment 4 uses only L1, L2 and L4.
            if item["level"] == 0:
                continue

            query = dict_to_query(item["query"])

            original_rules = []

            for rule_data in item["rules"]:
                rule = dict_to_rule(rule_data)
                original_rules.append(rule)

            # Keep only rules that actually apply.
            filtered_rules = []

            for rule in original_rules:
                if applies(rule, query):
                    filtered_rules.append(rule)

            result = solve(query, filtered_rules)

            if len(result) != 1:
                raise Exception(
                    "Experiment 4 item does not have one winner: "
                    + item["id"]
                )

            gold_answer = result[0].label

            prompt_id = (
                "exp4_"
                + item["family_id"]
                + "_L"
                + str(item["level"])
            )

            new_item = {
                "id": prompt_id,
                "experiment": 4,
                "family_id": item["family_id"],
                "level": item["level"],
                "source_exp1_id": item["id"],
                "query": item["query"],
                "rules": [],
                "gold_answer": gold_answer,
                "prompt": render_rule_task(
                    query,
                    filtered_rules
                )
            }

            for rule in filtered_rules:
                new_item["rules"].append(
                    rule_to_dict(rule)
                )

            dataset.append(new_item)

    return dataset


def save_dataset(dataset, filename):
    with open(filename, "w") as file:
        for item in dataset:
            file.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    dataset = build_exp4_dataset()

    save_dataset(
        dataset,
        "data/experiment_4_filtering.jsonl"
    )

    print(
        "Saved",
        len(dataset),
        "Experiment 4 prompts"
    )
