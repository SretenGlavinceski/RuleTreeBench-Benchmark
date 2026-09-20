import json

from src.models import Query, Rule
from src.solver import applies, failed_conditions, specificity, solve


def load_jsonl(filename):
    items = []

    with open(filename) as file:
        for line in file:
            items.append(json.loads(line))

    return items


def make_query(data):
    return Query(data)


def make_rule(data):
    return Rule(
        data["label"],
        data["conditions"]
    )


def check_no_duplicate_rules(rules):
    for i in range(len(rules)):
        for j in range(i + 1, len(rules)):
            assert rules[i].conditions != rules[j].conditions


def validate_exp1(items):
    assert len(items) == 400

    level_counts = {
        0: 0,
        1: 0,
        2: 0,
        4: 0
    }

    for item in items:
        level = item["level"]
        level_counts[level] += 1

        query = make_query(item["query"])

        rules = []

        for rule_data in item["rules"]:
            rules.append(make_rule(rule_data))

        assert len(rules) == 10

        check_no_duplicate_rules(rules)

        result = solve(query, rules)

        assert len(result) == 1
        assert result[0].label == item["gold_answer"]
        assert specificity(result[0]) == 4

        applicable_rules = []

        for rule in rules:
            if applies(rule, query):
                applicable_rules.append(rule)

        # Winner + L general rules
        assert len(applicable_rules) == 1 + level

        for rule in applicable_rules:
            if rule.label != item["gold_answer"]:
                assert specificity(rule) == 3

        # All invalid rules should fail at least 2 conditions
        for rule in rules:
            if not applies(rule, query):
                assert failed_conditions(rule, query) >= 2

    assert level_counts[0] == 100
    assert level_counts[1] == 100
    assert level_counts[2] == 100
    assert level_counts[4] == 100

    print("Experiment 1: PASSED")


def validate_exp2(items):
    assert len(items) == 400

    level_counts = {
        0: 0,
        1: 0,
        2: 0,
        4: 0
    }

    for item in items:
        level = item["level"]
        level_counts[level] += 1

        query = make_query(item["query"])

        rules = []

        for rule_data in item["rules"]:
            rules.append(make_rule(rule_data))

        assert len(rules) == 10

        check_no_duplicate_rules(rules)

        result = solve(query, rules)

        assert len(result) == 1
        assert result[0].label == item["gold_answer"]
        assert specificity(result[0]) == 4

        near_match_count = 0

        for rule in rules:
            failures = failed_conditions(rule, query)

            if failures == 1:
                near_match_count += 1
                assert specificity(rule) == 4

            if not applies(rule, query) and failures != 1:
                assert failures >= 2

        assert near_match_count == level

        first_near_match = make_rule(
            item["first_near_match"]
        )

        assert specificity(first_near_match) == 4
        assert failed_conditions(first_near_match, query) == 1

    assert level_counts[0] == 100
    assert level_counts[1] == 100
    assert level_counts[2] == 100
    assert level_counts[4] == 100

    print("Experiment 2: PASSED")


def validate_exp3_independent(items):
    assert len(items) == 400

    yes = 0
    no = 0

    spec3_yes = 0
    spec3_no = 0
    spec4_yes = 0
    spec4_no = 0

    probe_counts = {
        "valid_3": 0,
        "valid_4": 0,
        "one_error": 0,
        "two_error": 0
    }

    for item in items:
        query = make_query(item["query"])
        rule = make_rule(item["rule"])

        probe_type = item["probe_type"]
        probe_counts[probe_type] += 1

        failures = failed_conditions(rule, query)

        assert failures == item["failed_conditions"]
        assert specificity(rule) == item["specificity"]

        if applies(rule, query):
            assert item["gold_answer"] == "YES"
            yes += 1
        else:
            assert item["gold_answer"] == "NO"
            no += 1

        if probe_type == "valid_3":
            assert failures == 0
            assert specificity(rule) == 3

        if probe_type == "valid_4":
            assert failures == 0
            assert specificity(rule) == 4

        if probe_type == "one_error":
            assert failures == 1

        if probe_type == "two_error":
            assert failures == 2

        if specificity(rule) == 3:
            if item["gold_answer"] == "YES":
                spec3_yes += 1
            else:
                spec3_no += 1

        if specificity(rule) == 4:
            if item["gold_answer"] == "YES":
                spec4_yes += 1
            else:
                spec4_no += 1

    assert yes == 200
    assert no == 200

    assert spec3_yes == 100
    assert spec3_no == 100
    assert spec4_yes == 100
    assert spec4_no == 100

    assert probe_counts["valid_3"] == 100
    assert probe_counts["valid_4"] == 100
    assert probe_counts["one_error"] == 100
    assert probe_counts["two_error"] == 100

    print("Experiment 3 independent: PASSED")


def validate_exp3_linked(items, exp2_items):
    assert len(items) == 200

    exp2_by_id = {}

    for item in exp2_items:
        exp2_by_id[item["id"]] = item

    yes = 0
    no = 0

    for item in items:
        query = make_query(item["query"])
        rule = make_rule(item["rule"])

        source_id = item["source_exp2_id"]

        assert source_id in exp2_by_id

        source = exp2_by_id[source_id]

        assert source["level"] == 1
        assert item["query"] == source["query"]

        if item["probe_type"] == "winner":
            assert item["gold_answer"] == "YES"
            assert failed_conditions(rule, query) == 0
            assert rule.label == source["gold_answer"]

            yes += 1

        elif item["probe_type"] == "near_match":
            assert item["gold_answer"] == "NO"
            assert failed_conditions(rule, query) == 1

            assert item["rule"] == source["first_near_match"]

            no += 1

        else:
            assert False

    assert yes == 100
    assert no == 100

    print("Experiment 3 linked: PASSED")


def validate_exp4(items, exp1_items):
    assert len(items) == 300

    exp1_by_id = {}

    for item in exp1_items:
        exp1_by_id[item["id"]] = item

    expected_rule_counts = {
        1: 2,
        2: 3,
        4: 5
    }

    level_counts = {
        1: 0,
        2: 0,
        4: 0
    }

    for item in items:
        level = item["level"]
        level_counts[level] += 1

        assert len(item["rules"]) == expected_rule_counts[level]

        source_id = item["source_exp1_id"]

        assert source_id in exp1_by_id

        source = exp1_by_id[source_id]

        assert item["query"] == source["query"]
        assert item["gold_answer"] == source["gold_answer"]

        query = make_query(item["query"])

        rules = []

        for rule_data in item["rules"]:
            rule = make_rule(rule_data)
            rules.append(rule)

            assert applies(rule, query)

        result = solve(query, rules)

        assert len(result) == 1
        assert result[0].label == item["gold_answer"]
        assert specificity(result[0]) == 4

        for rule in rules:
            if rule.label != item["gold_answer"]:
                assert specificity(rule) == 3

    assert level_counts[1] == 100
    assert level_counts[2] == 100
    assert level_counts[4] == 100

    print("Experiment 4: PASSED")


def validate_unique_ids(all_items):
    ids = []

    for item in all_items:
        ids.append(item["id"])

    assert len(ids) == len(set(ids))

    print("All IDs unique: PASSED")


def main():
    exp1 = load_jsonl(
        "data/generated/experiment_1_general.jsonl"
    )

    exp2 = load_jsonl(
        "data/generated/experiment_2_nearmatch.jsonl"
    )

    exp3_independent = load_jsonl(
        "data/generated/experiment_3_independent.jsonl"
    )

    exp3_linked = load_jsonl(
        "data/generated/experiment_3_linked.jsonl"
    )

    exp4 = load_jsonl(
        "data/generated/experiment_4_filtering.jsonl"
    )

    validate_exp1(exp1)
    validate_exp2(exp2)

    validate_exp3_independent(
        exp3_independent
    )

    validate_exp3_linked(
        exp3_linked,
        exp2
    )

    validate_exp4(
        exp4,
        exp1
    )

    all_items = (
        exp1
        + exp2
        + exp3_independent
        + exp3_linked
        + exp4
    )

    assert len(all_items) == 1700

    validate_unique_ids(all_items)

    print()
    print("==============================")
    print("ALL DATASET VALIDATION PASSED")
    print("TOTAL PROMPTS:", len(all_items))
    print("==============================")


if __name__ == "__main__":
    main()
