import random

from src.generator_exp1 import (
    generate_query,
    generate_winner,
    generate_general_rules,
    generate_invalid_rule,
    generate_exp1_family,
    generate_exp1_variants
)
from src.solver import applies, failed_conditions, specificity, solve

def test_query_has_six_attributes():
    random.seed(42)

    query = generate_query()

    assert len(query.values) == 6


def test_winner_has_four_conditions():
    random.seed(42)

    query = generate_query()
    winner = generate_winner(query)

    assert specificity(winner) == 4
    assert applies(winner, query) == True


def test_four_general_rules_are_generated():
    random.seed(42)

    query = generate_query()
    winner = generate_winner(query)
    general_rules = generate_general_rules(winner)

    assert len(general_rules) == 4


def test_general_rules_have_three_conditions():
    random.seed(42)

    query = generate_query()
    winner = generate_winner(query)
    general_rules = generate_general_rules(winner)

    for rule in general_rules:
        assert specificity(rule) == 3
        assert applies(rule, query) == True


def test_general_rules_are_different():
    random.seed(42)

    query = generate_query()
    winner = generate_winner(query)
    general_rules = generate_general_rules(winner)

    conditions_list = []

    for rule in general_rules:
        conditions_list.append(rule.conditions)

    assert len(conditions_list) == 4

    for i in range(4):
        for j in range(i + 1, 4):
            assert conditions_list[i] != conditions_list[j]

def test_invalid_rule_with_four_conditions():
    random.seed(42)

    query = generate_query()
    rule = generate_invalid_rule(query, 4)

    assert specificity(rule) == 4
    assert applies(rule, query) == False
    assert failed_conditions(rule, query) == 2


def test_invalid_rule_with_three_conditions():
    random.seed(42)

    query = generate_query()
    rule = generate_invalid_rule(query, 3)

    assert specificity(rule) == 3
    assert applies(rule, query) == False
    assert failed_conditions(rule, query) == 2


def test_invalid_rule_with_two_conditions():
    random.seed(42)

    query = generate_query()
    rule = generate_invalid_rule(query, 2)

    assert specificity(rule) == 2
    assert applies(rule, query) == False
    assert failed_conditions(rule, query) == 2

def test_exp1_family_has_ten_rules():
    random.seed(42)

    family = generate_exp1_family()

    assert len(family["rules"]) == 10


def test_exp1_family_has_one_winner():
    random.seed(42)

    family = generate_exp1_family()

    query = family["query"]
    rules = family["rules"]

    result = solve(query, rules)

    assert len(result) == 1
    assert result[0] == family["winner"]


def test_exp1_family_has_four_replaceable_rules():
    random.seed(42)

    family = generate_exp1_family()

    assert len(family["replaceable_rules"]) == 4

    for rule in family["replaceable_rules"]:
        assert specificity(rule) == 3
        assert applies(rule, family["query"]) == False
        assert failed_conditions(rule, family["query"]) == 2

def test_exp1_family_has_no_duplicate_rules():
    random.seed(42)

    family = generate_exp1_family()
    rules = family["rules"]

    for i in range(len(rules)):
        for j in range(i + 1, len(rules)):
            assert rules[i].conditions != rules[j].conditions

def test_many_exp1_families_have_no_duplicates():
    random.seed(42)

    for i in range(100):
        family = generate_exp1_family()
        rules = family["rules"]

        assert len(rules) == 10

        for first in range(len(rules)):
            for second in range(first + 1, len(rules)):
                assert rules[first].conditions != rules[second].conditions


def test_exp1_has_four_levels():
    random.seed(42)

    family = generate_exp1_family()
    variants = generate_exp1_variants(family)

    assert len(variants) == 4

    assert 0 in variants
    assert 1 in variants
    assert 2 in variants
    assert 4 in variants


def test_every_exp1_variant_has_ten_rules():
    random.seed(42)

    family = generate_exp1_family()
    variants = generate_exp1_variants(family)

    for level in variants:
        assert len(variants[level]) == 10



def test_rule_labels_stay_in_same_positions():
    random.seed(42)

    family = generate_exp1_family()
    variants = generate_exp1_variants(family)

    labels_l0 = []
    labels_l1 = []
    labels_l2 = []
    labels_l4 = []

    for rule in variants[0]:
        labels_l0.append(rule.label)

    for rule in variants[1]:
        labels_l1.append(rule.label)

    for rule in variants[2]:
        labels_l2.append(rule.label)

    for rule in variants[4]:
        labels_l4.append(rule.label)

    assert labels_l0 == labels_l1
    assert labels_l0 == labels_l2
    assert labels_l0 == labels_l4


def test_correct_number_of_applicable_rules_per_level():
    random.seed(42)

    family = generate_exp1_family()
    variants = generate_exp1_variants(family)

    query = family["query"]

    expected_counts = {
        0: 1,
        1: 2,
        2: 3,
        4: 5
    }

    for level in variants:
        applicable_count = 0

        for rule in variants[level]:
            if applies(rule, query):
                applicable_count += 1

        assert applicable_count == expected_counts[level]


def test_winner_is_unique_at_every_level():
    random.seed(42)

    family = generate_exp1_family()
    variants = generate_exp1_variants(family)

    query = family["query"]

    for level in variants:
        result = solve(query, variants[level])

        assert len(result) == 1
        assert result[0].label == family["winner"].label

def test_exp1_variants_are_reproducible():
    random.seed(42)

    family = generate_exp1_family()

    variants_first = generate_exp1_variants(family)
    variants_second = generate_exp1_variants(family)

    for level in [0, 1, 2, 4]:
        rules_first = variants_first[level]
        rules_second = variants_second[level]

        for i in range(len(rules_first)):
            assert rules_first[i].label == rules_second[i].label
            assert rules_first[i].conditions == rules_second[i].conditions
