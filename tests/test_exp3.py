import random

from src.generator_exp3 import (
    generate_valid_rule,
    generate_invalid_rule,
    generate_exp3_independent_group
)

from src.generator_exp1 import generate_query

from src.solver import (
    applies,
    failed_conditions,
    specificity
)


def test_valid_three_condition_rule():
    random.seed(42)

    query = generate_query()
    rule = generate_valid_rule(query, 3)

    assert specificity(rule) == 3
    assert applies(rule, query) == True
    assert failed_conditions(rule, query) == 0


def test_valid_four_condition_rule():
    random.seed(42)

    query = generate_query()
    rule = generate_valid_rule(query, 4)

    assert specificity(rule) == 4
    assert applies(rule, query) == True
    assert failed_conditions(rule, query) == 0


def test_one_error_rule():
    random.seed(42)

    query = generate_query()
    rule = generate_invalid_rule(query, 4, 1)

    assert specificity(rule) == 4
    assert applies(rule, query) == False
    assert failed_conditions(rule, query) == 1


def test_two_error_rule():
    random.seed(42)

    query = generate_query()
    rule = generate_invalid_rule(query, 4, 2)

    assert specificity(rule) == 4
    assert applies(rule, query) == False
    assert failed_conditions(rule, query) == 2


def test_group_has_four_probe_types():
    random.seed(42)

    group = generate_exp3_independent_group(1)

    assert "valid_3" in group
    assert "valid_4" in group
    assert "one_error" in group
    assert "two_error" in group


def test_odd_group_invalid_lengths():
    random.seed(42)

    group = generate_exp3_independent_group(1)

    assert specificity(group["one_error"]) == 3
    assert specificity(group["two_error"]) == 4


def test_even_group_invalid_lengths():
    random.seed(42)

    group = generate_exp3_independent_group(2)

    assert specificity(group["one_error"]) == 4
    assert specificity(group["two_error"]) == 3


def test_one_hundred_groups_are_balanced():
    random.seed(42)

    one_error_spec3 = 0
    one_error_spec4 = 0

    two_error_spec3 = 0
    two_error_spec4 = 0

    for group_number in range(1, 101):
        group = generate_exp3_independent_group(group_number)

        if specificity(group["one_error"]) == 3:
            one_error_spec3 += 1
        else:
            one_error_spec4 += 1

        if specificity(group["two_error"]) == 3:
            two_error_spec3 += 1
        else:
            two_error_spec4 += 1

    assert one_error_spec3 == 50
    assert one_error_spec4 == 50

    assert two_error_spec3 == 50
    assert two_error_spec4 == 50
