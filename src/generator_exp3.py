import random

from src.models import Rule
from src.generator_exp1 import (
    ATTRIBUTES,
    generate_query,
    get_different_value
)


def generate_valid_rule(query, rule_specificity):
    selected_attributes = random.sample(
        ATTRIBUTES,
        rule_specificity
    )

    conditions = {}

    for attribute in selected_attributes:
        conditions[attribute] = query.values[attribute]

    return Rule("R1", conditions)


def generate_invalid_rule(query, rule_specificity, number_of_errors):
    selected_attributes = random.sample(
        ATTRIBUTES,
        rule_specificity
    )

    conditions = {}

    for attribute in selected_attributes:
        conditions[attribute] = query.values[attribute]

    wrong_attributes = random.sample(
        selected_attributes,
        number_of_errors
    )

    for attribute in wrong_attributes:
        conditions[attribute] = get_different_value(
            query.values[attribute]
        )

    return Rule("R1", conditions)


def generate_exp3_independent_group(group_number):
    query = generate_query()

    valid_3 = generate_valid_rule(query, 3)
    valid_4 = generate_valid_rule(query, 4)

    # We alternate rule lengths so the whole dataset is balanced.
    if group_number % 2 == 1:
        one_error_specificity = 3
        two_error_specificity = 4
    else:
        one_error_specificity = 4
        two_error_specificity = 3

    one_error = generate_invalid_rule(
        query,
        one_error_specificity,
        1
    )

    two_error = generate_invalid_rule(
        query,
        two_error_specificity,
        2
    )

    return {
        "query": query,
        "valid_3": valid_3,
        "valid_4": valid_4,
        "one_error": one_error,
        "two_error": two_error
    }
