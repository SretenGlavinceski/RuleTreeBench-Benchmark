from src.parser import (
    parse_rule_answer,
    parse_yes_no_answer
)


def test_rule_normal_answer():
    text = """
Some reasoning here.

ANSWER: R4
"""

    answer, fallback = parse_rule_answer(text)

    assert answer == "R4"
    assert fallback == False


def test_rule_ten():
    text = "ANSWER: R10"

    answer, fallback = parse_rule_answer(text)

    assert answer == "R10"
    assert fallback == False


def test_rule_fallback():
    text = """
Some reasoning.

R7
"""

    answer, fallback = parse_rule_answer(text)

    assert answer == "R7"
    assert fallback == True


def test_rule_does_not_parse_discussion():
    text = """
R1 does not apply.
R8 does not apply.
No final answer was given.
"""

    answer, fallback = parse_rule_answer(text)

    assert answer is None


def test_yes():
    text = """
The rule applies.

ANSWER: YES
"""

    answer, fallback = parse_yes_no_answer(text)

    assert answer == "YES"
    assert fallback == False


def test_no():
    text = "ANSWER: NO"

    answer, fallback = parse_yes_no_answer(text)

    assert answer == "NO"
    assert fallback == False


def test_yes_no_fallback():
    text = """
Some explanation.

NO
"""

    answer, fallback = parse_yes_no_answer(text)

    assert answer == "NO"
    assert fallback == True
