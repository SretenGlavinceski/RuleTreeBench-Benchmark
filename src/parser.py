import re


def parse_rule_answer(text):
    lines = text.strip().splitlines()

    # First look for the requested ANSWER: R# format.
    for line in reversed(lines):
        match = re.fullmatch(
            r"\s*ANSWER:\s*(R(?:10|[1-9]))\s*",
            line,
            re.IGNORECASE
        )

        if match:
            return match.group(1).upper(), False

    # Fallback:
    # only accept a standalone rule label on the final non-empty line.
    if len(lines) > 0:
        last_line = lines[-1].strip()

        match = re.fullmatch(
            r"R(?:10|[1-9])",
            last_line,
            re.IGNORECASE
        )

        if match:
            return last_line.upper(), True

    return None, False


def parse_yes_no_answer(text):
    lines = text.strip().splitlines()

    for line in reversed(lines):
        match = re.fullmatch(
            r"\s*ANSWER:\s*(YES|NO)\s*",
            line,
            re.IGNORECASE
        )

        if match:
            return match.group(1).upper(), False

    if len(lines) > 0:
        last_line = lines[-1].strip().upper()

        if last_line == "YES":
            return "YES", True

        if last_line == "NO":
            return "NO", True

    return None, False
