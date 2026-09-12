def render_query(query):
    lines = []

    for attribute in query.values:
        value = query.values[attribute]
        lines.append(attribute + "=" + value)

    return "\n".join(lines)


def render_rule(rule):
    conditions = []

    for attribute in rule.conditions:
        value = rule.conditions[attribute]
        conditions.append(attribute + "=" + value)

    conditions_text = " AND ".join(conditions)

    return rule.label + ": IF " + conditions_text


def render_rules(rules):
    lines = []

    for rule in rules:
        lines.append(render_rule(rule))

    return "\n".join(lines)


def render_rule_task(query, rules):
    prompt = """
You are given a query and a set of rules.

A rule applies only if ALL of its conditions match the query.

If more than one rule applies, choose the applicable rule with the most conditions.

Return your final answer on the last line exactly in this form:

ANSWER: R#

RULES:
{rules}

QUERY:
{query}
"""

    return prompt.format(
        rules=render_rules(rules),
        query=render_query(query)
    ).strip()


def render_applicability_task(query, rule):
    prompt = """
You are given a query and one rule.

A rule applies only if ALL of its conditions match the query.

Does the rule apply?

Return your final answer on the last line exactly as:

ANSWER: YES

or

ANSWER: NO

RULE:
{rule}

QUERY:
{query}
"""

    return prompt.format(
        rule=render_rule(rule),
        query=render_query(query)
    ).strip()



