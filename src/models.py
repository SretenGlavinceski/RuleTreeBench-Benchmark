class Query:
    def __init__(self, values):
        self.values = values


class Rule:
    def __init__(self, label, conditions):
        self.label = label
        self.conditions = conditions
