from utils.string_utils import split_on

from logic.Term import Term

class Rule:
    def __init__(self, expr, graph=None):
        parts = split_on(expr, ':-')
        self.head = Term(parts[0], graph=graph)
        self.goals = []
        if len(parts) == 2:
            sub_goals = split_on(parts[1], ',')
            for sub_goal in sub_goals:
                self.goals.append(Term(sub_goal, graph=graph))
    def __repr__(self):
        return str(self.head)