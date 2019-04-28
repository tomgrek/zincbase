import copy

class Goal:
    def __init__(self, rule, parent=None, bindings={}):
        self.rule = rule
        self.parent = parent
        self.bindings = copy.deepcopy(bindings)
        self.idx = 0