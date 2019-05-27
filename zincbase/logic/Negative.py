"""Internal ZincBase class for negative training examples"""

from zincbase.utils.string_utils import split_on

class Negative:
    def __init__(self, expr):
        sub_exprs = split_on(expr, '(', all=False)
        if len(sub_exprs) != 2:
            raise Exception('Syntax error')
        atoms = split_on(sub_exprs[1][:-1], ',')
        self.head = atoms[0]
        self.pred = sub_exprs[0]
        self.tail = atoms[1]
    def __repr__(self):
        return self.pred + '(' + self.head + ', ' + self.tail + ')'
