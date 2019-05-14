"""A base unit for ZincBase's Prolog-like implementation of 'facts'"""

from utils.string_utils import split_on

class Term:
    def __init__(self, expr, args=None, graph=None):
        if args:
            self.pred = expr
            self.args = args
        elif expr[-1] == ']':
            arr = split_on(expr[1:-1], ',')
            headtail = split_on(expr[1:-1], '|')
            if len(headtail) > 1:
                self.args = [Term(f, graph=graph) for f in headtail]
                self.pred = '__list__'
            else:
                arr.reverse()
                first = Term('__list__', [], graph=graph)
                for part in arr:
                    first = Term('__list__', [Term(part, graph=graph), first], graph=graph)
                self.pred = first.pred
                self.args = first.args
        elif expr[-1] == ')':
            sub_exprs = split_on(expr, '(', all=False)
            if len(sub_exprs) != 2:
                raise Exception('Syntax error')
            self.args = [Term(sub_expr, graph=graph) for sub_expr in split_on(sub_exprs[1][:-1], ',')]
            self.pred = sub_exprs[0]
        else:
            self.pred = expr
            self.args = []

        if graph is not None:
            for i, arg in enumerate(self.args):
                if arg:
                    if not graph.has_node(str(arg)):
                        graph.add_node(str(arg))
                    for arg2 in self.args[i+1:]:
                        if not graph.has_node(str(arg2)):
                            graph.add_node(str(arg2))
                        graph.add_edge(str(arg), str(arg2), pred=self.pred)

    def __repr__(self):
        if self.pred == '__list__':
            if not self.args:
                return '[]'
            first = self.args[1]
            if first.pred == '__list__' and first.args == []:
                return '[{}]'.format(str(self.args[0]))
            elif first.pred == '__list__':
                return '[{},{}]'.format(str(self.args[0]), str(self.args[1])[1:-1])
            else:
                return '[{}|{}]'.format(str(self.args[0]), str(self.args[1]))
        elif self.args:
            return '{}({})'.format(self.pred, ', '.join(map(str,self.args)))
        else:
            return self.pred
