def isVar(term):
    return term.args == [] and term.pred[0].isupper()

def isAtom(term):
    return term.args == [] and not term.pred[0].isupper()
