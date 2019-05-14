"""Under-the-hood details of ZincBase's Prolog-like implementation"""

import copy

from logic.Term import Term
from utils.type_checks import isVar, isAtom

def unify(src, src_bindings, dest, dest_bindings):
    if src.pred == '_' or dest.pred == '_':
        return True
    if isVar(src):
        tmp_src = process(src, src_bindings)
        if not tmp_src:
            return True
        else:
            return unify(tmp_src, src_bindings, dest, dest_bindings)
    if isVar(dest):
        tmp_dest = process(dest, dest_bindings)
        if tmp_dest:
            return unify(src, src_bindings, tmp_dest, dest_bindings)
        else:
            dest_bindings[dest.pred] = process(src, src_bindings)
            return True
    if len(src.args) != len(dest.args):
        return False
    elif src.pred != dest.pred:
        return False
    else:
        dest_bindings_copy = copy.deepcopy(dest_bindings)
        for i in range(len(src.args)):
            if not unify(src.args[i], src_bindings, dest.args[i], dest_bindings_copy):
                return False
        dest_bindings.update(dest_bindings_copy)
        return True

def process(term, bindings, graph=None):
    if isAtom(term):
        return term
    if isVar(term):
        ans = bindings.get(term.pred, None)
        if not ans:
            return None
        else:
            return process(ans, bindings)
    args = []
    for arg in term.args:
        a = process(arg, bindings)
        if not a:
            return None
        args.append(a)
    return Term(term.pred, args, graph=graph)
