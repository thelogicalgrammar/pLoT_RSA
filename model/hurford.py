import z3

def entails(expr1, expr2, solver):
    """
    Check if expr1 entails expr2.
    """
    solver.push()
    solver.add(z3.Not(z3.Implies(expr1, expr2)))
    if solver.check() == z3.unsat:
        output = True
    else:
        output = False
    solver.pop()
    return output

def hurford_constraint(expr, solver):
    """
    Recursively check if the expression or any sub-expression satisfies the Hurford constraint.
    """
    if z3.is_or(expr):
        disjuncts = expr.children()
        for i in range(len(disjuncts)):
            for j in range(len(disjuncts)):
                if i != j and entails(
                        disjuncts[i],
                        disjuncts[j],
                        solver
                    ):
                    return False
        # Recursively check each disjunct
        for disjunct in disjuncts:
            if not hurford_constraint(disjunct, solver):
                return False
    else:
        # Recursively check the children of the expression
        for child in expr.children():
            if not hurford_constraint(child, solver):
                return False
    return True

def check_hurford_constraints(expr_list, solver):
    """
    Check a list of Z3 expressions for the Hurford constraint.
    """
    results = []
    for expr in expr_list:
        results.append(hurford_constraint(expr, solver))
    return results
