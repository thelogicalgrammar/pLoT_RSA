import z3
import numpy as np
from copy import copy, deepcopy

from hurford import hurford_constraint
from MSSSolver import MSSSolver, enumerate_sets, all_smt
from Models import BooleanModel, ObjectsModel, colors
from grammar import Parse


def softmax(x, temp=1):
    """
    Compute the softmax of vector x with temperature parameter temp.
    
    Arguments
    ---------
    x: numpy.ndarray
        Input array.
    temp: float
        (Negative) temperature parameter (default is 1).
    
    Returns
    -------
    numpy.ndarray 
        Softmax output.
    """
    x = np.asarray(x)  # Ensure x is a numpy array
    if temp == 0:
        raise ValueError("Temperature parameter temp cannot be zero.")
    x = x * temp
    e_x = np.exp(x - np.max(x))  # Subtract the max value for numerical stability
    return e_x / e_x.sum(axis=0)


def flatten(items, seqtypes=(list, tuple)):
    for i, x in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            items[i:i+1] = items[i]
    return items


# Function to check if two formulas are equivalent
def are_equivalent(f1, f2, solver):
    # Check if f1 and f2 are logically equivalent by checking (f1 <=> f2)
    solver.push()
    solver.add(z3.Not(z3.And(f1 == f2)))
    result = solver.check() == z3.unsat
    solver.pop()
    return result


def equivalence_table(expr_list, solver):
    """
    Generate a table showing which combinations of Z3 boolean formulas are equivalent.
    """
    n = len(expr_list)
    table = [[False] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                table[i][j] = True
            else:
                table[i][j] = table[j][i] = are_equivalent(
                    expr_list[i],
                    expr_list[j],
                    solver
                )
    
    return table


def print_equivalence_table(expr_list, table):
    """
    Print the equivalence table.
    """
    header = " ".join(f"{i:5}" for i in range(len(expr_list)))
    print(f"{'':5} {header}")
    for i, row in enumerate(table):
        row_str = " ".join(f"{str(val):5}" for val in row)
        print(f"{i:5} {row_str}")


def remove_duplicates(formulas, solver, return_indices=False):
    # Function to remove duplicates
    unique_formulas = []
    if return_indices:
        indices = []
    for i, formula in enumerate(formulas):
        # NOTE: Do NOT simplify; the internal structure
        # is potentially important for recognizing hurford constraint!
        is_unique = True
        for unique_formula in unique_formulas:
            if are_equivalent(formula, unique_formula, solver):
                is_unique = False
                break
        if is_unique:
            unique_formulas.append(formula)
            if return_indices:
                indices.append(i)
    return (indices, unique_formulas) if return_indices else unique_formulas


def check_c(a, b): 
    # check if a is a child of b or viceversa
    adepth, bdepth = a.depth(), b.depth()
    if adepth == bdepth:
        return False
    (maybep, maybec) = (a, b) if adepth > bdepth else (b, a)
    # note that we check for identity, not just equality
    return any(maybep is y.parent for y in maybec.up_to())


def produce_possible_structures(phonform, grammar):
    """
    Return a list of the ways of putting exactly one EXH
    in the sentence.
    
    Arguments
    ---------
    phonform: LOTlib3.FunctionNode.FunctionNode
        a phonological form 
        i.e., parse without the silent morpheme EXH
    grammar: LOTlib3 grammar
        for global call, the grammar for the phonological form 
        (with or without EXH)
    Return
    ------
    list[LOTlib3.FunctionNode.FunctionNode]
        A list of possible parses
        i.e. ways of inserting EXH
    """

    # if it does not contain EXH yet, add it
    try:
        grammar.get_rule_by_name('EXH')
    except AssertionError:
        grammar = deepcopy(grammar)
        # add EXH
        grammar.add_rule('BOOL', 'EXH', ['BOOL'], 2.0)

    # create a different copy of the full sentence
    # for each subnode, and then add EXH to 
    # nodes of type BOOL
    subnodes = phonform.subnodes()
    copies = [
        copy(phonform)
        for _ in range(len(subnodes))
    ]

    possibilities = []
    for i, (original_subn, formcopy) in enumerate(zip(subnodes, copies)):
        # if this is of type BOOL, we can add EXH
        if original_subn.type() == 'BOOL':
            # put EXH between node and supernode
            subnode = formcopy.subnodes()[i]
            supernode = subnode.parent
            EXH_node = (
                grammar
                .get_rule_by_name('EXH')
                .make_FunctionNodeStub(grammar, None)
            )
            EXH_node.args = [subnode]
            subnode.parent = EXH_node
            EXH_node.parent = supernode
            try:
                newargs = []
                for arg in supernode.args:
                    if arg is subnode:
                        newargs.append(EXH_node)
                    else:
                        newargs.append(arg)
                supernode.args = newargs
            except AttributeError:
                pass
            possibilities.append(copies[i])
    
    return possibilities


def find_phonform_possible_structures(phonform, grammar, qud, model, solver,
                                      unique=True):
    """
    Grammar is the parse grammar

    Generate the possible parse by placing a single EXH
    everywhere it can be placed.
    Parses are instances of the Parse class
    """

    parses_values = produce_possible_structures(
        phonform, 
        grammar
    )

    possible_parses = [
        Parse(
            qud=qud, 
            model=model,
            grammar=grammar,
            solver=solver,
            value=value,
        )
        for value in parses_values
    ]

    possible_meanings = [
        x() 
        for x in possible_parses
    ]

    ### HURFORD CONSTRAINT ###
    results = []
    for m, uv in zip(possible_meanings, parses_values):
        if hurford_constraint(m, solver):
            results.append((m, uv))
        else:
            print(f"{uv} has been excluded because failed hurford!")
    possible_meanings, parses_values = zip(*results)

    if unique:
        indices_unique, possible_meanings = remove_duplicates(
            possible_meanings,
            solver,
            return_indices=True
        )
        parses_values = [parses_values[i] for i in indices_unique]

    return possible_meanings, parses_values


def print_possible_parses(parses_values, possible_meanings, 
                              n_props, solver):
    
    boolean_model = BooleanModel(n_props)
    print('terms: ', boolean_model.terms, '\n')
    for i in range(len(parses_values)):
        print(parses_values[i])
        m = possible_meanings[i]
        if solver is not None and n_props is not None:
            solver.push()
            solver.add(m)
            for model in all_smt(solver, boolean_model.terms):
                print('\t', model)
            solver.pop()
            print('\n')
        else:
            print('No solver provided for printing models!')

    print('Synonymy table:')
    table = equivalence_table(
        possible_meanings,
        solver
    )
    print_equivalence_table(
        possible_meanings,
        table
    )
    print()


