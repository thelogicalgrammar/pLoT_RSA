import z3
from z3 import Solver, Const, Int, Real, EnumSort, sat
import LOTlib3
from LOTlib3.Grammar import Grammar
from LOTlib3.Eval import primitive
from LOTlib3.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib3.Hypotheses.Likelihoods.BinaryLikelihood import BinaryLikelihood
import numpy as np
from functools import cache
import builtins
from copy import copy, deepcopy
from itertools import product
from pprint import pprint
from functools import reduce
from MSSSolver import MSSSolver, enumerate_sets, all_smt


try:
    # Use Enums for categorical data (color in this case)
    Color, colors = EnumSort(
        'Color', 
        ['red', 'blue', 'green']
    )
    
    red, blue, green = colors
except z3.Z3Exception:
    print('already defined!')

symmetric_ops = [
    'and_', 'or_'
]


def define_grammar(EXH=False):
    """
    __NOTE__: All function rules must have .name attribute that can be evaled into the corresponding function!
    This is needed because when interpreting the function I need to evaluate the argument by itself.
    """
    
    grammar = Grammar()

    # NOTE: Keep this as empty string name, because that's how 
    # it's recognized in the _exhaustify function
    grammar.add_rule('START', '', ['BOOL'], 1.0)
    
    grammar.add_rule('BOOL', 'land', ['BOOL', 'BOOL'], 0.5)
    grammar.add_rule('BOOL', 'lor', ['BOOL', 'BOOL'], 1.0)
    grammar.add_rule('BOOL', 'ifthen', ['BOOL', 'BOOL'], 1.0)
    grammar.add_rule('BOOL', 'lnot', ['BOOL'], 1.0)
    
    grammar.add_rule('BOOL', 'p', ['MODEL'], 2.0)
    grammar.add_rule('BOOL', 'q', ['MODEL'], 2.0)

    grammar.add_rule('MODEL', 'M', None, 1.0)
    
    # --> is_color_(OBJECT, 'red') --> OBJECT.color == 'red'
    # grammar.add_rule('BOOL', 'is_color',  ['OBJECT', 'COLOR'], 5.00)
    
    # for c in colors:
    #     grammar.add_rule('COLOR', c.sexpr(), None, 1.0)
    
    # for i in range(n_objects):
    #     grammar.add_rule('OBJECT', f'W[{i}]', None, 1.0)

    if EXH:
        # add EXH for bool
        # NOTE: this must be at the end!
        grammar.add_rule('BOOL', 'EXH', ['BOOL'], 2.0)

    return grammar


# make colors available to LOTlib3
for c in colors:
    builtins.__dict__[c.sexpr()] = c

@primitive
def p(model):
    return model('p')

@primitive
def q(model):
    return model('q')

@primitive
def land(p, q):
    return z3.And(p, q)

@primitive
def lor(p, q):
    return z3.Or(p, q)

@primitive
def lnot(p):
    return z3.Not(p)

@primitive
def ifthen(p, q):
    return z3.Implies(p, q)

@primitive
def is_color(obj, col):
    return obj.color == col

@primitive
def EXH(h):
    """
    Returns the exhaustified meaning.
    
    This is a bit tricky.
    The reason using EXH is complicated is that
    when using python's "eval" is not aware of the syntax
    and so cannot find structural alternatives.

    h: z3 formula
    """
    return h


class Obj:
    def __init__(self, index, color):
        """
        - Each object is modelled as a dictionary of z3 variables, corresponding to the various properties of the object.
        - Features can be of different types e.g. Enumerate, binary, etc.
        """
        self.color = Const(f'{index}:color', color)
        self.price = Real(f'{index}:price')
        self.speed = Int(f'{index}:speed')

    def __getitem__(self, propname):
        return getattr(self, propname)


class BooleanModel:
    def __init__(self):
        self.p = z3.Bool('p')
        self.q = z3.Bool('q')

    def __call__(self, i):
        # get a proposition
        return getattr(self, i)


class ObjectsModel:
    def __init__(self, n_objects):
        self.objects = [
            Obj(index, Color)
            for index in range(n_objects)
        ]

    def __getitem__(self, i):
        # get one or more objects
        return self.objects[i]

    def get_prop(self, i, prop):
        # get a property of a specific object
        return self.objects[i][prop]


def flatten(items, seqtypes=(list, tuple)):
    for i, x in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            items[i:i+1] = items[i]
    return items


# Function to check if two formulas are equivalent
def are_equivalent(f1, f2):
    solver = z3.Solver()
    # Check if f1 and f2 are logically equivalent by checking (f1 <=> f2)
    equivalence = z3.And(f1 == f2)
    solver.add(z3.Not(equivalence))
    return solver.check() == z3.unsat


# Function to remove duplicates
def remove_duplicates(formulas):
    unique_formulas = []
    for formula in formulas:
        # Simplify the formula to a canonical form
        # formula = z3.simplify(formula)
        is_unique = True
        for unique_formula in unique_formulas:
            if are_equivalent(formula, unique_formula):
                is_unique = False
                break
        if is_unique:
            unique_formulas.append(formula)
    return unique_formulas


def check_c(a, b): 
    # check if a is a child of b or viceversa
    adepth, bdepth = a.depth(), b.depth()
    if adepth == bdepth:
        return False
    (maybep, maybec) = (a, b) if adepth > bdepth else (b, a)
    # note that we check for identity, not just equality
    return any(maybep is y.parent for y in maybec.up_to())


class Exhaustifier:

    def __init__(self, grammar, qud, cache_fs=False):
        """
        Following Katzir, we define the structural alternatives of a sentence as the sentences that can be produced by 
            1. Deletion
            2. Contraction
            3. Replacement
        of its nodes with other nodes from the substitution source (which we assume here for simplicity is the lexicon).

        NOTE: This is a superclass of Utterance, so has access to self.QUD
        """

        self.grammar = grammar
        self.qud = qud

        # Do this so that instances can get garbage collected
        # see: https://rednafi.com/python/lru_cache_on_methods/
        # but NOTE: Caching gives wrong results atm!
        self.innocently_excludable = (
            cache(self._innocently_excludable)
            if cache_fs
            else self._innocently_excludable
        )
        self.exhaustify = (
            cache(self._exhaustify)
            if cache_fs
            else self._exhaustify
        )
        self.find_structural_alternatives = (
            cache(self._find_structural_alternatives)
            if cache_fs
            else self._find_structural_alternatives
        )

    def _innocently_excludable(self, x, negalts):
        solver = MSSSolver(x, negalts)
        mms = tuple(enumerate_sets(solver))
        intersection = reduce(lambda x, y: x & set(y), mms, set(mms[0]))
        # interpret as z3 expressions rather than boolean literals
        returnvalue = [solver.vars_dict[i] for i in intersection]
        return returnvalue

    def _exhaustify(self, x, M):
        """
        Computes the exhaustified addition to utterance x recursively.
        x is a LOTlib3 FunctionNode that potentially contains EXHs.
        
        1. If the node is an EXH:
            1. runs self.find_structural_alternatives on its arg
            2. selects the alts ALTS whose negation does not contradict p and is not entailed by p
            3. find the innocently excludable subset
            4. return a conjunction of their negations, which is the exhaustified meaning
        2. If the node is not EXH:
            1. Run self._exhaustify on each argument, pass up resulting meaning
                (to replace its literal meaning with its (potentially) exhaustified meaning, 
                    if it contains an EXH)

        Arguments
        ---------
        x: LOTlib3.FunctionNode
            The node we are exhaustifying

        Returns
        -------
        z3 formula
            The interpretation of the utterance
        """
        
        if x.name == 'EXH':
            # EXH has a single argument
            subnode = list(x.argFunctionNodes())[0]
            # get interpretation of the subnode
            x_int = self._exhaustify(subnode, M)
            s = z3.Solver()
            s.add(x_int)
            # find the recursively defined structural alternatives
            # and add the negation of each to conjs
            conjs = []
            for y in self.find_structural_alternatives(subnode):
                content = self._exhaustify(y,M)
                interpretation = z3.Not(content)
                s.push()
                s.add(interpretation)
                # check if the alternative is relevant to the QUD
                relevant = self.qud.is_relevant(interpretation)
                if relevant and (s.check() == z3.sat):
                    # conjs.append(z3.simplify(interpretation))
                    conjs.append(interpretation)
                s.pop()
            # find innocently excludable subset of those alts
            inn_exh = list(self.innocently_excludable(
                x_int, 
                tuple(conjs)
            ))
            returnvalue = z3.And([x_int] + inn_exh)
            return returnvalue
        else:
            # get the potentially enriched meaning for each 
            # argument of the expression, i.e.,
            # modify the arguments of the current top node
            # so that they have their exhaustified content.
            # E.g. not(exh(p)) --> not(p and not q)
            if x.is_nonfunction():
                return eval(str(x))

            # get exhaustified content for each argument
            exh_content = [
                self._exhaustify(y, M)
                for y 
                in x.argFunctionNodes()
            ]

            if x.name == '':
                # this is the top node, which does not do anything
                # so just run identity function
                node_int = lambda x: x
            elif x.name == 'M':
                return M
            else:
                # run top argument with exhaustified content
                # first get object
                node_int = eval(x.name)
            
            enriched_int = node_int(*exh_content)
            return enriched_int

    def _find_structural_alternatives(self, x):
        """
        Defined recursively for a LOTlib3 FunctionNode x!
        Takes a node and returns the structural alternatives to that node
        """
        nargs = x.nargs()
        xtype = x.get_rule_signature()[0]

        if x.name == 'EXH':
            yield from self.find_structural_alternatives(
                # EXH has a single argument!
                list(x.argFunctionNodes())[0]
            )
            # don't keep going!
            return
    
        if nargs == -1:
            # if x is a leaf, you return
            # each of the lexical alternatives of the same type
            for alt_r in self.grammar.get_rules(x.type()):
                alt = alt_r.make_FunctionNodeStub(self.grammar, None)
                if not alt.is_function():
                    yield alt
        
        else:
            
            # return the alts of each child, if of the same type as parent
            # This corresponds to Katzir's deletion and contraction
            # E.g., "p and q" -> "p"
            # E.g., "not p"   -> "p"
            for arg in x.args:
                if arg.get_rule_signature()[0] == xtype:
                    for i in self.find_structural_alternatives(arg):
                        yield copy(i)        
        
            # if x has one or more arguments,
            # return the alternatives obtained by
            # the product of each child's alts
            # & the node's own alts
    
            # loop through all combos of alternatives of arguments
            # (parents are none here for the arguments but are set below)
            args_alts_gens = [
                list(self.find_structural_alternatives(y))
                for y in x.args
            ]
            
            # Consider all possible alternative rules for self
            for r in self.grammar.get_rules(xtype):
                # make sure argument types match
                if r.to == x.argTypes():
                    for args in product(*args_alts_gens):
                        self_alt = r.make_FunctionNodeStub(self.grammar, None)
                        if self_alt.is_canonical_order(symmetric_ops) and self_alt.name!='EXH':
                            # copy because product reuses the objects
                            self_alt.args = [copy(a) for a in args]
                            for arg in self_alt.argFunctionNodes():
                                arg.parent = self_alt
                            yield self_alt 


class QUD:
    
    def __init__(self, props):
        """
        This class encodes a Question Under Discussion (QUD).

        Keep track of partition on the model set,
        each element being a possible answers to the question.
        See: https://centaur.reading.ac.uk/80434/2/Questions%20under%20discussion_final.pdf

        Arguments
        ---------
        props: list[z3 formula]
            Propositions that ought to be settled for the QUD;
            they partition the possible worlds.
        """
        self.props = props
        # check if the partition is valid
        assert self.check_partition(), "The partition is not valid!"

    def check_partition(self):
        """
        Check that the set of propositions self.prop is a
        partition of the model space.
        """
        # Check mutual exclusivity
        mutual_exclusivity = True
        for i in range(len(self.props)):
            for j in range(i + 1, len(self.props)):
                s = Solver()
                s.add(z3.And(self.props[i], self.props[j]))
                if s.check() == z3.sat:
                    mutual_exclusivity = False
                    break
            if not mutual_exclusivity:
                break

        # Check collective exhaustiveness
        s = z3.Solver()
        s.add(z3.Not(z3.Or(self.props)))
        collective_exhaustiveness = (s.check() == z3.unsat)

        # Output results
        if mutual_exclusivity and collective_exhaustiveness:
            return True
        else:
            if not mutual_exclusivity:
                print("The set of self.props is not mutually exclusive.")
            if not collective_exhaustiveness:
                print("The set of self.props is not collectively exhaustive.")

    def is_relevant(self, answer, solver=None):
        """
        Evaluates if the answer is relevant to the QUD. An answer is relevant if it:
            - Entails the negation of at least one alternative (partial answer).
            - Its negation is a partial answer.
        Note that "resolving answers" are also partial answers.

        Arguments
        ---------
        meaning: z3 formula 
            The meaning to evaluate the QUD on
        """
        if solver is None:
            solver = z3.Solver()
        for prop in self.props:
            h1 = self.implies(prop, answer, solver)
            h2 = self.implies(prop, z3.Not(answer), solver)
            if h1 in ['implies', 'impliesnot']:
                return True
            if h2 in ['implies', 'impliesnot']:
                return True
        return False

    def implies(self, prop, obs, solver):
        """
        Checks if an observation implies 
        one of the possible answers or their negation.
        """

        solver.push()
        # check if the observation implies the proposition
        solver.add(z3.Not(z3.Implies(obs, prop)))
        if solver.check() == z3.unsat:
            outcome = 'implies'
        else:
            solver.pop()
            solver.push()
            # check if the observation implies the negation of the proposition
            solver.add(z3.Not(z3.Implies(obs, z3.Not(prop))))
            outcome = (
                'impliesnot' 
                if solver.check() == z3.unsat 
                else 'norelation'
            )
        solver.pop()
        return outcome


class MeaningHypothesis(LOTHypothesis):

    def __init__(self, QUD, grammar=None, n_objects=5, **kwargs):
        """
        These encode underlying meanings
        """

        if grammar is None:
            # we don't need EXH for meanings
            grammar = define_grammar(EXH=False)
        
        LOTHypothesis.__init__(
            self,
            grammar=grammar,
            display='lambda M: %s',
            **kwargs
        )

        self.QUD = QUD
        self.n_objects = n_objects
        self.model = BooleanModel()

    def compute_single_likelihood(self, datum):
        # get interpretation as a z3 object
        f = self()
        utt = datum.input


class Utterance(LOTHypothesis, Exhaustifier):
    
    def __init__(self, qud, value=None, grammar=None, n_objects=5, **kwargs):
        """
        An utterance is the underlying representation of a phonological form
        including the silent morpheme EXH (and potentially more!).
        This is what the semantics sees.

        Arguments
        ---------
        qud: QUD
            The qud is a way of restricting which alternatives are considered.
        grammar: LOTlib3 grammar
            Grammar with EXH (since this is an utterance and not just
            a phonological form).
        value: LOTlib3.FunctionNode.FunctionNode
            The value of the utterance as a FunctionNode generated by the grammar.
        """

        if grammar is None:
            grammar = define_grammar(EXH=True)
        
        LOTHypothesis.__init__(
            self,
            grammar=grammar,
            display='lambda M: %s',
            value=value,
            **kwargs
        )

        self.n_objects = n_objects
        self.model = BooleanModel()

        # an exhaustifier to use in the likelihood calculation
        Exhaustifier.__init__(
            self,
            grammar=grammar,
            qud=qud
        )

    def __call__(self):
        """
        overwrite the exhaustify method from Exhaustifier so that it uses
        the value of the utterance as the input.
        """
        return self.exhaustify(self.value, self.model)

    def compute_single_likelihood(self, datum):
        """
        This doesn't need to be defined since we are not using these
        for inference.
        """
        pass


def produce_possible_structures(phonform, grammar):
    """
    Return a list of the ways of putting exactly one EXH
    in the sentence.
    
    Arguments
    ---------
    phonform: LOTlib3.FunctionNode.FunctionNode
        a phonological form 
        i.e., utterance without the silent morpheme EXH
    grammar: LOTlib3 grammar
        for global call, the grammar for the phonological form 
        (with or without EXH)
    Return
    ------
    list[LOTlib3.FunctionNode.FunctionNode]
        A list of possible utterances
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


def infer_meaning(phonform, qud, grammar, n_objects):

    # Generate the possible utterances by placing a single EXH
    # everywhere it can be placed.
    # Utteraces are instances of the Uttterance class
    possible_utterances_values = produce_possible_structures(
        phonform, 
        grammar
    )

    possible_utterances = [
        Utterance(qud, value=value, grammar=grammar, n_objects=n_objects)
        for value 
        in possible_utterances_values
    ]

    # loop through the possible literal utterances
    # (containing EXH)
    tn = []
    for utterance in possible_utterances:

        h0 = MeaningHypothesis(
            QUD=qud, 
            grammar=grammar, 
            n_objects=n_objects
        )
        # convert phonform to data
        data = [FunctionData(
            input=utterance, 
            output=None, 
            alpha=1
        )]
        # get an approximated posterior over interpretations given the utterance
        single_tn = TopN()
        # run inference
        for h in MetropolisHastingsSampler(h0, data, steps=10000):
            single_tn << h
        # store samples
        tn.append((utterance, single_tn))

    return tn
