import z3

from LOTlib3.Grammar import Grammar
from LOTlib3.Eval import primitive
from LOTlib3.Hypotheses.LOTHypothesis import LOTHypothesis
from LOTlib3.Miscellaneous import attrmem, Infinity

from math import log
import numpy as np
import builtins
from copy import copy, deepcopy
from itertools import product
from functools import reduce

from MSSSolver import MSSSolver, enumerate_sets, all_smt
from Models import BooleanModel, ObjectsModel, colors

symmetric_ops = [
    'land_', 'lor_'
]
    

def define_grammar(n_props, EXH=False, index=None):
    """
    This function can be used to define the grammar that defines
    the meanings and the grammar that defines the parses.
    
    __NOTE__: All function rules must have .name attribute 
    that can be evaled into the corresponding function/object!
    So we cannot define them simply as anonymous lambda functions.
    This is needed because when interpreting the function 
    I need to evaluate the argument by itself.

    Arguments
    ---------
    n_props: int
        Number of propositions to consider
    EXH: bool
        Whether to include EXH in the grammar
    index: None | int
        When this is an int:
            - The grammar defines the speaker's belief along with the index of the true parse.
            - Hypotheses take the form (int, bool). 
            - The value of the int argument encodes the number of possible parses
              (among the ones compatible with the phonological form).
              This depends on the phonological form!
            - The first element of the tuple encodes the index of the true parse.
            - The second element encodes the speaker's belief.
        When None:
            - Hypotheses take the form bool.
        
    """
    
    grammar = Grammar()

    # This effectively functions as a check for whether we are using the grammar 
    # to define beliefs or parses.
    if type(index) == int:
        # We are encoding beliefs!
        grammar.add_rule('START', 'tuple_', ['PARSEINDEX', 'BOOL'], 1.0)
        for i in range(index):
            grammar.add_rule('PARSEINDEX', f'u{i}', None, 1.0)
            builtins.__dict__[f'u{i}'] = i
        # We need notp because when encoding beliefs, we don't want
        # knowledge of false propositions to be less likely
        grammar.add_rule('BOOL', 'notp', ['INDEX', 'MODEL'], 5.0)
    else:
        # We are encoding parses!
        # NOTE: Keep this as empty string name, because that's how 
        # it's recognized in the _exhaustify function
        grammar.add_rule('START', '', ['BOOL'], 1.0)
    
    grammar.add_rule('BOOL', 'land', ['BOOL', 'BOOL'], 1.0)
    grammar.add_rule('BOOL', 'lor', ['BOOL', 'BOOL'], 1.0)
    # Can't add 'eitheror' because it competes directly with 'or'!
    # which means 'EXH(p or q)' == 'p or q'
    # (different from natural language!)
    # grammar.add_rule('BOOL', 'leitheror', ['BOOL', 'BOOL'], 1.0)
    # grammar.add_rule('BOOL', 'ifthen', ['BOOL', 'BOOL'], 1.0)
    grammar.add_rule('BOOL', 'lnot', ['BOOL'], 2.0)
    
    grammar.add_rule('BOOL', 'p', ['INDEX', 'MODEL'], 5.0)

    for i in range(n_props):
        grammar.add_rule('INDEX', f'i{i}', None, 1.0)
        builtins.__dict__[f'i{i}'] = i

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

    ##### DEFINE MEANINGS

    # make colors available to LOTlib3
    for c in colors:
        builtins.__dict__[c.sexpr()] = c

    @primitive
    def tuple_(index, model):
        return (index, model)

    @primitive
    def p(index, model):
        return model(index)

    @primitive
    def notp(index, model):
        return z3.Not(model(index))

    @primitive
    def land(p, q):
        return z3.And(p, q)

    @primitive
    def lor(p, q):
        return z3.Or(p, q)

    @primitive
    def leitheror(p, q):
        return z3.Or(z3.And(z3.Not(p), q), z3.And(p, z3.Not(q)))

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
        python's "eval" is not aware of the syntax
        and so cannot find structural alternatives.

        Therefore, EXH is handled below manually 
        in the Exhaustifier.

        h: z3 formula
        """
        return h

    return grammar


class Exhaustifier:

    def __init__(self, grammar, qud, solver, cache_fs=False):
        """
        Following Katzir, we define the structural alternatives of a sentence as the sentences that can be produced by 
            1. Deletion
            2. Contraction
            3. Replacement
        of its nodes with other nodes from the substitution source (which we assume here for simplicity is the lexicon).

        NOTE: This is a superclass of Parse, so has access to self.QUD
        """

        self.solver = solver
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

        solver = MSSSolver(
            x,
            negalts,
            # this is not very clean, but we don't want MSSSolver
            # to change the state of the solver
            deepcopy(self.solver)
        )
        mms = tuple(enumerate_sets(solver))
        intersection = reduce(lambda x, y: x & set(y), mms, set(mms[0]))
        # interpret as z3 expressions rather than boolean literals
        returnvalue = [
            solver.vars_dict[i] 
            for i in intersection
        ]
        return returnvalue

    def _exhaustify(self, x, M):
        """
        Computes the exhaustified addition to parse x recursively.
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
            The interpretation of the parse
        """
        
        if x.name == 'EXH':
            # EXH has a single argument
            arglist = list(x.argFunctionNodes())
            assert len(arglist) == 1, f"More than one arg to EXH: {arglist}"
            subnode = arglist[0]
            # get interpretation of the subnode
            # (that might itself contain exh operators)
            x_int = self._exhaustify(subnode, M)
            # find the recursively defined structural alternatives
            # and add the negation of each to conjs
            conjs = []
            for y in self.find_structural_alternatives(subnode):
                content = self._exhaustify(y,M)
                interpretation = z3.Not(content)
                self.solver.push()
                self.solver.add(interpretation)
                satisfiable = self.solver.check() == z3.sat
                self.solver.pop()
                # check if the alternative is relevant to the QUD
                # Only consider relevant alternatives
                relevant = self.qud.is_relevant(
                    interpretation,
                    self.solver
                )
                if relevant and satisfiable:
                    # NOTE: Do NOT simplify; the internal structure
                    # is important!
                    conjs.append(interpretation)
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
                # This is the model which is special because it is
                # a variable passed to the hypothesis 
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


class MeaningHypothesis(LOTHypothesis):

    def __init__(self, parses, qud, solver, temp, model,
                 grammar, print_log=False, **kwargs):
        """
        A MeaningHypothesis encodes underlying meanings as structured objects, a la pLoT.

        Parameters
        ----------
        parses:
            All the possible meanings of the phonological form.
            These are the "possible parses" in the RSA sense.
        qud: QUD
            The question under discussion
        solver: z3 solver
            A solver that may encode some known facts about the world
        temp: float
            Speaker's rationality parameter
        model:
            The model M's interpretation
        grammar: lotlib3 grammar
            Without EXH: this encodes a belief state (rather than a parse)!
            With index: this tracks the index of the parse 
                        among the ones compatible with the phonform!
        """

        LOTHypothesis.__init__(
            self,
            grammar=grammar,
            display='lambda M: %s',
            **kwargs
        )

        self.parses = parses
        self.print_log = print_log
        self.solver = solver
        self.qud = qud
        self.temp = temp
        self.model = model

        self.define_p_given_parse()

    def define_p_given_parse(self):
        """
        Find the distribution over the answers to the QUD 
        for an L0, given each possible parse.
        Shape: (parse, QUD answer)
        """
        p_given_parses = []
        n_compatible_parses = 0
        for u in self.parses:
            p_given_parse = []
            for prop in self.qud:
                self.solver.push()
                self.solver.add(z3.And(u, prop))
                compatible = self.solver.check() == z3.sat
                p_given_parse.append(compatible)
                n_compatible_parses += compatible
                self.solver.pop()

            p_given_parses.append([
                x/sum(p_given_parse) 
                for x in p_given_parse
            ])
        self.p_given_parses = p_given_parses
        self.n_compatible_parses = n_compatible_parses


    def compute_single_likelihood(self, datum):
        """
        This function computes the probability of a phonological form
        (which is the observation) given a speaker's belief state (which the hypothesis models).

        The belief state does not need to be identical to the interpreted parse,
        i.e., the parse might only express part of the belief state.
        Therefore, I need to keep the parse and the belief state distinct.
        
        Compute the likelihood P( phon form | speaker's belief state )
        by marginalizing across the possible meanings of the phon form.
        """

        # get listener's hypothesis about the speaker's state as a z3 formula,
        # which consists of the index of the true parse 
        # (according to the hypothesis)
        # and the belief state of the speaker.
        i, f = self(self.model)

        # Get just the parse indicated by the meaning
        # The parse is a Parse object
        u = self.parses[i]
        
        # we assume that the speaker does not say more than they believe to be true!
        # (e.g., if the speaker believes 'p', they might say 'p or q', which is also true in q,
        # but if the speaker believes 'p or q', they're not going to say 'p')
        # So we check that belief state is at least as strong as the parse.
        # Check that the belief state entails the parse.
        # (i.e., negation is unsatisfiable)
        self.solver.push()
        self.solver.add(z3.Not(z3.Implies(f, u)))
        implies = self.solver.check() == z3.unsat
        self.solver.pop()

        # We also assume the belief is consistent, so check *that*
        self.solver.push()
        self.solver.add(f)
        consistent = self.solver.check() == z3.sat
        self.solver.pop()

        if consistent and implies:

            # Belief is consistent and entails parse, so
            # it's a possible candidate!
            # Find probability of parse u given belief state f

            # p_answer_given_belief says for each possible answers to the QUD
            # whether the answer is compatible with the belief state
            p_answer_given_belief = []
            n_compatible_belief = 0
            for j, prop in enumerate(self.qud):
                if self.p_given_parses[i][j] == 1:
                    # since the belief implies the parse,
                    # if an answer is compatible with the belief,
                    # it will also be compatible with the parse
                    p_answer_given_belief.append(1)
                    
                else:
                    self.solver.push()
                    self.solver.add(z3.And(f, prop))
                    compatible = self.solver.check() == z3.sat
                    p_answer_given_belief.append(compatible)
                    self.solver.pop()
                    n_compatible_belief += compatible

            ##### calculation of the utility as KL(belief || parse)

            # Normalize to get the (approximate) probability of each answer
            # to the QUD given the belief state.
            # Since each element of p_answer_given_belief is a boolean, all probs
            # are going to be either 0 or 1/sum(p_answer_given_belief).
            p_answer_given_belief = [
                x/sum(p_answer_given_belief) 
                for x in p_answer_given_belief
            ]

            # compute the (negative) KL(belief || parse)
            # This is a measure of how much information the 
            # parse gives about the information in the speaker's belief 
            # that's relevant to answering the QUD
            util = -sum([
                x*(np.log(x) - np.log(y)) 
                if (x != 0 and y != 0) else 0
                for x,y in zip(p_answer_given_belief, self.p_given_parses[i])
            ])

        else:
            util = -Infinity

        # We don't need to consider the cost 
        # because we are only finding out which parse
        # the speaker intended, but they all have
        # the same utterance cost.
        prob = np.exp(np.array(util) * self.temp)

        if self.print_log:
            print()
            print('----------')
            print('hyp: ', self.__str__())
            print('qud: ', self.qud)
            print('util:', util)
            print('exp KL: ', prob)
        
        if np.isnan(prob) or prob==0:
            if self.print_log:
                print('inconsistent prob: ', prob)
                print()
            # if the speaker said something incoherent
            return log(1-datum.alpha)

        if self.print_log:
            print('prob belief: ', prob_belief)
            print('total lik: ', prob*prob_belief)

        prob_belief = self.qud.QUD_prior(f)
        return log(prob) + log(prob_belief)

    def __copy__(self, value=None):
        """
        Copied from LOTlib3 Hypothesis
        Returns a copy of the Hypothesis. Allows you to pass in value to set to that instead of a copy.
        """

        thecopy = type(self)(
            parses=self.parses,
            qud=self.qud,
            solver=self.solver,
            temp=self.temp,
            model=self.model,
            grammar=self.grammar,
        ) 

        # copy over all the relevant attributes and things.
        # Note objects like Grammar are not copied
        thecopy.__dict__.update(self.__dict__)

        # and then we need to explicitly *deepcopy* the value (in case its a dict or tree, or whatever)
        if value is None:
            value = deepcopy(self.value)

        thecopy.set_value(value)

        return thecopy


class Parse(LOTHypothesis, Exhaustifier):
    
    def __init__(self, qud, model, grammar, solver, value=None, **kwargs):
        """
        A parse is the underlying representation of a phonological form
        including the silent morpheme EXH (and potentially more!).
        This is what the semantics sees.

        Arguments
        ---------
        qud: QUD
            The qud is a way of restricting which alternatives are considered.
        grammar: LOTlib3 grammar
            Grammar with EXH (since this is a parse and not just
            a phonological form).
        value: LOTlib3.FunctionNode.FunctionNode
            The value of the parse as a FunctionNode generated by the grammar.
        """

        LOTHypothesis.__init__(
            self,
            grammar=grammar,
            display='lambda M: %s',
            value=value,
            **kwargs
        )

        self.model = model

        # an exhaustifier to use in the likelihood calculation
        Exhaustifier.__init__(
            self,
            grammar=grammar,
            qud=qud,
            solver=solver
        )

    def __call__(self):
        """
        overwrite the exhaustify method from Exhaustifier so that it uses
        the value of the parse as the input.
        """
        return self.exhaustify(self.value, self.model)

    def compute_single_likelihood(self, datum):
        """
        This doesn't need to be defined since we are not using these
        for inference.
        """
        pass

