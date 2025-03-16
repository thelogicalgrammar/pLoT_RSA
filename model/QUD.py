from math import factorial as fac

import z3

class QUD:
    
    def __init__(self, props, solver):
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
        self.solver = solver
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
                self.solver.push()
                self.solver.add(z3.And(self.props[i], self.props[j]))
                if self.solver.check() == z3.sat:
                    mutual_exclusivity = False
                    self.solver.pop()
                    break
                self.solver.pop()
            if not mutual_exclusivity:
                break

        # Check collective exhaustiveness
        self.solver.push()
        self.solver.add(z3.Not(z3.Or(self.props)))
        collective_exhaustiveness = (self.solver.check() == z3.unsat)
        self.solver.pop()

        # Output results
        if mutual_exclusivity and collective_exhaustiveness:
            return True
        else:
            if not mutual_exclusivity:
                print("The set of self.props is not mutually exclusive.")
            if not collective_exhaustiveness:
                print("The set of self.props is not collectively exhaustive.")

    def is_relevant(self, answer, solver):
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

    def QUD_prior(self, prop):
        """
        Implement this if part of the probability of a belief state
        depends on the QUD.
        """
        raise NotImplementedError

    def __iter__(self):
        return iter(self.props)

    def __len__(self):
        return len(self.props)


def binomial(x, y):
    try:
        return fac(x) // fac(y) // fac(x - y)
    except ValueError:
        return 0


def zero_truncated_binomial(n, p):
    """
    Generate a zero-truncated binomial distribution.
    When p=0, the P(1) = 1
    """

    probs = [
        binomial(n, k) * p**k * (1-p)**(n-k) 
        for k in range(1, n+1)
    ]
    # normalize
    probs = [
        x/sum(probs) 
        for x in probs
    ]
    return probs


class ProductQUD(QUD):

    def __init__(self, *atoms, solver, prob_knows=None):
        """
        Create a QUD from a product of atomic propositions.
        Each element is a proposition. This function builds
        a partition from the product of the propositions.
        Any two elements in the product are mutually exclusive.
        and collectively exhaustive.
        NOTE: Their negations also need to be explicitly included.
        NOTE: This does not scale well for large numbers of atoms.

        Create a partition from the product of the propositions
        e.g. for [p1,p2] -> [
            z3.And(p1, p2), 
            z3.And(z3.Not(p1), p2),
            z3.And(p1, z3.Not(p2)),
            z3.And(z3.Not(p1), z3.Not(p2))
        ]

        Arguments
        ---------
        prob_knows: float
            A probability, gives the probability of the speaker knowing
            whether each proposition is true (independently of the others)
        """

        # Create all possibilities
        n = len(atoms)
        props = []
        for i in range(2 ** n):
            combination = []
            for j in range(n):
                if (i >> j) & 1:
                    combination.append(atoms[j])
                else:
                    combination.append(z3.Not(atoms[j]))
            props.append(z3.And(combination))

        self.atoms = atoms
        self.prob_knows = prob_knows
        QUD.__init__(self, props, solver)

    def QUD_prior(self, prop, strategy='binomial'):
        """
        Calculate the probability of a belief state based on 
        one of the strategies.
        """

        if strategy == 'factorized':
            # the probability of knowing whether each proposition 
            # in the QUD is T/F. 
            # For each atom, check whether the proposition entails it
            # or its negation and if so, multiply the probability 
            # of knowing whether the atom is true, 
            # else by the probability of not knowing it
            prob = 1
            for i, atom in enumerate(self.atoms):
                if self.knows_whether(prop, atom):
                    prob *= self.prob_knows
                else:
                    prob *= (1 - self.prob_knows)

        elif strategy == 'binomial':
            # find the number of atoms that are entailed by the proposition
            n_true = 0
            for i, p in enumerate(self.props):
                if self.knows_whether(prop, p):
                    n_true += 1
                else:
                    pass
            prob = zero_truncated_binomial(
                len(self.props),
                self.prob_knows
            )[n_true-1]

        else:
            raise ValueError("Invalid kind of prior!")

        return prob

    def knows_whether(self, prop, atom):
        """
        Check if the prop (representing a belief state here)
        determines whether the atom is true or false.
        """

        self.solver.push()
        self.solver.add(z3.Not(z3.Implies(prop, atom)))
        ent = self.solver.check() == z3.unsat
        self.solver.pop()

        self.solver.push()
        self.solver.add(z3.Not(z3.Implies(prop, z3.Not(atom))))
        ent_not = self.solver.check() == z3.unsat
        self.solver.pop()

        return ent or ent_not
