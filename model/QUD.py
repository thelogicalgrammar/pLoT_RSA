from math import factorial as fac
import z3

class QUD:
    
    def __init__(self, partition_f, solver, minimum=None, maximum=None):
        """
        This class encodes a Question Under Discussion (QUD).
        The partition_f function is a z3 function
        that maps things in a Model into a partition cell
        represented as an integer.
        This gives us a general way of encoding partitions.
        See: https://centaur.reading.ac.uk/80434/2/Questions%20under%20discussion_final.pdf

        Arguments
        ---------
        partition: z3 formula
            Encodes a proposition that ought to be settled for the QUD;
            It partitions the possible worlds.
        """
        self.partition_f = partition_f
        self.solver = solver
        
        if minimum is None:
            self.min = self.find_min()
        else:
            self.min = minimum
        if maximum is None:
            self.max = self.find_max()
        else:
            self.max = maximum

        self.alternatives = self.find_alternatives()

    def find_min(self):
        """
        Uses the Optimize module to find the minimum value of partition_f
        under the constraints in the solver.
        """
        opt = z3.Optimize()
        # Add all constraints from the provided solver.
        for a in self.solver.assertions():
            opt.add(a)
        h_min = opt.minimize(self.partition_f)
        if opt.check() == z3.sat:
            lb = h_min.value().as_long()
            return lb
        else:
            raise Exception("Constraints are unsatisfiable when minimizing partition_f.")

    def find_max(self):
        """
        Uses the Optimize module to find the maximum value of partition_f
        under the constraints in the solver.
        """
        opt = z3.Optimize()
        for a in self.solver.assertions():
            opt.add(a)
        h_max = opt.maximize(self.partition_f)
        if opt.check() == z3.sat:
            ub = h_max.value().as_long()
            return ub
        else:
            raise Exception("Constraints are unsatisfiable when maximizing partition_f.")

    def find_alternatives(self):
        """
        Loops over the range [min, max] and collects those values for which
        there is a model. Each alternative is represented as the formula
        (partition_f == n).
        """
        alternatives = []
        # Iterate over the candidate partition cell values.
        for n in range(self.min, self.max + 1):
            s = z3.Solver()
            # Add all constraints from the original solver.
            for a in self.solver.assertions():
                s.add(a)
            s.add(self.partition_f == n)
            if s.check() == z3.sat:
                # Record the formula corresponding to this cell.
                alternatives.append(self.partition_f == n)
        return alternatives

    def is_relevant(self, answer, solver):
        """
        Evaluates if the answer is relevant to the QUD. 
        An answer is relevant if it or its negation 
        entails the negation of at least one alternative (partial answer).
        Note that "resolving answers" are also partial answers.

        Arguments
        ---------
        answer: z3 formula 
            The answer to evaluate the QUD on
        """
        
        # loop over the cells of the partition
        for prop in self.alternatives:
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
        return iter(self.alternatives)

    def __len__(self):
        return len(self.alternatives)


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
    
    def __init__(self, *atoms, solver, prob_knows=0.5, p_truth=0.5):
        """
        Create a QUD from a product of atomic propositions.
        Each element is an atomic proposition. The partition is built
        by encoding the truth values of the atoms as bits in an integer.
        For example, given [p1, p2] the partition function is:
        
            partition_f = If(p1, 1, 0) + If(p2, 2, 0)
        
        which yields values 0, 1, 2, or 3 corresponding to:
        
            [Not(p1) & Not(p2), p1 & Not(p2), Not(p1) & p2, p1 & p2]
        
        Arguments
        ---------
        atoms: Z3 BoolRef
            A sequence of atomic propositions.
        solver: z3.Solver
            A solver instance that already contains any additional constraints.
        prob_knows: float, optional
            The probability that the speaker knows whether each proposition is true.
        """
        self.atoms = atoms
        self.p_truth = p_truth
        self.prob_knows = prob_knows
        # Create the partition function by treating each atom as a bit.
        partition_expr = z3.Sum([z3.If(atom, 2**j, 0) for j, atom in enumerate(atoms)])
        # Call the parent constructor with the partition expression.
        super().__init__(partition_expr, solver)

    def QUD_prior(self, prop, strategy='partitions'):
        """
        Strategy refers to whether the prob_knows is one of knowing 
        whether atoms are true or false ('atoms')
        or whether partition alternatives are true or false ('partitions').

        The intuitive difference is that 'atoms' is less fine-grained,
        because it only captures whether the speaker's belief
        determines whole atomic propositions, rather than 
        combinations of their truth values.

        E.g., (1) 'p and not q' and (2) '(p and not q) and not(p and q)
        'atoms' find them equivalent, but for 'partitions' 
        the speaker knows more in (2).
        
        Parameters
        ----------
        prop : Z3 formula
            The belief state (a proposition) to evaluate.
        
        Returns
        -------
        float
            The computed probability of the belief state.
        """
        
        if strategy=='atoms':
            # Calculate the total probability of a belief state (prop) according to
            # the following factors:
            #   - For each atomic proposition that the belief state "knows" (i.e., entails
            #     either the atom or its negation), multiply by p_knowing * p_truth.
            #   - For each atom that the belief state does not resolve, multiply by (1 - p_knowing).
            # This product represents:
            # P(knowing the atoms) * P(known atoms having the known truth values)
            prob = 1.0
            for atom in self.atoms:
                # Check whether the belief state determines the truth of the atom.
                if self.knows_whether(prop, atom):
                    prob *= (self.prob_knows * self.p_truth)
                else:
                    prob *= (1 - self.prob_knows)
        
        elif strategy=='partitions':
            # Compute the total probability of a belief state (prop) according to:
            # P(belief) = (p_knowing * p_truth)^(# resolved alternatives)
            #             * (1 - p_knowing)^(# unresolved alternatives)
            # Here, an alternative (a partition cell in self.props) is "resolved" by prop if
            # either prop entails that cell or prop entails its negation.
            # This prior therefore distinguishes between belief states that rule out more of the partition.
            n = len(self.alternatives)
            resolved = 0
            for alt in self.alternatives:
                if self.knows_whether(prop, alt):
                    resolved += 1
            unresolved = n - resolved
            prob = (self.prob_knows * self.p_truth) ** resolved * (1 - self.prob_knows) ** unresolved
        
        return prob

    def knows_whether(self, prop, atom):
        """
        Check if the belief state (prop) determines whether the atom is true or false.
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
