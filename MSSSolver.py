from z3 import *

# Generating maximal consistent subsets with z3
# From: https://github.com/Z3Prover/blob/master/examples/python/mus/mss.py (but slightly modified to reset the solver state on each run of the function)

def enumerate_sets(solver):
    while True:
        if sat == solver.s.check():
            MSS = solver.grow()
            yield MSS
        else:
            break


def all_smt(s, initial_terms):
    """
    s: a solver (with maybe some constraints
    t: a list of terms
    
    From: https://stackoverflow.com/questions/11867611/y-checking-all-solutions-for-equation/70656700#70656700
    """
    def block_term(s, m, t):
        s.add(t != m.eval(t, model_completion=True))
    def fix_term(s, m, t):
        s.add(t == m.eval(t, model_completion=True))
    def all_smt_rec(terms):
        if sat == s.check():
           m = s.model()
           yield m
           for i in range(len(terms)):
               s.push()
               block_term(s, m, terms[i])
               for j in range(i):
                   fix_term(s, m, terms[j])
               yield from all_smt_rec(terms[i:])
               s.pop()   
    yield from all_smt_rec(list(initial_terms))


class MSSSolver:
    def __init__(self, hard, soft):
        self.s = Solver()
        self.varcache = {}
        self.idcache = {}
        self.n = len(soft)
        self.soft = soft
        self.hard = hard
        self.s.add(hard)
        self.vars_dict = dict()
        self.soft_vars = set([
            self.c_var(i) 
            for i in range(self.n)
        ])
        self.orig_soft_vars = set([
            self.c_var(i) 
            for i in range(self.n)
        ])
        self.s.add([
            (self.c_var(i) == soft[i]) 
            for i in range(self.n)
        ])

    def c_var(self, i):
        if i not in self.varcache:
            v = Bool(str(self.soft[abs(i)]))
            self.vars_dict[v] = self.soft[abs(i)]
            self.idcache[v] = abs(i)
            if i >= 0:
                self.varcache[i] = v
            else:
                self.varcache[i] = Not(v)
        return self.varcache[i]

    def update_unknown(self):
        self.model = self.s.model()
        new_unknown = set([])
        for x in self.unknown:
            if is_true(self.model[x]):
                self.mss.append(x)
            else:
                new_unknown.add(x)
        self.unknown = new_unknown

    def add_def(self, fml):
        name = Bool("%s" % fml)
        self.s.add(name == fml)
        return name

    def relax_core(self, Fs):
        assert(Fs <= self.soft_vars)
        prefix = BoolVal(True)
        self.soft_vars -= Fs
        Fs = [f for f in Fs]
        for i in range(len(Fs) - 1):
            prefix = self.add_def(And(Fs[i], prefix))
            self.soft_vars.add(self.add_def(Or(prefix, Fs[i + 1])))

    def resolve_core(self, core):
        new_core = set([])
        for x in core:
            if x in self.mcs_explain:
                new_core |= self.mcs_explain[x]
            else:
                new_core.add(x)
        return new_core

    def grow(self):
        self.mss = []
        self.mcs = []
        self.nmcs = []
        self.mcs_explain = {}
        self.unknown = self.soft_vars
        self.update_unknown()
        cores = []
        while len(self.unknown) > 0:
            x = self.unknown.pop()
            is_sat = self.s.check([self.hard] + self.mss + [x] + self.nmcs)
            if is_sat == sat:
                self.mss.append(x)
                self.update_unknown()
            elif is_sat == unsat:
                core = self.s.unsat_core()
                core = self.resolve_core(core)
                self.mcs_explain[Not(x)] = {
                    y 
                    for y in core 
                    if not eq(x, y)
                }
                self.mcs.append(x)
                self.nmcs.append(Not(x))
                cores += [core]
            else:
                print("solver returned %s" % is_sat)
                exit()
        mss, mcs = [], []
        for x in self.orig_soft_vars:
            if is_true(self.model[x]):
                mss.append(x)
            else:
                mcs.append(x)
        self.s.add(Or(mcs))
        core_literals = set([])
        cores.sort(key=lambda element: len(element))
        for core in cores:
            if len(core & core_literals) == 0:
                self.relax_core(core)
                core_literals |= core
        return mss

