from model.utilities import (
    find_phonform_possible_structures,
    print_possible_parses
)

from model.grammar import MeaningHypothesis

from tqdm import tqdm
from pprint import pprint
import html
from copy import deepcopy
from LOTlib3.Grammar import pack_string
from LOTlib3.DataAndObjects import FunctionData
import z3
import re
import matplotlib.pyplot as plt
import numpy as np


def already_defined(dictionary, fullmeaning, posterior_score, solver):
    """
    Find marginal distribution of meanings in the posterior
    """
    assert posterior_score >= 0, 'Posterior score in prob space, not logprob!'
    dictionary = deepcopy(dictionary)
    new = True
    parse_index, meaning = fullmeaning
    for (i, m), p in dictionary.items():
        solver.push()
        solver.add(z3.Not(meaning == m))
        if (solver.check() == z3.unsat) and (i == parse_index):
            # equivalent
            dictionary[(parse_index, m)] = p + posterior_score
            # it's not going to be synonymous
            # with anything else, given how
            # dictionary is constructed
            new = False
            solver.pop()
            break
        solver.pop()
    if new:
        dictionary[fullmeaning] = posterior_score

    return dictionary


def compress_meaning_dict(dictionary):
    """
    Take the output of already_defined and compress it by ignoring the 
    parse index, so that we get the marginal probability of meanings.
    """
    returnvalue = dict()
    # get unique meanings
    set_of_meanings = set(m for (i, m), p in dictionary.items())
    for m in set_of_meanings:
        # get the sum of all probabilities of m
        sum_ps = 0
        for (i, m2), p in dictionary.items():
            if m2 == m:
                sum_ps += p
        returnvalue[m] = sum_ps
    return returnvalue


def parse(s, grammar):
    """
    Parse a string into a FunctionNode.

    NOTE: I did not test this with lambda stuff
    NOTE: This assumes that at most the name of the start symbol 
    is an empty string
    """

    topempty = False
    inverse_dict = dict()
    for k, v in grammar.idx2rule().items():
        inverse_dict[v.name] = pack_string[k]
        if k == 0 and v.name == '':
            topempty = True
        if k != 0:
            assert v.name != '', 'Name of non-top is not empty string'

    split = [
        x
        for x in re.split(r'\(|\)|,', s.replace(' ', ''))
        if x != ''
    ]

    toascii = ''.join([
        str(inverse_dict[x])
        for x in split
    ])

    # NOTE: Adds top node manually
    if topempty:
        toascii = '0' + toascii

    return grammar.unpack_ascii(toascii)


def analyse_specific_hyps(s, hypotheses, n_props, grammar_phon, grammar_belief,
                          qud, temp, model, solver, print_log=True, **kwargs):
    """
    Prints information about the likelihood of specific hypotheses
    i.e., (parse indices, belief states)
    given a specific sentence s

    Example of grammar definition: 
        grammar_belief = define_grammar(
            n_props=n_props,
            EXH=False,
            index=len(possible_meanings)
        )
    """

    if isinstance(s, str):
        s = parse(s, grammar_phon)
    
    possible_meanings, parses_values = find_phonform_possible_structures(
        phonform=s,
        grammar=grammar_phon,
        qud=qud,
        model=model,
        solver=solver
    )
    
    print_possible_parses(
        parses_values,
        possible_meanings,
        n_props,
        solver
    )

    # convert phonform to data
    datum = FunctionData(
        input=[],
        output=None,
        alpha=1-1e-4,
    )
    
    output = dict()
    for h in tqdm(hypotheses):
        
        if isinstance(h, str):
            h = parse(h, grammar_belief)

        meaninghyp = MeaningHypothesis(
            parses=possible_meanings,
            solver=solver,
            grammar=grammar_belief,
            value=h,
            qud=qud,
            temp=temp,
            model=model,
            print_log=print_log,
            **kwargs
        )

        posterior_score = meaninghyp.compute_posterior([datum])
        output[meaninghyp] = posterior_score

    return output


def plot_results(dictionary, solver, dict_type='tn', first_n=None, ignore_parse=False):
    """
    Plot the output of analyse_specific_hyps
    """

    if dict_type == 'tn':

        meanings = dict()
        for h in tqdm(list(dictionary.get_all(sorted=True))[::-1]):
            m = h.model
            meanings = already_defined(
                meanings,
                h(m),
                np.exp(h.posterior_score),
                solver
            )

    elif dict_type == 'scores':

        sorted_posterior_scores = sorted(
            dictionary.items(),
            key=lambda item:item[1]
        )[::-1]

        meanings = dict()
        for i, (h, score) in enumerate(tqdm(sorted_posterior_scores)):
            m = h.model
            u_i, meaning = h(m)
            meanings = already_defined(
                meanings,
                (u_i, z3.simplify(meaning)),
                np.exp(score),
                solver
            )
            if first_n is not None and i == first_n:
                break

    pprint(meanings)

    if ignore_parse:
        meanings = compress_meaning_dict(meanings)

    keys, unnorm_ps = zip(*list(meanings.items()))
    if not ignore_parse:
        parses, expressions = zip(*keys)
    expressions = [z3.simplify(x) for x in expressions]
    unnorm_ps = np.array(unnorm_ps)
    unnorm_ps = unnorm_ps / unnorm_ps.sum()
    order = np.argsort(unnorm_ps)

    fig, ax = plt.subplots()
    ax.bar(
        np.arange(len(unnorm_ps)),
        unnorm_ps[order]
    )
    ax.set_xticks(
        np.arange(len(unnorm_ps))
    )

    xlabels = [
        html.unescape(x._repr_html_())
        for x in np.array(expressions)[order]
    ]

    if not ignore_parse:
        xlabels = [
            f"{parse}, {expr}"
            for parse,expr
            in zip(np.array(parses)[order],xlabels)
        ]
    
    ax.set_xticklabels(
        xlabels,
        rotation='vertical'
    )

    return fig, ax
