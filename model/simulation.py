import z3

from LOTlib3.DataAndObjects import FunctionData
from LOTlib3.TopN import TopN
from LOTlib3.Samplers.MetropolisHastings import MetropolisHastingsSampler

from tqdm import tqdm
from grammar import MeaningHypothesis


def infer_meaning(qud, model, solver, grammar_belief, grammar_EXH, utterance,
                  n_steps=10000, temp=5, prior_temperature=1, 
                  likelihood_temperature=1, alpha=1-1e-5, print_log=False):
    """
    example for grammar_belief:
        grammar_belief = define_grammar(
            n_props,
            False,
            len(parses_values)
        )


    Arguments
    ---------
    qud: QUD 
    model: Model object
    solver: z3 solver
    grammar_belief: LOTlib3 grammar
    n_steps: int
    temp: float>0
    alpha: float[0,1]
        Probability of a speaker getting the parse right
    """

    h0 = MeaningHypothesis(
        utterance=utterance,
        qud=qud, 
        solver=solver,
        temp=temp,
        model=model,
        grammar=grammar_belief, 
        grammar_EXH=grammar_EXH,
        prior_temperature=prior_temperature,
        likelihood_temperature=likelihood_temperature,
        print_log=print_log
    )

    # convert phonform to data
    data = [FunctionData(
        input=[], 
        output=None, 
        alpha=alpha,
    )]

    # get an approximated posterior over interpretations given the parse
    single_tn = TopN()

    with tqdm(total=n_steps) as pbar:
        # run inference
        for h in MetropolisHastingsSampler(h0, data, steps=n_steps):
            single_tn << h
            pbar.update()
        pbar.close()

    return single_tn
