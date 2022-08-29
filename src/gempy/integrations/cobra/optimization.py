import numpy as np

from pymoo.core.problem import Problem as PyMOOProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from bpyutils._compat import iteritems

ALGORITHMS = {
    "nsga2": {
        "class": NSGA2
    }
}

def _get_bounds(model, type_):
    attr    = "%s_bound" % type_
    bounds  = []

    for reaction in model.reactions:
        bound   = getattr(reaction, attr)
        bounds += [bound, bound]

    return bounds

def _get_obj_coeff_arr(model):
    reactions    = model.reactions
    n_reactions  = len(reactions)

    n_objectives = len(model.objectives)

    matrix       = np.zeros((n_reactions * 2, n_objectives))

    for i, (reaction_id, reaction) in enumerate(iteritems(model.objectives)):
        reaction_index = reactions.index(reaction_id) * 2
        matrix[reaction_index,     i] = -1
        matrix[reaction_index + 1, i] = 1 if reaction.reversibility else 0

    return matrix

class OptimizationProblem(PyMOOProblem):
    def __init__(self, *args, **kwargs):
        model = kwargs["model"]

        self._super = super(OptimizationProblem, self)
        self._super.__init__(
            n_var       = len(model.variables),
            n_constr    = len(model.constraints),
            n_obj       = len(model.objectives),
            xl          = _get_bounds(model, "lower"),
            xu          = _get_bounds(model, "upper")
        , *args, **kwargs)

        self._model = model

    @property
    def model(self):
        return self._model

    def _evaluate(self, x, out, *args, **kwargs):
        model    = self.model
        
        S        = model.sparse_stoichiometric_matrix

        X        = np.transpose(x)

        c        = _get_obj_coeff_arr(model)

        out["F"] = np.dot(np.transpose(c), X)
        out["G"] = np.dot(S, X)

class Problem:
    def __init__(self, model):
        self._model = model

    @property
    def model(self):
        return self._model

    def solve(self, *args, **kwargs):
        model     = self.model

        algorithm = kwargs.pop("algorithm", "nsga2")

        if algorithm not in ALGORITHMS:
            raise ValueError("Algorithm %s not found." % algorithm)

        algorithm_class     = ALGORITHMS[algorithm]["class"]
        algorithm_instance  = algorithm_class()

        problem = OptimizationProblem(model = model)

        result  = minimize(problem, algorithm_instance, *args, **kwargs)

        return result