from bpyutils.util.array import sequencify

import pandas as pd

from cobra.core.model    import Model as COBRAPyModel
from cobra.core.solution import Solution as COBRAPySolution

from dgemm.integrations.cobra.optimization import Problem
from dgemm.integrations.cobra.util import create_sparse_stoichiometric_matrix

MULTI = "multi"

class Model(COBRAPyModel):
    def __init__(self, *args, **kwargs):
        self._super = super(Model, self)
        self._super.__init__(*args, **kwargs)

        self._objectives = {}

    @property
    def objectives(self):
        objectives = {}

        if not len(self._objectives) and self.objective:
            self._objectives[self.objective.name] = self.objective

        return self._objectives

    @property
    def n_flat_reactions(self):
        return sum([2 if r.reversibility else 1 for r in self.reactions])

    @objectives.setter
    def objectives(self, value):
        self._objectives = {}

        value = sequencify(value)

        for v in value:
            if isinstance(v, str):
                reaction = self.reactions.get_by_id(v)
                self._objectives[reaction.id] = reaction
            else:
                self._objectives[v.id] = value

    @property
    def sparse_stoichiometric_matrix(self):
        return create_sparse_stoichiometric_matrix(self)

    def optimize(self, *args, **kwargs):
        algorithm = kwargs.pop("algorithm", "nsga2")
        verbose   = kwargs.pop("verbose", True)

        if len(self.objectives) == 1 and algorithm == "single":
            solution = self._super.optimize(*args, **kwargs)
        else:
            problem   = Problem(self)

            solution  = problem.solve(algorithm = algorithm, 
                verbose = verbose, *args, **kwargs)

            cobra_solution = COBRAPySolution(
                objective_value = 0,
                status = MULTI,
                fluxes = pd.DataFrame(data = solution.X)
            )

            solution  = cobra_solution

        return solution