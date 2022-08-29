from bpyutils.util.array import sequencify

from cobra.core.model import Model as COBRAPyModel

from gempy.integrations.cobra.optimization import Problem
from gempy.integrations.cobra.util import create_sparse_stoichiometric_matrix

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
        if len(self.objectives) == 1:
            solution = self._super.optimize(*args, **kwargs)
        else:
            algorithm = kwargs.pop("algorithm", "nsga2")

            problem   = Problem(self)

            solution  = problem.solve(algorithm = algorithm, *args, **kwargs)

        return solution