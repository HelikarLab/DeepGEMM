def _optimize_taxicab(model, *args, **kwargs):
    with model as m:
        prev_obj  = model.objectivec
        direction = kwargs.pop("objective_sense", prev_obj.direction)

        objective = model.problem.Objective(
            abs(prev_obj.expression), direction = direction
        )

        m.objective = objective

        return m.optimize(*args, **kwargs)

def optimize(model, type_ = None, *args, **kwargs):
    if not type_:
        return model.optimize(*args, **kwargs)
    elif type_ == "taxicab":
        return _optimize_taxicab(model, *args, **kwargs)