import random

from bpyutils.util.types import lfilter

MAXIMUM_KNOCKOUTS = 3

def get_random_model_object_sample(model, type_, n = MAXIMUM_KNOCKOUTS, exclude = None):
    exclude = exclude or []

    objekt = getattr(model, type_)
    objekt = lfilter(lambda x: x.id not in exclude, objekt)
    
    n      = max(1, min(n, len(objekt)))

    rand_n = random.randint(1, n)
    sample = random.sample(objekt, rand_n)

    return sample