from gempy.__attr__ import __name__ as NAME
from gempy.data import get_data_dir
from gempy import settings, dops

def build_model(artifacts_path = None):
    dropout_rate  = settings.get("dropout_rate")
    batch_norm    = settings.get("batch_norm")

    from deeply.model.gan import GAN

    model = GAN()
    # do something ...
    pass

def train(data_dir = None, artifacts_dir = None, check = False, *args, **kwargs):
    batch_size    = 1 if check else settings.get("batch_size")
    learning_rate = settings.get("learning_rate")
    epochs        = 1 if check else settings.get("epochs")

    data_dir = get_data_dir(NAME, data_dir)
    model    = build_model()
    
    dops.watch(model)

    # do something ...