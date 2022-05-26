from gempy.__attr__ import __name__ as NAME
from gempy.data import get_data_dir

def build_model():
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    from deeply.model.gan import GAN

    model = GAN() 
=======
    pass
>>>>>>> template/master
=======
    pass
>>>>>>> template/master
=======
    pass
>>>>>>> template/master
    # do something ...

def train(data_dir = None, artifacts_dir = None, *args, **kwargs):
    data_dir = get_data_dir(NAME, data_dir)
    model    = build_model()
    # do something ...