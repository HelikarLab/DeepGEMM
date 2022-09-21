# from gempy.__attr__ import __name__ as NAME
# from gempy.data import get_data_dir
# from gempy import settings, dops

# def build_model(artifacts_path = None):
#     dropout_rate  = settings.get("dropout_rate")
#     batch_norm    = settings.get("batch_norm")

#     from deeply.model.gan import GAN

#     model = GAN()
#     # do something ...
#     pass

# def train(data_dir = None, artifacts_dir = None, check = False, *args, **kwargs):
#     batch_size    = 1 if check else settings.get("batch_size")
#     learning_rate = settings.get("learning_rate")
#     epochs        = 1 if check else settings.get("epochs")

#     data_dir = get_data_dir(NAME, data_dir)
#     model    = build_model()
    
#     dops.watch(model)

#     # do something ...

import os.path as osp

# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.metrics    import binary_accuracy

# from deeply.model.gan     import GAN

# from deeply.datasets.util import SPLIT_TYPES
# from deeply.generators    import ImageMaskGenerator
# from deeply.losses        import dice_loss

from bpyutils.util.ml     import get_data_dir

from gempy import settings, __name__ as NAME

def build_model(artifacts_path = None):
    encoder_dropout_rate = settings.get("encoder_dropout_rate")
    encoder_batch_norm   = settings.get("encoder_batch_norm")
    decoder_dropout_rate = settings.get("decoder_dropout_rate")
    decoder_batch_norm   = settings.get("decoder_batch_norm")

    # width, height = settings.get("image_width"), \
    #     settings.get("image_height")

    # unet = UNet(x = width, y = height, channels = 3, n_classes = 1,
    #     final_activation = "sigmoid", batch_norm = batch_norm, 
    #     dropout_rate = dropout_rate, padding = "same",
    #     backbone = "efficient-net-b7", backbone_weights = "imagenet")

    # # unet.
    
    # if artifacts_path:
    #     path_plot = osp.join(artifacts_path, "model.png")
    #     unet.plot(to_file = path_plot)

    # return unet

# STRATEGIES = [
#     { ""}
# ]

def train(check = False, data_dir = None, artifacts_path = None, *args, **kwargs):
    batch_size    = 1 if check else settings.get("batch_size")
    learning_rate = settings.get("learning_rate")
    epochs        = 1 if check else settings.get("epochs")
    
    model = build_model()
    # model.compile(optimizer = Adam(learning_rate = learning_rate),
    #     loss = dice_loss, metrics = [binary_accuracy])

    # dops.watch(model)

    # output_shape = model.output_shape[1:-1]

    # width, height = settings.get("image_width"), \
    #     settings.get("image_height")

    # data_dir = get_data_dir(NAME, data_dir)

    # path_img = osp.join(data_dir, "%s", "images")
    # path_msk = osp.join(data_dir, "%s", "masks")

    # args = dict(
    #     batch_size = batch_size,
    #     # color_mode = "grayscale",
    #     image_size = (width, height),
    #     mask_size  = output_shape,
    #     shuffle    = True
    # )

    # train_, val, test = [
    #     ImageMaskGenerator(path_img % type_, path_msk % type_, **args)
    #         for type_ in SPLIT_TYPES
    # ]

    # trainer = Trainer(artifacts_path = artifacts_path)
    # history = trainer.fit(model, train_, val = val, epochs = epochs, batch_size = batch_size)