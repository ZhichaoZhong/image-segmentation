
import mrcnn.model as modellib
from mrcnn.config import Config

def load_model(weight_path, maskrcnn_config=None, model_dir='./models'):
    """Load mask-rcnn models from a h5 file.
    :params
        weight_path:
    """
    if not maskrcnn_config:
        class InferenceConfig(Config):
            NAME = 'segmentation'
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NUM_CLASSES = 1 + 1
            IMAGE_MIN_DIM = 128
            IMAGE_MAX_DIM = 512

        inference_config = InferenceConfig()
    else:
        inference_config = maskrcnn_config

    inference_model = modellib.MaskRCNN(mode="inference",
                                        config=inference_config,
                                        model_dir=model_dir)
    print("Loading weights from ", weight_path)
    inference_model.load_weights(weight_path, by_name=True)
    return inference_model

