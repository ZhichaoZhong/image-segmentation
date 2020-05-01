from sklearn.metrics import jaccard_score
import mrcnn.model as modellib
from mrcnn.config import Config
import logging

logger = logging.getLogger('image-seg')
#-------------------
# for mask-rcnn
#-------------------
def load_model(weight_path, maskrcnn_config=None, model_dir='./models'):
    """Load mask-rcnn models from a h5 file.
    :params
        weight_path:
    """
    # Define the config is not defined
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
    logger.info(f"Loading weights from {weight_path}")
    inference_model.load_weights(weight_path, by_name=True)
    return inference_model


# ------------------------
# Evaluation Metrics
# ------------------------
def get_iou(mask_pred, mask_true):
    """
    Return intersection of union of two binary images.
    This is an alias call to sklearn.metrics.jaccard_score.
    :param mask_pred: binary numpy array.
    :param mask_gt: binary numpy array.
    :return: score: float
    """

    # assert list(np.unique(mask_pred)) == [0, 1]
    iou = jaccard_score(mask_true, mask_pred)
    return iou

