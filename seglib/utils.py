import numpy as np
import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn.utils import  compute_overlaps_masks
import logging
from matplotlib import pyplot as plt

logger = logging.getLogger('image-seg')
#-------------------
# for mask-rcnn
#-------------------
def load_mrcnn_model(weight_path, model_dir='./models', maskrcnn_config=None):
    """Load mask-rcnn models from a h5 file.
    :params
        weight_path:
    """
    # Define the config is not defined
    if not maskrcnn_config:
        class InferenceConfig(Config):
            NAME = 'segmentation'
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1 # BATCH_SIZE = IMAGES_PER_GPU * GPU_COUNT
            NUM_CLASSES = 1 + 1
            IMAGE_MIN_DIM = 128
            IMAGE_MAX_DIM = 512
            BATCH_SIZE = 1
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
    This is an alias call to mrcnn.utils.compute_overlaps_masks
    :param mask_pred: binary numpy array or arrays. Shape could be [M, N] or [M, N, n_instances]
    :param mask_gt: binary numpy array or arrays. Shape could be [M, N] or [M, N, n_instances]
    :return: score: float
    """

    # assert list(np.unique(mask_pred)) == [0, 1]
    assert mask_pred.shape==mask_true.shape
    if len(mask_true.shape)==2:
        mask_true = np.expand_dims(mask_true, -1)
        mask_pred = np.expand_dims(mask_pred, -1)
        iou = compute_overlaps_masks(mask_true, mask_pred).squeeze()
        return float(iou)
    else:
        return compute_overlaps_masks(mask_true, mask_pred).squeeze()


def test_get_iou():
    mask_pred = np.eye(3)
    mask_true = np.eye(3)
    assert  1 == get_iou(mask_pred, mask_true)

# -------------------------
# Visualizaiton
# -------------------------

def display_one_image(img, title):
    fig, ax = plt.subplots(1, 1)
    ax.imshow(img, cmap='gray')
    ax.set_axis_off()
    ax.set_title(title, fontsize=12)
    fig.show()

