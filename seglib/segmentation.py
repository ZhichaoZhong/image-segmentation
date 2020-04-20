import numpy as np
import logging
import mrcnn.model as modellib
from mrcnn.config import Config

from skimage.filters import sobel, gaussian
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.color import rgb2gray

logger = logging.getLogger('image-seg')


def apply_mask(image, mask, reverse=False):
    """
    Apply a mask on an image:
    :param image: numpy array. dim should be 2 or 3.
    :param mask: 2d numpy array. Data range should be [0, 1].
    :param reverse: bool. To reverse the mask or not.
    :return:
    """
    try:
        assert mask.max() <= 1 and mask.min() >= 0
    except AssertionError:
        logger.exception('Scale of mask must be between 0 and 1.')
    try:
        assert mask.shape == image.shape[:2]
    except AssertionError:
        logger.exception('Shape of mask and image must be the same.')

    if reverse:
        mask = mask.max() - mask

    image = image.copy()
    if len(image.shape) == 2:
        return image * mask
    for c in range(image.shape[2]):
        image[:, :, c] = image[:, :, c] * mask
    return image


def load_maskrcnn_model(weight_path, model_dir='./models', maskrcnn_config=None):
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
            IMAGE_MAX_DIM = 1024

        inference_config = InferenceConfig()
    else:
        inference_config = maskrcnn_config

    inference_model = modellib.MaskRCNN(mode="inference",
                                        config=inference_config,
                                        model_dir=model_dir)
    print("Loading weights from ", weight_path)
    inference_model.load_weights(weight_path, by_name=True)
    return inference_model


def predict_mask(model, img):
    """
    Predict mask for an image
    :param model: mask-rcnn mask instance.
    :param img: 3D numpy array.
    :return: 2D binary array.
    """
    assert len(img.shape) == 3, 'Input image must have channel dimension.'
    pred_result = model.detect([img], verbose=1)
    pred_mask = pred_result[0]['masks'].astype(np.int).squeeze()
    return pred_mask


def sobel_watershed(img, bg_threshold, fg_threshold):
    """
    Segment an image using a sobel+watershed approach.
    :param img: gray image. 2D numpy array of scale range [0, 1].
    :param bg_threshold:
    :param fg_threshold:
    :return:
    Reference:
    """
    data = img.copy()
    edges = sobel(data)

    markers = np.zeros_like(data)
    foreground, background = 1, 2
    markers[data > bg_threshold] = background
    markers[data < fg_threshold] = foreground

    ws = watershed(edges, markers)
    mask = label(ws == foreground)
    return mask


# ------------------
# Segmenter Class
# ------------------

class BasicSegmenter(object):
    def __init__(self):
        # TODO: logger?
        pass

    def segment(self, image):
        pass


class BoostedSegmenter(BasicSegmenter):
    """
    Class to boost the coarse mask produced by mask-rcnn model
    using conventional segmentaion algorithms.
    """

    def __init__(self,
                 weight_path,
                 gaussian_sigma=30,
                 bg_threshold=0.05,
                 fg_threshold=0.80,
                 model_dir='./models',
                 maskrcnn_config=None):
        super().__init__()
        self.bg_threshold = bg_threshold
        self.fg_threshold = fg_threshold
        self.model = load_maskrcnn_model(weight_path, model_dir, maskrcnn_config)
        self.sigma = gaussian_sigma

    def segment(self, img):
        processed_image = self.apply_soft_mask(img)
        seg = sobel_watershed(processed_image, self.bg_threshold, self.fg_threshold)
        return seg

    @staticmethod
    def apply_soft_mask(self, img):
        # Get a coarse mask using the mask-rcnn model
        pred_mask = predict_mask(self.model, img)

        # Create a soft mask using the softmask
        soft_mask = gaussian(pred_mask, self.sigma)
        soft_mask -= soft_mask.min()
        soft_mask /= soft_mask.max()

        # Apply the soft mask on the image
        processed_image = apply_mask(img, soft_mask, True)
        return processed_image


class MaskRcnnSegmenter(BasicSegmenter):
    def __init__(self,
                 weight_path,
                 model_dir='./models',
                 maskrcnn_config=None):
        super().__init__()
        self.model = load_maskrcnn_model(weight_path, model_dir, maskrcnn_config)

    def segment(self, img):
        seg = predict_mask(self.model, img)
        return seg


class WatershedSegmenter(BasicSegmenter):
    def __init__(self,
                 bg_threshold=0.05,
                 fg_threshold=0.80):
        super().__init__()
        self.bg_threshold = bg_threshold
        self.fg_threshold = fg_threshold

    def segment(self, img):
        img = rgb2gray(img)
        seg = sobel_watershed(img, self.bg_threshold, self.fg_threshold)
        return seg
