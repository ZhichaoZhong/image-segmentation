# scikit-image version 0.17
import numpy as np
import logging
import mrcnn.model as modellib
from mrcnn.config import Config

from skimage.filters import sobel, gaussian
from skimage.segmentation import watershed
from skimage.color import rgb2gray

logger = logging.getLogger('image-seg')

def normalize_image_scale(img):
    img -= img.min()
    img /= img.max()
    return img

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

    if len(image.shape) == 2:
        return image * mask

    image_masked = np.zeros_like(image)
    for c in range(image.shape[2]):
        image_masked[:, :, c] = image[:, :, c] * mask
    return image_masked


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
    assert img.max() <= 1.0 and img.min() >= 0.0
    assert len(img.shape) == 2

    edges = sobel(img)
    markers = np.zeros_like(img)
    foreground, background = 1, 2
    markers[img > bg_threshold] = background
    markers[img < fg_threshold] = foreground

    ws = watershed(edges, markers)
    mask = (ws == foreground).astype(np.int)
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
                 model,
                 sigma_inner=30,
                 sigma_outer=60,
                 bg_threshold=0.05,
                 fg_threshold=0.80):
        super().__init__()
        self.bg_threshold = bg_threshold
        self.fg_threshold = fg_threshold
        self.model = model
        self.sigma_inner = sigma_inner
        self.sigma_outer = sigma_outer

    def segment(self, img):
        processed_image = self._apply_soft_mask(img)
        seg = sobel_watershed(processed_image, self.bg_threshold, self.fg_threshold)
        return seg

    def _get_mrcnn_mask(self, img):
        return predict_mask(self.model, img)

    def _get_soft_mask(self, img):
        """
        Return a soft mask based on the mrcnn predicted mask
        :param img:
        :return:
        """
        # Get a coarse mask using the mask-rcnn model
        pred_mask = self._get_mrcnn_mask(img)
        # Apply a Gaussain filter to smooth the mask boundaries
        soft_mask_inner = gaussian(pred_mask, self.sigma_inner)
        soft_mask_inner = normalize_image_scale(soft_mask_inner)
        soft_mask_outer = gaussian(pred_mask, self.sigma_outer)
        soft_mask_outer = normalize_image_scale(soft_mask_outer)

        return soft_mask_inner, soft_mask_outer

    def _apply_soft_mask(self, img):
        """
        Apply a soft mask so as to darken or lighten the
         object to segment
        :param img: np.array(), 3D rgb image.
        :return: np.array(), 2D gray image.
        """
        soft_mask_inner, soft_mask_outer = self._get_soft_mask(img)
        # Transform the image to gray image
        gray_img = normalize_image_scale(rgb2gray(img))
        # Wipe out the background
        gray_img = np.where(soft_mask_outer>0.1, gray_img, 1.0)
        # Apply the soft mask on the image
        processed_image = apply_mask(gray_img, soft_mask_inner, True)
        processed_image /= processed_image.max()
        return processed_image


class MaskRcnnSegmenter(BasicSegmenter):
    def __init__(self,
                 model):
        super().__init__()
        self.model = model

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
        img = normalize_image_scale(img)
        print(img.max(), img.min())
        seg = sobel_watershed(img, self.bg_threshold, self.fg_threshold)
        return seg
