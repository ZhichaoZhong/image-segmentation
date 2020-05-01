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


def predict_mask(model, img):
    """
    Predict the mask for an image using a maskrcnn model
    :param model: mask-rcnn mask instance.
    :param img: 3D numpy array.
    :return: 2D binary array.
    """
    assert len(img[0].shape) == 3, 'Input image must have channel dimension.'
    pred_result = model.detect(img, verbose=0)
    pred_mask = [p['masks'].astype(np.int).squeeze() for p in pred_result]
    return pred_mask

def sobel_watershed(img, bg_threshold, fg_threshold):
    """
    Segment an image using a sobel+watershed approach.
    :param img: gray image of shape [M, N] and scale range [0.0, 1.0].
    :param bg_threshold:
    :param fg_threshold:
    :return: a binary image of shape [M, N]
    Reference:
    https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.watershed
    """
    assert img.max() <= 1.0 and img.min() >= 0.0
    assert len(img.shape) == 2

    data = img
    # Detect the edges of the image
    edges = sobel(data)

    # Define the initial markers for the watershed algorithm
    markers = np.zeros_like(data)
    foreground, background = 1, 2
    markers[data > bg_threshold] = background
    markers[data < fg_threshold] = foreground

    ws = watershed(edges, markers)
    mask = (ws == foreground).astype(np.int)
    return mask


# ------------------
# Segmenter Class
# ------------------

class BasicSegmenter(object):
    def __init__(self):
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
                 gaussian_sigma=30,
                 bg_threshold=0.05,
                 fg_threshold=0.80):
        super().__init__()
        self.bg_threshold = bg_threshold
        self.fg_threshold = fg_threshold
        self.model = model
        self.sigma = gaussian_sigma

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
        soft_mask = gaussian(pred_mask, self.sigma)
        soft_mask = normalize_image_scale(soft_mask)
        return soft_mask

    def _apply_soft_mask(self, img, ):
        """
        Apply a soft mask so as to darken or lighten the
         object to segment
        :param img: np.array(), 3D rgb image.
        :return: np.array(), 2D gray image.
        """
        soft_mask = self._get_soft_mask(img)
        # Transform the image to gray image
        # gray_img = normalize_image_scale(rgb2gray(img.copy()))
        gray_img = normalize_image_scale(rgb2gray(img))
        # Apply the soft mask on the image
        processed_image = apply_mask(gray_img, soft_mask, True)
        processed_image /= processed_image.max()
        return processed_image

class MaskRcnnSegmenter(BasicSegmenter):
    def __init__(self, model):
        """
        A segmenter class that use mask-rcnn model to segment
        :param model: a maskrcnn model instance.
        """
        super().__init__()
        self.model = model

    def segment(self, img):
        """
        Segment the input images.
        :param img: One or a list of rgb images of dimension [M, N, C]
        :return: One or a list of binary images of dimension [M, N]
        """
        if isinstance(img, list):
            pass
        else:
            return_1 = True
            img = [img]

        seg = predict_mask(self.model, img)
        if return_1:
            return seg[0]
        return seg


class WatershedSegmenter(BasicSegmenter):
    def __init__(self,
                 bg_threshold=0.01,
                 fg_threshold=0.60):
        """
        A segmenter class that uses the watershed segmentation method
        :param bg_threshold: background grayscale threshold. Default is 0.01.
        :param fg_threshold: foreground grayscale threshold. Default is 0.60.
        reference:
        https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.watershed
        """
        super().__init__()
        self.bg_threshold = bg_threshold
        self.fg_threshold = fg_threshold

    def segment(self, img):
        """
        Segment the input images.
        :param img: One or a list of rgb images of dimension [M, N, C]
        :return: One or a list of binary images of dimension [M, N]
        """
        if isinstance(img, list):
            pass
        else:
            return_1 = True
            img = [img]

        seg = []
        for image in img:
            # Convert the input image to grayscale image between [0.0, 1.0]
            image = normalize_image_scale(rgb2gray(image))
            seg.append(sobel_watershed(image, self.bg_threshold, self.fg_threshold))

        if return_1:
            return seg[0]
        return seg
