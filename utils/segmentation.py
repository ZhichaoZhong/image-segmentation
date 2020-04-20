
import numpy as np
import logging

logger = logging.getLogger('image-seg')


def apply_mask(image, mask, reverse = False):
    """
    Apply a mask on an image:
    :param image: numpy array. dim should be 2 or 3.
    :param mask: 2d numpy array. Data range should be [0, 1].
    :param reverse: bool. To reverse the mask or not.
    :return:
    """
    try:
        assert mask.max() <= 1 and mask.min()>=0
    except AssertionError as error:
        logger.exception('Scale of mask must be between 0 and 1.')
    try:
        assert mask.shape == image.shape[:2]
    except AssertionError as error:
        logger.exception('Shape of mask and image must be the same.')

    if reverse:
        mask = mask.max()-mask

    image = image.copy()
    if len(image.shape) ==2:
        return image*mask
    for c in range(image.shape[2]):
        image[:, :, c] = image[:, :, c] * mask
    return image

