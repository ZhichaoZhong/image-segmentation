
from sklearn.metrics import jaccard_score
import numpy as np

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

