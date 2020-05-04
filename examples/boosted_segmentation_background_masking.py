"""
This example shows how to use the boostedSegmenter class to segment an image with the boosted approach.
Compared to the example boosted_segmentation.py, this example enable the masking background feature,
which helps to suppress the non-white background pixels.
"""

from seglib.segmentation import BoostedSegmenter, apply_mask
from seglib.utils import load_mrcnn_model, get_iou
from skimage import io as skio
from skimage.transform import rescale
from mrcnn.config import Config
from matplotlib import pyplot as plt
import numpy as np
import os


# Only required on a macbook
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load the data in your way
image_dir = "./local_data/image_data/photostudio/validation/"
image_name = "000062"
image_mask = skio.imread(f'{image_dir}/pngs/{image_name}.png')
image_mask = (image_mask[:, :, 3]!=0).astype(np.int)
image_orig = skio.imread(f'{image_dir}/jpgs/{image_name}.jpg')
mask_orig = rescale(image_mask, 0.25, preserve_range=True, multichannel=False).astype(np.int)
image_orig = rescale(image_orig, 0.25, preserve_range=True, multichannel=True).astype(np.int)

# Path to the mask-rcnn model weight
model_path = "./local_data/models/2020-04-15_GPU_EAL_deepfashion2-fotostudio_laplace_20200415T1525/mask_rcnn_fotostudio_0030.h5"

class InferenceConfig(Config):
    NAME = 'segmentation'
    NUM_CLASSES = 1 + 1
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 512
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
model = load_mrcnn_model(model_path, './model_log_dir', config)

# Initialize a BoostedSegmenter instance with the following parameters
boosted_seg = BoostedSegmenter(model=model,
                               gaussian_sigma=24,
                               gaussian_sigma_outer=24,
                               gaussian_threshold_outer=0.2,
                               bg_threshold=0.6,
                               fg_threshold=0.05,
                               mask_background=True)

# Get the mask from the instance
mask_bst = boosted_seg.segment(image_orig)

# Get the mrcnn mask that is used in the intermediate process
mask_mrcnn = boosted_seg._get_mrcnn_mask([image_orig])[0]

# Get the softly masked image that is fed into the watershed segmentation
soft_image = boosted_seg._apply_soft_mask(image_orig, mask_mrcnn)
# The final segmented image
masked_img = apply_mask(image_orig, mask_bst)

# The intersection over union scores
iou_bst = get_iou(mask_bst, mask_orig)
iou_mrcnn = get_iou(mask_mrcnn, mask_orig)

# Display the segmentation result
fig, axes = plt.subplots(2, 3, figsize=(10, 10))
ax = axes.flatten()
def plot_i(ax, i, img, title, cmap=None):
    ax[i].imshow(img, cmap)
    ax[i].set_axis_off()
    ax[i].set_title(title, fontsize=12)

i = 0; plot_i(ax, i, image_orig, "Raw image")
i += 1; plot_i(ax, i, mask_mrcnn, f"Maskrcnn mask. IOU: {iou_mrcnn: 0.4f}", cmap='gray')
i += 1; plot_i(ax, i, soft_image, "Softly-masked image", cmap='gray')
i += 1; plot_i(ax, i, mask_bst, f"Boosted-seg mask. IOU: {iou_bst: 0.4f}", cmap='gray')
i += 1; plot_i(ax, i, masked_img, "Masked image")
i += 1; plot_i(ax, i, mask_orig, "Ground-truth mask")
fig.show()
