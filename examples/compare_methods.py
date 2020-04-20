from seglib.segmentation import BoostedSegmenter, MaskRcnnSegmenter, WatershedSegmenter, load_maskrcnn_model
from skimage import io as skio
from skimage.transform import rescale
from matplotlib import pyplot as plt
import numpy as np
from mrcnn.config import Config
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# Load the data in your way
image_dir = "./local_data/image_data/photostudio/validation/"
image_name = "000222"
image_mask = skio.imread(f'{image_dir}/pngs/{image_name}.png')
image_mask = (image_mask[:,:,3]!=0).astype(np.int)
image_orig = skio.imread(f'{image_dir}/jpgs/{image_name}.jpg')
image_mask = rescale(image_mask, 0.25, preserve_range=True, multichannel=False)
image_orig = rescale(image_orig, 0.25, preserve_range=True, multichannel=True).astype(np.int)

# Path to the mask-rcnn model weight
model_path = "./local_data/models/2020-04-15_GPU_EAL_deepfashion2-fotostudio_laplace_20200415T1525/mask_rcnn_fotostudio_0030.h5"

watershed_seg = WatershedSegmenter(bg_threshold=0.6, fg_threshold=0.80)
mask_ws = watershed_seg.segment(image_orig)
skio.imshow(mask_ws)


class InferenceConfig(Config):
    NAME = 'segmentation'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 256
config = InferenceConfig()

model = load_maskrcnn_model(model_path, './model_log_dir', config)

maskrcnn_seg = MaskRcnnSegmenter(model=model)
mask_mr = maskrcnn_seg.segment(image_orig)
skio.imshow(mask_mr)

boosted_seg = BoostedSegmenter(model=model,
                              gaussian_sigma=30,
                              bg_threshold=0.05,
                              fg_threshold=0.80)
mask_bs = boosted_seg.segment(image_orig)
skio.imshow(mask_bs)


def display_result(image, seg, gt, figsize=(15, 15)):
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax = axes.flatten()

    ax[0].imshow(image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].set_title("Original Image", fontsize=12)

    ax[1].imshow(seg, cmap="gray")
    ax[1].set_axis_off()
    title = "Segmentaion".format(len(cv[2]))
    ax[1].set_title(title, fontsize=12)

    ax[2].imshow(gt, cmap="gray")
    ax[2].set_axis_off()
    ax[2].set_title("Ground truth", fontsize=12)
    fig.show()

    ax[3].imshow(gt - seg, cmap="gray")
    ax[3].set_axis_off()
    ax[3].set_title("Difference", fontsize=12)
    fig.show()
