from seglib.segmentation import BoostedSegmenter, MaskRcnnSegmenter, WatershedSegmenter
from skimage import io as skio
from matplotlib import pyplot as plt

# Load the data in your way
image_dir = "./local_data/image_data/photostudio/validation/"
image_name = "000222"
image_mask = skio.imread(f'{image_dir}/pngs/{image_name}.png')
image_orig = skio.imread(f'{image_dir}/jpgs/{image_name}.jpg')

# Path to the mask-rcnn model weight
model_path = "./local_data/models/2020-04-15_GPU_EAL_deepfashion2-fotostudio_laplace_20200415T1525/mask_rcnn_fotostudio_0030.h5"

boosted_seg = BoostedSegmenter(weight_path=model_path,
                              gaussian_sigma=30,
                              bg_threshold=0.05,
                              fg_threshold=0.80,
                              model_dir='./model_log_dir',
                              maskrcnn_config=None)

maskrcnn_seg = MaskRcnnSegmenter(weight_path=model_path,
                              model_dir='./model_log_dir',
                              maskrcnn_config=None)

watershed_seg = WatershedSegmenter(bg_threshold=0.05, fg_threshold=0.80)

mask_mr = maskrcnn_seg.segment(image_orig)
mask_ws = watershed_seg.segment(image_orig)
mask_bs = boosted_seg(image_orig)


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
