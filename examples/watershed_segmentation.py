from seglib.segmentation import WatershedSegmenter, apply_mask
from skimage import io as skio
from skimage.transform import rescale
import numpy as np
import os
from matplotlib import pyplot as plt

# Only required on a macbook
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load the data in your way
image_dir = "./local_data/image_data/photostudio/validation/"
image_name = "000062"
image_mask = skio.imread(f'{image_dir}/pngs/{image_name}.png')
image_mask = (image_mask[:, :, 3]!=0).astype(np.int)
image_orig = skio.imread(f'{image_dir}/jpgs/{image_name}.jpg')
mask_orig = rescale(image_mask, 0.25, preserve_range=True, multichannel=False)
image_orig = rescale(image_orig, 0.25, preserve_range=True, multichannel=True).astype(np.int)

# Initialize a watershed segmentation instance
watershed_seg = WatershedSegmenter(bg_threshold=0.50, fg_threshold=0.80)
mask = watershed_seg.segment(image_orig)
masked_img = apply_mask(image_orig, mask)


# Display the segmentation result
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
ax = axes.flatten()

ax[0].imshow(image_orig)
ax[0].set_axis_off()
ax[0].set_title("Original image", fontsize=12)

ax[1].imshow(mask, cmap="gray")
ax[1].set_axis_off()
ax[1].set_title("Segmentation output", fontsize=12)
fig.show()

ax[2].imshow(masked_img, cmap="gray")
ax[2].set_axis_off()
ax[2].set_title("Masked image", fontsize=12)

ax[3].imshow(mask_orig, cmap="gray")
ax[3].set_axis_off()
ax[3].set_title("Ground-truth mask", fontsize=12)
fig.show()
