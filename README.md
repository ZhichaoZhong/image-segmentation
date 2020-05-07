# Image segmentation for product images
This repository contains the codes to segment a raw images taken at the photo studio.  

## Getting Started:
[boosted_segmentation.py](./examples/boosted_segmentation.py) 
gives you the basic example to start with.

### Prerequisites
1. [The mask-rcnn inplementation](https://github.com/matterport/Mask_RCNN/blob/master/requirements.txt)
2. Check `requirements.txt`  

### Boosted segmentation method
#### Semantic segmentation task
This library is used to perform a simple semantic segmentation task: given a photo of a clothing, segment the clothing 
from the background with a high resolution. Below is an example:  
<img src="assets/task.png" width="360" height="200" alt="Segmentation task"/>
  
#### Mask-rcnn
One of the state-of-the-art techniques is to [use neural-network models](https://arxiv.org/pdf/1809.10198.pdf). In our case, we trained a 
[mask-rcnn](https://github.com/matterport/Mask_RCNN) model on our own dateset. The mask-rcnn model can generalize very well
on different images. However, the final output mask is interpolated from a 28X28 feature map, 
and therefore the accuracy is very low compared the resolution of our images (2400X1600). The size of the feature map can be 
increased to  56X56, or even 112X112, but hardly higher due to the limitation of GPU memory.    
<img src="assets/mrcnn_example.png" width="400" height="200" alt="Segmentation task"/>
  
#### Watershed
We looked at some [segmentation techniques](https://scikit-image.org/docs/dev/api/skimage.segmentation.html) that can potential perform well on high resolution image. 
Among them, we found the [watershed approach](https://scikit-image.org/docs/dev/user_guide/tutorial_segmentation.html) most interesting. The watershed approach can be 
summarized in three steps:  
1. Detect the edges using image filters (e.g. Sobel, Laplace).
2. Label markers as foreground and background (e.g. using thresholds). 
3. Flood the markers till reaching the edges.
Here is an example:  
<img src="assets/watershed_example.png" width="450" height="160" alt="Segmentation task"/>
The watershed approach is straightforward and performs well on the edges by nature. However, it is difficult to label the markers 
in an automatic manner. Also, the edges inside instance to be segmented can be troublesome. Below is a bad example:
<img src="assets/watershed_example_bad.png" width="460" height="160" alt="Segmentation task"/>
 
#### The boosted approach
The boosted approach dedicates itself to compensate the drawbacks of both approach. It can be summarized by the following procedures:
1. Generate a mrcnn-mask using the mask-rcnn model.  
2. Softly mask the image to make the object edges dominant (by masking the interiors).  
3. Use the mrcnn mask as the markers.  
4. Apply the watershed algorithms on the softly-masked image.
The following image describes this approach the best:
<img src="assets/boosted_approach_explain.png" width="460" height="320" alt="Segmentation task"/>

### Benchmark
ToDo
### ToDo:
1. [x] Merge the two-soft branch to master 
2. [x] Edit the example notebooks
3. Finish the readme:  
    a. Descriptions  
    b. Explain the algorithm  
    c. Include examples and benchmark  
4. add requirements.txt