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

![Example](./assets/task.png)

### ToDo:
1. [x] Merge the two-soft branch to master 
2. [x] Edit the example notebooks
3. Finish the readme:  
    a. Descriptions  
    b. Explain the algorithm  
    c. Include examples and benchmark  
4. add requirements.txt