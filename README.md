# Dental Decay Object Detection
This project trains a Faster R-CNN model to detect dental decay from bitewing images used in dentistry.  
## Features
* Uses Pytorch Lightning
* COCO dataset style standardized data input
* Faster R-CNN implementation
* Metrics
 * Calculation of mAP on test data implemented in model class
 * Manually calculating precision, recall, accuracy, specificity
  * True Positive, False Negative, ... values calculated for each class
  
## Purpose  
* main.py  
 * All in one file
  * In later larger projects, code will be separated into several files in order to be easier to read.
 * Contains dataset class, dataloader, model, training, etc.  
  * Visualize function is used to visualize the prediction and ground truth
* convert_data_to_coco.py  
 * Converts out custom data from .nrrd (3dslicer output) to coco style JPEG and JSON.  
---
# Demo
![](images/demo.jpg)
