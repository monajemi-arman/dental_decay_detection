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

# Usage
1. Clone this repository
2. Prepare a JSON and a directory for each of train, val, test. The directory contains images, and each JSON has COCO style annotation data of the images.
3. Use the main.py to train a model on your data. Checkpoint of the model will be saved in lightning_logs directory.
4. **To Be Implemented** Use infer.py after creating the model checkpoint for inference.
---
# Demo
![](images/demo.jpg)
