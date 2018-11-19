import os
import sys
import itertools
import math
import logging
import json
import re
import random
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

import cv2


os.environ['KMP_DUPLICATE_LIB_OK']='True'

root = "./"
ROOT_DIR = root + "nets/"
dataroot = root + "data_oneofeach/"


sys.path.append(root)  
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log


from mrcnn.config import Config


import pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
 


COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_shapes3.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
    
MODEL_DIR = os.path.join(ROOT_DIR, "logs")





class PiecesConfig(Config):
    """Configuration for training on the dataset.
    Derives from the base Config class and overrides values specific
    to the dataset.
    """
    # Give the configuration a recognizable name
    NAME = "legobasic"

    # Train on 1 GPU and 1 images per GPU. We can put multiple images on each
    # GPU. Batch size is (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 + 1 + 1

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 64, 128, 256)  # anchor side in pixels

    # Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 20
        
    MASK_POOL_SIZE = 32
    MASK_SHAPE = [64, 64]

    # set number of epoch
    STEPS_PER_EPOCH = 100

    # set validation steps 
    VALIDATION_STEPS = 1
    
    



class InferenceConfig(PiecesConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inf_config = InferenceConfig()
    
config = PiecesConfig()
config.display()



def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax



##Dataset setup

class LegoDataset(utils.Dataset):
  
    def load_lego(self, subset, class_ids=None,
                  class_map=None, return_coco=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """
        
        coco = COCO(root + "{}.json".format(subset))

        image_dir = dataroot + "{}_oneofeach".format(subset)

        print(sorted(coco.getCatIds()))
        
        class_ids = coco.getCatIds()

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
                
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
          
            # All images
            image_ids = list(coco.imgs.keys())
            
        print(image_ids)

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])
            
        print(coco.imgs)
        
        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        
            
        if return_coco:
            return coco



    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CocoDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CocoDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return a link to the image in the COCO Website."""
        info = self.image_info[image_id]
        if info["source"] == "coco":
            return "http://cocodataset.org/#explore?id={}".format(info["id"])
        else:
            super(CocoDataset, self).image_reference(image_id)

    # The following two functions are from pycocotools with a few changes.

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

#Training dataset
dataset_train = LegoDataset()
dataset_train.load_lego("train")
dataset_train.prepare()

# Validation dataset
dataset_val = LegoDataset()
dataset_val.load_lego("val")
dataset_val.prepare()

dataset_test = LegoDataset()
dataset_test.load_lego("test")
dataset_test.prepare()
'''
# Load and display random samples
image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
'''

"""## Ceate Model"""

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

"""## Training


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE, 
            epochs=3, 
            layers='heads')

# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2, 
            layers="all")

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes3.h5")
model.keras_model.save_weights(model_path)
print(model_path)
"""


# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inf_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes3.h5")

#print(model_path)

# Load trained weights (fill in path to trained weights here)
#assert model_path != "", "Provide path to trained weights"
#print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

# Test on a random image
image_id = random.choice(dataset_test.image_ids)


original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset_test, inf_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

#visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
#                            dataset_test.class_names, figsize=(8, 8))


#original_image = cv2.imread(dataroot + "ontable.jpeg",1)
#original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)


results = model.detect([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax())







print("Writing...")
for i in range(r['masks'].shape[2]):
    mmask = r['masks'][:,:,i]
    newimg = np.zeros((512, 512,3))

    for y in range(512):
        for x in range(512):
            b = mmask[x][y]
            newimg[x][y] = (original_image[x][y])
            if not b:  
                newimg[x][y] = [255,255,255]

    cv2.imwrite("region-" + str(i) + ".png", newimg)

print("Written...")
  













































"""## Evaluation

# Compute VOC-Style mAP @ IoU=0.5
# Running on 30 images. Increase for better accuracy.
image_ids = np.random.choice(dataset_val.image_ids, 30)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))

X_test = np.zeros((len(test_ids), config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], 3), dtype=np.uint8)
sizes_test = []
_test_ids = []

print('Getting and resizing test images ... ')
#sys.stdout.flush()
for n, id_ in enumerate(test_ids):
    _test_ids.append([id_])
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:3]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]), mode='constant', preserve_range=True)
    X_test[n] = img

print("checking a test image with masks ...")
results = model.detect([X_test[7]], verbose=1)

r = results[0]
visualize.display_instances(X_test[7], r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax())

# Run-length encoding for submission
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths
  
  
def prob_to_rles(x):
   
    for i in range(0, len(x[1, 1, :])):
        y = x[:, :, i]
        yield rle_encoding(np.squeeze(y))

new_test_ids = []
rles = []

print(len(test_ids))

for n in range(0, len(test_ids)):

    results = model.detect([X_test[n]], verbose=0)
     
    r = results[0]

    number_of_masks = len(r['masks'][1, 1, :])
    
    pred_masks = np.zeros([sizes_test[n][0], sizes_test[n][1], number_of_masks], dtype=np.uint8)
   
    for m in range(0, number_of_masks):
        pred_mask_temp = r['masks'][:, :, m]
        pred_masks[:, :, m] = resize(np.squeeze(pred_mask_temp), (sizes_test[n][0], sizes_test[n][1]), mode='constant', preserve_range=True)
        
    # Handle occlusions
    occlusion = np.logical_not(pred_masks[:, :, -1]).astype(np.uint8)
    for i in range(number_of_masks-2, -1, -1):
        pred_masks[:, :, i] = pred_masks[:, :, i] * occlusion
        occlusion = np.logical_and(occlusion, np.logical_not(pred_masks[:, :, i]))
        
    rle = list(prob_to_rles(pred_masks))
        
            
    em = 0
    while (em < len(rle)):
          if len(rle[em]) == 1:
              print ("mask has one pixel")
          if len(rle[em]) == 0:
              print ("mask is empty. deleting mask")
              del rle[em]
          else:
              em = em + 1
    
    
    k = 0
    
    for m in range(0, len(rle)):
        while (k < len(rle[m]) ):
            if rle[m][k] + rle[m][k+1] >= sizes_test[n][0] * sizes_test[n][1]:
                print("Index was outside the bounds of the array.    index = ", rle[m][k] + rle[m][k+1], "    array size = ", sizes_test[n][0] * sizes_test[n][1] )
                rle[m][k+1] = sizes_test[n][0] * sizes_test[n][1] - rle[m][k] - 1
                #print("run length fixed ")
                print("run length fixed.    index = ", rle[m][k] + rle[m][k+1], "    array size = ", sizes_test[n][0] * sizes_test[n][1] )
                if rle[m][k+1] < 0:
                    print("run lngth is negative. deleting the index and run length   ")
                    del rle[m][k]
                    del rle[m][k]
                if rle[m][k] < 0:
                    print("index is negative. deleting the index and run length   ")
                    del rle[m][k]
                    del rle[m][k]
                    
           
                    
                 
                    
                #check that pixels are ordered 
                if rle[m][k] > rle[m][k+2]:
                   print ("pixel not ordered")
                
                #check that a pixel is not duplicated
                if rle[m][k] + rle[m][k+1] >= rle[m][k+2]:
                   print ("pixel duplicated")
                
            k = k + 2
    k = 0           
    for m in range(0, len(rle)):
        
        while (k < len(rle[m]) ):
            if rle[m][k] + rle[m][k+1] >= sizes_test[n][0] * sizes_test[n][1]:
                print("Index was outside the bounds of the array.    index = ", rle[m][k] + rle[m][k+1], "    array size = ", sizes_test[n][0] * sizes_test[n][1] )
            k = k + 2   
    
    rles.extend(rle)
    new_test_ids.extend(_test_ids[n] * len(rle))


# Create submission DataFrame

sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018.csv', index=False)

files.download('sub-dsbowl2018.csv')"""