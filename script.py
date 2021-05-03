"""
Mask R-CNN
Train on the toy bottle dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python script.py train --dataset=glitch --weights=coco

    # Resume training a model that you had trained earlier
    python script.py train --dataset=glitch --weights=last

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class CustomConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Configuration name
    NAME = "glitch"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # Background + glitches

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    BATCH_SIZE = 16

    # Adjust as necessary
    LOSS_WEIGHTS = {'rpn_class_loss': 1.0, 'rpn_bbox_loss': 1.0, 'mrcnn_class_loss': 1.0, 'mrcnn_bbox_loss': 1.0, 'mrcnn_mask_loss': 1.0}


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset, Fold):
        """Load the glitch dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        self.add_class("glitch", 1, "missing")
        self.add_class("glitch", 2, "low_resolution")
        self.add_class("glitch", 3, "stretched")

        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region

        annotations = []

        if (subset=="train"):
            annotations1 = json.load(open(os.path.join("via_region_data_train.json")))
        else:
            annotations1 = json.load(open(os.path.join("via_region_data_test.json")))

        annotation = list(annotations1.values())  # don't need the dict keys

        # 4-Fold CV:
        k_fold = KFold(n_splits = 4, random_state = 42, shuffle = True)

        for i, (train, val) in enumerate(k_fold.split(annotation)):
            if subset == "train" and Fold == i:
                for index in train:
                    annotations.append(annotation[index])

            elif subset == "val" and Fold == i:
                for index in val:
                    annotations.append(annotation[index])


        annotations = [a for a in annotations]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            if a['regions']:
                polygons = [r['shape_attributes'] for r in a['regions']]
                class_ids =  [r['region_attributes'] for r in a['regions']]

                # load_mask() needs the image size to convert polygons to masks.
                # Unfortunately, VIA doesn't include it in JSON, so we must read
                # the image. This is only managable since the dataset is tiny.
                image_path = os.path.join(dataset_dir, a['filename'])
                image = skimage.io.imread(image_path, plugin='matplotlib')
                height, width = image.shape[:2]

                self.add_image(
                    "glitch",
                    image_id=a['filename'],
                    path=image_path,
                    width=width, height=height,
                    polygons=polygons, class_ids=class_ids)
            else:
                image_path = os.path.join(dataset_dir, a['filename'])
                image = skimage.io.imread(image_path, plugin='matplotlib')
                height, width = image.shape[:2]

                self.add_image(
                    "glitch",  ## for a single class just add the name here
                    image_id=a['filename'],  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    polygons=[], class_ids=[])

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a glitch dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "glitch":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        masks = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        returnclasses = list()
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            masks[rr, cc, i] = 1
            if info["class_ids"][i]["name"] == "missing":
                returnclasses.append(1)
            elif info["class_ids"][i]["name"] == "low_resolution":
                returnclasses.append(2)
            else:
                returnclasses.append(3)

        # Return mask, and array of class IDs of each instance.
        return masks, np.array(returnclasses, dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "glitch":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.

    for i in range(4):

        print("Training fold", i)

        dataset_train = CustomDataset()
        dataset_train.load_custom(args.dataset, "train", i)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CustomDataset()
        dataset_val.load_custom(args.dataset, "val", i)
        dataset_val.prepare()

        # Training - Stage
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE,
                        epochs=10,
                        layers='heads')

         # Training - Stage 2
         # Finetune layers from ResNet stage 4 and up
        print("Fine tuning Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE,
                        epochs=40,
                        layers='4+')

         # Training - Stage 3
         # Fine tune all layers
        print("Fine tuning all layers")
        model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE / 10,
                        epochs=50,
                        layers='all')


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom class.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CustomConfig()
    else:
        class InferenceConfig(CustomConfig):
            # Set batch size to 1 since we'll be running inference on one image at a time.
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
        # weights_path = model.find_last()[1]
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    if args.command == "train":
        train(model)
