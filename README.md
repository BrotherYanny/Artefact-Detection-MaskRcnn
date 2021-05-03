# Using a Mask R-CNN to Detect Video-Game Artefacts
This is code adapted from [Matterport's Mask R-CNN implementation](https://github.com/matterport/Mask_RCNN#readme) to detect Textural Artefacts in Video-Games; in particular Missing, Low Resolution and Stretched. The weights provided are trained on samples from the game The Elder Scrolls V: Skyrim.


## Installation
Using python 3.7, run
```
pip install requirements.txt
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
cd ..
git clone https://github.com/BrotherYanny/
```
A virtual environment is recommended.

A few changes to the mrcnn library need to be made:

In mrcnn/model.py,
in data_generation, comment out
```
if not np.any(gt_class_ids > 0):
continue
```
in build_rpn_targets, add
```
if gt_class_ids.shape[0] == 0: rpn_match = -1 * np.ones([anchors.shape[0]], dtype=np.int32) rpn_bbox = generate_random_rois(image_shape, \ config.RPN_TRAIN_ANCHORS_PER_IMAGE, gt_class_ids, gt_boxes) return rpn_match, rpn_bbox
```
in generate_random_rois, change
```
rois_per_box = int(0.9 * count / gt_boxes.shape[0])
```
to
```
rois_per_box = int(0.9 * count / (gt_boxes.shape[0] + 0.000001))
```


## Usage
After collecting an adequate dataset, place in the glitch/train and glitch/test directories
and the annotation JSONs in the root folder

To configure the model, load data, train and run:
```
python script.py train --dataset=glitch --weights=coco
```

After getting trained weights (or using the pre-trained weights supplied), use the Jupyter Notebook 'test.ipynb' to test the model on your testing dataset (adjust the weight path if necessary).


## Notes
