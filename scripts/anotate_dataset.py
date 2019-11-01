from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances

import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import json_tricks
import os

# define 81 classes that the coco model knowns about
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

# Declare config object that defines hyperparameter for model 
class InferenceConfig(Config):
    # Set batch size to 1 to inference one image at time.
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80

def plot_bounding_box_with_image(img, boxes, class_ids, scores):
    ax = plt.gca() # get current axis
    # ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.imshow(img)
    ax.axis("off")

    for i, box in enumerate(boxes):
        if class_ids[i] != class_names.index('car'):
            continue
        # lower left
        y1, x1, y2, x2 = box
        # Compute width and height
        width, height = x2 - x1, y2 - y1
        
        # Create rectangle
        rect = Rectangle((x1, y1), width, height, fill=False, color='red')
        # Draw rect
        ax.add_patch(rect)

        # Write confidence
        ax.text(x1, y1, "{:.2f}".format(scores[i]), fontsize=12, color='red')
    
    plt.show()

pretrained_weights_dir = "./mask_rcnn_coco.h5"
data_dir = "/Users/haophung/OneDrive - VNU-HCMUS/car_dataset"
conf_threshold = 0.8 # Confidence threshold
area_threshold = 0.4 # Area threshold for small objq
anot_data = {}

# Define model to inference, model_dir is directory to write logs
rcnn = MaskRCNN(mode='inference', model_dir='./', config=InferenceConfig())

# Load pretrained weights on coco
rcnn.load_weights(pretrained_weights_dir, by_name=True)


image_paths = {}
for directory in os.listdir(data_dir):
    # Pass if this is a file
    if "." in directory:
        continue
    image_paths[directory] = os.listdir(os.path.join(data_dir, directory))
# print(json_tricks.dumps(image_paths, indent=4))

car_ids = [class_names.index('car'), class_names.index('truck')]

for car_model, imgnames in image_paths.items():
    for imgname in imgnames:
        if imgname.startswith(".") or imgname.startswith("Icon"):
            continue
        # Get img_dir
        img_dir = os.path.join(data_dir, car_model, imgname)
        
        # load image
        img = cv2.imread(img_dir)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        assert(img is not None)
        # Fit model
        """res is list of dictionary [dict]. Len of res is the number of images that you fit into

        Where dict contain these keys:
            "rois": the bounding boxes or ROI for detected objects (contain lower left and higher right points)
            "masks": The masks for detected objects
            "class_ids": The class index
            "scores": The confidence level of predicted class
        """

        res_list = rcnn.detect([img], verbose=0)

        res = res_list[0]

        # Get boxes for image
        boxes = res['rois']
        class_ids = res['class_ids']
        scores = res['scores']

        anot_key = os.path.join(car_model, imgname)
        anot_data[anot_key] = {"boxes":[], "scores":[], "scaled_area":[]}

        fig = plt.figure()
        ax = fig.gca()
        ax.imshow(img)

        img_height, img_width = img.shape[:2]

        for i, box in enumerate(boxes):
            if class_ids[i] not in car_ids or scores[i] < conf_threshold:
                continue
            # lower left
            y1, x1, y2, x2 = box
            # Compute width and height
            width, height = x2 - x1, y2 - y1

            # Filter out small bounding boxes
            scaled_area = (width * height) / (img_width * img_height)
            if scaled_area < area_threshold:
                continue
            
            # Create rectangle
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            # Draw rect
            ax.add_patch(rect)

            # Write confidence
            ax.text(x1, y1, "{:.2f}".format(scores[i]), fontsize=12, color='red')

            # Save "scaled_area", "boxes" and "scores"
            anot_data[anot_key]["scaled_area"].append(scaled_area)
            anot_data[anot_key]["boxes"].append(box)
            anot_data[anot_key]["scores"].append(scores[i])
            
        # # Plot image with bounding boxes
        # plot_bounding_box_with_image(img, boxes, class_ids, scores)
        break
    break

plt.show()

# Save anot_data
with open(os.path.join(data_dir, "anotations.json"), "w") as json_file:
    json_tricks.dump(anot_data, json_file, indent=4)