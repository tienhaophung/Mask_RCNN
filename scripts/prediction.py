from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.visualize import display_instances
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

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
img_dir = "/Users/haophung/Documents/Mask_RCNN/images/mercedes_benz.jpg" # "/Users/haophung/Documents/Mask_RCNN/images/12283150_12d37e6389_z.jpg"

# Define model to inference, model_dir is directory to write logs
rcnn = MaskRCNN(mode='inference', model_dir='./', config=InferenceConfig())

# Load pretrained weights on coco
rcnn.load_weights(pretrained_weights_dir, by_name=True)

# load image
img = cv2.imread(img_dir)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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

# Plot image with bounding boxes
plot_bounding_box_with_image(img, boxes, class_ids, scores)

display_instances(img, res['rois'], res['masks'], res['class_ids'], class_names, res['scores'])



