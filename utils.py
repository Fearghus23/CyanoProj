# utils.py
import os
import json
import cv2

def create_annotations(data_dir, annotations_dir, classes):
    """
    Create annotations for all images in the training directory.
    Each annotation contains information about the image and its class.
    """
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)
    annotations = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            annotation = {
                'filename': img_path,
                'width': w,
                'height': h,
                'class': cls,
                'bbox': [0, 0, w, h]  # Full image
            }
            annotations.append(annotation)
    with open(os.path.join(annotations_dir, 'annotations.json'), 'w') as f:
        json.dump(annotations, f)

