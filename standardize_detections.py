# {'bbox : [int(x), y, w, h]',
#  'conf': float('4 decimal'),
# 'class':'3'}
import os

import numpy as np


def standardize_format(detections, classes, threshold, frame_no, vid_name):
    boxes = detections["boxes"].cpu().numpy()
    labels = detections["labels"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()
    lbl_mask = np.isin(labels, classes)
    scores = scores[lbl_mask]
    mask = scores > threshold
    boxes = boxes[lbl_mask][mask]
    scores = scores[mask]
    labels = labels[lbl_mask][mask]

    # Prepare the filename based on video name and frame number
    output_dir = 'bounding_box_annotations/'
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{vid_name}_{frame_no}.txt"
    file_path = os.path.join(output_dir, filename)


    with open(file_path, 'w') as f:
        for i, box in enumerate(boxes):
            class_name = labels[i]
            confidence = scores[i]
            # Append ([x, y, w, h], score, label_string).
            x_min, y_min, x_max, y_max = box

            # Convert to (x_center, y_center, width, height)
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            f.write(f"{class_name} {x_center} {y_center} {width} {height} {confidence}\n")









