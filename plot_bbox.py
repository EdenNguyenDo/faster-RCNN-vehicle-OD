import cv2
import numpy as np

from config.VEHICLE_CLASS import VEHICLE_CLASSES

COLORS = np.random.randint(0, 255, size=(12, 3))

def draw_boxes(detections, frame, classes, threshold):

    boxes = detections["boxes"].cpu().numpy()
    labels = detections["labels"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()

    lbl_mask = np.isin(labels, classes)
    scores = scores[lbl_mask]
    mask = scores > threshold
    boxes = boxes[lbl_mask][mask]
    scores = scores[mask]
    labels = labels[lbl_mask][mask]


    # read the image
    for i, box in enumerate(boxes):
        label = labels[i]
        class_name = VEHICLE_CLASSES[label]
        confidence =  scores[i]

        # Ensure the color is a tuple of integers
        color = tuple(int(c) for c in COLORS[labels[i]])
        # Define unique colors for each label

        cv2.rectangle(
            frame,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        class_name_confidence = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, class_name_confidence, (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)


    return frame