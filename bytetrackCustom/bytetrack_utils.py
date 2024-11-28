import cv2
import numpy as np
import torch

from config.VEHICLE_CLASS import VEHICLE_CLASSES

"""
This script contains utility function for byte tracker to work.
"""

COLORS = np.random.randint(0, 255, size=(len(VEHICLE_CLASSES), 3))


def calculate_centroid(tl_x, tl_y, w, h):
    mid_x = int(tl_x + w / 2)
    mid_y = int(tl_y + h / 2)
    return mid_x, mid_y


def convert_output(outputs: torch.Tensor):
    # Output of format []
    return


def convert_history_to_dict(track_history):
    history_dict = {}
    for frame_content in track_history:
        obj_ids, tlwhs, _ = frame_content
        for obj_id, tlwh in zip(obj_ids, tlwhs):
            tl_x, tl_y, w, h = tlwh
            mid_x, mid_y = calculate_centroid(tl_x, tl_y, w, h)

            if obj_id not in history_dict.keys():
                history_dict[obj_id] = [[mid_x, mid_y]]
            else:
                history_dict[obj_id].append([mid_x, mid_y])

    return history_dict

def count_tracks(track_history):
    obj_ids, tlwhs, class_ids = track_history[-1]

    num_detections = len(tlwhs)

    return num_detections

def plot_tracking(image, track_history, args):
    obj_ids, tlwhs, class_ids = track_history[-1]
    history_dict = convert_history_to_dict(track_history)

    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    num_detections = len(tlwhs)
    # label_count = {class_name: 0 for class_name in args.cls}
    # for label_idx in class_ids:
    #     label_count[VEHICLE_CLASSES[label_idx]] += 1

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        class_id = class_ids[i]
        id_text = '{}'.format(int(obj_id))
        color = tuple(int(c) for c in COLORS[class_ids[i]])
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=1)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, 1, color,
                    thickness=1)
        cv2.putText(im, VEHICLE_CLASSES[class_id], (intbox[0], intbox[3] + 20), cv2.FONT_HERSHEY_PLAIN, 1, color,
                    thickness=1)

        for idx in range(len(history_dict[obj_id]) - 1):
            prev_point, next_point = history_dict[obj_id][idx], history_dict[obj_id][idx + 1]
            cv2.line(im, prev_point, next_point, color, 2)

    return im


def transform_detection_output(detections, classes):
    """
    Convert the detection output of Faster R CNN to the format used by ByteTrack

    :param classes:
    :param detections:
    :return: outputs - tensorized detections
    """
    boxes = detections['boxes'].cpu().numpy()
    labels = detections['labels'].cpu().numpy()
    scores = detections['scores'].cpu().numpy()
    lbl_mask = np.isin(labels, classes)
    scores = scores[lbl_mask]
    labels = labels[lbl_mask]
    boxes = boxes[lbl_mask]
    outputs = []

    for i, box in enumerate(boxes):
        label = labels[i]
        score =  scores[i]
        xmin, ymin, xmax, ymax = boxes[i]
        output = [xmin, ymin, xmax, ymax, score, label]
        outputs.append(output)

    outputs = torch.tensor(outputs, dtype=torch.float32)

    return outputs

