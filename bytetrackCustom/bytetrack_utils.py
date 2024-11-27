import cv2
import numpy as np
import torch

from infer_video_saveDET import COLORS
"""
This script contains utility function for byte tracker to work.
"""

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


def plot_tracking(image, track_history):
    obj_ids, tlwhs, class_ids = track_history[-1]
    history_dict = convert_history_to_dict(track_history)

    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    num_detections = len(tlwhs)
    label_count = {class_name: 0 for class_name in args.cls}
    for label_idx in class_ids:
        label_count[ID2CLASSES[label_idx]] += 1

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        class_id = class_ids[i]
        id_text = '{}'.format(int(obj_id))
        color = COLORS[class_id]
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=1)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, 1, color,
                    thickness=1)
        cv2.putText(im, ID2CLASSES[class_id], (intbox[0], intbox[3] + 20), cv2.FONT_HERSHEY_PLAIN, 1, color,
                    thickness=1)

        for idx in range(len(history_dict[obj_id]) - 1):
            prev_point, next_point = history_dict[obj_id][idx], history_dict[obj_id][idx + 1]
            cv2.line(im, prev_point, next_point, color, 2)

    return im
