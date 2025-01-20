import cv2
import numpy as np
import os
import csv
from config.VEHICLE_CLASS import VEHICLE_CLASSES
import torch
from torchvision.ops.boxes import box_area, nms
from loguru import logger


"""
Utilities for bounding box manipulation and GIoU.
"""


COLORS = np.random.randint(0, 255, size=(len(VEHICLE_CLASSES), 3))

def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def write_results_no_score(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},-1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),
                                          h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def vectorized_iou(boxes1, boxes2):
    '''
        this is not a standard implementation, but to incorporate with the main function
    '''
    x11, y11, x12, y12 = boxes1
    x21, y21, x22, y22 = boxes2

    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.maximum(x12, np.transpose(x22))
    yB = np.maximum(y12, np.transpose(y22))

    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)

    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)

    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def batch_iou(a, b, epsilon=1e-5):
    """ Given two arrays `a` and `b` where each row contains a bounding
        box defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union scores for each corresponding
        pair of boxes.

    Args:
        a:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        b:          (numpy array) each row containing [x1,y1,x2,y2] coordinates
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (numpy array) The Intersect of Union scores for each pair of bounding
        boxes.
    """
    # COORDINATES OF THE INTERSECTION BOXES
    x1 = np.array([a[:, 0], b[:, 0]]).max(axis=0)
    y1 = np.array([a[:, 1], b[:, 1]]).max(axis=0)
    x2 = np.array([a[:, 2], b[:, 2]]).min(axis=0)
    y2 = np.array([a[:, 3], b[:, 3]]).min(axis=0)

    # AREAS OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)

    # handle case where there is NO overlap
    width[width < 0] = 0
    height[height < 0] = 0

    area_overlap = width * height

    # COMBINED AREAS
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined + epsilon)
    return iou


def clip_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    union = area1 + area2 - inter
    iou = (inter + 1e-6) / (union + 1e-6)
    # generalized version
    # iou=iou-(inter-union)/inter
    return iou


def multi_iou(boxes1, boxes2):
    lt = torch.max(boxes1[..., :2], boxes2[..., :2])
    rb = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    wh = (rb - lt).clamp(min=0)
    wh_1 = boxes1[..., 2:] - boxes1[..., :2]
    wh_2 = boxes2[..., 2:] - boxes2[..., :2]
    inter = wh[..., 0] * wh[..., 1]
    union = wh_1[..., 0] * wh_1[..., 1] + wh_2[..., 0] * wh_2[..., 1] - inter
    iou = (inter + 1e-6) / (union + 1e-6)
    return iou


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = (inter + 1e-6) / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - ((area - union) + 1e-6) / (area + 1e-6)


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks
    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.
    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)



def create_track_file(output_dir, folder_path, track_id):

    full_path = os.path.join(output_dir, folder_path.split('/')[-2])

    # Ensure the directory exists
    os.makedirs(full_path, exist_ok=True)

    filename = f"{track_id}.csv"

    filepath = os.path.join(full_path, filename)

    return filepath


def apply_nms(detections_tensor, iou_threshold):

    # Extract bounding boxes, scores, and labels
    boxes = detections_tensor[0][:, :4]  # Bounding box coordinates (x_min, y_min, x_max, y_max)
    scores = detections_tensor[0][:, 4]  # Confidence scores
    labels = detections_tensor[0][:, 5]  # Class labels

    # Apply Non-Maximum Suppression (NMS)
    keep_inds = nms(boxes, scores, iou_threshold)

    # Keep the detections based on NMS filtering
    kept_boxes = boxes[keep_inds]
    kept_scores = scores[keep_inds]
    kept_labels = labels[keep_inds]

    # Filter the detections that passed NMS
    kept_detections = torch.cat((kept_boxes, kept_scores.unsqueeze(1), kept_labels.unsqueeze(1)), dim=1)
    detections_nms = [kept_detections]

    return detections_nms



def calculate_centroid(tl_x, tl_y, w, h):
    """
    Calculate the x,y centre of the bounding boxes from top-left coordinates and width and height of frames
    :param tl_x:
    :param tl_y:
    :param w:
    :param h:
    :return:
    """
    mid_x = int(tl_x + w / 2)
    mid_y = int(tl_y + h / 2)
    return mid_x, mid_y


def convert_history_to_dict(track_history):
    """
    Convert the tracking history to a dictionary for access within ByteTrack
    :param track_history:
    :return:
    """
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
    """
    Plot tracking bounding box for each object when ByteTrack is running.

    :param image:
    :param track_history:
    :return:
    """
    obj_ids, tlwhs, class_ids = track_history[-1]
    history_dict = convert_history_to_dict(track_history)

    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

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


def transform_detection_output(detections, classes, detect_threshold):
    """
    Convert the detection output of Faster R-CNN to the format used by ByteTrack.

    :param detect_threshold:
    :param classes:
    :param detections:
    :return: outputs - tensorized detections
    """
    boxes = detections['boxes'].cpu().numpy()
    labels = detections['labels'].cpu().numpy()
    scores = detections['scores'].cpu().numpy()
    lbl_mask = np.isin(labels, classes) & (scores > detect_threshold)
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

def convert_frcnn_detections(output, classes):
    """
    Convert faster R CNN model to OCSort input
    :param output:
    :return:
    """
    # Extract components from the Faster R-CNN output and move to CPU
    boxes = output['boxes'].cpu().numpy()  # Shape: [N, 4]
    labels = output['labels'].cpu().numpy()  # Shape: [N]
    scores = output['scores'].cpu().numpy()  # Shape: [N]

    # Filter detections based on the given classes
    lbl_mask = np.isin(labels, classes)

    boxes = boxes[lbl_mask]
    labels = labels[lbl_mask]
    scores = scores[lbl_mask]

    # Convert filtered results back to tensors
    boxes = torch.tensor(boxes)
    labels = torch.tensor(labels)
    scores = torch.tensor(scores)

    # Concatenate the tensors to form the desired output
    converted = [torch.cat((boxes, scores.unsqueeze(1), labels.unsqueeze(1)), dim=1)]

    return converted



def save_detections(filedir, frame_number, detections, classes, detect_threshold=None):

    boxes = detections['boxes'].cpu().numpy()
    labels = detections['labels'].cpu().numpy()
    scores = detections['scores'].cpu().numpy()

    lbl_mask = np.isin(labels, classes)

    scores = scores[lbl_mask]
    labels = labels[lbl_mask]
    boxes = boxes[lbl_mask]

    outputs = []
    for i, box in enumerate(boxes):
        label = int(labels[i])  # Ensure labels are converted to Python int
        score = float(scores[i])  # Convert Numpy float32 to Python float
        xmin, ymin, xmax, ymax = map(float, box)  # Convert all box coordinates to Python float
        output = [xmin, ymin, xmax, ymax, score, label]
        outputs.append(output)

    # Ensure the output directory exists
    os.makedirs(filedir, exist_ok=True)

    # Create a unique file path for the frame number
    filepath = os.path.join(filedir, f"{frame_number}.csv")

    # Only write to the file if it does not already exist
    if not os.path.exists(filepath):
        with open(filepath, mode='w', newline='') as file:  # 'w' ensures the file is created
            writer = csv.writer(file)
            # Write the header
            writer.writerow(['xmin', 'ymin', 'xmax', 'ymax', 'score', 'label'])
            # Write each detection output
            writer.writerows(outputs)


