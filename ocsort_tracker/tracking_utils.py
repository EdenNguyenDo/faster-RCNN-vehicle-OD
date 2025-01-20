import cv2
import numpy as np
import os
import csv
from config.VEHICLE_CLASS import VEHICLE_CLASSES
import torch
from torchvision.ops.boxes import box_area, nms


def create_track_file(output_dir, track_id,video_path = None, detection_folder = None):

    if video_path is not None and detection_folder is None:
        full_path = os.path.join(output_dir, "track_results", video_path.split('/')[-2], video_path.split('/')[-1].split(".")[0])
    elif detection_folder is not None and video_path is None:
        full_path = os.path.join(output_dir, "track_results", detection_folder.split('/')[-2])
    else:
        full_path = os.path.join(output_dir, "track_results")

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


