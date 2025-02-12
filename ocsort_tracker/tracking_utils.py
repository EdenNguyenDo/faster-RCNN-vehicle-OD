import cv2
import numpy as np
import os
import csv
from config.VEHICLE_CLASS import VEHICLE_CLASSES
import torch
from torchvision.ops.boxes import box_area, nms

from config.coco_classes import COCO_91_CLASSES


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

    # Handle empty detections
    if boxes.shape[0] == 0:  # No detections
        outputs = []
    else:
        # Filter detections by the specified classes
        lbl_mask = np.isin(labels, classes)
        scores = scores[lbl_mask]
        labels = labels[lbl_mask]
        boxes = boxes[lbl_mask]

        # Prepare outputs
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



COLORS = np.random.randint(0, 255, size=(len(COCO_91_CLASSES), 3))


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


def save_tracks(track_file, frame_number, tid, class_id, tlwh):
    with open(track_file, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            csv_writer.writerow(
                ["frame_number", "track_id", "class_id", "score", "x_topleft", "y_topleft",
                 "width", "height"])

        csv_writer.writerow([
            frame_number, int(tid), int(class_id), 0,
            round(tlwh[0], 1), round(tlwh[1], 1),
            round(tlwh[2], 1), round(tlwh[3], 1)
        ])


















































# def save_detections_h5(filedir, frame_number, detections, classes, detect_threshold=None):
#     boxes = detections['boxes'].cpu().numpy()
#     labels = detections['labels'].cpu().numpy()
#     scores = detections['scores'].cpu().numpy()
#
#     lbl_mask = np.isin(labels, classes)
#     scores = scores[lbl_mask]
#     labels = labels[lbl_mask]
#     boxes = boxes[lbl_mask]
#
#     # Prepare outputs as a structured numpy array
#     outputs = []
#     for i, box in enumerate(boxes):
#         label = int(labels[i])
#         score = float(scores[i])
#         xmin, ymin, xmax, ymax = map(float, box)
#         outputs.append([xmin, ymin, xmax, ymax, score, label])
#
#     outputs = np.array(outputs, dtype=np.float32)
#
#     # Ensure the output directory exists
#     os.makedirs(filedir, exist_ok=True)
#
#     # HDF5 file path for the video
#     h5_path = os.path.join(filedir, f"detections.h5")
#
#     # Append detections to the HDF5 file
#     with h5py.File(h5_path, 'a') as h5file:
#         frame_key = str(frame_number)  # Use plain numeric keys as strings
#         if frame_key not in h5file:
#             h5file.create_dataset(frame_key, data=outputs, compression="gzip", chunks=True)
#
#
#
# def save_detections_parquet_optimized(filedir, frame_number, detections, classes, buffer_limit=1000, flush=False):
#     """
#     Saves frame detections to a Parquet file more efficiently by using a buffer.
#     Handles cases where there are no detections by saving empty rows.
#     Flushes remaining data in the buffer at the end of the video.
#
#     Args:
#         filedir (str): Directory to save the Parquet file.
#         frame_number (int): Frame number of the detections.
#         detections (dict): Detections for the frame.
#         classes (list): List of classes to include.
#         buffer_limit (int): Number of frames to buffer before writing to Parquet.
#         flush (bool): If True, flushes the buffer to the main Parquet file.
#     """
#     os.makedirs(filedir, exist_ok=True)
#
#     # Buffer file path (to accumulate data temporarily)
#     buffer_path = os.path.join(filedir, "buffer.parquet")
#     parquet_path = os.path.join(filedir, "p_detections.parquet")
#
#     # Define the schema (column names and dtypes)
#     schema = {
#         "frame_number": "int",
#         "xmin": "float",
#         "ymin": "float",
#         "xmax": "float",
#         "ymax": "float",
#         "score": "float",
#         "label": "float"
#     }
#
#     # If not flushing, process the current frame
#     if not flush:
#         if len(detections['labels']) > 0:
#             boxes = detections['boxes'].cpu().numpy()
#             labels = detections['labels'].cpu().numpy()
#             scores = detections['scores'].cpu().numpy()
#
#             lbl_mask = np.isin(labels, classes)
#             scores = scores[lbl_mask]
#             labels = labels[lbl_mask]
#             boxes = boxes[lbl_mask]
#
#             # Prepare detections DataFrame
#             records = []
#             for i, box in enumerate(boxes):
#                 label = int(labels[i])
#                 score = float(scores[i])
#                 xmin, ymin, xmax, ymax = map(float, box)
#                 records.append([frame_number, xmin, ymin, xmax, ymax, score, label])
#
#             df = pd.DataFrame(records, columns=schema.keys()).astype(schema)
#         else:
#             # Create an empty DataFrame for frames without detections
#             df = pd.DataFrame({
#                 "frame_number": [frame_number],
#                 "xmin": [np.nan],
#                 "ymin": [np.nan],
#                 "xmax": [np.nan],
#                 "ymax": [np.nan],
#                 "score": [np.nan],
#                 "label": [np.nan]
#             }).astype(schema)
#
#         # Append to buffer
#         if os.path.exists(buffer_path):
#             buffer_df = pd.read_parquet(buffer_path, engine="pyarrow")
#             buffer_df = pd.concat([buffer_df, df], ignore_index=True)
#         else:
#             buffer_df = df
#
#         # Write buffer back to temporary file
#         buffer_df.to_parquet(buffer_path, engine="pyarrow",compression="snappy", index=False)
#
#         # If buffer exceeds limit, flush to main Parquet file
#         if len(buffer_df) >= buffer_limit:
#             if os.path.exists(parquet_path):
#                 existing_df = pd.read_parquet(parquet_path, engine="pyarrow")
#                 combined_df = pd.concat([existing_df, buffer_df], ignore_index=True)
#             else:
#                 combined_df = buffer_df
#
#             # Write combined DataFrame back to the main file
#             combined_df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)
#
#             # Clear the buffer
#             os.remove(buffer_path)
#
#     # If flushing, write remaining buffer to the main Parquet file
#     else:
#         if os.path.exists(buffer_path):
#             buffer_df = pd.read_parquet(buffer_path, engine="pyarrow")
#             if os.path.exists(parquet_path):
#                 existing_df = pd.read_parquet(parquet_path, engine="pyarrow")
#                 combined_df = pd.concat([existing_df, buffer_df], ignore_index=True)
#             else:
#                 combined_df = buffer_df
#
#             # Write combined DataFrame back to the main file
#             combined_df.to_parquet(parquet_path, engine="pyarrow",compression="snappy", index=False)
#
#             # Clear the buffer
#             os.remove(buffer_path)
