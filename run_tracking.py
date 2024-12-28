import argparse
import os
import json
import numpy as np
import torch
import yaml
import csv
from ByteTrack.bytetrackCustom.bytetrack_args import ByteTrackArgument
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from helpers.save_count_data import get_base_directory


def setup_byteTracker(yaml_file):
    """Set up the argument parser for video inference."""

    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description='Arguments for inference using fine-tuned model')

    parser.add_argument(
        "--trackers",
        default = config.get('number_of_trackers'))

    parser.add_argument(
        "--detection_folder",
        default = config.get('detection_folder'))

    parser.add_argument(
        "--detection_folder_list",
        default = config.get('detection_folder_list'))

    parser.add_argument(
        "--aspect_ratio_thresh",
        default = config.get('aspect_ratio_thresh'))

    parser.add_argument(
        "--min_box_area",
        default = config.get('min_box_area'))

    return parser



def create_track_file(folder_path, track_id):
    base_dir = get_base_directory()

    dir_name = folder_path.split('/')[-3]

    parent = os.path.join(base_dir, 'tracks')

    full_path = os.path.join(parent, dir_name, folder_path.split('/')[-2])

    # Ensure the directory exists
    os.makedirs(full_path, exist_ok=True)

    filename = f"{track_id}.csv"

    filepath = os.path.join(full_path, filename)

    return filepath



def read_detections_from_csv_folder(folder_path):
    """
    Reads detections from multiple CSV files in the specified folder.
    Each CSV file corresponds to detections for a single frame.

    Args:
        folder_path (str): Path to the folder containing CSV files.

    Returns:
        dict: A dictionary where keys are frame numbers (int) derived from CSV filenames,
        and values are lists of detections (xmin, ymin, xmax, ymax, score, label).
    """
    detection_bytetrack = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            try:
                # Extract frame number from the file name (assumes frame_<frame_number>.csv format)
                frame_number = int(file_name.split('.')[0])
                file_path = os.path.join(folder_path, file_name)

                # Initialize the list of detections for this frame
                detection_bytetrack[frame_number] = []

                with open(file_path, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    next(csv_reader, None)  # Skip the first row (header)

                    for row in csv_reader:
                        try:
                            # Each row is expected to contain: xmin, ymin, xmax, ymax, score, label
                            # time = float(row[0])
                            xmin = float(row[0])
                            ymin = float(row[1])
                            xmax = float(row[2])
                            ymax = float(row[3])
                            score = float(row[4])
                            label = int(row[5])
                            detection_bytetrack[frame_number].append([xmin, ymin, xmax, ymax, score, label])
                        except (ValueError, IndexError):
                            print(f"Skipping invalid row in file {file_name}: {row}")

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    return detection_bytetrack, folder_path



def run_byteTrack(args):
    """
    Runs ByteTrack for each frame and saves tracking results into track files.

    Args:
        args: Command-line arguments containing configuration and parameters.

    Returns:
        None
    """
    all_tlwhs = []
    all_ids = []
    all_classes = []
    results = []

    for folder_path in args.detection_folder_list:


        trackers = [BYTETracker(ByteTrackArgument) for _ in range(args.trackers)]


        # Read detections from the specified folder
        detections_bytetrack, current_folder = read_detections_from_csv_folder(folder_path)

        # Process each frame (frame numbers may be non-continuous)
        for frame_number in sorted(detections_bytetrack.keys()):
            detections_list = detections_bytetrack[frame_number]

            if len(detections_list)>0:

                # Convert detections_list to a NumPy array
                detections_array = np.array(detections_list)

                for class_id, tracker in enumerate(trackers):
                    # Filter detections for the current class ID
                    class_outputs = detections_array[detections_array[:, 5] == class_id][:, :5]
                    if len(class_outputs) > 0:
                        online_targets = tracker.update(class_outputs)
                        online_tlwhs = []
                        online_ids = []
                        online_scores = []
                        online_classes = [class_id] * len(online_targets)

                        for t in online_targets:
                            # Tracker bounding box coordinates are pixels (not normalized)
                            tlwh = t.tlwh
                            tlbr = t.tlbr
                            tid = t.track_id

                            vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                                online_tlwhs.append(tlwh)

                                # Save tracking data into per-track CSV files
                                track_file = create_track_file(current_folder, tid)

                                with open(track_file, 'a', newline='') as csvfile:
                                    csv_writer = csv.writer(csvfile)
                                    if csvfile.tell() == 0:
                                        csv_writer.writerow(["frame_number","track_id", "class_id", "score", "x_topleft", "y_topleft", "width", "height"])
                                    csv_writer.writerow([frame_number, tid, class_id, round(t.score,2), round(tlwh[0],1), round(tlwh[1],1), round(tlwh[2],1), round(tlwh[3],1)])



                                online_ids.append(tid)
                                online_scores.append(t.score)


if __name__ == '__main__':
    args = setup_byteTracker('./tracks/track_config.yaml').parse_args()

    run_byteTrack(args)