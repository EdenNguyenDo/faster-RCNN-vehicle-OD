import csv
import os
import pandas as pd
import h5py
import numpy as np


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
    detection_dict = {}

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            try:
                # Extract frame number from the file name (assumes frame_<frame_number>.csv format)
                frame_number = int(file_name.split('.')[0])

                file_path = os.path.join(folder_path, file_name)

                # Initialize the list of detections for this frame
                detection_dict[frame_number] = []

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
                            detection_dict[frame_number].append([xmin, ymin, xmax, ymax, score, label])
                        except (ValueError, IndexError):
                            print(f"Skipping invalid row in file {file_name}: {row}")

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    return detection_dict


def read_detections_from_h5(h5_filepath):
    """
    Reads detections from an HDF5 file.

    Args:
        h5_filepath (str): Path to the HDF5 file.

    Returns:
        dict: A dictionary where keys are frame numbers (int) and values are numpy arrays of detections.
    """
    detections = {}
    with h5py.File(h5_filepath, 'r') as h5file:
        for frame_key in h5file.keys():
            frame_number = int(frame_key)  # Convert string keys back to integers
            detections[frame_number] = h5file[frame_key][:]
    return detections


def read_detections_from_parquet(parquet_path):
    """
    Reads detections from a Parquet file and returns them sorted by frame number.

    Args:
        parquet_path (str): Path to the Parquet file.

    Returns:
        dict: A dictionary where keys are frame numbers (int) and values are DataFrames of detections.
    """
    df = pd.read_parquet(parquet_path)
    df = df.sort_values(by="frame_number")  # Ensure rows are sorted by frame number

    # Group by frame number and return as a dictionary
    detections_by_frame = {
        frame_number: group.drop(columns=["frame_number"]).to_numpy()
        for frame_number, group in df.groupby("frame_number")
    }
    return detections_by_frame
