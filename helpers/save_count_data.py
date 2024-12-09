import csv
import sys
import os
import time
from datetime import datetime

"""
This script saves count data in a format that is needed for analysis
"""

def get_base_directory():
    """
    Get the base directory where the application is running.
    """
    if getattr(sys, 'frozen', False):  # Check if running as a bundled executable
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Use script's original directory
    return base_dir

def create_count_files(args):
    """
    This function creates count files for a video inference.
    The file name would consist of the camera or footage name + the datetime when inferences are produced.
    :param args:
    :return:
    """
    live = args.live
    current_time = time.localtime()
    formatted_time = time.strftime("%Y_%m%d_%H%M%S", current_time)
    base_dir = get_base_directory()

    if not live:
        video_name = args.input_video.split('/')[-1].split('.')[0]
        date_video_name = video_name + "_" + formatted_time
    else:
        video_name = "cam1"
        date_video_name = video_name + "_" + formatted_time

    # Construct the path for saved counts in the application's directory
    saved_counts_dir = os.path.join(base_dir, 'saved_counts')
    directory_path = os.path.join(saved_counts_dir, video_name)

    # Ensure the directory exists
    os.makedirs(directory_path, exist_ok=True)

    filename = f"{date_video_name}_counts.csv"
    json_filename = f"{date_video_name}_total_counts.json"

    filepath = os.path.join(directory_path, filename)
    json_filepath = os.path.join(directory_path, json_filename)

    return filepath, json_filepath

def save_count_data(args, filepath, region_counts, direction, class_id, track_id, frame_number):
    """
    This function saves the counts_by_lines into the file created by the function above.
    :param args:
    :param filepath:
    :param region_counts:
    :param direction:
    :param class_id:
    :param track_id:
    :param frame_number:
    :return:
    """
    live = args.live
    date = datetime.now().strftime("%Y-%m-%d")

    if not live:
        video_name = args.input_video.split('/')[-1].split('.')[0]
        time_mili = f"{video_name}_{round(frame_number/30, 3)}"
    else:
        time_mili = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Strip milliseconds for better readability

    # Collect data to write
    data_to_write = []
    counts_by_lines = region_counts[class_id]  # Access the item using class_id
    accumulate_count = sum(counts_by_lines)
    data_to_write.append([
        date,  # Use generated timestamp
        time_mili,
        class_id,
        track_id,
        direction,
        accumulate_count
    ])

    # Write the data to a CSV file
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header only if the file does not exist or is empty
        if file.tell() == 0:
            writer.writerow(["date", "time", "class_id", "track_id", "direction", "count"])
        writer.writerows(data_to_write)  # Write data rows
