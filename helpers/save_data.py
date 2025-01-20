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
        # Always set the base directory to the persistent storage location
        home_dir = os.path.expanduser("~")
        base_dir = os.path.join(home_dir, "COUNT_FILES", "tupi-ai-realtime")
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Use script's original directory

    return base_dir




def create_log_files(live, video, video_out_dir):
    """
    This function creates count files for a video inference.
    The file name would consist of the camera or footage name + the datetime when inferences are produced.
    :param args:
    :return:
    """
    current_time = time.localtime()
    formatted_time = time.strftime("%Y_%m%d_%H%M%S", current_time)
    base_dir = get_base_directory()

    if not live:
        video_name = video.split('/')[-1].split('.')[0]
        video_dir = video.split('/')[-2]
        date_video_name = video_name + "_" + formatted_time
    else:
        video_name = "cam1"
        video_dir = "live_webcam"
        date_video_name = video_name + "_" + formatted_time

    # Construct the path for saved counts in the application's directory
    saved_log_dir = os.path.join(base_dir, 'saved_inferred_log', video_dir)
    directory_path = os.path.join(video_out_dir,"inferred_videos", video_dir)

    # Ensure the directory exists
    os.makedirs(directory_path, exist_ok=True)

    filename = f"{date_video_name}_log.csv"

    filepath = os.path.join(directory_path, filename)

    return filepath, directory_path


def create_detection_directory(live, video, output_dir):
    """
    This function creates detection files for a video inference in JSON format.
    The file name would consist of the camera or footage name + the datetime when inferences are produced.
    :param args:
    :return:
    """
    current_time = time.localtime()
    formatted_time = time.strftime("%Y_%m%d_%H%M%S", current_time)
    base_dir = output_dir


    if not live:
        video_name = video.split('/')[-1].split('.')[0]
        video_dir = video.split('/')[-2]
        date_video_name = video_name + "_" + formatted_time
    else:
        video_name = "live_cam"
        video_dir = video.split('/')[-2]
        date_video_name = video_name + "_" + formatted_time

    # Construct the path for saved counts in the application's directory
    # saved_det_dir = os.path.join(base_dir, 'saved_all_filtered_detection', video_dir)

    saved_raw_det_dir = os.path.join(base_dir, 'raw_detections', video_dir)

    #det_directory_path = os.path.join(saved_det_dir, video_name)
    raw_directory_path = os.path.join(saved_raw_det_dir, video_name)

    # Ensure the directory exists
    #os.makedirs(det_directory_path, exist_ok=True)
    os.makedirs(raw_directory_path, exist_ok=True)

    #det_filename = f"{date_video_name}_detection"
    raw_filename = f"{date_video_name}_raw_detection"

    #det_filepath = os.path.join(det_directory_path, det_filename)
    raw_filepath = os.path.join(raw_directory_path, raw_filename)

    return raw_filepath




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



def save_log(live, video, filepath, class_id, track_id, tlwh = None):
    """
    This function saves the counts_by_lines into the file created by the function above.
    :param video:
    :param live:
    :param tlwh:
    :param tlbr:
    :param args:
    :param filepath:
    :param class_id:
    :param track_id:
    :param frame_number:
    :return:
    """
    date = datetime.now().strftime("%Y-%m-%d")

    if not live:
        #video_name = video.split('/')[-1].split('.')[0]
        #time_mili = f"{video_name}_{round(frame_number/30, 3)}"
        time_mili = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Strip milliseconds for better readability
    else:
        time_mili = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Strip milliseconds for better readability

    # Collect data to write
    data_to_write = []

    data_to_write.append([
        date,  # Use generated timestamp
        time_mili,
        class_id,
        track_id,
        tlwh[0], tlwh[1], tlwh[2], tlwh[3]
    ])

    # Write the data to a CSV file
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header only if the file does not exist or is empty
        if file.tell() == 0:
            writer.writerow(["date", "timestamp", "class_id", "track_id", "x", "y", "w", "h"])
        writer.writerows(data_to_write)  # Write data rows

