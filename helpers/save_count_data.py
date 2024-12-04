import csv
import os
import time

"""
This script saves count data in a format that is needed for analysis
"""
def create_count_files(args):
    """
    This function create count files for a video inference.
    The file name would consist of the camera or footage name + the datetime when inferences is produced
    :param args:
    :return:
    """
    live = args.live
    current_time = time.localtime()
    formatted_time = time.strftime("%Y_%m%d_%H%M%S", current_time)

    if not live:
        video_name = args.input_video.split('/')[-1].split('.')[0]
        date_video_name = video_name + "_" + formatted_time
    else:
        date_video_name = "cam1_" + formatted_time

    filename = f"{date_video_name}_counts.csv"
    filepath = os.path.join('./saved_counts/', filename)

    return filepath



def save_count_data(args, filepath, region_counts, direction, class_id, track_id, frame_number):
    """
    This function saves the counts into the file created by the function above

    :param args:
    :param filepath:
    :param region_counts:
    :param direction:
    :param track_id:
    :param class_id:
    :param frame_number:
    :return:
    """
    live = args.live
    if not live:
        video_name = args.input_video.split('/')[-1].split('.')[0]
        timestamp = str(video_name)+str(frame_number*30)
    else:
        current_time = time.localtime()
        formatted_time = time.strftime("%Y_%m%d_%H%M%S", current_time)
        milliseconds = int(time.time() * 1000) % 1000
        timestamp = f"{formatted_time}_{milliseconds:03d}"


    # Collect data to write
    data_to_write = []
    item = region_counts[class_id]  # Directly access the item using class_id
    for line_id, count in enumerate(item):  # Iterate over the specific item's counts
        data_to_write.append([
            timestamp,  # Use generated timestamp
            class_id,
            track_id,
            line_id,
            direction[track_id],  # Access direction through track_id
            count
        ])



    # Write the data to a CSV file
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write header only if the file does not exist or is empty
        if file.tell() == 0:
            writer.writerow(["timestamp", "class_id", "track_id", "line_id", "direction", "count"])
        writer.writerows(data_to_write)  # Write data rows

