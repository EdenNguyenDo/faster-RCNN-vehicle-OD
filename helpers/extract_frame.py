import os
import cv2
"""
This script save the infer image in their corresponding 
"""

def extract_frame(output_folder, frame, frame_no, video_name):

    # Create the output directory if it doesn't exist
    output_dir = os.path.join(output_folder, video_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"{video_name}_{frame_no}.jpg"
    frame_filename = os.path.join(output_dir, filename)
    cv2.imwrite(frame_filename, frame)

