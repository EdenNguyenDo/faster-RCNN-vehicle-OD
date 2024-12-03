"""This file help extract the trajectory by deep SORT through each frame"""

import argparse
from collections import defaultdict

import cv2
import torch
import torchvision
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
from torchvision.transforms import ToTensor
from config.coco_classes import COCO_91_CLASSES
from helpers.helper import Helper
from infer_video_DS import COLORS, device
import json
import matplotlib.pyplot as plt


model = getattr(torchvision.models.detection, "fasterrcnn_resnet50_fpn")(weights='DEFAULT')

def infer_video(args):
    print(f"Tracking: {[COCO_91_CLASSES[3]]}")
    print("Detector: fasterrcnn_resnet50_fpn")
    print("Re-ID embedder: mobilenet")

    # Load model.
    model.eval().to(device)

    # Initialize a SORT tracker object.
    tracker = DeepSort(30)

    VIDEO_PATH = args.input_videos
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_fps = int(cap.get(5))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    frame_count = 0  # To count total frames.
    total_fps = 0  # To get the final frames per second.

    # Create a dictionary to store trajectories
    # Create dictionaries to store actual and predicted trajectories
    actual_trajectories = defaultdict(list)
    predicted_trajectories = defaultdict(list)

    while cap.isOpened():
        # Read a frame
        ret, frame = cap.read()
        if ret:
            if args.imgsz != None:
                resized_frame = cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    (args.imgsz, args.imgsz)
                )
            else:
                resized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert frame to tensor and send it to device (cpu or cuda).
            frame_tensor = ToTensor()(resized_frame).to(device)

            start_time = time.time()
            # Feed frame to model and get detections.
            with torch.no_grad():
                detections = model([frame_tensor])[0]

            # Convert detections to Deep SORT format.
            detections = Helper.convert_detections(detections, args.threshold, args.cls)

            # Update tracker with detections.
            tracks = tracker.update_tracks(detections, frame=frame)

            # Store trajectories
            for track in tracks:
                object_id = track.track_id
                bbox = track.to_tlbr()  # Get bounding box coordinates

                # Actual position: center of the bounding box
                actual_center_x = (bbox[0] + bbox[2]) / 2
                actual_center_y = (bbox[1] + bbox[3]) / 2


                # Add actual and predicted positions to their respective lists
                actual_trajectories[object_id].append({
                    'frame': frame_count,
                    'x': actual_center_x,
                    'y': actual_center_y,
                    'type': 'actual'
                })
                predicted_trajectories[object_id].append({
                    'frame': frame_count,
                    'x': predicted_center_x,
                    'y': predicted_center_y,
                    'type': 'predicted'
                })

            # Annotate frame with tracking information
            if len(tracks) > 0:
                frame = Helper.annotate(
                    tracks,
                    frame,
                    resized_frame,
                    frame_width,
                    frame_height,
                    COLORS
                )

            # Increment frame count.
            frame_count += 1
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            total_fps += fps
            print(f"Frame {frame_count}/{frames}, FPS: {fps:.1f}")

        else:
            break

    # After the video is processed, you can save the trajectories to a file (e.g., CSV, JSON)
    save_trajectories(trajectories)
    plot_traject()

def save_trajectories(trajectories):
    """Save the extracted trajectories to a file."""
    print(trajectories)
    # Convert NumPy ndarrays to lists
    serializable_trajectories = {k: [bbox.tolist() for bbox in v] for k, v in trajectories.items()}

    # Save trajectories to a JSON file
    with open("trajectories.json", "w") as f:
        json.dump(serializable_trajectories, f, indent=4)

    print("Trajectories saved to trajectories.json")

def plot_traject():


    # Load the trajectories from the saved JSON file
    with open("trajectories.json", "r") as f:
        trajectories = json.load(f)

    # Example of plotting the trajectory of a specific object (track_id=0)
    track_id = 0
    if track_id in trajectories:
        trajectory = trajectories[track_id]
        xs = [bbox[0] for bbox in trajectory]  # x1
        ys = [bbox[1] for bbox in trajectory]  # y1

        plt.plot(xs, ys, label=f"Track {track_id}")
        plt.scatter(xs, ys, label=f"Track {track_id} Points", s=10, color='red')

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.show()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments for inference using fine-tuned model')
    parser.add_argument(
        '--input_videos',
        default='input_videos/shoret.mp4',
        help='path to input video',
    )
    parser.add_argument(
        '--imgsz',
        default=None,
        help='image resize, 640 will resize images to 640x640',
        type=int
    )
    parser.add_argument(
        '--threshold',
        default=0.7,
        help='score threshold to filter out detections',
        type=float
    )
    parser.add_argument(
        '--cls',
        nargs='+',
        default=[3],
        help='which classes to track',
        type=int
    )

    args = parser.parse_args()
    infer_video(args)

