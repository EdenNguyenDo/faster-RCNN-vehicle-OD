import csv
from collections import deque

import numpy as np
import torch
import torchvision
import cv2
import os
import time
import argparse

from bytetrackCustom.bytetrack_main import ByteTracker
from config.VEHICLE_CLASS import VEHICLE_CLASSES
from config.argument_config import setup_argument_parser
from helpers.line_counter import LineCounter, process_count, read_lines_from_csv
from helpers.save_count_data import create_count_files
from helpers.standardize_detections import standardize_to_txt
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor
from config.coco_classes import COCO_91_CLASSES
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from bytetrackCustom.bytetrack_args import ByteTrackArgument
from bytetrackCustom.bytetrack_utils import transform_detection_output, plot_tracking, count_tracks, cross_product_line

"""
Running inference with object tracking with faster R-CNN model

- This script run inference with video and plot bounding boxes
- It saves annotated video while saving the model annotation output for each frame into txt and xml format for other purposes.
- The script also saved annotated frames into folder for further assessment. 
- This script is able to run object tracking using ByteTrack algorithm. 
"""

np.random.seed(3101)

OUT_DIR = 'output_frcnn-ds'
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLORS = np.random.randint(0, 255, size=(len(COCO_91_CLASSES), 3))



def load_model():
    """
    This function load model faster R-CNN with trained weight file
    """
    faster_rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                             min_size=600,
                                                                             max_size=1000,
                                                                             box_score_thresh=0.22,
                                                                             )
    faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
        faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
        num_classes=7)

    faster_rcnn_model.eval()
    faster_rcnn_model.to(device)

    faster_rcnn_model.load_state_dict(torch.load(
        '../vtod/tv_frcnn_r50fpn_faster_rcnn_vtod.pth',
        map_location=device))

    return faster_rcnn_model


results = []
def infer_video(args):
    """
    This function runs inference using loaded model frame by frame while saving the annotation of each frame into a predefined format
    Then each individual frame is aggregated to create an annotated video.
    """

    # Define the line (start and end points)
    #todo read in lines from svg file
    #todo create some way of pairing the lines into A and B hoses
    #todo function which reads lines from svg, and outputs arrays of line_start and line_end points,

    args.live = False
    count_filepath = create_count_files(args)

    #(630, 200), (300, 450)

    main_tracker = ByteTracker(args)


    print(f"Tracking: {[COCO_91_CLASSES[idx] for idx in args.classes_to_track]}")
    print(f"Detector: {args.pretrained_model}")
    print(f"Re-ID embedder: {args.embedder}")

    # Load model.
    model = getattr(torchvision.models.detection, args.pretrained_model)(weights='DEFAULT')
    #model = load_model()

    # Set model to evaluation mode.
    model.eval().to(device)


    VIDEO_PATH = args.input_video
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_fps = int(cap.get(5))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = VIDEO_PATH.split(os.path.sep)[-1].split('.')[0].split("/")[-1]
    saved_frame_dir = 'inference_dataset/images/'
    saved_annotated_frame_dir = 'inference_dataset/annotated_images'

    # Define codec and create VideoWriter object.
    out = cv2.VideoWriter(
        f"{OUT_DIR}/{video_name}_{args.pretrained_model}_{args.embedder}.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'), frame_fps,
        (frame_width, frame_height)
    )


    frame_count = 0  # To count total frames.

    # # Get the video's FPS (frames per second)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    #
    # # Set the frame interval to capture one frame every 3 seconds
    # frame_interval = int(fps * 3)  # For 30 FPS, this equals 90 frames being skipped before one is saved


    class_count_dict = {"human": 0,
                        "vehicle type 1 - bicycle": 0,
                        "vehicle type 1 - car": 0,
                        "vehicle type 3": 0,
                        "vehicle type 5": 0}

    line_counter = LineCounter(args.lines_data)

    try:
        completed_successfully = True  # Initialize the flag at the start of the try block.

        while cap.isOpened():

            # Read a frame
            ret, frame = cap.read()

            if not ret:
                break

            # draw user defined lines on page
            line_counter.draw_lines(frame)

            # if frame_count % frame_interval ==0:

            if args.img_size is not None:
                resized_frame = cv2.resize(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    (args.imgsz, args.imgsz)
                )
            else:
                resized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert frame to tensor and send it to device (cpu or cuda).
            frame_tensor = ToTensor()(resized_frame).to(device)

            # Feed frame to model and get detections - these are in xyxy format, not normalised.
            det_start_time = time.time()
            with torch.no_grad():
                detections = model([frame_tensor])[0]
            det_end_time = time.time()
            det_fps = 1 / (det_end_time - det_start_time)

            ################################################################################################################
            ########################################## Byte Track Integration ##############################################
            ################################################################################################################

            # Transform detection output to ones to be used by bytetracker - xyxy px,
            detections_bytetrack = transform_detection_output(detections, args.classes_to_track)


            if len(detections_bytetrack)>0:
            # if detections_bytetrack.dim() > 1:
                online_im, region_counts = main_tracker.startTrack(frame, detections_bytetrack, frame_count, count_filepath)
                class_count_dict = process_count(region_counts, args.classes_to_track)
            else:
                online_im = frame

            ################################################################################################################
            ################################################################################################################
            ################################################################################################################


            # Extract original frame
            # extract_frame(saved_frame_dir, frame, frame_count, video_name)

            # #Plot bounding boxes
            # annotated_frame = draw_boxes(detections, frame, args.cls, 0.9)

            # Saved annotated vehicles from the image.
            # standardize_to_txt(detections, args.classes_to_track, args.score_threshold, frame_count, video_name, frame_width, frame_height)
            # standardize_to_xml(detections, args.cls, frame_count, video_name, frame_width, frame_height)


            # Extract annotated frame
            # extract_frame(saved_frame_dir, annotated_frame, frame_count, video_name)


            cv2.putText(
                online_im,
                f"FPS: {det_fps:.1f}",
                (int(20), int(40)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255),
                thickness=1,
                lineType=cv2.LINE_AA
            )

            # Display each class with its count
            y_position = 80  # Starting y-position for text display
            for class_name, count in class_count_dict.items():
                cv2.putText(
                    online_im,
                    f"{class_name}: {count}",
                    (int(20), int(y_position)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,  # Smaller font size
                    color=(255, 0, 0),  # Blue color: (B, G, R)
                    thickness=1,
                    lineType=cv2.LINE_AA
                )
                y_position += 20  # Move down for the next class

            frame_count += 1

            print(f"Frame {frame_count}/{frames}",
                  f"Detection FPS: {det_fps:.1f}")

            out.write(online_im)

            if args.show:
                # Display or save output frame.
                cv2.imshow("Output", online_im)
                # Press q to quit.
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    completed_successfully = False
                    break

            # frame_count+=1

    except Exception as e:
        completed_successfully = False  # Set the flag to False if an exception is caught.
        print(f"An interruption occurred: {e}")

    finally:
        # Release resources.
        cap.release()
        out.release()  # Ensure the VideoWriter is released.

        cv2.destroyAllWindows()

        # Only write the completion message if processing was completed successfully.
        if completed_successfully:
            with open(count_filepath, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["Counting and saving completed smoothly without interruption"])


if __name__ == '__main__':

    args = setup_argument_parser().parse_args()
    history = deque()

    infer_video(args)

