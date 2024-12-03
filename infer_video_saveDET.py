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
    # main_tracker = ByteTracker(args)


    # Define the line (start and end points)
    #todo read in lines from svg file
    #todo create some way of pairing the lines into A and B hoses
    #todo function which reads lines from svg or csv, and outputs arrays of line_start and line_end points,


    #(630, 200), (300, 450)

    trackers = [BYTETracker(ByteTrackArgument) for _ in range(14)]
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

    # previous_side = None
    # start_point_normalized = ((line_start[0] / frame_width), (line_start[1] / frame_height))
    # end_point_normalized = ((line_end[0] / frame_width), (line_end[1] / frame_height))

    # global cross_product
    # global current_side
    # global previous_side
    # previous_side = [[0 for _ in range(len(lines_end))] for _ in range(10000)]
    # current_side = [[0 for _ in range(len(lines_end))] for _ in range(10000)]
    # cross_product = [[0 for _ in range(len(lines_end))] for _ in range(10000)]
    region_counts = None

    line_counter = LineCounter(args.lines_data)

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


        # img_height, img_width = detections_bytetrack[0].boxes.orig_shape
        # detections_bytetrack = detections_bytetrack[0].boxes.boxes

        # all_tlwhs = []
        # all_ids = []
        # all_classes = []
        # detection_output_xml = {'trackIDs': [], 'boxes': [], 'labels': [], 'scores': []}
        #
        # for class_id, tracker in enumerate(trackers):
        #     detections_bytetrack = np.array(detections_bytetrack)
        #     class_outputs = detections_bytetrack[detections_bytetrack[:, 5] == class_id][:, :5]
        #     if class_outputs is not None:
        #         online_targets = tracker.update(class_outputs)
        #         online_tlwhs = []
        #         online_ids = []
        #         online_scores = []
        #         online_classes = [class_id] * len(online_targets)
        #         for t in online_targets:
        #             # tracker box coordinates are given in pixels, not normalised
        #             tlwh = t.tlwh
        #             tlbr = t.tlbr
        #             tid = t.track_id
        #             vertical = tlwh[2] / tlwh[3] > ByteTrackArgument.aspect_ratio_thresh
        #             if tlwh[2] * tlwh[3] > ByteTrackArgument.min_box_area and not vertical:
        #                 online_tlwhs.append(tlwh)
        #
        #                 # use the trackings' output bbox locations to detect objects, with
        #                 region_counts = line_counter.perform_count_line_detections(class_id, tid, tlbr)
        #
        #                 # Get the xml output for saving into annotation file
        #                 tlbr_box = [tlbr[0], tlbr[1], tlbr[2], tlbr[3]]
        #                 detection_output_xml['trackIDs'].append(tid)
        #                 detection_output_xml['boxes'].append(tlbr_box)
        #                 detection_output_xml['labels'].append(class_id)
        #                 detection_output_xml['scores'].append(t.score)
        #
        #                 online_ids.append(tid)
        #                 online_scores.append(t.score)
        #                 tlwh_box = (tlwh[0], tlwh[1], tlwh[2], tlwh[3])
        #
        #                 results.append(
        #                     # frame_id, track_id, tl_x, tl_y, w, h, score = obj_prob * class_prob, class_idx, dummy, dummy, dummy
        #                     f"{frame_count},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f}, {class_id}, -1,-1,-1\n"
        #                 )
        #
        #         all_tlwhs += online_tlwhs
        #         all_ids += online_ids
        #         all_classes += online_classes
        #
        # if len(history) < 30:
        #     history.append((all_ids, all_tlwhs, all_classes))
        # else:
        #     history.popleft()
        #     history.append((all_ids, all_tlwhs, all_classes))
        #
        # if len(all_tlwhs) > 0:
        #     online_im = plot_tracking(
        #         frame, history
        #     )
        # else:
        #     online_im = frame

        ################################################################################################################
        ################################################################################################################
        ################################################################################################################

        online_im, region_counts = main_tracker.startTrack(frame, detections_bytetrack, frame_count)

        # Extract original frame
        # extract_frame(saved_frame_dir, frame, frame_count, video_name)

        # #Plot bounding boxes
        # annotated_frame = draw_boxes(detections, frame, args.cls, 0.9)

        # Saved annotated vehicles from the image.
        standardize_to_txt(detections, args.classes_to_track, args.score_threshold, frame_count, video_name, frame_width, frame_height)
        # standardize_to_xml(detections, args.cls, frame_count, video_name, frame_width, frame_height)


        # Extract annotated frame
        # extract_frame(saved_frame_dir, annotated_frame, frame_count, video_name)

        frame_count += 1

        print(f"Frame {frame_count}/{frames}",
                f"Detection FPS: {det_fps:.1f}")

        cv2.putText(
            online_im,
            f"FPS: {det_fps:.1f}",
            (int(20), int(40)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )

        count = process_count(region_counts)
        
        cv2.putText(
            online_im,
            f"Count of cars:  {count[3] if region_counts is not None else 0}", #{region_counts:.1f}
            (int(20), int(60)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 250, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )

        out.write(online_im)

        if args.show:
            # Display or save output frame.
            cv2.imshow("Output", online_im)
            # Press q to quit.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # frame_count+=1

    # Release resources.
    cap.release()
    out.release()  # Ensure the VideoWriter is released.
    cv2.destroyAllWindows()


if __name__ == '__main__':

    args = setup_argument_parser().parse_args()
    history = deque()

    infer_video(args)

