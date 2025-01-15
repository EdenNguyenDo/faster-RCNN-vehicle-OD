import csv
import numpy as np
import torch
import torchvision
import cv2
import os
import time

from torchvision.ops import nms
from ByteTrack.bytetrackCustom.bytetrack_main import ByteTracker
from helpers.setup_infer_config import setup_argument_parser
from helpers.line_counter import LineCounter, process_count
from helpers.save_count_data import create_count_files, create_log_files, create_detection_directory
from torchvision.transforms import ToTensor
from config.coco_classes import COCO_91_CLASSES
from ByteTrack.bytetrackCustom.bytetrack_utils import transform_detection_output, save_detections

"""
Running inference with object tracking with faster R-CNN model

- This script run inference with video and plot bounding boxes
- It saves annotated video while saving the model annotation output for each frame into txt and xml format for other purposes.
- The script also saved annotated frames into folder for further assessment. 
- This script is able to run object tracking using ByteTrack algorithm. 
"""

np.random.seed(3101)
# OUT_DIR = 'output_frcnn-ds'
# os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
COLORS = np.random.randint(0, 255, size=(len(COCO_91_CLASSES), 3))
results = []


def infer(args):
    """
    This function runs inference using loaded model frame by frame while saving the annotation of each frame into a predefined format
    Then each individual frame is aggregated to create an annotated video.
    """

    print(f"Tracking: {[COCO_91_CLASSES[idx] for idx in args.classes_to_track]}")
    print(f"Detector: {args.pretrained_model}")
    print(f"Re-ID embedder: {args.embedder}")


    main_tracker = ByteTracker(args)
    line_counter = LineCounter(args.lines_data)


    # Load model.
    model = getattr(torchvision.models.detection, args.pretrained_model)(weights='DEFAULT')
    # Set model to evaluation mode.
    model.eval().to(device)


    '''
    If the video stream is realtime from config file
    '''
    if args.live:
        cap = cv2.VideoCapture(args.camera_index)
    else:
        for video in args.video_list:
            count_filepath, total_count_filepath = create_count_files(args.live, video)
            log_filepath, video_directory_path = create_log_files(args.live,video)
            det_file_dir, raw_det_file_dir = create_detection_directory(args.live, video, args.detection_output_dir)


            cap = cv2.VideoCapture(video)

            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            frame_fps = int(cap.get(5))
            VIDEO_PATH = video
            video_name = VIDEO_PATH.split(os.path.sep)[-1].split('.')[0].split("/")[-1]


            saved_frame_dir = 'inference_dataset/images/'
            saved_annotated_frame_dir = 'inference_dataset/annotated_images'

            # Define codec and create VideoWriter object.
            out = cv2.VideoWriter(
                f"{video_directory_path}/{video_name}_{args.pretrained_model}_{args.embedder}.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'), frame_fps,
                (frame_width, frame_height)
            )

            frame_count = 0  # To count total frames.
            class_count_dict = {"pedestrian": 0,
                                "vehicle type 1 - bicycle": 0,
                                "vehicle type 1 - car": 0,
                                "vehicle type 3 - bus": 0,
                                "vehicle type 5 - truck": 0}


            try:
                completed_successfully = True
                while cap.isOpened():

                    # if frame_count % frame_interval ==0:

                    # Read a frame
                    ret, frame = cap.read()

                    if not ret:
                        break

                    # draw user defined lines on page
                    #line_counter.draw_lines(frame)

                    # if frame_count % frame_interval ==0:

                    if args.img_size != 0:
                        resized_frame = cv2.resize(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            (args.img_size, args.img_size)
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
                    det_time = (det_end_time - det_start_time)



                    # Save raw detection if there is detections
                    if len(detections['labels'])>0:
                        save_detections(raw_det_file_dir, frame_count, detections, args.classes_to_track, args.detect_threshold, raw=True)

                    ################################################################################################################
                    ########################################## Byte Track Integration ##############################################
                    ################################################################################################################

                    # Transform detection output to ones to be used by bytetracker - xyxy px,
                    detections_bytetrack = transform_detection_output(detections, args.classes_to_track, args.detect_threshold)

                    if len(detections_bytetrack) > 0:

                        online_im, region_counts = main_tracker.startTrack(detections_bytetrack, count_filepath, video, frame, frame_count, log_filepath)

                        class_count_dict = process_count(region_counts, args.classes_to_track)

                        # if args.save_det:
                        #     save_detections(det_file_dir, frame_count, detections, args.classes_to_track, args.detect_threshold, raw=False)

                        # Write the total count dictionary to the JSON file, overwriting existing total counts
                        # with open(total_count_filepath, 'w', encoding='utf-8') as json_file:
                        #     json.dump(class_count_dict, json_file, indent=4, ensure_ascii=False)
                    else:
                        online_im = frame


                        # if track_time>0:
                        #     with open("track_time.csv", "a", newline='', encoding='utf-8') as file:
                        #         writer = csv.writer(file)
                        #         writer.writerow([round(track_time, 5)])


                    # Extract original frame
                    # extract_frame(saved_frame_dir, frame, frame_count, video_name)

                    # #Plot bounding boxes
                    # annotated_frame = draw_boxes(detections, frame, args.cls, 0.9)

                    # Saved annotated vehicles from the image.
                    # standardize_to_txt(detections, args.classes_to_track, args.score_threshold, frame_count, video_name, frame_width, frame_height)
                    # standardize_to_xml(detections, args.cls, frame_count, video_name, frame_width, frame_height)

                    # Extract annotated frame
                    # extract_frame(saved_frame_dir, annotated_frame, frame_count, video_name)

                    frame_count += 1

                    det_fps = 1 / (time.time() - det_start_time)

                    if args.debug_mode is True:
                        print(f"Frame {frame_count}",
                              f"Detection FPS: {det_fps:.2f}")

                    cv2.putText(
                        online_im,
                        f"{det_fps:.1f}- Frame:{frame_count:.3f}",
                        (int(20), int(40)),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=(0, 0, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA
                    )

                    # # Display each class with its count
                    # y_position = 80  # Starting y-position for text display
                    # for class_name, count in class_count_dict.items():
                    #     cv2.putText(
                    #         online_im,
                    #         f"{class_name}: {count}",
                    #         (int(20), int(y_position)),
                    #         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    #         fontScale=0.5,  # Smaller font size
                    #         color=(255, 0, 0),  # Blue color: (B, G, R)
                    #         thickness=1,
                    #         lineType=cv2.LINE_AA
                    #     )
                    #     y_position += 20  # Move down for the next class

                    out.write(online_im)

                    if args.show:
                        # Display or save output frame.
                        cv2.imshow("Output", online_im)
                        # Press q to quit.
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            completed_successfully = False
                            break

                # frame_count+=1



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
    args = setup_argument_parser('config/infer_config.yaml').parse_args()
    infer(args)