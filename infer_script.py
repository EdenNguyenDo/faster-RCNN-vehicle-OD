from time import perf_counter

import numpy as np
import torch
import torchvision
import cv2
import os
import time
from helpers.save_data import create_log_files, create_detection_directory
from torchvision.transforms import ToTensor
from config.coco_classes import COCO_91_CLASSES

from ocsort_tracker.run_tracking import OCS_tracker
from ocsort_tracker.tracking_utils import save_detections

"""
Running inference with object tracking with faster R-CNN model and a tracker

- This script run inference with video and plot bounding boxes
- It saves annotated video while saving the model annotation output for each frame into txt and xml format for other purposes.
- The script also saved annotated frames into folder for further assessment. 
- This script is able to run object tracking using ByteTrack algorithm. 
"""

np.random.seed(3101)
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


    detection_videos_output_dir = args.detection_videos_output_dir
    detection_files_output_dir = args.detection_files_output_dir

    if "\\" in detection_videos_output_dir:
        detection_videos_output_dir = args.detection_videos_output_dir.replace("\\", "/")
    if "\\" in detection_files_output_dir:
        detection_files_output_dir = args.detection_files_output_dir.replace("\\", "/")



    # Load torchvision model.
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
            if "\\" in video:
                video = video.replace("\\", "/")


            log_filepath, video_directory = create_log_files(args.live,video, detection_videos_output_dir)
            raw_det_file_dir = create_detection_directory(args.live, video, detection_files_output_dir)

            tracker = OCS_tracker(args)

            cap = cv2.VideoCapture(video)

            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            frame_fps = int(cap.get(5))
            VIDEO_PATH = video
            video_name = VIDEO_PATH.split(os.path.sep)[-1].split('.')[0].split("/")[-1]


            # Define codec and create VideoWriter object.
            out = cv2.VideoWriter(
                f"{video_directory}/{video_name}.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'), frame_fps,
                (frame_width, frame_height)
            )

            frame_dim = (frame_width, frame_height)

            frame_count = 0  # To count total frames.

            try:
                completed_successfully = True
                while cap.isOpened():

                    # Read a frame
                    ret, frame = cap.read()

                    if not ret:
                        break

                    if args.resize != 0:
                        resized_frame = cv2.resize(
                            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                            (640,640)
                        )
                    else:
                        resized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert frame to tensor and send it to device (cpu or cuda).
                    frame_tensor = ToTensor()(resized_frame).to(device)

                    # Feed frame to model and get detections - these are in xyxy format, not normalised.
                    det_start_time = perf_counter()
                    with torch.no_grad():
                        detections = model([frame_tensor])[0]
                    det_fps = 1 / (perf_counter() - det_start_time)


                    # Save detections to csv file for each frame
                    save_detections(raw_det_file_dir, frame_count, detections, args.classes_to_track, args.detect_threshold)

                    #frame = cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR)

                    ################################################################################################################
                    ######################################## OC-Sort tracker integration ###########################################
                    ################################################################################################################

                    online_im = tracker.operate_tracking(detections, frame, frame_count, frame_dim, video)



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

                    out.write(online_im)

                    if args.show:
                        # Display or save output frame.
                        cv2.imshow("Output", online_im)
                        # Press q to quit.
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            completed_successfully = False
                            break

                    frame_count+=1


            finally:
            # Release resources.
                cap.release()
                out.release()  # Ensure the VideoWriter is released.

                cv2.destroyAllWindows()

                # Only write the completion message if processing was completed successfully.
                # if completed_successfully:
                #     with open(log_filepath, 'a', newline='', encoding='utf-8') as file:
                #         writer = csv.writer(file)
                #         writer.writerow(["Counting and saving completed smoothly without interruption"])


