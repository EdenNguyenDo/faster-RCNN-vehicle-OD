import numpy as np
import torch
import torchvision
import cv2
import os
import time
import argparse
from plot_bbox import draw_boxes
from standardize_detections import standardize_to_txt, standardize_to_xml
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor
from helpers.extract_frame import extract_frame
from deepSORT.coco_classes import COCO_91_CLASSES
"""
Running inference with faster R-CNN model

This script run inference with video and save annotated video while saving the model output for each frame into
a different format for other purposes
"""



np.random.seed(3101)


OUT_DIR = 'output_frcnn-ds'
os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLORS = np.random.randint(0, 255, size=(len(COCO_91_CLASSES), 3))


"""
This function load model faster R-CNN with trained weight file
"""
def load_model():
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


"""
This function runs inference using loaded model frame by frame while saving the annotation of each frame into a predefined format
Then each individual frame is aggregated to create an annotated video

"""
def infer_video(args):
    print(f"Tracking: {[COCO_91_CLASSES[idx] for idx in args.cls]}")
    print(f"Detector: {args.pretrained_model}")
    print(f"Re-ID embedder: {args.embedder}")

    # Load model.
    model = getattr(torchvision.models.detection, args.pretrained_model)(weights='DEFAULT')
    #model = load_model()

    # Set model to evaluation mode.
    model.eval().to(device)


    VIDEO_PATH = args.input_videos
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_fps = int(cap.get(5))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = VIDEO_PATH.split(os.path.sep)[-1].split('.')[0].split("/")[-1]
    saved_frame_dir = 'inference_dataset/images/'

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
    # # Set the frame interval to capture one frame every 5 seconds
    # frame_interval = int(fps * 3)  # For 30 FPS, this equals 90 frames being skipped before one is saved


    while cap.isOpened():

        # Read a frame
        ret, frame = cap.read()

        if not ret:
            break

        # if frame_count % frame_interval ==0:

        if args.imgsz is not None:
            resized_frame = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                (args.imgsz, args.imgsz)
            )
        else:
            resized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert frame to tensor and send it to device (cpu or cuda).
        frame_tensor = ToTensor()(resized_frame).to(device)

        # Feed frame to model and get detections.
        det_start_time = time.time()
        with torch.no_grad():
            detections = model([frame_tensor])[0]
        det_end_time = time.time()
        det_fps = 1 / (det_end_time - det_start_time)


        #Plot bounding boxes
        draw_boxes(detections, frame, args.cls, 0.9)

        # Saved annotated vehicles from the image.
        standardize_to_txt(detections, args.cls, args.threshold, frame_count, video_name)
        standardize_to_xml(detections, args.cls, frame_count, video_name, frame_width, frame_height)
        extract_frame(saved_frame_dir, frame, frame_count, video_name)

        frame_count += 1

        print(f"Frame {frame_count}/{frames}",
                f"Detection FPS: {det_fps:.1f}")

        cv2.putText(
            frame,
            f"FPS: {det_fps:.1f}",
            (int(20), int(40)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )

        out.write(frame)

        if args.show:
            # Display or save output frame.
            cv2.imshow("Output", frame)
            # Press q to quit.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # frame_count+=1


    # Release resources.
    cap.release()
    out.release()  # Ensure the VideoWriter is released.
    cv2.destroyAllWindows()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Arguments for inference using fine-tuned model')
    parser.add_argument(
        '--input_videos',
        default="input_videos/vid2.mp4",
        help='path to input video',
    )
    parser.add_argument(
        '--imgsz',
        default=None,
        help='image resize, 640 will resize images to 640x640',
        type=int
    )
    parser.add_argument(
        '--pretrained_model',
        default='fasterrcnn_resnet50_fpn',
        help='model name',
        choices=[
            'fasterrcnn_resnet50_fpn_v2',
            'fasterrcnn_resnet50_fpn',
            'fasterrcnn_mobilenet_v3_large_fpn',
            'fasterrcnn_mobilenet_v3_large_320_fpn',
            'fcos_resnet50_fpn',
            'ssd300_vgg16',
            'ssdlite320_mobilenet_v3_large',
            'retinanet_resnet50_fpn',
            'retinanet_resnet50_fpn_v2'
        ]
    )
    parser.add_argument(
        '--threshold',
        default=0.7,
        help='score threshold to filter out detections',
        type=float
    )

    parser.add_argument(
        '--max_age',
        default=30,
        help='type of feature extractor to use',
        type = int
    )

    parser.add_argument(
        '--embedder',
        default='mobilenet',
        help='type of feature extractor to use',
        choices=[
            "mobilenet",
            "torchreid",
            "clip_RN50",
            "clip_RN101",
            "clip_RN50x4",
            "clip_RN50x16",
            "clip_ViT-B/32",
            "clip_ViT-B/16"
        ]
    )
    parser.add_argument(
        '--show',
        action='store_false',
        help='visualize results in real-time on screen'
    )
    parser.add_argument(
        '--cls',
        nargs='+',
        default=[3, 6, 8],
        help='which classes to track',
        type=int
    )

    parser.add_argument(
        '--evaluate',
        dest='evaluate',
        default=True,
        type=bool)

    parser.add_argument(
        '--infer_samples', dest='infer_samples',
        default=True,
        type=bool)


    args = parser.parse_args()

    infer_video(args)
