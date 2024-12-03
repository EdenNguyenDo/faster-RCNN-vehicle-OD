import argparse

# Extracting default values as constants for clarity
DEFAULT_INPUT_VIDEO_PATH = "./input_videos/2024_0323_120137_100A.MP4"
DEFAULT_PRETRAINED_MODEL = "fasterrcnn_resnet50_fpn"
DEFAULT_THRESHOLD = 0.9
DEFAULT_MAX_AGE = 30
DEFAULT_EMBEDDER = "mobilenet"
DEFAULT_LINES_DATA_PATH = "./lines_data.csv"


def setup_argument_parser():
    """Set up the argument parser for video inference."""
    parser = argparse.ArgumentParser(description='Arguments for inference using fine-tuned model')

    parser.add_argument(
        '--live',
        default = True,
        type = bool,
        help='live inference or not'
    )

    parser.add_argument(
        '--input_video',
        default=DEFAULT_INPUT_VIDEO_PATH,
        help='Path to input video'
    )
    parser.add_argument(
        '--img_size',
        default=None,
        type=int,
        help='Image resize; e.g., 640 will resize images to 640x640'
    )
    parser.add_argument(
        '--pretrained_model',
        default=DEFAULT_PRETRAINED_MODEL,
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
        ],
        help='Model name'
    )
    parser.add_argument(
        '--score_threshold',
        default=DEFAULT_THRESHOLD,
        type=float,
        help='Score threshold to filter out detections'
    )
    parser.add_argument(
        '--max_age',
        default=DEFAULT_MAX_AGE,
        type=int,
        help='Maximum age of a track before it is discarded'
    )
    parser.add_argument(
        '--embedder',
        default=DEFAULT_EMBEDDER,
        choices=[
            "mobilenet",
            "torchreid",
            "clip_RN50",
            "clip_RN101",
            "clip_RN50x4",
            "clip_RN50x16",
            "clip_ViT-B/32",
            "clip_ViT-B/16"
        ],
        help='Type of feature extractor to use'
    )
    parser.add_argument(
        '--show',
        action='store_false',
        help='Visualize results in real-time on screen'
    )
    parser.add_argument(
        '--classes_to_track',
        nargs='+',
        default=[1, 2, 3, 6, 8],
        type=int,
        help='Which classes to track'
    )
    parser.add_argument(
        '--evaluate',
        dest='evaluate',
        default=True,
        type=bool,
        help='Flag to enable evaluation mode'
    )
    parser.add_argument(
        '--infer_samples',
        dest='infer_samples',
        default=True,
        type=bool,
        help='Flag to enable inference on samples'
    )
    parser.add_argument(
        '--lines_data',
        default='./live_lines.csv',
        choices=["./live_lines.csv"],
        help='Path to CSV file containing lines to be drawn on video'
    )
    return parser

