import argparse
import yaml

def setup_argument_parser(config_path):
    """Set up the argument parser for video inference."""

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser(description='Arguments for inference using fine-tuned model')

    parser.add_argument(
        '--debug_mode',
        default=config.get('debug'),
        help='Indicator of debug mode activation'
    )

    parser.add_argument(
        '--camera_index',
        default=config.get('camera_index'),
        help='Index of the camera to be accessed by the model for real-time inference'
    )

    parser.add_argument(
        '--lines_data',
        default=config.get('lines_file_path'),
        help='Path to CSV file containing lines to be drawn on video'
    )

    parser.add_argument(
        '--live',
        default = config.get('live'),
        type = bool,
        help='live inference or not'
    )

    parser.add_argument(
        '--input_video',
        default=config.get('input_video'),
        help='Path to input video'
    )

    parser.add_argument(
        '--img_size',
        default=config.get('img_size'),
        type = int,
        help='Image resize; e.g., 640 will resize images to 640x640'
    )

    parser.add_argument(
        '--pretrained_model',
        default=config.get('pretrained_model'),
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
        default=config.get('score_threshold'),
        type=float,
        help='Score threshold to filter out detections'
    )

    parser.add_argument(
        '--max_frame_track',
        default=config.get('max_frame_track'),
        type=int,
        help='Maximum age of a track before it is discarded'
    )

    parser.add_argument(
        '--embedder',
        default=config.get('byte_track_embedder'),
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
        default= config.get('show'),
        action='store_false',
        help='Visualize results in real-time on screen'
    )

    parser.add_argument(
        '--classes_to_track',
        nargs='+',
        default=config.get('cls_to_track'),
        type=int,
        help='Which classes to track'
    )

    parser.add_argument(
        '--evaluate',
        dest='evaluate',
        default=config.get('evaluate'),
        type=bool,
        help='Flag to enable evaluation mode'
    )

    parser.add_argument(
        '--infer_samples',
        dest='infer_samples',
        default=config.get('infer_samples'),
        type=bool,
        help='Flag to enable inference on samples'
    )

    return parser

