import argparse
import yaml

def make_parser(config_path=None):
    # Initialize config to None by default
    config = None
    if config_path is not None:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

    # Fallback function to safely get values from config or return a default
    def get_config_value(key, default=None):
        if config is not None:
            return config.get(key, default)
        return default

    parser = argparse.ArgumentParser("OC-SORT parameters")

    parser.add_argument(
        '--classes_to_track',
        nargs='+',
        default=config.get('cls_to_track'),
        type=int,
        help='Which classes to track'
    )

    # Distributed Training Specific
    parser.add_argument("--local_rank", default=get_config_value('local_rank', 0), type=int, help="Local rank for distributed training")
    parser.add_argument("--num_machines", default=get_config_value('num_machines', 1), type=int, help="Number of nodes for training")
    parser.add_argument("--machine_rank", default=get_config_value('machine_rank', 0), type=int, help="Node rank for multi-node training")


    # Detection Parameters
    parser.add_argument("-c", "--ckpt", default=get_config_value('ckpt'), type=str, help="Checkpoint for evaluation")
    parser.add_argument("--conf", default=get_config_value('conf', 0.12), type=float, help="Test confidence threshold")
    parser.add_argument("--nms", default=get_config_value('nms', 0.1), type=float, help="Non-maximum suppression threshold")
    parser.add_argument("--tsize", default=get_config_value('tsize'), type=int, help="Test image size")
    parser.add_argument("--seed", default=get_config_value('seed'), type=int, help="Evaluation seed")

    # Tracking Parameters
    parser.add_argument("--track_thresh", type=float, default=get_config_value('track_thresh', 0.8), help="Detection confidence threshold")
    parser.add_argument("--lower_track_thresh", type=float, default=get_config_value('lower_bound_det_threshold', 0.5), help="Detection confidence threshold for filtering with BYTE")

    parser.add_argument("--nms_iou_thresh", type=float, default=get_config_value('nms_iou_thresh_det', 0.45), help="NMS IOU threshold for filtering overlapping boxes")
    parser.add_argument("--iou_thresh", type=float, default=get_config_value('iou_thresh', 0.2), help="IOU threshold for SORT matching")
    parser.add_argument("--min_hits", type=int, default=get_config_value('min_hits', 5), help="Minimum hits to create track in SORT")
    parser.add_argument("--inertia", type=float, default=get_config_value('inertia', 0.4), help="Weight of VDC term in cost matrix")
    parser.add_argument("--deltat", type=int, default=get_config_value('deltat', 1), help="Time step difference to estimate direction")
    parser.add_argument("--track_buffer", type=int, default=get_config_value('track_buffer', 60), help="Frames for keeping lost tracks")
    parser.add_argument("--max_exist", type=int, default=get_config_value('max_exist', 3001), help="Frames for keeping tracking parked vehicles")
    parser.add_argument("--match_thresh", type=float, default=get_config_value('match_thresh', 0.85), help="Matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=get_config_value('min-box-area', 150), help='Filter out tiny boxes')
    parser.add_argument("--gt-type", type=str, default=get_config_value('gt-type', "_val_half"), help="Suffix to find the GT annotation")
    parser.add_argument("--mot20", dest="mot20", default=get_config_value('mot20', False), action="store_true", help="Test MOT20.")
    parser.add_argument("--public", action="store_true", help="Use public detection")
    parser.add_argument('--asso', default=get_config_value('asso', "giou"), help="Similarity function: iou/giou/diou/ciou/ctdis")
    parser.add_argument("--use_byte", dest="use_byte", default=get_config_value('use_byte', True), action="store_true", help="Use byte in tracking.")


    parser.add_argument("--hp", action="store_true", help="Use head padding to add missing objects during initializing the tracks (offline).")

    # Demo Video Settings
    parser.add_argument("--demo_type", default=get_config_value('demo_type', "video"), help="Demo type (image, video, or webcam)")
    parser.add_argument("--path", default=get_config_value('path', "./videos/dance_demo.mp4"), help="Path to images or video")
    parser.add_argument("--camid", type=int, default=get_config_value('camid', 0), help="Webcam demo camera ID")
    parser.add_argument("--save_result", default=True, help="Whether to save the inference result of image/video")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=get_config_value('aspect_ratio_thresh', 1.5), help="Threshold for filtering out boxes of which aspect ratio is above the given value.")
    parser.add_argument('--min_box_area', type=float, default=get_config_value('min_box_area', 8), help='Filter out tiny boxes')
    parser.add_argument("--device", default=get_config_value('device', "gpu"), type=str, help="Device to run the model (cpu or gpu)")

    # Detection File Path
    parser.add_argument('--detection_input_folder', default=get_config_value("detection_input_folder", "./detections_folder/detections_folder1"), type=str, help="Path to the raw detection file")
    parser.add_argument('--track_output_dir', default=get_config_value("track_output_dir","../output/track_folder"), type=str, help="Path to the output directory")



    parser.add_argument(
        '--debug_mode',
        default=get_config_value('debug', False),
        help='Indicator of debug mode activation'
    )

    parser.add_argument(
        '--save_det',
        default=get_config_value('save_detection', True),
        help='Indicator of savinga all detections produced'
    )

    parser.add_argument(
        '--camera_index',
        default=get_config_value('camera_index', 0),
        help='Index of the camera to be accessed by the model for real-time inference'
    )

    parser.add_argument(
        '--live',
        default = config.get('live', False),
        type = bool,
        help='live inference or not'
    )

    parser.add_argument(
        '--detection_files_output_dir',
        default=get_config_value('detection_files_output_dir', "../output/detection_files")
    )

    parser.add_argument(
        '--detection_videos_output_dir',
        default=get_config_value('detection_videos_output_dir', "../output/inferred_videos")
    )

    parser.add_argument(
        '--video_list',
        nargs='+',
        default=get_config_value('video_list', "./video/video_traffic_2.mp4"),
        help='video to be ran on'
    )

    parser.add_argument(
        '--img_size',
        default=get_config_value('img_size', 0),
        type = int,
        help='Image resize; e.g., 640 will resize images to 640x640'
    )

    parser.add_argument(
        '--pretrained_model',
        default=get_config_value('pretrained_model', "fasterrcnn_resnet50_fpn"),
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
        '--detect_threshold',
        default=get_config_value('detect_threshold', 0.8),
        type=float,
        help='Score threshold to filter out track objects'
    )


    parser.add_argument(
        '--show',
        default= get_config_value('show', True),
        action='store_false',
        help='Visualize results in real-time on screen'
    )


    parser.add_argument(
        '--evaluate',
        dest='evaluate',
        default=get_config_value('evaluate', False),
        type=bool,
        help='Flag to enable evaluation mode'
    )

    parser.add_argument(
        '--video_track',
        default=get_config_value('video_track', True),
        type=bool,
        help='set to run tracking and inferencing with videos on TRUE'
    )

    return parser
