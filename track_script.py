import csv

import torch

from ocsort_tracker.helpers.track_with_det_files import read_detections_from_csv_folder, read_detections_from_h5, read_detections_from_parquet
from ocsort_tracker.src.ocsort import OCSort
from ocsort_tracker.tracking_utils import apply_nms, create_track_file


def run_track(args):


    detection_data_filepath = args.detection_input_folder
    output = args.track_output_dir

    if "\\" in detection_data_filepath:
        detection_data_filepath = args.detection_input_folder.replace("\\", "/")
    if "\\" in output:
        output = args.output.replace("\\", "/")


    tracker = OCSort(det_thresh=args.track_thresh, lower_det_thresh=args.lower_track_thresh, iou_threshold=args.iou_thresh, use_byte=args.use_byte,
                     inertia=args.inertia, min_hits=args.min_hits, max_age=args.track_buffer, asso_func=args.asso, delta_t=args.deltat)
    results = []

    line_count_dict = {}

    # Read detections from the specified folder
    # detections = read_detections_from_csv_folder(detection_data_filepath)
    detections = read_detections_from_h5("C:/transmetric/AI_system/output/raw_detections/067-00006_Mon_Wed_44hrs_2000/2024_1203_065010_023A/2024_1203_065010_023A_2025_0121_154759_raw_detection/detections.h5")
    # detections = read_detections_from_parquet("C:/transmetric/AI_system/output/raw_detections/067-00006_Mon_Wed_44hrs_2000/2024_1203_065010_023A/2024_1203_065010_023A_2025_0121_153808_raw_detection/detections.h5")

    sorted_keys = sorted(detections.keys())

    # Process each frame (frame numbers may be non-continuous)
    for frame_number in sorted_keys:

        outputs_by_frame = detections[frame_number]

        if len(outputs_by_frame) > 0:

            detections_tensor = [torch.tensor(outputs_by_frame, dtype=torch.float32)]

            # Apply NMS to remove overlapping boxes.
            detections_nms = apply_nms(detections_tensor, args.nms_iou_thresh)

            if detections_tensor[0] is not None:
                online_targets = tracker.update(detections_nms[0], 480, 640)
                online_tlwhs = []
                online_ids = []
                online_classes = []
                for t in online_targets:
                    tlwh = [int(t[0]), int(t[1]), int(t[2] - t[0]), int(t[3] - t[1])]
                    tid = int(t[4])
                    class_id = int(t[5])

                    vertical = tlwh[2] < tlwh[3]

                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_classes.append(class_id)

                        # Update line count for the current tid
                        if tid not in line_count_dict:
                            line_count_dict[tid] = 0

                        # Only proceed if the line count is below the threshold
                        if line_count_dict[tid] <= args.max_exist:
                            track_file = create_track_file(output, int(tid), detection_folder=detection_data_filepath)

                            with open(track_file, 'a', newline='') as csvfile:
                                csv_writer = csv.writer(csvfile)
                                if csvfile.tell() == 0:
                                    csv_writer.writerow(
                                        ["frame_number", "track_id", "class_id", "score", "x_topleft", "y_topleft",
                                         "width", "height"])

                                csv_writer.writerow([
                                    frame_number, int(tid), int(class_id), 0,
                                    round(tlwh[0], 1), round(tlwh[1], 1),
                                    round(tlwh[2], 1), round(tlwh[3], 1)
                                ])

                                # Increment line count for tid
                                line_count_dict[tid] += 1
