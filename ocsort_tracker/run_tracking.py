
from collections import deque
import cv2
import numpy as np
import csv
from config.VEHICLE_CLASS import VEHICLE_CLASSES
from config.coco_classes import COCO_91_CLASSES
from ocsort_tracker.src.ocsort import OCSort
from ocsort_tracker.tracking_utils import convert_frcnn_detections, apply_nms, create_track_file, convert_history_to_dict


class OCS_tracker:

    def __init__(self, track_args):
        self.classes = track_args.classes_to_track
        self.results = []
        self.all_classes = []
        self.all_ids = []
        self.all_tlwhs = []

        self.tracker = OCSort(det_thresh=track_args.track_thresh, lower_det_thresh=track_args.lower_track_thresh,
                         iou_threshold=track_args.iou_thresh, use_byte=track_args.use_byte,
                         inertia=track_args.inertia, min_hits=track_args.min_hits, max_age=track_args.track_buffer,
                         asso_func=track_args.asso, delta_t=track_args.deltat)

        self.history = deque()

        self.track_args = track_args



    def operate_tracking(self, detections, frame, frame_number, frame_dim, video):

        output = self.track_args.track_output_dir

        if "\\" in output:
            output = self.track_args.output.replace("\\", "/")

        self.all_classes = []
        self.all_ids = []
        self.all_tlwhs = []


        line_count_dict = {}

        converted_detections = convert_frcnn_detections(detections,self.classes)

        # Apply NMS to remove overlapping boxes.
        detections_nms = apply_nms(converted_detections, self.track_args.nms_iou_thresh)

        # if detections_nms[0] is not None:
        online_targets = self.tracker.update(detections_nms[0], frame_dim[0], frame_dim[1])
        online_tlwhs = []
        online_ids = []
        online_classes =[]
        for t in online_targets:
            tlwh = [int(t[0]), int(t[1]), int(t[2] - t[0]), int(t[3] - t[1])]
            tid = int(t[4])
            class_id = int(t[5])

            vertical = tlwh[2] < tlwh[3]

            if tlwh[2] * tlwh[3] > self.track_args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_classes.append(class_id)

                # Update line count for the current tid
                if tid not in line_count_dict:
                    line_count_dict[tid] = 0
                # Only proceed to save track if the line count is below the threshold
                if line_count_dict[tid] <= self.track_args.max_exist and self.track_args.save_result:
                    track_file = create_track_file(output, int(tid), video_path=video)
                    save_tracks(track_file,frame_number,tid,tlwh)


        self.all_tlwhs += online_tlwhs
        self.all_ids += online_ids
        self.all_classes += online_classes


        if len(self.history) < self.track_args.track_buffer:
            self.history.append((self.all_ids, self.all_tlwhs, self.all_classes))
        else:
            self.history.popleft()
            self.history.append((self.all_ids, self.all_tlwhs,self.all_classes))

        if len(self.all_tlwhs) > 0:
            online_im = plot_tracking(
                frame, self.history
            )
        else:
            online_im = frame

        return online_im




COLORS = np.random.randint(0, 255, size=(len(COCO_91_CLASSES), 3))


def plot_tracking(image, track_history):
    """
    Plot tracking bounding box for each object when ByteTrack is running.

    :param image:
    :param track_history:
    :return:
    """
    obj_ids, tlwhs, class_ids = track_history[-1]
    history_dict = convert_history_to_dict(track_history)

    im = np.ascontiguousarray(np.copy(image))
    im_h, im_w = im.shape[:2]

    top_view = np.zeros([im_w, im_w, 3], dtype=np.uint8) + 255

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        class_id = class_ids[i]
        id_text = '{}'.format(int(obj_id))
        color = tuple(int(c) for c in COLORS[class_ids[i]])
        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=1)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, 1, color,
                    thickness=1)
        cv2.putText(im, VEHICLE_CLASSES[class_id], (intbox[0], intbox[3] + 20), cv2.FONT_HERSHEY_PLAIN, 1, color,
                    thickness=1)

        for idx in range(len(history_dict[obj_id]) - 1):
            prev_point, next_point = history_dict[obj_id][idx], history_dict[obj_id][idx + 1]
            cv2.line(im, prev_point, next_point, color, 2)

    return im


def save_tracks(track_file, frame_number, tid, tlwh):
    with open(track_file, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            csv_writer.writerow(
                ["frame_number", "track_id", "class_id", "score", "x_topleft", "y_topleft",
                 "width", "height"])

        csv_writer.writerow([
            frame_number, int(tid), 0, 0,
            round(tlwh[0], 1), round(tlwh[1], 1),
            round(tlwh[2], 1), round(tlwh[3], 1)
        ])

