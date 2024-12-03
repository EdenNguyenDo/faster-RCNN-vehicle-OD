from collections import deque

import numpy as np
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from bytetrackCustom.bytetrack_args import ByteTrackArgument
from bytetrackCustom.bytetrack_utils import plot_tracking, transform_detection_output
from config.VEHICLE_CLASS import VEHICLE_CLASSES
from helpers.line_counter import LineCounter


class ByteTracker:

    def __init__(self, args):
        self.detection_output_xml = None
        self.results = None
        self.all_classes = None
        self.all_ids = None
        self.all_tlwhs = None
        self.trackers = [BYTETracker(ByteTrackArgument) for _ in range(14)]
        self.line_counter = LineCounter(args.lines_data)

        self.history = deque()
        self.region_counts = [[0] * len(self.line_counter.lines) for _ in range(len(VEHICLE_CLASSES))]
        self.live = args.live

        if args.live:
            self.video_name = "Cam1"
        else:
            self.video_name = args.input_video.split('/')[-1].split('.')[0]



    def startTrack(self, frame, detections_bytetrack, frame_count):
        self.all_tlwhs = []
        self.all_ids = []
        self.all_classes = []
        self.results = []
        self.detection_output_xml = {'trackIDs': [], 'boxes': [], 'labels': [], 'scores': []}

        for class_id, tracker in enumerate(self.trackers):
            detections_bytetrack = np.array(detections_bytetrack)
            class_outputs = detections_bytetrack[detections_bytetrack[:, 5] == class_id][:, :5]
            if class_outputs is not None:
                online_targets = tracker.update(class_outputs)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                online_classes = [class_id] * len(online_targets)
                for t in online_targets:
                    # tracker box coordinates are given in pixels, not normalised
                    tlwh = t.tlwh
                    tlbr = t.tlbr
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > ByteTrackArgument.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > ByteTrackArgument.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)

                        # use the trackings' output bbox locations to detect objects, with
                        self.region_counts = self.line_counter.perform_count_line_detections(class_id, tid, tlbr, self.video_name)

                        # Get the xml output for saving into annotation file
                        tlbr_box = [tlbr[0], tlbr[1], tlbr[2], tlbr[3]]
                        self.detection_output_xml['trackIDs'].append(tid)
                        self.detection_output_xml['boxes'].append(tlbr_box)
                        self.detection_output_xml['labels'].append(class_id)
                        self.detection_output_xml['scores'].append(t.score)

                        online_ids.append(tid)
                        online_scores.append(t.score)
                        tlwh_box = (tlwh[0], tlwh[1], tlwh[2], tlwh[3])

                        self.results.append(
                            # frame_id, track_id, tl_x, tl_y, w, h, score = obj_prob * class_prob, class_idx, dummy, dummy, dummy
                            f"{frame_count},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f}, {class_id}, -1,-1,-1\n"
                        )

                self.all_tlwhs += online_tlwhs
                self.all_ids += online_ids
                self.all_classes += online_classes

        if len(self.history) < 30:
            self.history.append((self.all_ids, self.all_tlwhs, self.all_classes))
        else:
            self.history.popleft()
            self.history.append((self.all_ids, self.all_tlwhs, self.all_classes))

        if len(self.all_tlwhs) > 0:
            online_im = plot_tracking(
                frame, self.history
            )
        else:
            online_im = frame


        return online_im, self.region_counts