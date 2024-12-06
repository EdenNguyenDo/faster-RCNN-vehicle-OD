from collections import deque

import numpy as np
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from bytetrackCustom.bytetrack_args import ByteTrackArgument
from bytetrackCustom.bytetrack_utils import plot_tracking
from config.VEHICLE_CLASS import VEHICLE_CLASSES
from helpers.line_counter import LineCounter
from helpers.save_count_data import save_count_data


class ByteTracker:

    """
    Class for tracking objects across video frames using BYTE tracking algorithm.

    This class processes video frames to track and count objects as well as perform
    lane detection. It utilizes multiple BYTETrackers, one for each class, and
    integrates detection outputs with tracking results. The tracking history is
    maintained to visualize the tracking results.

    :ivar detection_output_xml: Storage for tracking results in XML format.
    :type detection_output_xml: dict
    :ivar results: List of formatted string results for tracked objects.
    :type results: list
    :ivar all_classes: List of all object classes detected in current frame.
    :type all_classes: list
    :ivar all_ids: List of all object IDs tracked in current frame.
    :type all_ids: list
    :ivar all_tlwhs: List of all bounding box coordinates (top-left width-height)
        for tracked objects.
    :type all_tlwhs: list
    :ivar trackers: List of BYTETracker objects, one for each object class to be tracked.
    :type trackers: list
    :ivar line_counter: Object responsible for counting lines crossings and lane detecting by tracked objects.
    :type line_counter: LineCounter
    :ivar history: Deque maintaining tracking history of objects for visualization purposes.
    :type history: collections.deque
    :ivar region_counts: Array tracking the count of objects detected in predefined regions.
    :type region_counts: list
    :ivar direction_list: List maintaining directional data for each tracked object.
    :type direction_list: list
    :ivar lane_list: List maintaining lane assignment data for each tracked object.
    :type lane_list: list
    :ivar args: Arguments passed to the tracker, typically containing configuration settings.
    :type args: Any
    """

    def __init__(self, args):
        self.detection_output_xml = None
        self.results = None
        self.all_classes = None
        self.all_ids = None
        self.all_tlwhs = None
        self.trackers = [BYTETracker(ByteTrackArgument) for _ in range(9)]
        self.line_counter = LineCounter(args.lines_data)

        self.history = deque()
        self.region_counts = [[0 for _ in range(self.line_counter.count_lines)] for _ in range(len(VEHICLE_CLASSES))]
        self.direction_list = ["_" for _ in range(10000)]
        self.lane_list = ["_" for _ in range(10000)]
        self.args = args



    def startTrack(self, frame, detections_bytetrack, frame_count, count_filepath):
        """
        Starts the tracking process for identified objects in a video frame. It updates
        trackers for each object class, processes the detection outputs, counts objects,
        and generates tracking results for visualization and further analysis.
        Additionally, it saves detection and tracking information for further usage.

        :param frame: Video frame data for the current time step in the tracking process.
        :param detections_bytetrack: Array containing transformed detections made by the faster R-CNN model
            for the current frame, where each detection includes bounding box coordinates,
            confidence scores, and class identifiers.
        :param frame_count: Integer representing the current frame count, used to track the
            frame number during the sequence.
        :param count_filepath: The file path where the object count data is stored and
            updated as new frames are processed.
        :return: A tuple containing:
            - The updated video frame with tracking information visualized.
            - Current counts of objects tracked per line.
        """
        self.all_tlwhs = []
        self.all_ids = []
        self.all_classes = []
        self.results = []
        self.detection_output_xml = {'trackIDs': [], 'boxes': [], 'labels': [], 'scores': []}

        for class_id, tracker in enumerate(self.trackers):
            detections_bytetrack = np.array(detections_bytetrack)
            class_outputs = detections_bytetrack[detections_bytetrack[:, 5] == class_id][:, :5]
            if len(class_outputs)>0:
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

                        # use the trackings' output bbox locations to detect objects, with counts and direction detection
                        self.region_counts, self.direction_list, hit, newly_tracked = self.line_counter.perform_count_line_detections(class_id, tid, tlbr, frame)

                        if hit and newly_tracked:
                            # save the count into the file
                            save_count_data(self.args, count_filepath, self.region_counts, self.direction_list, class_id, tid, frame_count)


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