import cv2

from bytetrackCustom.bytetrack_utils import cross_product_line
from config.VEHICLE_CLASS import VEHICLE_CLASSES
import pandas as pd
import ast

class LineCounter:
    # todo save the counts by object id, line id and timestamp (frame number) into file with name based on the input video filename

    def __init__(self, filePath):
        # Lines will be stored as a dictionary with keys as identifiers
        # and values as tuples containing start and end points.
        self.lines = read_lines_from_csv(filePath)
        self.lines_start = [value['start'] for value in self.lines.values()]
        self.lines_end = [value['end'] for value in self.lines.values()]

        self.region_counts = [[0] * len(self.lines)] * len(VEHICLE_CLASSES)
        self.previous_side = [[0 for _ in range(len(self.lines))] for _ in range(10000)]
        self.current_side = [[0 for _ in range(len(self.lines))] for _ in range(10000)]
        self.cross_product = [[0 for _ in range(len(self.lines))] for _ in range(10000)]

        self.line_color = (0, 255, 0)  # Green line color
        self.line_thickness = 2


    def draw_lines(self, frame):
        # Assuming there is a function or method to draw lines
        # For example, using OpenCV's cv2.line
        for start, end in zip(self.lines_start, self.lines_end):
            cv2.line(frame, start, end, color=(0, 255, 0), thickness=2)

    def perform_count_line_detections(self, class_id, tid, tlbr):
        # Line intersection/counting section
        ### Calculate centroids in px, count if in regions
        x_centre = (tlbr[2] - tlbr[0]) / 2 + tlbr[0]
        y_centre = (tlbr[3] - tlbr[1]) / 2 + tlbr[1]
        # cv2.line(frame, (int(x_centre), int(y_centre)), line_start[0], line_color, line_thickness)
        # cv2.line(frame, (int(x_centre), int(y_centre)), line_start[1], (0,0,255), line_thickness)

        # iterate over user defined boundary lines
        for line_id, (start, end) in enumerate(zip(self.lines_start, self.lines_end)):
            # Initialize lists for line start and end points based on the self.lines dictionary

            # todo tidy up - split into functions
            # todo save the counts by object id, line id and timestamp (frame number) into file with name based on the input video filename
            # todo
            self.cross_product[tid][line_id] = cross_product_line((x_centre, y_centre), start, end)

            if self.cross_product[tid][line_id] >= 0:
                self.current_side[tid][line_id] = 'positive'
            elif self.cross_product[tid][line_id] < 0:
                self.current_side[tid][line_id] = 'negative'

            # Check if the object has crossed the line
            if self.previous_side[tid][line_id] != 0:  # check that it isn't a brand new track
                if self.previous_side[tid][line_id] != self.current_side[tid][line_id]:
                    print(
                        f"Object {class_id} has crossed the line with id {line_id}! Final side: {self.current_side[tid][line_id]}")
                    self.region_counts[class_id][line_id]+= 1
            self.previous_side[tid][line_id] = self.current_side[tid][line_id]

        return self.region_counts



def read_lines_from_csv(filePath):
    lines = {}
    reader = pd.read_csv(filePath)

    for index, row in reader.iterrows():


        try:
            # Direct use of the line_id from the CSV without 'line' prefix
            line_id = str(row['line_id'])
            startX = row['start_x']
            startY = row['start_y']
            endX = row['end_x']
            endY = row['end_y']
            lines[line_id] = {
                'start': (int(startX), int(startY)),
                'end': (int(endX), int(endY))
            }

        except KeyError as e:
            print(f"Missing expected column in row {index}: {e}")
        except ValueError as e:
            print(f"Invalid data format in row {index}: {e}")

    return lines


def process_count(region_counts):
    # Creating a list to store the maximum count for each class
    max_counts = [max(class_count) for class_count in region_counts]

    return max_counts
