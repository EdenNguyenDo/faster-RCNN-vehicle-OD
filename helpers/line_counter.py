import csv
import os
import time
from operator import index

import cv2
from config.VEHICLE_CLASS import VEHICLE_CLASSES
import pandas as pd
from collections import namedtuple


class LineCounter:
    # todo save the counts by object id, line id and timestamp (frame number) into file with name based on the input video filename
    #todo read in lines from svg file
    #todo create some way of pairing the lines into A and B hoses
    #todo function which reads lines from svg, and outputs arrays of line_start and line_end points,
    """

    """
    def __init__(self, filePath):
        # Lines will be stored as a dictionary with keys as identifiers
        # and values as tuples containing start and end points.
        """
        region_counts
        previous_side
        current_side
        cross_product


        Initializes data structures for handling and processing line data from a CSV file.
        The class is designed to store information about lines with specific start and end
        points, manage vehicle categorization data, and set configurations for visual
        representation of the lines.

        :param filePath: The path to the CSV file containing line data. The file must be
                         structured in a way that each line contains an identifier and
                         associated start and end point coordinates.
        :type filePath: str
        """
        self.lines = read_lines_from_csv(filePath)
        self.lines_start = [value['start'] for value in self.lines.values()]
        self.lines_end = [value['end'] for value in self.lines.values()]

        self.Line = namedtuple("Line", ["start", "end"])
        # Store intersections in the class
        self.line_intersections = self.compute_line_intersections(self.lines)

        self.region_counts = [[0] * len(self.lines) for _ in range(len(VEHICLE_CLASSES))]
        self.previous_side = [[0 for _ in range(len(self.lines))] for _ in range(10000)]
        self.current_side = [[0 for _ in range(len(self.lines))] for _ in range(10000)]
        self.cross_product = [[0 for _ in range(len(self.lines))] for _ in range(10000)]
        self.direction_list = ["_" for _ in range(10000)]
        self.lane_list = ["_" for _ in range(10000)]

        self.line_color = (0, 255, 0)  # Green line color
        self.line_thickness = 2


    def draw_lines(self, frame):
        # Assuming there is a function or method to draw lines
        # For example, using OpenCV's cv2.line
        for start, end in zip(self.lines_start, self.lines_end):
            cv2.line(frame, start, end, color=(0, 255, 0), thickness=2)

    def perform_count_line_detections(self, class_id, tid, tlbr, frame):
        # Line intersection/counting section
        ### Calculate centroids in px, count if in regions
        global lane
        x_centre = (tlbr[2] - tlbr[0]) / 2 + tlbr[0]
        y_centre = (tlbr[3] - tlbr[1]) / 2 + tlbr[1]

        # iterate over user defined count lines
        # Compute the cross product for the center of box vector with the first 2 lines
        for line_id in range(2):
            start, end = self.lines_start[line_id], self.lines_end[line_id]
            self.cross_product[tid][line_id] = cross_product_line((x_centre, y_centre), start, end)

            cv2.line(frame, (int(x_centre), int(y_centre)), end, color=(0, 10, 255), thickness=1)

            if self.cross_product[tid][line_id] >= 0:
                self.current_side[tid][line_id] = 'positive'
            elif self.cross_product[tid][line_id] < 0:
                self.current_side[tid][line_id] = 'negative'

            # Check if the object has crossed the line
            if self.previous_side[tid][line_id] != 0:  # check that it isn't a new track
                if self.previous_side[tid][line_id] != self.current_side[tid][line_id]:
                    print(f"Object {class_id} has crossed the line with id {line_id}! Final side: {self.current_side[tid][line_id]}")
                    self.region_counts[class_id][line_id] += 1

            # Determine direction based on changes
            if self.previous_side[tid][line_id] == 'negative' and self.current_side[tid][line_id] == 'positive':
                self.direction_list[tid] = 'N to P'
            elif self.previous_side[tid][line_id] == 'positive' and self.current_side[tid][line_id] == 'negative':
                self.direction_list[tid] = 'P to N'
            else:
                self.direction_list[tid] = "_"  # No crossing detected
            self.previous_side[tid][line_id] = self.current_side[tid][line_id]

        # Check the latter 3 lines after a side change
        for line_id in range(2, 5):
            start, end = self.lines_start[line_id], self.lines_end[line_id]
            self.cross_product[tid][line_id] = cross_product_line((x_centre, y_centre), start, end)

        # Determine lane classification based on the cross product for the latter 3 lines
        if ((self.cross_product[tid][2] < 0 and
                self.cross_product[tid][3] < 0 and
                self.cross_product[tid][4] > 0) or
                (self.cross_product[tid][2] < 0 and
                self.cross_product[tid][3] > 0 and
                self.cross_product[tid][4] > 0)):
            lane = 'left'
            self.lane_list[tid] = lane

        elif ((self.cross_product[tid][2] > 0 and
              self.cross_product[tid][3] > 0 and
              self.cross_product[tid][4] < 0) or
              (self.cross_product[tid][2] > 0 and
             self.cross_product[tid][3] < 0 and
             self.cross_product[tid][4] < 0)):
            lane = 'right'
            self.lane_list[tid] = lane

        else:
            lane = 'unknown'
            self.lane_list[tid] = lane

        return self.region_counts, self.direction_list, self.lane_list



    def intersection(self, line1, line2):
        # Calculates the intersection point of two lines
        x1, y1 = line1.start
        x2, y2 = line1.end
        x3, y3 = line2.start
        x4, y4 = line2.end

        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0:
            return None  # Lines are parallel

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom

        if 0 <= ua <= 1 and 0 <= ub <= 1:
            # Calculate the intersection point
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            return (round(x,3), round(y,3))
        else:
            return None  # No intersection within the line segments

    def compute_line_intersections(self, lines):
        intersections = {}

        boundary_lines = {key: self.Line(value['start'], value['end']) for key, value in lines.items() if
                          'boundary' in value['line_des']}
        count_lines = {key: self.Line(value['start'], value['end']) for key, value in lines.items() if
                       'count' in value['line_des']}

        for b_id, b_line in boundary_lines.items():
            for c_id, c_line in count_lines.items():
                inter_point = self.intersection(b_line, c_line)
                if inter_point:
                    intersections[(b_id, c_id)] = inter_point

        return intersections



def cross_product_line(point, line_start, line_end):
    # Calculate the direction vector of the line
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]

    # Vector from line start to the point
    # dpx = point[0] - line_start[0]
    # dpy = point[1] - line_start[1]
    dpx = line_end[0] - point[0]
    dpy = line_end[1] - point[1]
    #cx=0; cy=0; cz=

    # Cross product to determine which side of the line the point is on
    cross_product = dx * dpy - dy * dpx

    return cross_product


def read_lines_from_csv(filePath):
    lines = {}
    reader = pd.read_csv(filePath)
    for index, row in reader.iterrows():
        try:
            # Direct use of the line_id from the CSV without 'line' prefix
            line_id = row['line_id']
            line_des = str(row['line_des'])
            startX = row['start_x']
            startY = row['start_y']
            endX = row['end_x']
            endY = row['end_y']
            lines[line_id] = {
                'line_des': line_des,
                'start': (int(startX), int(startY)),
                'end': (int(endX), int(endY))
            }

        except KeyError as e:
            print(f"Missing expected column in row {index}: {e}")
        except ValueError as e:
            print(f"Invalid data format in row {index}: {e}")
    return lines


def process_count(region_counts, classes):
    # Creating a list to store the maximum count for each class
    # Mapping class indices to their names in VEHICLE_CLASSES and fetching max count
    class_max_counts = {VEHICLE_CLASSES[class_id]: max(region_counts[class_id]) for class_id in classes}
    return class_max_counts


#
# def perform_count_line_detections(self, class_id, tid, tlbr, frame):
#     # Line intersection/counting section
#     ### Calculate centroids in px, count if in regions
#     global lane
#     x_centre = (tlbr[2] - tlbr[0]) / 2 + tlbr[0]
#     y_centre = (tlbr[3] - tlbr[1]) / 2 + tlbr[1]
#
#     # iterate over user defined count lines
#     # Compute the cross product for the center of box vector with the first 2 lines
#     for line_id in range(2):
#         start, end = self.lines_start[line_id], self.lines_end[line_id]
#         self.cross_product[tid][line_id] = cross_product_line((x_centre, y_centre), start, end)
#
#         cv2.line(frame, (int(x_centre), int(y_centre)), end, color=(0, 10, 255), thickness=1)
#
#         if self.cross_product[tid][line_id] >= 0:
#             self.current_side[tid][line_id] = 'positive'
#         elif self.cross_product[tid][line_id] < 0:
#             self.current_side[tid][line_id] = 'negative'
#
#         # Check if the object has crossed the line
#         if self.previous_side[tid][line_id] != 0:  # check that it isn't a new track
#             if self.previous_side[tid][line_id] != self.current_side[tid][line_id]:
#                 print(f"Object {class_id} has crossed the line with id {line_id}! Final side: {self.current_side[tid][line_id]}")
#                 self.region_counts[class_id][line_id] += 1
#
#         # Determine direction based on changes
#         if self.previous_side[tid][line_id] == 'negative' and self.current_side[tid][line_id] == 'positive':
#             direction = 'N to P'
#         elif self.previous_side[tid][line_id] == 'positive' and self.current_side[tid][line_id] == 'negative':
#             direction = 'P to N'
#         else:
#             direction = "_"  # No crossing detected
#         self.previous_side[tid][line_id] = self.current_side[tid][line_id]
#
#     # Check the latter 3 lines after a side change
#     for line_id in range(2, 5):
#         start, end = self.lines_start[line_id], self.lines_end[line_id]
#         self.cross_product[tid][line_id] = cross_product_line((x_centre, y_centre), start, end)
#
#     # Determine lane classification based on the cross product for the latter 3 lines
#     if (self.cross_product[tid][2] < 0 and
#             self.cross_product[tid][3] < 0 and
#             self.cross_product[tid][4] > 0):
#         lane = 'left'
#     elif (self.cross_product[tid][2] > 0 and
#           self.cross_product[tid][3] > 0 and
#           self.cross_product[tid][4] < 0):
#         lane = 'right'
#     else:
#         lane = 'unknown'
#
#
#
#     self.lane_list[tid] = lane
#     self.direction_list[tid] = direction
#     self.previous_side[tid][line_id] = self.current_side[tid][line_id]
#
#
#     return self.region_counts, self.direction_list, self.lane_list
#
