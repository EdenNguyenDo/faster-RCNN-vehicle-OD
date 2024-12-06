
import cv2
import numpy as np
from sympy import false

from config.VEHICLE_CLASS import VEHICLE_CLASSES
import pandas as pd
from collections import namedtuple



class LineCounter:
    #todo create some way of pairing the lines into A and B hoses
    """
    Represents a class used to handle and analyze line data from a given CSV file
    to perform operations such as line drawing, vehicle counting, and lane detection.

    The LineCounter class processes line data stored in a specified file,
    initializes necessary data structures, and utilizes them to work with video
    frames for drawing lines and detecting objects crossing these lines. It
    maintains records of vehicle directions and lane occupancy based on their
    interactions with the lines. The class handles various configurations needed
    for visual representation and analysis.
    """
    def __init__(self, filePath):
        # Lines will be stored as a dictionary with keys as identifiers
        # and values as tuples containing start and end points.
        """
        region_counts
        previous_side
        current_side
        cross_product

        The line data file name would contain the number of count lines, boundary lines and reserve lines respectively
        separated by underscore '_'.



        Initializes data structures for handling and processing line data from a CSV file.
        The class is designed to store information about lines with specific start and end
        points, manage vehicle categorization data, and set configurations for visual
        representation of the lines.

        :param filePath: The path to the CSV file containing line data. The file must be
                         structured in a way that each line contains an identifier and
                         associated start and end point coordinates.
        :type filePath: str
        """
        self.lines, self.count_lines,self.bound_lines, self.res_lines  = read_lines_from_csv(filePath)
        self.lines_start = []
        self.lines_end = []
        for value in self.lines.values():
            self.lines_start.append(value['start'])
            self.lines_end.append(value['end'])

        self.Line = namedtuple("Line", ["start", "end"])
        # Store intersections in the class
        self.line_intersections = self.compute_line_intersections(self.lines)

        self.region_counts = [[0 for _ in range(self.count_lines)] for _ in range(len(VEHICLE_CLASSES))]
        self.previous_side = [[0 for _ in range(len(self.lines))] for _ in range(10000)]
        self.current_side = [[0 for _ in range(len(self.lines))] for _ in range(10000)]
        self.cross_product = [[0 for _ in range(len(self.lines))] for _ in range(10000)]
        self.direction_list = ["_" for _ in range(10000)]
        self.lane_list = ["_" for _ in range(10000)]

        self.line_color = (0, 255, 0)  # Green line color
        self.line_thickness = 2


    def draw_lines(self, frame):
        """
        Draws lines on the provided frame. This method iterates through pairs of
        start and end points (stored as class attributes) and draws lines on the
        frame using a specified color and thickness. The lines are rendered using OpenCV.

        :param frame: The frame on which the lines will be drawn. This is generally
                      an image or a video frame represented as an array.
        :return: None
        """
        # Assuming there is a function or method to draw lines
        # For example, using OpenCV's cv2.line
        for start, end in zip(self.lines_start, self.lines_end):
            cv2.line(frame, start, end, color=(0, 255, 0), thickness=2)


    def perform_count_line_detections(self, class_id, tid, tlbr, frame):
        """
        This method perform counting if the object passed the count lines
        Then direction is extracted based on the side change

        The lane detection assuming the furthest lane is 1 and increments as the lane get closer
        The boundary lines index has the same logic.

        There will always be just 2 reserve lines in total for each of the side in a road.

        The lane numbering logic is based on the index of the cross product result when the sign changed.
        i.e. (-,+,+) change index is 1 and the lane where the box centre stands, is also 1.


        Analyzes the trajectory of objects in a video frame to determine if they cross
        pre-defined lines, which may represent lanes or counting boundaries. The method
        calculates object centroids and uses cross product operations to determine
        intersections with count lines and boundary lines. Based on these calculations,
        it updates the count of objects that have crossed specific lines, detects the
        direction of movement, and assigns lane numbers to tracked objects.

        :param class_id: Identifier for the class of the detected object
        :type class_id: int
        :param tid: Unique track identifier for the detected object
        :type tid: int
        :param tlbr: Tuple representing the top-left and bottom-right coordinates of the object
        :type tlbr: tuple
        :param frame: Current video frame where objects are detected
        :type frame: np.ndarray
        :return: A tuple containing updated region counts, direction list, and lane list
                 for tracked objects
        :rtype: tuple
        """
        # Line intersection/counting section
        ### Calculate centroids in px, count if in regions
        global lane
        hit = False
        newly_tracked = False
        prev_id = None
        x_centre = (tlbr[2] - tlbr[0]) / 2 + tlbr[0]
        y_centre = (tlbr[3] - tlbr[1]) / 2 + tlbr[1]


        """
        This loop performs counting by computing cross product between count lines end point and box centre
        """
        # iterate over user defined count lines
        # Compute the cross product for the center of box vector with the first 2 lines
        for count_line_id in range(self.count_lines):
            start, end = self.lines_start[count_line_id], self.lines_end[count_line_id]
            self.cross_product[tid][count_line_id] = cross_product_line((x_centre, y_centre), start, end)

            current_cp = self.cross_product[tid][count_line_id]
            # cv2.line(frame, (int(x_centre), int(y_centre)), end, color=(0, 10, 255), thickness=1)
            if current_cp >= 0:
                self.current_side[tid][count_line_id] = 'positive'
            elif current_cp < 0:
                self.current_side[tid][count_line_id] = 'negative'


            # Check if the object has crossed the line, track has to be new
            if self.previous_side[tid][count_line_id] != 0:  # check that it isn't a new track
                # It is not a new track
                if self.previous_side[tid][count_line_id] != self.current_side[tid][count_line_id]:
                    print(f"Object {class_id} has crossed the line with id {count_line_id}! Final side: {self.current_side[tid][count_line_id]}")
                    self.region_counts[class_id][count_line_id] += 1
                    hit = True
                    if tid != prev_id:
                        newly_tracked = True

            if hit:
                # Determine direction based on changes
                if self.previous_side[tid][count_line_id] == 'negative' and self.current_side[tid][count_line_id] == 'positive':
                    object_direction = 'N to P (L-R)'
                    self.direction_list[tid] = object_direction
                elif self.previous_side[tid][count_line_id] == 'positive' and self.current_side[tid][count_line_id] == 'negative':
                    object_direction = 'P to N (R-L)'
                    self.direction_list[tid] = object_direction

            self.previous_side[tid][count_line_id] = self.current_side[tid][count_line_id]
            # Set the id to the current id
            prev_id = tid


        # self.lane_list = self.detect_lane(frame, tid, x_centre, y_centre)


        return self.region_counts, self.direction_list, hit, newly_tracked  #, self.lane_list






















    def detect_lane(self, frame, tid, x_centre, y_centre):


        """
        This loop computes cross product between end point of boundary lines and box centre
        """
        bound_list_CP = []
        # Check the boundary lines
        for bound_line_id in range(self.count_lines, self.bound_lines*2-1):
            start, end = self.lines_start[bound_line_id], self.lines_end[bound_line_id]
            self.cross_product[tid][bound_line_id] = cross_product_line((x_centre, y_centre), start, end)
            cv2.line(frame, (int(x_centre), int(y_centre)), end, color=(0, 10, 255), thickness=1)
            bound_list_CP.append(self.cross_product[tid][bound_line_id])


        """
        This loop performs lane detection and numbering using cross product
        """
        current_sign = 'DEFAULT_sign'
        for cp in bound_list_CP:
            index_cp = bound_list_CP.index(cp)
            next_sign = int(np.sign(cp))
            # If there is a sign differences (-,+,+) or (-,-,+) and it is not the first number
            if next_sign != current_sign and current_sign != 'DEFAULT_sign':
                # changed_idx will defo start at 1
                # changed_idx determines lanes number
                changed_idx = index_cp
                self.lane_list[tid] = changed_idx

                # If there is a sign difference, break and assure that the lanes is determined
                break
            elif next_sign == 0:
                self.lane_list[tid] = f"this vehicle is in the middle of two lanes: {index_cp} and {index_cp - 1}"
            else:
                current_sign = next_sign
                all_sign = next_sign
        else:
            # We know that the centre of box is outside the boundaries
            # Use reserve lines for checking
            # If all positive, use line 1 (further). Otherwise line 2 (closer)
            if int(all_sign) == 1:
                start, end = self.lines_start[-2], self.lines_end[-2]
                additional_CP = cross_product_line((x_centre, y_centre), start, end)
                if additional_CP > 0:
                    self.lane_list[tid] = f"this vehicle is outside the lanes on the FURTHER side"
                else:
                    self.lane_list[tid] = 1
            elif int(all_sign) == -1:
                start, end = self.lines_start[-1], self.lines_end[-1]
                additional_CP = cross_product_line((x_centre, y_centre), start, end)
                if additional_CP < 0:
                    self.lane_list[tid] = f"this vehicle is outside the lanes on the CLOSER side"
                else:
                    self.lane_list[tid] = self.bound_lines - 1

        return self.lane_list


    def compute_line_intersections(self, lines):
        """
        Computes the intersection points between boundary and count lines.

        This function iterates over pairs of boundary and count lines, determines their
        intersection points, and stores these points in a dictionary. The function relies
        on the method `intersection` to calculate the intersection point of two lines.

        :param lines: A dictionary where keys represent line identifiers and values
                      contain a dictionary with 'start', 'end', and 'line_des' keys.
                      The 'start' and 'end' keys define the coordinates of the line
                      endpoints, and 'line_des' describes the type of the line
                      ('boundary' or 'count').
        :return: A dictionary with keys as tuples of boundary and count line identifiers
                 and values as the intersection points of the respective lines.
        """

        intersections = {}

        boundary_lines = {key: self.Line(value['start'], value['end']) for key, value in lines.items() if
                          'boundary' in value['line_des']}
        count_lines = {key: self.Line(value['start'], value['end']) for key, value in lines.items() if
                       'count' in value['line_des']}

        for b_id, b_line in boundary_lines.items():
            for c_id, c_line in count_lines.items():
                inter_point = intersection(b_line, c_line)
                if inter_point:
                    intersections[(b_id, c_id)] = inter_point

        return intersections


def intersection(line1, line2):
    """
    Calculates the intersection point of two line segments, provided as input
    parameters. The function determines if the line segments intersect within
    their respective endpoints, and returns the intersection point when
    applicable. If the lines are parallel or there is no intersection within
    the segments, the function returns None. The intersection point, if
    found, is returned as a tuple of floats rounded to three decimal places.

    :param line1: The first line segment represented by a structure containing
                  start and end points. Each point is a tuple of floats
                  representing (x, y) coordinates.
    :param line2: The second line segment represented by a structure containing
                  start and end points. Each point is a tuple of floats
                  representing (x, y) coordinates.
    :return: A tuple containing the x and y coordinates of the intersection
             point, each rounded to three decimal places, or None if the lines
             are parallel or do not intersect within the segments.
    """
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


def cross_product_line(point, line_start, line_end):
    """
    Computes the cross product of a vector from a line start point to a given
    point with the line's direction vector. It is used to determine the
    relative orientation of the point to the line, indicating if the point
    lies to the left, right, or on the line in a 2D plane.

    :param point: Coordinates of the point as a tuple (or list) of two
                  numerical values, representing x and y.
    :type point: tuple[float, float]
    :param line_start: Coordinates of the starting point of the line as a tuple
                       (or list) of two numerical values, representing x and y.
    :type line_start: tuple[float, float]
    :param line_end: Coordinates of the ending point of the line as a tuple
                     (or list) of two numerical values, representing x and y.
    :type line_end: tuple[float, float]
    :return: A numerical value indicating the orientation:
             - Positive if the point is to the left of the line,
             - Negative if the point is to the right,
             - Zero if the point is on the line.
    :rtype: float
    """
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
    """
    Reads line data from a CSV file and processes it into a dictionary format.
    The function extracts lines described by start and end coordinates from
    each row of the CSV. The file path is assumed to encode some metadata
    regarding counts in its filename, which are extracted and returned as well.
    If any expected columns are missing or contain invalid data, an error
    message is printed, and processing continues for remaining rows.

    :param filePath: The path to the CSV file containing the data.
    :type filePath: str
    :return: A tuple containing a dictionary of lines and count metadata.
             The dictionary has line IDs as keys, with each corresponding
             value being a dictionary containing 'line_des', 'start' 2-tuple,
             and 'end' 2-tuple. The tuple of metadata includes count_line_count,
             bound_line_count, and res_line_count integers extracted from the
             file name.
    :rtype: Tuple[Dict[int, Dict[str, Union[str, Tuple[int, int]]]], int, int, int]
    """
    lines = {}
    reader = pd.read_csv(filePath)
    fileName_arr = filePath.split('/')[-1].split('.')[0].split('_')
    res_line_count = int(fileName_arr[-1])
    bound_line_count = int(fileName_arr[-2])
    count_line_count = int(fileName_arr[-3])
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
    return lines, count_line_count, bound_line_count, res_line_count


def process_count(region_counts, classes):
    """
    Determines the maximum count of occurrences for each specified class
    within given region counts.

    This function maps class indices from the `classes` list to their
    corresponding names defined in the global `VEHICLE_CLASSES` and calculate
    the maximum count for each class by analyzing the provided `region_counts`.

    :param region_counts: A dictionary where keys are class identifiers and
                          values are lists of counts that represent the
                          occurrence of each class in different regions.
    :type region_counts: dict[int, list[int]]
    :param classes: A list of class indices to be processed, where each index
                    corresponds to a particular class in `VEHICLE_CLASSES`.
    :type classes: list[int]
    :return: A dictionary mapping class names to their maximum count found in
             respective region counts.
    :rtype: dict[str, int]
    """
    # Creating a list to store the maximum count for each class
    # Mapping class indices to their names in VEHICLE_CLASSES and fetching max count
    class_max_counts = {VEHICLE_CLASSES[class_id]: max(region_counts[class_id]) for class_id in classes}
    return class_max_counts


