
from xml.dom import minidom

import numpy as np
import xml.etree.ElementTree as ET
import os
from config.VEHICLE_CLASS import VEHICLE_CLASSES
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
"""
This file contains methods for converting output data of the faster R-CNN of the model to text file and xml file

XML annotations for training purpose

"""


def standardize_to_txt(detections, classes, threshold, frame_no, vid_name):
    """
    This method convert the output of the model to text format in the format of
    class x_center y_center width height confidence

    Note: the coordinates have not yet been normalised
    """
    boxes = detections["boxes"].cpu().numpy()
    labels = detections["labels"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()
    lbl_mask = np.isin(labels, classes)
    scores = scores[lbl_mask]
    mask = scores > threshold
    boxes = boxes[lbl_mask][mask]
    scores = scores[mask]
    labels = labels[lbl_mask][mask]

    # Prepare the filename based on video name and frame number
    output_dir = 'bounding_box_annotations/bbox_txt_file/'
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{vid_name}_{frame_no}.txt"
    file_path = os.path.join(output_dir, filename)


    with open(file_path, 'w') as f:
        for i, box in enumerate(boxes):
            class_name = labels[i]
            confidence = scores[i]
            # Append ([x, y, w, h], score, label_string).
            x_min, y_min, x_max, y_max = box

            # Convert to (x_center, y_center, width, height)
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            f.write(f"{class_name} {x_center} {y_center} {width} {height} {confidence}\n")





def standardize_to_xml(detections, classes, frame_no, vid_name, width, height):
    """
    This method convert the output of the model to xml format for training data.

    Note: the coordinates have not yet been normalised.
    """
    # Function to convert detection results to XML format
    # Extract boxes, labels, and scores from the detection dictionary
    saved_annotation_folder = 'inference_dataset/annotations'

    boxes = detections['boxes'].cpu().numpy()  # Convert to numpy array
    labels = detections['labels'].cpu().numpy()
    lbl_mask = np.isin(labels, classes)
    labels = labels[lbl_mask]
    boxes = boxes[lbl_mask]

    # Create the XML structure
    annotation = ET.Element('annotation')

    # Add basic information
    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'video_frames'

    filename = ET.SubElement(annotation, 'filename')
    filename.text = f'frame{frame_no}.png'

    path = ET.SubElement(annotation, 'path')
    path.text = f'frame{frame_no}.png'

    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'video_database'

    size = ET.SubElement(annotation, 'size')
    width_tag = ET.SubElement(size, 'width')
    width_tag.text = str(width)
    height_tag = ET.SubElement(size, 'height')
    height_tag.text = str(height)
    depth_tag = ET.SubElement(size, 'depth')
    depth_tag.text = '3'  # Assuming RGB images

    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'

    # Add objects (detections)
    for box, label in zip(boxes, labels):
        object_element = ET.SubElement(annotation, 'object')

        name = ET.SubElement(object_element, 'name')
        name.text = VEHICLE_CLASSES[label] # Use label lookup for meaningful names

        pose = ET.SubElement(object_element, 'pose')
        pose.text = 'Unspecified'

        truncated = ET.SubElement(object_element, 'truncated')
        truncated.text = '0'

        difficult = ET.SubElement(object_element, 'difficult')
        difficult.text = '0'

        occluded = ET.SubElement(object_element, 'occluded')
        occluded.text = '0'

        bndbox = ET.SubElement(object_element, 'bndbox')

        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max = box
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = str(int(x_min))

        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = str(int(y_min))

        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = str(int(x_max))

        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = str(int(y_max))

    # Ensure the output directory exists

    # Save the XML file
    output_dir = os.path.join(saved_annotation_folder, vid_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"{vid_name}_{frame_no}.xml"
    file_path = os.path.join(output_dir, filename)

    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="     ")

    # Write the pretty-printed XML to file
    with open(file_path, "w") as xml_file:
        xml_file.write(xml_str)

    print(f"XML saved to {file_path}")








