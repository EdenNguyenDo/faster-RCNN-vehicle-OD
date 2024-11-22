import json
import os
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

def create_pascal_voc_xml(image_filename, width, height, depth, objects):
    """
    Creates a Pascal VOC XML annotation string for a given image and its objects.
    """
    annotation = Element('annotation')

    folder = SubElement(annotation, 'folder')
    folder.text = ''

    filename = SubElement(annotation, 'filename')
    filename.text = image_filename

    path = SubElement(annotation, 'path')
    path.text = image_filename

    source = SubElement(annotation, 'source')
    database = SubElement(source, 'database')
    database.text = 'roboflow.com'

    size = SubElement(annotation, 'size')
    width_elem = SubElement(size, 'width')
    width_elem.text = str(width)
    height_elem = SubElement(size, 'height')
    height_elem.text = str(height)
    depth_elem = SubElement(size, 'depth')
    depth_elem.text = str(depth)

    segmented = SubElement(annotation, 'segmented')
    segmented.text = '0'

    for obj in objects:
        obj_elem = SubElement(annotation, 'object')
        name = SubElement(obj_elem, 'name')
        name.text = obj['name']

        pose = SubElement(obj_elem, 'pose')
        pose.text = 'Unspecified'

        truncated = SubElement(obj_elem, 'truncated')
        truncated.text = '0'

        difficult = SubElement(obj_elem, 'difficult')
        difficult.text = '0'

        occluded = SubElement(obj_elem, 'occluded')
        occluded.text = '0'

        bndbox = SubElement(obj_elem, 'bndbox')
        xmin = SubElement(bndbox, 'xmin')
        xmin.text = str(obj['xmin'])
        xmax = SubElement(bndbox, 'xmax')
        xmax.text = str(obj['xmax'])
        ymin = SubElement(bndbox, 'ymin')
        ymin.text = str(obj['ymin'])
        ymax = SubElement(bndbox, 'ymax')
        ymax.text = str(obj['ymax'])

    # Convert to a pretty XML string
    xml_str = tostring(annotation)
    pretty_xml = parseString(xml_str).toprettyxml(indent="    ")
    return pretty_xml

def convert_jsonl_to_voc(jsonl_file, output_dir):
    """
    Converts a Florence 2 JSONL file to Pascal VOC XML files.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the JSONL file
    with open(jsonl_file, 'r') as file:
        for line in file:
            annotation = json.loads(line.strip())
            image_filename = annotation['image']
            suffix = annotation['suffix']

            # Placeholder image dimensions
            image_width, image_height, image_depth = 1024, 768, 3

            # Extract objects
            objects = []
            parts = suffix.split(">")
            current_object = {'name': None, 'bbox': []}

            for part in parts:
                part = part.strip()
                if part.startswith("<loc_"):  # Bounding box data
                    # Extract coordinate value (e.g., "123" from "<loc_123")
                    coord = int(part.split("_")[1])
                    if current_object is not None:  # Ensure an object is initialized
                        current_object['bbox'].append(coord)
                        if len(current_object['bbox']) == 4:  # Full bounding box collected
                            xmin, ymin, xmax, ymax = current_object['bbox']
                            objects.append({
                                'name': current_object['name'],
                                'xmin': xmin,
                                'xmax': xmax,
                                'ymin': ymin,
                                'ymax': ymax
                            })
                            current_object = {'name': None, 'bbox': []}    # Reset bounding box for this object
                elif part:  # Object name
                    current_object['name'] = part.split("<")[0]
                    current_object['bbox'].append(part.split("<loc_")[1])

            # Generate Pascal VOC XML
            xml_content = create_pascal_voc_xml(image_filename, image_width, image_height, image_depth, objects)

            # Save the XML file
            xml_filename = os.path.join(output_dir, os.path.splitext(image_filename)[0] + '.xml')
            with open(xml_filename, 'w') as xml_file:
                xml_file.write(xml_content)
            print(f"Saved: {xml_filename}")

# Usage
jsonl_file = 'annotation.jsonl'  # Replace with the path to your JSONL file
output_dir = '../dataset/valid'  # Replace with your desired output directory
convert_jsonl_to_voc(jsonl_file, output_dir)
