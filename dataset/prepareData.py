import glob
import os
import random

import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET
"""
This script create functions for transforming annotations and loading data.
"""

def load_images_and_anns(im_dir, ann_dir, label2idx):
    """
    Method to get the xml files and for each file
    get all the objects and their ground truth detection
    information for the dataset
    :param im_dir: Path of the images
    :param ann_dir: Path of annotation xmlfiles
    :param label2idx: Class Name to index mapping for dataset
    :return:
    """
    im_infos = []
    for ann_file in tqdm(glob.glob(os.path.join(ann_dir, '*.xml'))):
        im_info = {}
        im_info['img_id'] = os.path.basename(ann_file).split('.xml')[0]
        im_info['filename'] = os.path.join(im_dir, '{}.png'.format(im_info['img_id']))
        ann_info = ET.parse(ann_file)
        root = ann_info.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        im_info['width'] = width
        im_info['height'] = height
        detections = []

        for obj in ann_info.findall('object'):
            det = {}
            label = label2idx[obj.find('name').text]
            bbox_info = obj.find('bndbox')
            bbox = [
                int(float(bbox_info.find('xmin').text)) - 1,
                int(float(bbox_info.find('ymin').text)) - 1,
                int(float(bbox_info.find('xmax').text)) - 1,
                int(float(bbox_info.find('ymax').text)) - 1
            ]
            det['label'] = label
            det['bbox'] = bbox
            detections.append(det)
        im_info['detections'] = detections
        im_infos.append(im_info)
    print('Total {} images found'.format(len(im_infos)))
    return im_infos


class VtodDataset(Dataset):
    """
    A dataset class for loading images and corresponding annotations for a vehicle
    type object detection task.

    This dataset handles images and their respective annotations based on the
    provided directory paths, maps class labels to indices, and enables access
    to image data and its metadata. The dataset supports data augmentation during
    training, specifically horizontal flipping of images.

    :ivar split: The split type of the dataset (e.g., 'train', 'val', 'test'). Determines
        if augmentation is applied.
    :type split: str
    :ivar im_dir: The directory path containing image files.
    :type im_dir: str
    :ivar ann_dir: The directory path containing annotation files.
    :type ann_dir: str
    :ivar label2idx: A dictionary mapping class labels to numeric indices.
    :type label2idx: dict
    :ivar idx2label: A dictionary mapping numeric indices to class labels.
    :type idx2label: dict
    :ivar images_info: A list containing metadata for each image and its detections.
        The metadata includes filename, bounding boxes, and class labels.
    :type images_info: list
    """
    def __init__(self, split, im_dir, ann_dir):
        self.split = split
        self.im_dir = im_dir
        self.ann_dir = ann_dir
        classes = [
            'wheel', 'vehicle type 5', 'vehicle type 1', 'vehicle type 2', 'vehicle type 4', 'vehicle type 3'
        ]
        classes = sorted(classes)
        classes = ['background'] + classes
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        print(self.idx2label)
        self.images_info = load_images_and_anns(im_dir, ann_dir, self.label2idx)

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, index):
        im_info = self.images_info[index]
        im = Image.open(im_info['filename']).convert('RGB')
        to_flip = False
        if self.split == 'train' and random.random() < 0.5:
            to_flip = True
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im_tensor = torchvision.transforms.ToTensor()(im)
        targets = {}
        targets['bboxes'] = torch.as_tensor([detection['bbox'] for detection in im_info['detections']])
        targets['labels'] = torch.as_tensor([detection['label'] for detection in im_info['detections']])
        if to_flip:
            for idx, box in enumerate(targets['bboxes']):
                x1, y1, x2, y2 = box
                w = x2 - x1
                im_w = im_tensor.shape[-1]
                x1 = im_w - x1 - w
                x2 = x1 + w
                targets['bboxes'][idx] = torch.as_tensor([x1, y1, x2, y2])
        return im_tensor, targets, im_info['filename']
