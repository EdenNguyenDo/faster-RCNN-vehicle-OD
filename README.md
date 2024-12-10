# Traffic data processing and couting system

### Description 

The functionality of the system is to process video and real time video stream
for count extracting.

### Methodology

This system features vehicle and pedestrian tracking and counting with the use of an object detection model (OD) 
and a multi object tracking mechanism (MOT).

The video or realtime stream is processed frame by frame where each frame is analyzed and be ran through with the two models mentioned.

#### Object detection model

In respect to the OD model, Faster R-CNN is used as the main model for outputting detections with bounding boxes.
It is used together with resnet-50 that serves as the backbone network for feature extraction. 

With the use of Faster R-CNN, the models accuracy and mAP is significantly improved compared to other two-shot detection model.
However, the trade-off between speed and accuracy is considerable as it computing speed is slower compared to other single-shot
model such as SSD or YOLO variations.

#### Multi object tracking

Bytetrack algorithm and model is being utilised as a way to assign ID to keep track of objects/bounding boxes in frames.

Bytetrack is integrated with the system as it evaluates and tracks the objects detected by the OD model.

The usage of tracking here is crucial as it plays an important role in counting objects or vehicles. Without the usage of 
tracking, the model does not have a unique identifier for each object inside a frame or multiple frames leading to the
failure of accurate counting system.

