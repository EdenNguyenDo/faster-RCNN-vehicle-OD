import cv2
import matplotlib
import numpy

# pairs of edges for 17 of the keypoints detected ...
# ... these show which points to be connected to which point ...
# ... we can omit any of the connecting points if we want, basically ...
# ... we can easily connect less than or equal to 17 pairs of points ...
# ... for keypoint RCNN, not  mandatory to join all 17 keypoint pairs
edges = [
    (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10),
    (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
    (12, 14), (14, 16), (5, 6)
]

def draw_keypoints_and_boxes(outputs, image):
    # the `outputs` is list which in-turn contains the dictionary
    for i in range(len(outputs[0]['keypoints'])):
        # get the detected keypoints
        keypoints = outputs[0]['keypoints'][i].cpu().detach().numpy()
        # get the detected bounding boxes
        boxes = outputs[0]['boxes'][i].cpu().detach().numpy()
        # proceed to draw the lines and bounding boxes
        if outputs[0]['scores'][i] > 0.9: # proceed if confidence is above 0.9
            keypoints = keypoints[:, :].reshape(-1, 3)
            for p in range(keypoints.shape[0]):
                # draw the keypoints
                cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                            3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            # draw the lines joining the keypoints
            for ie, e in enumerate(edges):
                # get different colors for the edges
                rgb = matplotlib.colors.hsv_to_rgb([
                    ie/float(len(edges)), 1.0, 1.0
                ])
                rgb = rgb*255
                # join the keypoint pairs to draw the skeletal structure
                cv2.line(image, (keypoints[e, 0][0], keypoints[e, 1][0]),
                        (keypoints[e, 0][1], keypoints[e, 1][1]),
                        tuple(rgb), 2, lineType=cv2.LINE_AA)
            # draw the bounding boxes around the objects
            cv2.rectangle(image, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])),
                          color=(0, 255, 0),
                          thickness=2)
        else:
            continue
    return image