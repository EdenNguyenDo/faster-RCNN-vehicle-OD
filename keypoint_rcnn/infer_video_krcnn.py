import torch
import cv2
import argparse
import utils_krcnn
import time
from PIL import Image
from torchvision.transforms import transforms as transforms
from infer_image_krcnn import get_model
from deep_sort_realtime.deepsort_tracker import DeepSort
from deepSORT.coco_classes import COCO_91_CLASSES
from helpers.helper import Helper


# construct the argument parser to parse the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', default='../input_videos/mvmhat_1_1.mp4',
                    help='path to the input data')
parser.add_argument('-m', '--min-size', dest='min_size', default=800,
                    help='path to the input data')
args = vars(parser.parse_args())

tracker = DeepSort(30)


# transform to convert the image to tensor
transform = transforms.Compose([
    transforms.ToTensor()
])

# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the model on to the computation device and set to eval mode
model = get_model(min_size=args['min_size']).to(device).eval()

cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
# get the video frames' width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# set the save path
save_path = f"../outputs/{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}.mp4"
# define codec and create VideoWriter object
out = cv2.VideoWriter(save_path,
                      cv2.VideoWriter_fourcc(*'mp4v'), 20,
                      (frame_width, frame_height))
frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second

# read until end of video
while (cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:

        pil_image = Image.fromarray(frame).convert('RGB')
        orig_frame = frame
        # transform the image
        image = transform(pil_image)
        # add a batch dimension
        image = image.unsqueeze(0).to(device)
        # get the start time
        start_time = time.time()
        # get the detections, forward pass the frame through the model
        with torch.no_grad():
            outputs = model(image)
        # get the end time
        end_time = time.time()
        output_image = utils_krcnn.draw_keypoints_and_boxes(outputs, orig_frame)
        # get the fps
        fps = 1 / (end_time - start_time)
        # add fps to total fps
        total_fps += fps


        # increment frame count
        frame_count += 1
        wait_time = max(1, int(fps / 4))
        cv2.imshow('Pose detection frame', output_image)
        out.write(output_image)
        # press `q` to exit
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    else:
        break

# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")














