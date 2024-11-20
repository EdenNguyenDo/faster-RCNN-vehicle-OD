import torch
import cv2
import os
import argparse

import yaml
from torchvision import models
from torchvision.models.detection import FasterRCNN
from torch.utils.data.dataloader import DataLoader
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_return_video_data(video_path):
    cap = cv2.VideoCapture(video_path)
    # Get the video's frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    assert (frame_width != 0 and frame_height !=0), 'Please check video path...'
    return cap, frame_width, frame_height



def load_model_and_dataset(args):
    ###### Read the config file #####
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    #################################

    train_config = config['train_params']

    if args.use_resnet50_fpn:
        final_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                                 min_size=600,
                                                                                 max_size=1000,
                                                                                 box_score_thresh=0.5,
        )
        final_model.roi_heads.box_predictor = FastRCNNPredictor(
            final_model.roi_heads.box_predictor.cls_score.in_features,
            num_classes=7)
    else:
        backbone = torchvision.models.resnet34(pretrained=True, norm_layer=torchvision.ops.FrozenBatchNorm2d)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-3])
        backbone.out_channels = 256
        roi_align = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        rpn_anchor_generator = AnchorGenerator()
        final_model = torchvision.models.detection.FasterRCNN(backbone,
                                                                    num_classes=7,
                                                                    min_size=600,
                                                                    max_size=1000,
                                                                    rpn_anchor_generator=rpn_anchor_generator,
                                                                    box_roi_pool=roi_align,
                                                                    rpn_pre_nms_top_n_train=12000,
                                                                    rpn_pre_nms_top_n_test=6000,
                                                                    box_batch_size_per_image=128,
                                                                    box_score_thresh=0.7,
                                                                    rpn_post_nms_top_n_test=300)
    final_model.eval()
    final_model.to(device)

    if args.use_resnet50_fpn:
        final_model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                                  'tv_frcnn_r50fpn_' + train_config['ckpt_name']),
                                                     map_location=device))
    else:
        final_model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                                  'tv_frcnn_' + train_config['ckpt_name']),
                                                     map_location=device))
    return final_model





def infer_video(args):
    # Load the model
    final_model = load_model_and_dataset(args)

    # # Open video capture
    # cap = cv2.VideoCapture(args.video_input_path)
    # if not cap.isOpened():
    #     print("Error: Unable to open video.")
    #     return
    #
    # # Get video properties
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = cap.get(cv2.CAP_PROP_FPS)

    # Define video writer to save the output
    output_video_path = args.video_output_path

    cap, frame_width, frame_height = read_return_video_data(args.video_input_path)
    save_name = output_video_path.split(os.path.sep)[-1].split('.')[0]+"___Output"
    # Define codec and create VideoWriter object.
    out = cv2.VideoWriter(f"{output_video_path}/{save_name}.mp4",
                        cv2.VideoWriter_fourcc(*'mp4v'), 30,
                        (frame_width, frame_height))


    frame_count = 0 # To count total frames.
    total_fps = 0 # To get the final frames per second.


    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break

        image = frame.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert frame to tensor
        im_tensor = torchvision.transforms.ToTensor()(image)
        im_tensor = im_tensor.unsqueeze(0).float().to(device)


        # Getting predictions from trained model
        frcnn_output = final_model(im_tensor, None)[0]
        boxes = frcnn_output['boxes']
        labels = frcnn_output['labels']
        scores = frcnn_output['scores']
        im_copy = image.copy()


        # Draw boxes on the frame
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box.detach().cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(image, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            cv2.rectangle(im_copy, (x1, y1), (x2, y2), thickness=2, color=[0, 0, 255])
            text = f"{labels}: {scores:.2f}"

            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
            text_w, text_h = text_size
            cv2.rectangle(im_copy, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), [255, 255, 255], -1)
            cv2.putText(image, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            cv2.putText(im_copy, text=text,
                        org=(x1 + 5, y1 + 15),
                        thickness=1,
                        fontScale=1,
                        color=[0, 0, 0],
                        fontFace=cv2.FONT_HERSHEY_PLAIN)
            # x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
            # label = labels[idx].detach().cpu().item()
            # score = scores[idx].detach().cpu().item()
            #
            # # Draw bounding box and label
            # cv2.rectangle(im_copy, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            # text = f"{label}: {score:.2f}"
            # cv2.putText(im_copy, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Write the frame to output video
        out.write(im_copy)

    # Release resources
    cap.release()
    out.release()
    print(f"Output video saved at {output_video_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for inference using torchvision code faster rcnn')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    parser.add_argument('--infer_samples', dest='infer_samples',
                        default=True, type=bool)
    parser.add_argument('--use_resnet50_fpn', dest='use_resnet50_fpn',
                        default=True, type=bool)
    parser.add_argument('--video_output_path', dest='video_output_path',
                        default="C://transmetric//dev//python//AI camera//trial//faster-R-CNN-model//output_videos//iv_1201124", type=str)
    parser.add_argument('--video_input_path', dest='video_input_path',
                        default="C://Users//User//Downloads//vid2.mp4", type=str)
    args = parser.parse_args()

    if args.infer_samples:
        infer_video(args)
    else:
        print('Not Inferring for samples as `infer_samples` argument is False')

