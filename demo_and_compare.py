import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# COCO class labels (for model pre-trained on COCO)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A',
    'backpack', 'umbrella', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'N/A', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Load a pre-trained Faster R-CNN model with ResNet-50 backbone
def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()  # Set to evaluation mode
    return model


# Preprocess image for Faster R-CNN model
def transform_image(image_path):
    # Load image
    image = Image.open(image_path).convert("RGB")
    # Convert image to tensor
    image_tensor = F.to_tensor(image)
    # Normalize with pre-trained Faster R-CNN mean and std
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor


# Run inference on the image
def infer(model, image_tensor):
    with torch.no_grad():  # Disable gradients for inference
        prediction = model(image_tensor)
    return prediction


# Draw bounding boxes on the image and save result
def save_inferred_image(image, prediction, save_path):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()


    # Draw boxes
    for element in range(len(prediction[0]['boxes'])):
        box = prediction[0]['boxes'][element].cpu().numpy()
        score = prediction[0]['scores'][element].cpu().numpy()
        label = prediction[0]['labels'][element].cpu().numpy()

        if score > 0.5:  # Only consider boxes with a score above a threshold (e.g., 0.5)
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Add label and score text near the bounding box
            label_name = COCO_CLASSES[label]  # Get the class name
            ax.text(xmin, ymin, f"{label_name}: {score:.2f}",
                    bbox=dict(facecolor='yellow', alpha=0.5), fontsize=10, color='black')

    # Save the image with bounding boxes
    plt.axis('off')  # Hide axes
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)


# Process all images in the input folder and save results to output folder
def process_images(input_folder, output_folder):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load pre-trained Faster R-CNN model
    model = load_model()

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
            # Load and preprocess image
            image_tensor = transform_image(file_path)

            # Run inference
            prediction = infer(model, image_tensor)

            # Load the image for visualization
            image = Image.open(file_path)

            # Save the original image
            original_image_path = os.path.join(output_folder, f"original_{filename}")
            image.save(original_image_path)

            # Save the inferred image (with bounding boxes)
            inferred_image_path = os.path.join(output_folder, f"based_model_inferred_{filename}")
            save_inferred_image(image, prediction, inferred_image_path)

            print(f"Processed: {filename}")


# Example to use the model for a folder of images
def main():
    input_folder = 'dataset/test/test_images'  # Set this to the path of the input folder
    output_folder = 'output_test'  # Set this to the path of the output folder

    # Process all images in the folder
    process_images(input_folder, output_folder)


# Run the script
if __name__ == '__main__':
    main()
