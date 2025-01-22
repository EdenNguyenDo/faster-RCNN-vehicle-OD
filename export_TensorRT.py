from torch2trt import torch2trt
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
from PIL import Image


# Load the Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True).eval().cuda()

# Preprocessing function for the image
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((800, 800)),  # Resize to 800x800
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img)
    return [img_tensor]  # Return as a list of tensors


# Wrapper to handle input/output for Torch2TRT compatibility
class FasterRCNNWrapper(torch.nn.Module):
    def __init__(self, model):
        super(FasterRCNNWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Convert the 4D input tensor to a list of 3D tensors
        x = x.squeeze(0)  # Remove the batch dimension
        return self.model([x])[0]  # Return only the first output (detections)


class Wrapper(torch.nn.Module):
    """
    A wrapper around fasterrcnn_resnet50_fpn so we can:
      1. Accept a standard batch tensor of shape [N, 3, H, W].
      2. Convert it to a list of 3D tensors (each [3, H, W]) that the model expects.
      3. Return bounding boxes, scores, and labels separately.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch_images):
        # batch_images shape: [N, 3, H, W]
        # Convert this into a list of images [img0, img1, ...], each [3, H, W]
        images_list = []
        for i in range(batch_images.size(0)):
            images_list.append(batch_images[i])

        # The model returns a list of dicts, one per image in the batch
        # Each dict has keys ["boxes", "labels", "scores", ...]
        outputs = self.model(images_list)  # list of dict, length = N

        # For ONNX export, we often flatten out the dictionary so we have fixed outputs.
        # We'll assume a single image in the batch for simplicity below,
        # but you can handle multiple images if needed.
        # If you do have multiple images, you'll need a more sophisticated approach
        # to handle variable number of detections across images.
        out = outputs[0]  # get the dict for the first image
        return out["boxes"], out["scores"], out["labels"]

def main(img_path):
    # Preprocess the image
    img_list = preprocess_image(img_path)
    img_tensor = img_list[0]  # Extract the tensor from the list

    wrapped = Wrapper(model)

    dummy_batch = torch.randn(1, 3, 800, 800).cuda()

    # 3) Export to ONNX
    #    We'll produce 3 outputs for bounding boxes, scores, and labels
    #    Typically, Faster R-CNN actually returns a list of dicts.
    #    We'll rely on PyTorch to flatten that into separate outputs.
    torch.onnx.export(
        wrapped,
        dummy_batch,
        "fasterrcnn_resnet50_fpn.onnx",
        input_names=["images"],
        output_names=["boxes", "scores", "labels"],  # simplified naming
        opset_version=11,  # >=11 recommended for detection models
        do_constant_folding=True,
        dynamic_axes={
            "images": {0: "num_channels", 1: "height", 2: "width"},
            "boxes": {0: "num_detections"},
            "scores": {0: "num_detections"},
            "labels": {0: "num_detections"}
        }
    )

    print("Faster R-CNN has been exported to ONNX successfully!")

    # Print the output for verification


if __name__ == "__main__":
    main("C:/transmetric/dev/python/AI_camera/trial/FRCNN+Bytetrack/inference_dataset/images/2024_0323_120137_100A/2024_0323_120137_100A_118.jpg")
