import cv2
import torch
import argparse
import os
import subprocess
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

# Load the model with updated weights argument
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)

# Define available models
MODEL_MAP = {
    "fish": "C:/Users/krupal/Coding/FLL/pytorch-objectDetection/datasets/fish/fish-faster-rcnn.pth",  # Custom model path
}

# Supported image file extensions
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Class names for your custom model
class_names = ['background', 'fish']  # Replace with your actual class names if different


def install_requirements():
    """Ensure all required libraries are installed."""
    try:
        import torch
        import torchvision
    except ImportError:
        print("PyTorch or torchvision is not installed. Installing now...")
        subprocess.check_call(["pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"])
    try:
        import cv2
    except ImportError:
        print("OpenCV is not installed. Installing now...")
        subprocess.check_call(["pip", "install", "opencv-python"])


def load_custom_model(model_path, model_fn, num_classes):
    """Load a custom-trained model."""
    model = model_fn(pretrained=False)  # Do not load the pretrained weights for custom model
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)  # Load the model weights
    model.eval()  # Set the model to evaluation mode
    return model


def ensure_output_directory():
    """Ensure the output directory exists."""
    if not os.path.exists("output-object_detect"):
        os.makedirs("output-object_detect")


def process_image(model, image_path):
    """Process a single image."""
    _, ext = os.path.splitext(image_path)
    if ext.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        print(f"Unsupported image format: {ext}. Supported formats: {SUPPORTED_IMAGE_EXTENSIONS}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to read image: {image_path}")
        return

    frame = detect_and_draw(frame, model)
    output_path = os.path.join("output", os.path.basename(image_path))
    cv2.imwrite(output_path, frame)
    print(f"Processed image saved to: {output_path}")


def process_video(model, video_path):
    """Process a single video."""
    cap = cv2.VideoCapture(video_path)
    output_path = os.path.join("output", os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_and_draw(frame, model)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Processed video saved to: {output_path}")


def process_directory(model, directory_path):
    """Process all images and videos in a directory."""
    if not os.path.exists(directory_path):
        print(f"Directory does not exist: {directory_path}")
        return

    files = os.listdir(directory_path)
    for file in files:
        file_path = os.path.join(directory_path, file)
        _, ext = os.path.splitext(file)

        if ext.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            print(f"Processing image: {file_path}")
            process_image(model, file_path)
        elif ext.lower() in {".mp4", ".avi", ".mov", ".mkv"}:  # Add more video extensions as needed
            print(f"Processing video: {file_path}")
            process_video(model, file_path)
        else:
            print(f"Skipping unsupported file: {file_path}")


def detect_and_draw(frame, model):
    """Perform object detection and draw results on the frame."""
    image_tensor = F.to_tensor(frame).unsqueeze(0)
    with torch.no_grad():
        predictions = model(image_tensor)

    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        if score > 0.9:  # Only consider predictions with score > 0.5
            if label.item() == 1:  # Valid "fish" class
                x1, y1, x2, y2 = box
                label_name = class_names[1]  # Always map label 1 to "fish"
                confidence = score.item() * 100  # Convert score to percentage
                label_text = f"{label_name}: {confidence:.2f}%"
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Put label with confidence above the bounding box
                cv2.putText(frame, label_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def main():
    install_requirements()
    ensure_output_directory()

    parser = argparse.ArgumentParser(description="PyTorch Object Detection")
    parser.add_argument("--source", type=str, required=True, 
                        help="Input source: 'image:<path>', 'video:<path>', or 'directory:<path>'")
    parser.add_argument("--model", type=str, required=True, choices=MODEL_MAP.keys(),
                        help="Choose the model: 'fish' (Custom Model), 'fcnn', or 'retinanet'")
    args = parser.parse_args()

    try:
        model_fn = MODEL_MAP[args.model]
        if args.model == "fish":  # Custom model path
            num_classes = len(class_names)
            model = load_custom_model(model_fn, fasterrcnn_resnet50_fpn, num_classes)
        else:
            model = model_fn(pretrained=True)
    except KeyError:
        print(f"Model '{args.model}' is not supported. Available options: {list(MODEL_MAP.keys())}")
        return

    model.eval()

    if args.source.startswith("image:"):
        image_path = args.source.split(":", 1)[1]
        process_image(model, image_path)
    elif args.source.startswith("video:"):
        video_path = args.source.split(":", 1)[1]
        process_video(model, video_path)
    elif args.source.startswith("directory:"):
        directory_path = args.source.split(":", 1)[1]
        process_directory(model, directory_path)
    else:
        print("Invalid source. Use 'image:<path>', 'video:<path>', or 'directory:<path>'.")


if __name__ == "__main__":
    main()