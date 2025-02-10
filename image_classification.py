import os
import torch
import argparse
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Function to modify the Faster R-CNN model for classification
def modify_model_for_classification(model, num_classes=2):
    # Modify the classification head to match the number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Load the pre-trained Faster R-CNN model
def load_model(custom_model_path=None):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model = modify_model_for_classification(model, num_classes=2)  # 2 classes: fish and background
    
    # Load your custom-trained model if the path is provided
    if custom_model_path:
        model.load_state_dict(torch.load(custom_model_path, map_location=torch.device('cpu')))
        print(f"Custom model loaded from {custom_model_path}")
    
    # Force use of CPU since you don't have a GPU
    device = torch.device("cpu")  # Use CPU
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model, device

# Define a transform to convert image to tensor
transform = transforms.Compose([transforms.ToTensor()])

# Function to classify an image with confidence threshold
def classify_image(image_path, model, device, threshold=0.65):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        prediction = model(image_tensor)

    # Filter predictions based on the confidence threshold
    scores = prediction[0]['scores'].cpu().numpy()  # Get the confidence scores
    labels = prediction[0]['labels'].cpu().numpy()  # Get the predicted labels

    # Find the class with the highest score above the threshold
    high_confidence_preds = [(label, score) for label, score in zip(labels, scores) if score > threshold]

    if high_confidence_preds:
        # Sort predictions by score in descending order
        high_confidence_preds.sort(key=lambda x: x[1], reverse=True)
        predicted_class = high_confidence_preds[0][0]  # Get the class of the highest-scoring prediction
        return "Fish" if predicted_class == 1 else "Nothing found here"
    else:
        return "Nothing found here"

# Process all images in the directory
def classify_images_in_directory(image_dir, model, device, threshold=0.9):
    # List all files in the directory
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('jpg', 'jpeg', 'png'))]
    
    if not image_files:
        print("No images found in the directory.")
        return
    
    print(f"Classifying {len(image_files)} images in '{image_dir}'...\n")
    
    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)
        predicted_label = classify_image(image_path, model, device, threshold)
        print(f"Image: {image_name}, Predicted label: {predicted_label}")

# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Classify all images in a directory using Faster R-CNN.")
    parser.add_argument('--image_dir', type=str, required=True, help="Directory containing images to classify.")
    return parser.parse_args()

if __name__ == "__main__":
    # Define the custom model path as 'fish'
    fish = '../datasets/fish/fish-faster-rcnn.pth'  # Put your custom model path here
    
    # Parse arguments
    args = parse_args()

    # Load the pre-trained Faster R-CNN model with your custom weights (using 'fish')
    model, device = load_model(custom_model_path=fish)

    # Classify images in the specified directory
    classify_images_in_directory(args.image_dir, model, device)
