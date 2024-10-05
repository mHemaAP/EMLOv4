import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from models.catdog_classifier import CatDogClassifier
from models.dogbreed_classifier import DogBreedClassifier  # Import the Dog Breed Classifier
from utils.logging_utils import setup_logger, task_wrapper, get_rich_progress

# Define dog breed labels
dog_breed_labels = [
    'beagle', 'boxer', 'bulldog', 'dachshund', 'german_shepherd',
    'golden_retriever', 'labrador_retriever', 'poodle', 'rottweiler', 'yorkshire_terrier'
]

@task_wrapper
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return img, transform(img).unsqueeze(0)

@task_wrapper
def infer(model, image_tensor, image_type):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    if image_type == "catdog":
        class_labels = ['cat', 'dog']
    else:
        class_labels = dog_breed_labels
        
    predicted_label = class_labels[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    return predicted_label, confidence

@task_wrapper
def save_prediction_image(image, predicted_label, confidence, output_path):
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Predicted: {predicted_label} (Confidence: {confidence:.2f})")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

@task_wrapper
def main(args):
    if args.image_type == "catdog":
        model = CatDogClassifier.load_from_checkpoint(args.ckpt_path)
    else:
        model = DogBreedClassifier.load_from_checkpoint(args.ckpt_path)

    model.eval()

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    image_files = list(input_folder.glob('*'))
    with get_rich_progress() as progress:
        task = progress.add_task("[green]Processing images...", total=len(image_files))
        
        for image_file in image_files:
            if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img, img_tensor = load_image(image_file)
                predicted_label, confidence = infer(model, img_tensor.to(model.device), args.image_type)
                
                output_file = output_folder / f"{image_file.stem}_prediction.png"
                save_prediction_image(img, predicted_label, confidence, output_file)
                
                progress.console.print(f"Processed {image_file.name}: {predicted_label} ({confidence:.2f})")
                progress.advance(task)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer using trained Classifier")
    
    # Set default input folder to 'samples'
    parser.add_argument("--input_folder", type=str, default="samples", help="Path to input folder containing images")
    
    # Set default output folder to 'output'
    parser.add_argument("--output_folder", type=str, default="output", help="Path to output folder for predictions")
    
    # Set default checkpoint to the last DogBreedClassifier checkpoint
    last_checkpoint_path = sorted(Path("logs/dogbreed_classification/checkpoints").glob("*.ckpt"), reverse=True)[0]
    parser.add_argument("--ckpt_path", type=str, default=last_checkpoint_path, help="Path to model checkpoint")
    
    parser.add_argument("--image_type", type=int, choices=[1, 2], default=2, help="1 for Classifying Cat & Dog images, 2 for Classifying Dog Breed images")
    
    args = parser.parse_args()

    log_dir = Path(__file__).resolve().parent.parent / "logs"
    setup_logger(log_dir / "infer_log.log")

    # Convert classifier type to string for easier handling
    args.image_type = "catdog" if args.image_type == 1 else "dogbreed"

    main(args)
