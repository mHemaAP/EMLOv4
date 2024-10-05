import argparse
import os
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, RichModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from datamodules.catdog_datamodule import CatDogImageDataModule
from datamodules.dogbreed_datamodule import DogBreedImageDataModule  
from models.catdog_classifier import CatDogClassifier
from models.dogbreed_classifier import DogBreedClassifier 
from utils.logging_utils import setup_logger, task_wrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Choose the image set for validation.")
    parser.add_argument(
        "--image_type",
        type=int,
        choices=[1, 2],
        default=2,  # Default to Dog Breed
        help="1 for Classifying Cat & Dog images, 2 for Classifying Dog Breed images"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to the checkpoint file. If not specified, the latest one will be used."
    )
    return parser.parse_args()

@task_wrapper
def load_and_validate(checkpoint_path, data_module, model_class, trainer):
    """Load the model from checkpoint and validate it."""
    model = model_class.load_from_checkpoint(checkpoint_path)
    print(f"Loaded model from {checkpoint_path}")
    
    # Validate the model on the validation dataset
    val_results = trainer.validate(model, data_module)
    
    # Print the validation metrics
    print("Validation Metrics: ", val_results)

def main():
    # Parse command-line arguments
    args = parse_args()
    image_type = "catdog" if args.image_type == 1 else "dogbreed"

    # Set up paths
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    log_dir = base_dir / "logs"
    
    # Set up logger
    setup_logger(log_dir / "validate_log.log")

    # Initialize DataModule and Model based on user input
    if image_type == "catdog":
        data_module = CatDogImageDataModule(dl_path=data_dir, batch_size=32, num_workers=0)
        model_class = CatDogClassifier
        checkpoint_dir = log_dir / "catdog_classification" / "checkpoints"
        logger_name = "catdog_classification"
    else:
        data_module = DogBreedImageDataModule(dl_path=data_dir, batch_size=32, num_workers=0)
        model_class = DogBreedClassifier
        checkpoint_dir = log_dir / "dogbreed_classification" / "checkpoints"
        logger_name = "dogbreed_classification"

    # List available checkpoints
    checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoint_files:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Specified checkpoint does not exist: {checkpoint_path}")
            return
    else:
        # Select the latest checkpoint by modification time
        checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
        print(f"Using the latest checkpoint: {checkpoint_path.name}")

    # Initialize Trainer with Rich Progress Bar and Model Summary
    trainer = L.Trainer(
        accelerator="auto",
        logger=TensorBoardLogger(save_dir=log_dir, name=logger_name),
        callbacks=[
            RichProgressBar(),
            RichModelSummary(max_depth=2)
        ]
    )

    # Load the model from the checkpoint and validate
    load_and_validate(checkpoint_path, data_module, model_class, trainer)

if __name__ == "__main__":
    main()
