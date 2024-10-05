import argparse
import os
from pathlib import Path
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from datamodules.catdog_datamodule import CatDogImageDataModule
from datamodules.dogbreed_datamodule import DogBreedImageDataModule  
from models.catdog_classifier import CatDogClassifier
from models.dogbreed_classifier import DogBreedClassifier 
from utils.logging_utils import setup_logger, task_wrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Choose the image set for training the classification model")
    parser.add_argument(
        "--image_type",
        type=int,
        choices=[1, 2],
        default=2,  # Default to Dog Breed
        help="1 for Classifying Cat & Dog images, 2 for Classifying Dog Breed images"
    )
    return parser.parse_args()

@task_wrapper
def train_and_test(data_module, model, trainer):
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

def main():
    # Parse command-line arguments
    args = parse_args()
    image_type = "catdog" if args.image_type == 1 else "dogbreed"

    # Set up paths
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"
    log_dir = base_dir / "logs"
    
    # Set up logger
    setup_logger(log_dir / "train_log.log")

    # Initialize DataModule and Model based on user input
    if image_type == "catdog":
        data_module = CatDogImageDataModule(dl_path=data_dir, batch_size=32, num_workers=0)
        model = CatDogClassifier(lr=1e-3)
        checkpoint_dir = log_dir / "catdog_classification" / "checkpoints"
        logger_name = "catdog_classification"
    else:
        data_module = DogBreedImageDataModule(dl_path=data_dir, batch_size=32, num_workers=0)
        model = DogBreedClassifier(lr=1e-3)
        checkpoint_dir = log_dir / "dogbreed_classification" / "checkpoints"
        logger_name = "dogbreed_classification"

    # Set up checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch={epoch:02d}-val_loss={val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    # Initialize Trainer
    trainer = L.Trainer(
        max_epochs=1,
        callbacks=[
            checkpoint_callback,
            RichProgressBar(),
            RichModelSummary(max_depth=2)
        ],
        accelerator="auto",
        logger=TensorBoardLogger(save_dir=log_dir, name=logger_name),
    )

    # Train and test the model
    train_and_test(data_module, model, trainer)

if __name__ == "__main__":
    main()
