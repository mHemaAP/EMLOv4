import gdown
from pathlib import Path
from typing import Union
import lightning as L
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import zipfile
import os
import shutil
from sklearn.model_selection import train_test_split

class DogBreedImageDataModule(L.LightningDataModule):
    def __init__(self, dl_path: Union[str, Path] = "data", num_workers: int = 0, batch_size: int = 8, val_split: float = 0.2):
        super().__init__()
        self._dl_path = dl_path
        self._num_workers = num_workers
        self._batch_size = batch_size
        self.val_split = val_split

    def prepare_data(self):
        """Download the dataset from gdown and extract it."""
        # Extract the Google Drive file ID from the URL
        # file_id = "15O6OEGzBUQ46jNxFGaNT6Cl4u614STvz"  # This is the ID extracted from the provided link
        file_id = "1XmwKdyUgU98lazb0IaiqPL_1q86hTHlc"
        url = f"https://drive.google.com/uc?id={file_id}"  # Correct format for gdown

        output = Path(self._dl_path).joinpath("dog_breed_image_dataset.zip")
        
        # Download the dataset using gdown
        gdown.download(url, str(output), quiet=False)

        # Extract the downloaded dataset
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(self._dl_path)

        # Rename 'dog_breed_image_dataset' folder to 'dog_breeds'
        extracted_folder = Path(self._dl_path).joinpath("dog_breed_image_dataset")
        # new_folder = Path(self._dl_path).joinpath("dog_breeds")
        # if extracted_folder.exists() and not new_folder.exists():
        #     extracted_folder.rename(new_folder)
        
        # Create train and validation split
        self.split_data(extracted_folder)

    def split_data(self, dataset_path: Path):
        """Split dataset into train and validation."""
        train_dir = dataset_path / 'train'
        val_dir = dataset_path / 'validation'
        
        # Ensure train and validation folders exist
        if not train_dir.exists():
            train_dir.mkdir()
        if not val_dir.exists():
            val_dir.mkdir()

        # For each breed, move a portion of the images to the validation folder
        for breed_dir in dataset_path.iterdir():
            if breed_dir.is_dir() and breed_dir.name not in ['train', 'validation']:
                images = list(breed_dir.glob('*'))
                train_images, val_images = train_test_split(images, test_size=self.val_split)

                # Create breed folders in train and validation directories
                train_breed_dir = train_dir / breed_dir.name
                val_breed_dir = val_dir / breed_dir.name
                train_breed_dir.mkdir(exist_ok=True)
                val_breed_dir.mkdir(exist_ok=True)

                # Move images to respective directories
                for img in train_images:
                    shutil.move(str(img), train_breed_dir / img.name)
                for img in val_images:
                    shutil.move(str(img), val_breed_dir / img.name)
        
        # After splitting, remove empty breed folders in the root dataset
        for breed_dir in dataset_path.iterdir():
            if breed_dir.is_dir() and breed_dir.name not in ['train', 'validation']:
                shutil.rmtree(breed_dir)

    @property
    def data_path(self):
        # return Path(self._dl_path).joinpath("dog_breeds")
        return Path(self._dl_path).joinpath("dog_breed_image_dataset")

    @property
    def normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @property
    def train_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            self.normalize_transform,
        ])

    @property
    def valid_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize_transform
        ])

    def create_dataset(self, root, transform):
        return ImageFolder(root=root, transform=transform)

    def __dataloader(self, train: bool):
        """Train/validation/test loaders."""
        if train:
            dataset = self.create_dataset(self.data_path.joinpath("train"), self.train_transform)
        else:
            dataset = self.create_dataset(self.data_path.joinpath("validation"), self.valid_transform)
        return DataLoader(dataset=dataset, batch_size=self._batch_size, num_workers=self._num_workers, shuffle=train)

    def train_dataloader(self):
        return self.__dataloader(train=True)

    def val_dataloader(self):
        return self.__dataloader(train=False)

    def test_dataloader(self):
        return self.__dataloader(train=False)  # Using validation dataset for testing
