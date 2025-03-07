import math

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SubsetRandomSampler

from .base_provider import DataProvider
from mcunet.utils.my_dataloader import MyRandomResizedCrop, MyDistributedSampler
from mcunet.utils.my_dataloader.my_data_loader import MyDataLoader

__all__ = ["AuroraDataProvider", "AuroraDataset"]

class AuroraDataProvider(DataProvider):
    DEFAULT_PATH = "/home/vilatsut/Desktop/archive"

    def __init__(
        self,
        data_path=None,
        train_batch_size=256,
        test_batch_size=256,
        valid_size=None,
        test_size=0.2,
        n_worker=2,
        resize_scale=0.08,
        distort_color=None,
        image_size=224,
        num_replicas=None,
        rank=None,
        seed=42
    ):
        self._data_path = data_path
        self.valid_size =valid_size
        self.test_size = test_size
        self.seed = seed
        self.image_size = image_size  # int or list of int    
        self.distort_color = "None" if distort_color is None else distort_color
        self.resize_scale = resize_scale

        self.dataset_mean = [0.23280394, 0.24616548, 0.26092353]
        self.dataset_std = [0.16994016, 0.17286949, 0.16250615]

        self._valid_transform_dict = {}

        if isinstance(self.image_size, list):
            self.image_size.sort()
            MyRandomResizedCrop.IMAGE_SIZE_LIST = self.image_size.copy()
            MyRandomResizedCrop.ACTIVE_SIZE = max(self.image_size)

            for img_size in self.image_size:
                self._valid_transform_dict[img_size] = self.build_valid_transform(
                    img_size
                )
            self.active_img_size = max(self.image_size)
            valid_transforms = self._valid_transform_dict[self.active_img_size]
            train_loader_class = MyDataLoader # randomly sample image size for each batch of training image
        else:
            self.active_img_size = self.image_size
            valid_transforms = self.build_valid_transform()
            train_loader_class = torch.utils.data.DataLoader

        train_transforms = self.build_train_transform(image_size=image_size)

        train_dataset = AuroraDataset(
            root=self.data_path,
            transform=train_transforms,
            split="train",
            test_size=test_size,  # 20% of the data is used for testing
            valid_size=valid_size,
            random_seed=seed
        )

        if valid_size is not None:
            valid_dataset = AuroraDataset(
                root=self.data_path,
                transform=valid_transforms,
                split="val",
                test_size=test_size,
                valid_size=valid_size,
                random_seed=seed
            )
        else:
            valid_dataset = None

        test_dataset = AuroraDataset(
            root=self.data_path,
            transform=valid_transforms,
            split="test",
            test_size=test_size,
            valid_size=valid_size,
            random_seed=seed
        )

        if num_replicas is not None:
            train_sampler = DistributedSampler(
                train_dataset, num_replicas, rank, True
            )
            test_sampler = DistributedSampler(
                test_dataset, num_replicas, rank, True
            )
            if valid_dataset is not None:
                valid_sampler = DistributedSampler(
                    valid_dataset, num_replicas, rank, True
                )
            else:
                valid_sampler = None
        else:
            train_sampler = RandomSampler(train_dataset)
            test_sampler = RandomSampler(test_dataset)
            if valid_dataset is not None:
                valid_sampler = RandomSampler(valid_dataset)
            else:
                valid_sampler = None

        self.train = train_loader_class(
            train_dataset,
            batch_size=train_batch_size,
            sampler=train_sampler,
            num_workers=n_worker,
            pin_memory=True,
        )

        if valid_dataset is not None:
            self.valid = DataLoader(
                valid_dataset,
                batch_size=test_batch_size,
                sampler=valid_sampler,
                num_workers=n_worker,
                pin_memory=True,
            )
        else:
            self.valid = None

        self.test = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            sampler=test_sampler,
            num_workers=n_worker,
            pin_memory=True
        )
        if self.valid is None:
            self.valid = self.test

    @staticmethod
    def name():
        return "aurora"

    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W

    @property
    def n_classes(self):
        return 2

    @property
    def data_path(self):
        if self._data_path is None:
            self._data_path = self.DEFAULT_PATH
            if not os.path.exists(self._data_path):
                self._data_path = os.path.expanduser("~/aurora")
        return self._data_path

    @property
    def data_url(self):
        raise ValueError("unable to download %s" % self.name())

    def build_train_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size
        
        if isinstance(image_size, list):
            resize_transform_class = MyRandomResizedCrop
        else:
            resize_transform_class = transforms.RandomResizedCrop
        
        # color augmentation (optional)
        color_transform = None
        if self.distort_color == "torch":
            color_transform = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
        elif self.distort_color == "tf":
            color_transform = transforms.ColorJitter(
                brightness=32.0 / 255.0, saturation=0.5
            )

        return transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.uint8),
            resize_transform_class(image_size, scale=(self.resize_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
            color_transform,
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(self.dataset_mean, self.dataset_std)
        ])

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size
        return transforms.Compose([
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.uint8),
            transforms.Resize(int(math.ceil(image_size / 0.875))),
            transforms.CenterCrop(image_size),            
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(self.dataset_mean, self.dataset_std)
        ])
        
    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[
                self.active_img_size
            ] = self.build_valid_transform()
        # change the transform of the valid and test set
        self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
        self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]

    def build_sub_train_loader(
        self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None
    ):
        # used for resetting BN running statistics
        if self.__dict__.get("sub_train_%d" % self.active_img_size, None) is None:
            if num_worker is None:
                num_worker = self.train.num_workers

            n_samples = len(self.train.dataset)
            g = torch.Generator()
            g.manual_seed(DataProvider.SUB_SEED)
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()
            chosen_indexes = rand_indexes[:n_images]

            train_transforms = self.build_train_transform(image_size=self.active_img_size)
            new_train_dataset = AuroraDataset(
                root=self.data_path,
                transform=train_transforms,
                split="train",
                test_size=self.test_size,  # 20% of the data is used for testing
                valid_size=self.valid_size,
                random_seed=self.seed
            )
            
            if num_replicas is not None:
                sub_sampler = MyDistributedSampler(
                    new_train_dataset,
                    num_replicas,
                    rank,
                    True,
                    np.array(chosen_indexes),
                )
            else:
                sub_sampler = SubsetRandomSampler(chosen_indexes)

            sub_data_loader = DataLoader(
                new_train_dataset,
                batch_size=batch_size,
                sampler=sub_sampler,
                num_workers=num_worker,
                pin_memory=True,
            )
            
            self.__dict__["sub_train_%d" % self.active_img_size] = []
            for images, labels in sub_data_loader:
                self.__dict__["sub_train_%d" % self.active_img_size].append(
                    (images, labels)
                )
        return self.__dict__["sub_train_%d" % self.active_img_size]


from sklearn.model_selection import train_test_split
import torch
from torchvision import datasets
from typing import Union, Callable, Optional, Tuple, List
from pathlib import Path
from PIL import Image
import os

class AuroraDataset(datasets.VisionDataset):
    """
    Aurora Dataset with a single folder structure.
    
    Args:
        root (str or pathlib.Path): Root directory of the dataset containing `aurora` and `no_aurora` folders.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        split (str, optional): One of ['train', 'val', 'test'].
        test_size (float, optional): Fraction of the dataset used for testing (default 0.2).
        valid_size (float, optional): Fraction of the training set used for validation (default 0.1).
        random_seed (int, optional): Seed for reproducibility.
    """

    classes = ["no_aurora", "aurora"]

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        download: Optional[bool] = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        test_size: Optional[float] = 0.2,
        valid_size: Optional[float] = 0.1,
        random_seed: Optional[int] = 101
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)

        if download:
            self._download_data()
        self.data, self.targets = self._load_data()

        if split not in ["train", "val", "test"]:
            raise ValueError("split must be one of ['train', 'val', 'test']")

        # Split into train and test
        train_data, test_data, train_targets, test_targets = train_test_split(
            self.data, self.targets, test_size=test_size, stratify=self.targets, random_state=random_seed
        )

        if valid_size is not None:
            # Split train further into train and validation
            train_data, val_data, train_targets, val_targets = train_test_split(
                train_data, train_targets, test_size=valid_size, stratify=train_targets, random_state=random_seed
            )

        # Assign data based on requested split
        if split == "train":
            self.data, self.targets = train_data, train_targets
        elif split == "val":
            if valid_size is None:
                raise ValueError("Validation split requested but valid_size is None")
            self.data, self.targets = val_data, val_targets
        else:
            self.data, self.targets = test_data, test_targets

    def _download_data(self) -> None:
        """
        Downloads the dataset from Kaggle and extracts it to the root directory.
        """

        # Set up Kaggle API and download the dataset
        kaggle_dataset = "villehokkinen/aurora"
        try:
            if not os.path.exists(str(self.root)):
                kaggle.api.dataset_download_files(
                    kaggle_dataset,
                    path=str(self.root),
                    unzip=True,
                )
                print(f"Dataset {kaggle_dataset} downloaded and extracted to {self.root}.")
            else:
                print(f"Dataset {kaggle_dataset} already exists in {self.root}")
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset {kaggle_dataset} from Kaggle: {e}")


    def _load_data(self) -> Tuple[List[str], List[int]]:
        """
        Loads the dataset from the specified root directory.
        Expects subdirectories `aurora` and `no_aurora`.
        """
        data = []
        targets = []

        for class_index, class_name in enumerate(self.classes):
            class_path = os.path.join(self.root, class_name)
            if not os.path.exists(class_path):
                raise RuntimeError(f"Class folder {class_name} not found at {class_path}")

            for file_name in os.listdir(class_path):
                if file_name.lower().endswith(("png", "jpg", "jpeg", "bmp", "tiff")):
                    data.append(os.path.join(class_path, file_name))
                    targets.append(class_index)

        return data, targets

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        """
        Returns an image and its label.
        """
        img_path = self.data[index]
        target = self.targets[index]

        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target
    
    def __get_items__(self, indices: List[int]) -> List[Tuple[Image.Image, int]]:
        """
        Returns a list of images and their labels for the given list of indices.
        """
        items = []
        for index in indices:
            img_path = self.data[index]
            target = self.targets[index]

            img = Image.open(img_path)

            if self.transform:
                img = self.transform(img)
            if self.target_transform:
                target = self.target_transform(target)

            items.append((img, target))

        return items
    
    def __len__(self) -> int:
        return len(self.data)