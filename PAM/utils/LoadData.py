# ------------------------------------------------------------------------------
# Reference: https://github.com/qjadud1994/DRS/blob/main/utils/LoadData.py
# ------------------------------------------------------------------------------

from .transforms import transforms
from torch.utils.data import DataLoader

# import torch
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image


def train_data_loader(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]

    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    tsfm_train = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
            ),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean_vals, std_vals),
        ]
    )

    train_list = os.path.join(args.root_dir, "ImageSets/Segmentation/train_cls.txt")

    img_train = VOCDataset(
        train_list,
        crop_size,
        root_dir=args.root_dir,
        num_classes=args.num_classes,
        transform=tsfm_train,
        mode="train",
    )

    train_loader = DataLoader(
        img_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    return train_loader


def test_data_loader(args):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]

    # input_size = int(args.input_size)
    crop_size = int(args.crop_size)

    tsfm_test = transforms.Compose(
        [
            transforms.Resize(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean_vals, std_vals),
        ]
    )

    test_list = os.path.join(args.root_dir, "ImageSets/Segmentation/train_cls.txt")

    img_test = VOCDataset(
        test_list,
        crop_size,
        root_dir=args.root_dir,
        num_classes=args.num_classes,
        transform=tsfm_test,
        mode="test",
    )

    test_loader = DataLoader(
        img_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return test_loader


class VOCDataset(Dataset):
    def __init__(
        self,
        datalist_file,
        input_size,
        root_dir,
        num_classes=20,
        transform=None,
        mode="train",
    ):
        self.root_dir = root_dir
        self.mode = mode
        self.datalist_file = datalist_file
        self.transform = transform
        self.num_classes = num_classes

        self.image_list, self.label_list = self.read_labeled_image_list(
            self.root_dir, self.datalist_file
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(img_name).convert("RGB")

        meta = {"img_name": img_name, "ori_size": image.size}

        if self.transform is not None:
            image = self.transform(image)

        return image, self.label_list[idx], meta

    def read_labeled_image_list(self, data_dir, data_list):
        img_dir = os.path.join(data_dir, "JPEGImages")

        with open(data_list, "r") as f:
            lines = f.readlines()

        img_name_list = []
        img_labels = []

        for line in lines:
            fields = line.strip().split()
            image = fields[0] + ".jpg"

            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields) - 1):
                index = int(fields[i + 1])
                labels[index] = 1.0

            img_name_list.append(os.path.join(img_dir, image))
            img_labels.append(labels)

        return img_name_list, img_labels


class CocoDataset(Dataset):
    """ Prepare data in coco-format dataset

    Args:
        datalist_file (str): text file image-class_ids data
        root_dir (str): text file image[.jpg]-class_ids data
        num_classes (int): number of categories
        transform (object): image transform functions
        mode (str): process mode: train, val

    """
    def __init__(
        self,
        datalist_file: str,
        root_dir: str,
        num_classes=80,
        transform=None,
        mode="train",
    ):
        self.datalist_file = datalist_file
        self.root_dir = root_dir
        self.num_classes = num_classes
        self.transform = transform
        self.mode = mode

        self.image_list, self.label_list = self.read_labeled_image_list()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        image = Image.open(img_name).convert("RGB")

        meta = {"img_name": img_name, "ori_size": image.size}

        if self.transform is not None:
            image = self.transform(image)

        return image, self.label_list[idx], meta

    def read_labeled_image_list(self):
        img_dir = os.path.join(self.root_dir, f"image_{self.mode}")

        with open(self.datalist_file, "r") as f:
            lines = f.readlines()

        img_name_list = []
        img_labels = []

        for line in lines:
            fields = line.strip().split()
            image = fields[0] + ".jpg"

            labels = np.zeros((self.num_classes,), dtype=np.float32)
            for i in range(len(fields) - 1):
                index = int(fields[i + 1])
                labels[index] = 1.0

            img_name_list.append(os.path.join(img_dir, image))
            img_labels.append(labels)

        return img_name_list, img_labels