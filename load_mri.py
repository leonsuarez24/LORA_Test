import requests
import shutil
import os
import zipfile
import torch
import torchvision
import numpy as np
from torchvision import transforms
from PIL import Image
from io import BytesIO
from tqdm import tqdm


class MRIData(torch.utils.data.Dataset):
    """fastMRI dataset (knee subset)."""

    def __init__(
        self, root_dir, train=True, sample_index=None, tag=900, transform=None
    ):
        x = torch.load(str(root_dir) + ".pt")
        x = x.squeeze()
        self.transform = transform

        if train:
            self.x = x[:tag]
        else:
            self.x = x[tag:, ...]

        self.x = torch.stack([self.x, torch.zeros_like(self.x)], dim=1)

        if sample_index is not None:
            self.x = self.x[sample_index].unsqueeze(0)

    def __getitem__(self, index):
        x = self.x[index]

        if self.transform is not None:
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.x)


def get_git_root():
    import git

    git_repo = git.Repo(".", search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root


def get_image_dataset_url(dataset_name, file_type="zip"):
    return (
        "https://huggingface.co/datasets/deepinv/images/resolve/main/"
        + dataset_name
        + "."
        + file_type
        + "?download=true"
    )


def get_degradation_url(file_name):
    return (
        "https://huggingface.co/datasets/deepinv/degradations/resolve/main/"
        + file_name
        + "?download=true"
    )


def get_image_url(file_name):
    return (
        "https://huggingface.co/datasets/deepinv/images/resolve/main/"
        + file_name
        + "?download=true"
    )


def load_dataset(
    dataset_name, data_dir, transform, download=True, url=None, train=True
):
    dataset_dir = data_dir / dataset_name
    if dataset_name == "fastmri_knee_singlecoil":
        file_type = "pt"
    else:
        file_type = "zip"
    if download and not dataset_dir.exists():
        dataset_dir.mkdir(parents=True, exist_ok=True)
        if url is None:
            url = get_image_dataset_url(dataset_name, file_type)
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        print("Downloading " + str(dataset_dir) + f".{file_type}")
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        with open(str(dataset_dir) + f".{file_type}", "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if file_type == "zip":
            with zipfile.ZipFile(str(dataset_dir) + ".zip") as zip_ref:
                zip_ref.extractall(str(data_dir))
            # remove temp file
            os.remove(str(dataset_dir) + f".{file_type}")
            print(f"{dataset_name} dataset downloaded in {data_dir}")
        else:
            shutil.move(
                str(dataset_dir) + f".{file_type}",
                str(dataset_dir / dataset_name) + f".{file_type}",
            )
    if dataset_name == "fastmri_knee_singlecoil":
        dataset = MRIData(
            train=train, root_dir=dataset_dir / dataset_name, transform=transform
        )
    else:
        dataset = torchvision.datasets.ImageFolder(
            root=dataset_dir, transform=transform
        )
    return dataset