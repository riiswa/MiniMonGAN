import os

import torch
from torch.utils.data import IterDataPipe
from torchvision.io import ImageReadMode
from torchvision.transforms import Resize, InterpolationMode
from torchvision import io
import torch.utils.data.datapipes.iter as pipes


def read_image(path: str):
    return io.read_image(path, ImageReadMode.RGB_ALPHA)


def normalize_image(image: torch.Tensor):
    return 2 * (image / 255.0) - 1


def extract_left_part(image: torch.Tensor):
    return image[:, :, : image.shape[2] // 2]


def build_data_pipe(root_path: str = "data/") -> IterDataPipe:
    icons_path = os.path.join(root_path, "Icons")
    front_path = os.path.join(root_path, "Front")
    fn1, fn2 = pipes.IterableWrapper(set(os.listdir(icons_path)).intersection(set(os.listdir(front_path)))).fork(2)

    front_sprites = (
        fn1.map(lambda f: os.path.join(front_path, f))
        .map(read_image)
        .map(Resize((64, 64), InterpolationMode.NEAREST))
        .map(normalize_image)
    )

    icon_sprites = (
        fn2.map(lambda f: os.path.join(icons_path, f))
        .map(read_image)
        .map(extract_left_part)
        .map(Resize((32, 32), InterpolationMode.NEAREST))
        .map(normalize_image)
    )

    return pipes.Zipper(front_sprites, icon_sprites)