import os
from random import sample
from typing import Tuple

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


def build_data_pipe(root_path: str = "data/") -> Tuple[IterDataPipe, IterDataPipe]:
    icons_path = os.path.join(root_path, "Icons")
    front_path = os.path.join(root_path, "Front")
    img_paths = set(os.listdir(icons_path)).intersection(set(os.listdir(front_path)))

    valid_img_paths = set(sample(img_paths, int(len(img_paths) * 0.05)))
    train_img_paths = img_paths - valid_img_paths

    fn1, fn2 = pipes.IterableWrapper(train_img_paths).fork(2)

    train_front_sprites = (
        fn1.map(lambda f: os.path.join(front_path, f))
        .map(read_image)
        .map(Resize((96, 96), InterpolationMode.NEAREST))
        .map(normalize_image)
    )

    train_icon_sprites = (
        fn2.map(lambda f: os.path.join(icons_path, f))
        .map(read_image)
        .map(extract_left_part)
        .map(Resize((64, 64), InterpolationMode.NEAREST))
        .map(normalize_image)
    )

    fn1, fn2 = pipes.IterableWrapper(valid_img_paths).fork(2)

    valid_front_sprites = (
        fn1.map(lambda f: os.path.join(front_path, f))
        .map(read_image)
        .map(Resize((96, 96), InterpolationMode.NEAREST))
        .map(normalize_image)
    )

    valid_icon_sprites = (
        fn2.map(lambda f: os.path.join(icons_path, f))
        .map(read_image)
        .map(extract_left_part)
        .map(Resize((64, 64), InterpolationMode.NEAREST))
        .map(normalize_image)
    )

    return pipes.Zipper(train_front_sprites, train_icon_sprites), pipes.Zipper(
        valid_front_sprites, valid_icon_sprites
    )
