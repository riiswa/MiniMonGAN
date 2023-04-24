import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_pipelines import build_data_pipe
from model import Generator, Discriminator


def unormalize_image(image: torch.Tensor):
    return (((image + 1) / 2) * 255).long()


def visualize_sprites(fronts, icons, n_rows=4):
    plt.tight_layout()
    fronts = fronts.permute(0, 2, 3, 1)
    icons = icons.permute(0, 2, 3, 1)
    n_cols = fronts.size(0) // n_rows
    fig, ax = plt.subplots(n_rows * 2,  n_cols, figsize=(10, 9))
    for i in range(n_rows * 2):
        for j in range(n_cols):
            img_index = i//2 + n_rows * j
            if i % 2 == 0:
                ax[i, j].imshow(unormalize_image(fronts[img_index]).detach().cpu().numpy())
            else:
                ax[i, j].imshow(unormalize_image(icons[img_index]).detach().cpu().numpy())
            ax[i, j].axis('off')
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    args = parser.parse_args()

    epochs = args.epochs

    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, n_images = build_data_pipe()
    train_dataset, valid_dataset = dataset.random_split(total_length=n_images, weights={"train": 0.95, "valid": 0.05})
    train_dataset = train_dataset.shuffle()
    valid_dataset = valid_dataset.shuffle()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    generator = Generator()
    generator.to(device)
    discriminator = Discriminator()
    discriminator.to(device)

    generator_optimizer = torch.optim.Adam(
        generator.parameters(), lr=2e-4, betas=(0.5, 0.999)
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)
    )

    l1 = nn.L1Loss()
    bce = nn.BCELoss()

    train_test_fronts, _ = next(iter(train_dataloader))
    train_test_fronts = train_test_fronts.to(device)

    valid_test_fronts, _ = next(iter(valid_dataloader))
    valid_test_fronts = valid_test_fronts.to(device)

    writer = SummaryWriter()

    for epoch in tqdm(range(epochs)):
        generator_epoch_loss = torch.tensor(0.0)
        discriminator_epoch_loss = torch.tensor(0.0)
        for i, (fronts, icons) in enumerate(train_dataloader):
            fronts = fronts.to(device)
            icons = icons.to(device)
            fake_target = torch.zeros((fronts.size(0) * 16 * 16, 1, 2, 2), device=device)
            valid_target = torch.ones((fronts.size(0) * 16 * 16, 1, 2, 2), device=device)

            generator_optimizer.zero_grad()
            fake_icons = generator.forward(fronts)
            generator_loss = bce(
                discriminator.forward(fake_icons, fronts), valid_target
            ) + 100 * l1(fake_icons, icons)
            generator_loss.backward()
            generator_optimizer.step()
            generator_epoch_loss += generator_loss.item()

            discriminator_optimizer.zero_grad()
            discriminator_loss = 0.5 * (
                bce(discriminator.forward(icons, fronts), valid_target)
                + bce(discriminator.forward(fake_icons.detach(), fronts), fake_target)
            )
            discriminator_loss.backward()
            discriminator_optimizer.step()
            discriminator_epoch_loss += discriminator_loss.item()
        writer.add_scalar("train generator loss", generator_epoch_loss / i, epoch)
        writer.add_scalar(
            "train discriminator loss", discriminator_epoch_loss / i, epoch
        )

        if i % 10 == 0:
            tests_icons = generator.forward(train_test_fronts)
            fig = visualize_sprites(train_test_fronts, tests_icons)
            writer.add_figure("train generated images", fig, epoch)
            plt.close(fig)

            tests_icons = generator.forward(valid_test_fronts)
            fig = visualize_sprites(valid_test_fronts, tests_icons)
            writer.add_figure("valid generated images", fig, epoch)
            plt.close(fig)

        generator_epoch_loss = torch.tensor(0.0)
        discriminator_epoch_loss = torch.tensor(0.0)
        for i, (fronts, icons) in enumerate(valid_dataloader):
            fronts = fronts.to(device)
            icons = icons.to(device)
            fake_target = torch.zeros((fronts.size(0) * 16 * 16, 1, 2, 2), device=device)
            valid_target = torch.ones((fronts.size(0) * 16 * 16, 1, 2, 2), device=device)

            fake_icons = generator.forward(fronts)
            generator_loss = bce(
                discriminator.forward(fake_icons, fronts), valid_target
            ) + 100 * l1(fake_icons, icons)
            generator_epoch_loss += generator_loss.item()

            discriminator_loss = 0.5 * (
                    bce(discriminator.forward(icons, fronts), valid_target)
                    + bce(discriminator.forward(fake_icons.detach(), fronts), fake_target)
            )
            discriminator_epoch_loss += discriminator_loss.item()
        writer.add_scalar("valid generator loss", generator_epoch_loss / i, epoch)
        writer.add_scalar(
            "valid discriminator loss", discriminator_epoch_loss / i, epoch
        )
