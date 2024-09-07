import argparse
import json
import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_class import BrainSegmentationDataset as Dataset
from logger import Logger
from loss_function import DiceCoefficientLoss
from transform import apply_transforms
from unet import UNet
from utils import log_visualizations, compute_dsc


def train_unet_model(args):
    create_directories(args)
    save_arguments(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    train_loader, valid_loader = prepare_data_loaders(args)
    loaders = {"train": train_loader, "valid": valid_loader}

    unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
    unet.to(device)

    dsc_loss_fn = DiceCoefficientLoss()
    best_dsc = 0.0

    optimizer = optim.Adam(unet.parameters(), lr=args.lr)

    logger = Logger(args.logs)
    train_loss = []
    valid_loss = []

    step = 0

    for epoch in tqdm(range(args.epochs), total=args.epochs):
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            valid_predictions = []
            valid_ground_truths = []

            for i, batch in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                x, y_true = batch
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)

                    loss = dsc_loss_fn(y_pred, y_true)

                    if phase == "valid":
                        valid_loss.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        valid_predictions.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])
                        y_true_np = y_true.detach().cpu().numpy()
                        valid_ground_truths.extend([y_true_np[s] for s in range(y_true_np.shape[0])])

                        if (epoch % args.vis_freq == 0) or (epoch == args.epochs - 1):
                            if i * args.batch_size < args.vis_images:
                                tag = f"image/{i}"
                                num_images = args.vis_images - i * args.batch_size
                                logger.image_list_summary(
                                    tag,
                                    log_images(x, y_true, y_pred)[:num_images],
                                    step,
                                )

                    if phase == "train":
                        train_loss.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    log_loss(logger, train_loss, step)
                    train_loss = []

            if phase == "valid":
                log_loss(logger, valid_loss, step, prefix="val_")
                mean_dsc = np.mean(
                    compute_dsc_per_volume(
                        valid_predictions,
                        valid_ground_truths,
                        valid_loader.dataset.patient_slice_index,
                    )
                )
                logger.scalar_summary("val_dsc", mean_dsc, step)
                if mean_dsc > best_dsc:
                    best_dsc = mean_dsc
                    torch.save(unet.state_dict(), os.path.join(args.weights, "unet.pt"))
                valid_loss = []

    print(f"Best validation DSC: {best_dsc:.4f}")


def prepare_data_loaders(args):
    train_dataset, valid_dataset = load_datasets(args)

    def init_worker(worker_id):
        np.random.seed(42 + worker_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=init_worker,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=init_worker,
    )

    return train_loader, valid_loader


def load_datasets(args):
    train_dataset = Dataset(
        images_dir=args.images,
        subset="train",
        image_size=args.image_size,
        transform=apply_transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=0.5),
    )
    valid_dataset = Dataset(
        images_dir=args.images,
        subset="validation",
        image_size=args.image_size,
        random_sampling=False,
    )
    return train_dataset, valid_dataset


def compute_dsc_per_volume(predictions, truths, slice_indices):
    dsc_scores = []
    num_slices = np.bincount([p[0] for p in slice_indices])
    index = 0
    for p in range(len(num_slices)):
        pred_vol = np.array(predictions[index : index + num_slices[p]])
        true_vol = np.array(truths[index : index + num_slices[p]])
        dsc_scores.append(compute_dsc(pred_vol, true_vol))
        index += num_slices[p]
    return dsc_scores


def log_loss(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)


def create_directories(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


def save_arguments(args):
    args_path = os.path.join(args.logs, "args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for brain MRI segmentation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--ep
