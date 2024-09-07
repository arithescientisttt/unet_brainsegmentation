import argparse
import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from medpy.filter.binary import largest_connected_component
from skimage.io import imsave
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_class import BrainSegmentationDataset as Dataset
from unet_model import UNet
from utils import compute_dsc, convert_gray_to_rgb, add_outline


def run_inference(args):
    create_directories(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    data_loader = load_data(args)

    with torch.no_grad():
        unet_model = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
        state_dict = torch.load(args.weights, map_location=device)
        unet_model.load_state_dict(state_dict)
        unet_model.eval()
        unet_model.to(device)

        inputs, predictions, ground_truths = [], [], []

        for i, batch in tqdm(enumerate(data_loader)):
            x, y_true = batch
            x, y_true = x.to(device), y_true.to(device)

            y_pred = unet_model(x)
            pred_np = y_pred.cpu().numpy()
            predictions.extend([pred_np[s] for s in range(pred_np.shape[0])])

            true_np = y_true.cpu().numpy()
            ground_truths.extend([true_np[s] for s in range(true_np.shape[0])])

            input_np = x.cpu().numpy()
            inputs.extend([input_np[s] for s in range(input_np.shape[0])])

    volume_data = process_volumes(inputs, predictions, ground_truths, data_loader.dataset.patient_slice_index, data_loader.dataset.patients)

    dsc_scores = calculate_dsc_distribution(volume_data)

    dsc_figure = create_dsc_plot(dsc_scores)
    imsave(args.figure, dsc_figure)

    save_predictions(volume_data, args.predictions)


def load_data(args):
    dataset = Dataset(
        images_dir=args.images,
        subset="validation",
        image_size=args.image_size,
        random_sampling=False,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, drop_last=False, num_workers=1
    )
    return loader


def process_volumes(inputs, predictions, ground_truths, patient_slice_indices, patients):
    volume_data = {}
    num_slices = np.bincount([p[0] for p in patient_slice_indices])
    idx = 0
    for p in range(len(num_slices)):
        input_vol = np.array(inputs[idx: idx + num_slices[p]])
        pred_vol = np.round(np.array(predictions[idx: idx + num_slices[p]])).astype(int)
        pred_vol = largest_connected_component(pred_vol)
        true_vol = np.array(ground_truths[idx: idx + num_slices[p]])
        volume_data[patients[p]] = (input_vol, pred_vol, true_vol)
        idx += num_slices[p]
    return volume_data


def calculate_dsc_distribution(volume_data):
    dsc_scores = {}
    for patient_id in volume_data:
        y_pred = volume_data[patient_id][1]
        y_true = volume_data[patient_id][2]
        dsc_scores[patient_id] = dsc(y_pred, y_true, lcc=False)
    return dsc_scores


def create_dsc_plot(dsc_scores):
    y_pos = np.arange(len(dsc_scores))
    sorted_dsc = sorted(dsc_scores.items(), key=lambda x: x[1])
    dsc_values = [x[1] for x in sorted_dsc]
    labels = ["_".join(l.split("_")[1:-1]) for l in [x[0] for x in sorted_dsc]]
    fig = plt.figure(figsize=(12, 8))
    canvas = FigureCanvasAgg(fig)
    plt.barh(y_pos, dsc_values, align="center", color="skyblue")
    plt.yticks(y_pos, labels)
    plt.xticks(np.arange(0.0, 1.0, 0.1))
    plt.xlim([0.0, 1.0])
    plt.axvline(np.mean(dsc_values), color="tomato", linewidth=2)
    plt.axvline(np.median(dsc_values), color="forestgreen", linewidth=2)
    plt.xlabel("Dice Coefficient", fontsize="x-large")
    plt.grid(axis='x', linestyle="--", color="silver", alpha=0.5)
    plt.tight_layout()
    canvas.draw()
    plt.close()
    s, (width, height) = canvas.print_to_buffer()
    return np.frombuffer(s, np.uint8).reshape((height, width, 4))


def save_predictions(volume_data, output_dir):
    for patient_id in volume_data:
        x, y_pred, y_true = volume_data[patient_id]
        for s in range(x.shape[0]):
            image = gray2rgb(x[s, 1])  # channel 1 is for FLAIR
            image = outline(image, y_pred[s, 0], color=[255, 0, 0])
            image = outline(image, y_true[s, 0], color=[0, 255, 0])
            filename = f"{patient_id}-{str(s).zfill(2)}.png"
            filepath = os.path.join(output_dir, filename)
            imsave(filepath, image)


def create_directories(args):
    os.makedirs(args.predictions, exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference for brain MRI segmentation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for inference (default: cuda:0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the model weights file",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="./kaggle_3m",
        help="Directory containing the input images",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Target size for input images (default: 256)",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default="./predictions",
        help="Directory to save prediction output images",
    )
    parser.add_argument(
        "--figure",
        type=str,
        default="./dsc.png",
        help="Filepath to save the DSC distribution plot",
    )

    args = parser.parse_args()
    run_inference(args)
