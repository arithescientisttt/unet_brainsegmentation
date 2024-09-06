import numpy as np
from medpy.filter.binary import largest_connected_component
from skimage.exposure import rescale_intensity
from skimage.transform import resize


# Function to compute Dice Similarity Coefficient (DSC)
def compute_dsc(prediction, ground_truth, apply_lcc=True):
    if apply_lcc and np.any(prediction):
        prediction = np.round(prediction).astype(int)
        ground_truth = np.round(ground_truth).astype(int)
        prediction = largest_connected_component(prediction)
    intersection = np.sum(prediction[ground_truth == 1])
    return (2.0 * intersection) / (np.sum(prediction) + np.sum(ground_truth))


# Crop the volume and mask to relevant areas
def crop_image(sample):
    volume, mask = sample
    volume[volume < np.max(volume) * 0.1] = 0

    # Find non-zero regions in each axis (z, y, x)
    z_projection = np.max(np.max(np.max(volume, axis=-1), axis=-1), axis=-1)
    z_nonzero_indices = np.nonzero(z_projection)
    z_min, z_max = np.min(z_nonzero_indices), np.max(z_nonzero_indices) + 1

    y_projection = np.max(np.max(np.max(volume, axis=0), axis=-1), axis=-1)
    y_nonzero_indices = np.nonzero(y_projection)
    y_min, y_max = np.min(y_nonzero_indices), np.max(y_nonzero_indices) + 1

    x_projection = np.max(np.max(np.max(volume, axis=0), axis=0), axis=-1)
    x_nonzero_indices = np.nonzero(x_projection)
    x_min, x_max = np.min(x_nonzero_indices), np.max(x_nonzero_indices) + 1

    return (
        volume[z_min:z_max, y_min:y_max, x_min:x_max],
        mask[z_min:z_max, y_min:y_max, x_min:x_max],
    )


# Pad the volume and mask to ensure square dimensions
def pad_image(sample):
    volume, mask = sample
    y_dim, x_dim = volume.shape[1], volume.shape[2]

    if y_dim == x_dim:
        return volume, mask

    diff = (max(y_dim, x_dim) - min(y_dim, x_dim)) / 2.0
    if y_dim > x_dim:
        padding = ((0, 0), (0, 0), (int(np.floor(diff)), int(np.ceil(diff))))
    else:
        padding = ((0, 0), (int(np.floor(diff)), int(np.ceil(diff))), (0, 0))

    mask = np.pad(mask, padding, mode="constant", constant_values=0)
    volume = np.pad(volume, padding + ((0, 0),), mode="constant", constant_values=0)
    return volume, mask


# Resize the volume and mask to the target size
def resize_image(sample, target_size=256):
    volume, mask = sample
    volume_shape = volume.shape
    mask_resized = resize(
        mask,
        output_shape=(volume_shape[0], target_size, target_size),
        order=0,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    )
    volume_resized = resize(
        volume,
        output_shape=(volume_shape[0], target_size, target_size, volume_shape[3]),
        order=2,
        mode="constant",
        cval=0,
        anti_aliasing=False,
    )
    return volume_resized, mask_resized


# Normalize the volume to a standard intensity range
def normalize_intensity(volume):
    p10, p99 = np.percentile(volume, 10), np.percentile(volume, 99)
    volume = rescale_intensity(volume, in_range=(p10, p99))
    mean_val = np.mean(volume, axis=(0, 1, 2))
    std_val = np.std(volume, axis=(0, 1, 2))
    volume = (volume - mean_val) / std_val
    return volume


# Visualize and log images along with predictions and ground truth
def log_visualizations(volume, true_labels, predicted_labels, channel_idx=1):
    images_list = []
    volume_np = volume[:, channel_idx].cpu().numpy()
    true_np = true_labels[:, 0].cpu().numpy()
    pred_np = predicted_labels[:, 0].cpu().numpy()

    for i in range(volume_np.shape[0]):
        img = convert_gray_to_rgb(np.squeeze(volume_np[i]))
        img = add_outline(img, pred_np[i], color=[255, 0, 0])
        img = add_outline(img, true_np[i], color=[0, 255, 0])
        images_list.append(img)

    return images_list


# Convert a grayscale image to an RGB format
def convert_gray_to_rgb(image):
    h, w = image.shape
    image = image - np.min(image)
    image = image / np.max(image) if np.max(image) > 0 else image
    rgb_image = np.empty((h, w, 3), dtype=np.uint8)
    rgb_image[:, :, 0] = rgb_image[:, :, 1] = rgb_image[:, :, 2] = image * 255
    return rgb_image


# Add outlines of a mask onto an image with a specified color
def add_outline(image, mask, color):
    mask = np.round(mask)
    y_indices, x_indices = np.nonzero(mask)
    
    for y, x in zip(y_indices, x_indices):
        surrounding_pixels = mask[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
        if 0.0 < np.mean(surrounding_pixels) < 1.0:
            image[y, x] = color
    return image
