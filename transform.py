import numpy as np
from skimage.transform import rescale, rotate
from torchvision.transforms import Compose

# Function to combine multiple transformations (scaling, rotating, flipping)
def apply_transforms(scaling_factor=None, rotation_angle=None, flip_probability=None):
    transform_sequence = []

    if scaling_factor is not None:
        transform_sequence.append(ScaleTransform(scaling_factor))
    if rotation_angle is not None:
        transform_sequence.append(RotateTransform(rotation_angle))
    if flip_probability is not None:
        transform_sequence.append(HorizontalFlipTransform(flip_probability))

    return Compose(transform_sequence)

# Class for scaling transformation
class ScaleTransform(object):
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, sample):
        image, mask = sample
        original_size = image.shape[0]

        # Random scaling within the range [1.0 - scale, 1.0 + scale]
        scaling_value = np.random.uniform(low=1.0 - self.scale_factor, high=1.0 + self.scale_factor)

        # Apply scaling to both image and mask
        image = rescale(
            image,
            (scaling_value, scaling_value),
            multichannel=True,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )
        mask = rescale(
            mask,
            (scaling_value, scaling_value),
            order=0,
            multichannel=True,
            preserve_range=True,
            mode="constant",
            anti_aliasing=False,
        )

        # Padding or cropping based on scale value
        if scaling_value < 1.0:
            pad_size = (original_size - image.shape[0]) / 2.0
            padding = ((int(np.floor(pad_size)), int(np.ceil(pad_size))),) * 2 + ((0, 0),)
            image = np.pad(image, padding, mode="constant", constant_values=0)
            mask = np.pad(mask, padding, mode="constant", constant_values=0)
        else:
            crop_min = (image.shape[0] - original_size) // 2
            crop_max = crop_min + original_size
            image = image[crop_min:crop_max, crop_min:crop_max, ...]
            mask = mask[crop_min:crop_max, crop_min:crop_max, ...]

        return image, mask


# Class for rotating transformation
class RotateTransform(object):
    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, sample):
        image, mask = sample

        # Random rotation between -angle and +angle
        random_angle = np.random.uniform(low=-self.max_angle, high=self.max_angle)
        image = rotate(image, random_angle, resize=False, preserve_range=True, mode="constant")
        mask = rotate(mask, random_angle, resize=False, order=0, preserve_range=True, mode="constant")

        return image, mask


# Class for horizontal flip transformation
class HorizontalFlipTransform(object):
    def __init__(self, flip_probability):
        self.flip_probability = flip_probability

    def __call__(self, sample):
        image, mask = sample

        # Flip the image and mask with the specified probability
        if np.random.rand() > self.flip_probability:
            return image, mask

        # Apply horizontal flip
        image = np.fliplr(image).copy()
        mask = np.fliplr(mask).copy()

        return image, mask
