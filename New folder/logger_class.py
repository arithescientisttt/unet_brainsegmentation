from io import BytesIO
from PIL import Image
import tensorflow as tf


class TensorboardLogger(object):
    
    def __init__(self, log_dir):
        # Initialize the TensorBoard writer for the specified log directory
        self.writer = tf.summary.FileWriter(log_dir)

    def log_scalar(self, tag, value, step):
        # Create a summary for scalar values (e.g., loss, accuracy)
        scalar_summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(scalar_summary, step)
        self.writer.flush()

    def log_image(self, tag, image, step):
        # Convert the image into a byte stream
        buffer = BytesIO()
        Image.fromarray(image).save(buffer, format="png")

        # Create an Image summary object for TensorBoard
        img_summary = tf.Summary.Image(
            encoded_image_string=buffer.getvalue(),
            height=image.shape[0],
            width=image.shape[1],
        )

        # Create and write the summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, image=img_summary)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_image_list(self, tag, images, step):
        if not images:
            return

        img_summaries = []
        for idx, img in enumerate(images):
            buffer = BytesIO()
            Image.fromarray(img).save(buffer, format="png")

            # Create an Image summary object for each image
            img_summary = tf.Summary.Image(
                encoded_image_string=buffer.getvalue(),
                height=img.shape[0],
                width=img.shape[1],
            )

            # Append the image summary with a unique tag
            img_summaries.append(
                tf.Summary.Value(tag="{}/{}".format(tag, idx), image=img_summary)
            )

        # Write all image summaries to TensorBoard
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()
