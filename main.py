import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image


class ImageEvaluator:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.images = [
            os.path.join(image_dir, img)
            for img in os.listdir(image_dir)
            if img.endswith((".jpg", ".jpeg", ".png"))
        ]
        self.rg_ness_images = []

    def compute_rg_ness_hsv(self):
        for img_path in self.images:
            im = Image.open(img_path).convert("HSV")
            Hue = np.array(im.getchannel("H"))
            mask = np.zeros_like(Hue, dtype=np.uint8)
            # Set all green pixels to 1
            mask[(Hue < 120)] = 1
            rg_ness_percentage = mask.mean() * 100
            self.rg_ness_images.append((img_path, Hue, rg_ness_percentage))

    def plot_rg_ness(self):
        num_images = len(self.rg_ness_images)
        cols = 6
        rows = num_images

        plt.figure(figsize=(20, 4 * rows), layout="constrained")

        for index, (img_path, rg_ness_mask, rg_ness_percentage) in enumerate(
            self.rg_ness_images
        ):
            img = plt.imread(img_path)
            plt.subplot(rows, cols, 2 * index + 1)
            plt.title("Original Image")
            plt.imshow(img)
            plt.axis("off")

            plt.subplot(rows, cols, 2 * index + 2)
            plt.title(f"rg_ness Mask - {rg_ness_percentage:.2f}%")
            plt.imshow(rg_ness_mask, cmap="gray")
            plt.colorbar(label="rg_ness Intensity")
            plt.axis("off")

        plt.show()


if __name__ == "__main__":
    image_dir = "samples"
    evaluator = ImageEvaluator(image_dir)
    evaluator.compute_rg_ness_hsv()
    evaluator.plot_rg_ness()
