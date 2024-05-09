# Image Evaluator Tool

The Image Evaluator tool is a Python utility designed for analyzing images, particularly focusing on evaluating their greenness percentage.

## Features

- **Greenness Evaluation:** Determine the percentage of green in images using the HSV color space.

## Dependencies

- Python 3.x
- NumPy
- Matplotlib
- Pillow

Ensure these libraries are installed in your Python environment. You can install them using pip:

```bash
pip install numpy matplotlib pillow
```


## Usage

1. **Prepare Your Image Dataset:**
   Ensure all images you wish to evaluate are stored in a single directory. Supported image formats are `.jpg`, `.jpeg`, and `.png`.

2. **Run the Tool:**
   Before running the script, set the `image_dir` variable to the path of your images directory:

   ```python
   image_dir = "path/to/your/image/directory"
   ```

   Run the script in your terminal or command prompt:

   ```bash
   python image_evaluator.py
   ```

3. **View Results:**
   The script will compute the greenness of each image (as a percentage) and display both the original image and a mask highlighting its green regions. Additionally, placeholders for future features (sharpness and noisiness) are present.

## How It Works

- **Greenness Evaluation:** Each image is converted to HSV color space, and the hue channel is analyzed to determine the percentage of green in the image.
