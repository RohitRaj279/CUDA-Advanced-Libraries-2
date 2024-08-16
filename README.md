# CUDA Edge Detection Project

This project performs edge detection on black and white images using CUDA and Python. It uses the Sobel filter to detect edges efficiently on the GPU.

## Project Structure

- **bin/**: Contains compiled binaries and executables.
- **data/**: Contains input `.tiff` images and output `.png` images.
- **lib/**: Contains additional libraries.
- **src/**: Contains the source code for edge detection and utility functions.
- **README.md**: Provides an overview of the project.
- **INSTALL**: Contains installation instructions.
- **Makefile**: Provides build automation.
- **run.sh**: Script to execute the edge detection.

## Usage

1. Place your input `.tiff` images in the `data/input/` directory.
2. Run `./run.sh` to execute the edge detection.
3. The output images will be saved in the `data/output/` directory.

## Requirements

- Python 3.x
- PyCUDA
- OpenCV
- NumPy
