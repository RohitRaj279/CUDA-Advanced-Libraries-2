import cv2

def load_image(image_path):
    """Loads an image from a given path in grayscale mode."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def save_image(image, output_path):
    """Saves an image to a specified path."""
    cv2.imwrite(output_path, image)
