import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import cv2
import os

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def save_image(image, output_path):
    cv2.imwrite(output_path, image)

def edge_detection(input_image):
    # Sobel kernel for edge detection
    sobel_kernel = np.array([[-1, -2, -1],
                             [0,  0,  0],
                             [1,  2,  1]], dtype=np.float32)

    # Prepare GPU memory
    image_gpu = cuda.mem_alloc(input_image.nbytes)
    output_gpu = cuda.mem_alloc(input_image.nbytes)
    kernel_gpu = cuda.mem_alloc(sobel_kernel.nbytes)

    # Copy data to GPU
    cuda.memcpy_htod(image_gpu, input_image)
    cuda.memcpy_htod(kernel_gpu, sobel_kernel)

    # CUDA kernel for edge detection
    mod = SourceModule("""
    __global__ void sobel_filter(unsigned char *input, unsigned char *output, float *kernel, int width, int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        if (x >= width || y >= height) return;

        float sum = 0.0;
        int ksize = 3;
        int half_k = ksize / 2;

        for (int ky = -half_k; ky <= half_k; ky++) {
            for (int kx = -half_k; kx <= half_k; kx++) {
                int img_x = min(max(x + kx, 0), width - 1);
                int img_y = min(max(y + ky, 0), height - 1);
                sum += input[img_y * width + img_x] * kernel[(ky + half_k) * ksize + (kx + half_k)];
            }
        }
        output[y * width + x] = min(max(int(sum), 0), 255);
    }
    """)

    func = mod.get_function("sobel_filter")

    # Define grid and block sizes
    block_size = (16, 16, 1)
    grid_size = (int(np.ceil(input_image.shape[1] / block_size[0])),
                 int(np.ceil(input_image.shape[0] / block_size[1])))

    # Run the CUDA kernel
    func(image_gpu, output_gpu, kernel_gpu,
         np.int32(input_image.shape[1]), np.int32(input_image.shape[0]),
         block=block_size, grid=grid_size)

    # Copy the result back to host
    output_image = np.empty_like(input_image)
    cuda.memcpy_dtoh(output_image, output_gpu)

    return output_image

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(input_folder):
        if image_name.endswith('.tiff'):
            input_path = os.path.join(input_folder, image_name)
            output_path = os.path.join(output_folder, image_name.replace('.tiff', '.png'))

            print(f"Processing {image_name}...")

            # Load and process the image
            input_image = load_image(input_path)
            output_image = edge_detection(input_image)

            # Save the result
            save_image(output_image, output_path)

if __name__ == "__main__":
    input_folder = "../data/input/"
    output_folder = "../data/output/"
    process_images(input_folder, output_folder)
