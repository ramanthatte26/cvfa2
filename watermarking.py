import numpy as np
import pywt
import cv2
from scipy.linalg import svd
from tqdm import tqdm
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class WatermarkEmbedder:
    def __init__(self, config):
        self.config = config

    def embed_watermark(self, input_dir: str, output_dir: str) -> None:
        """Embed watermark into images from the input directory and save results to the output directory."""
        logging.info("Searching for images in input directory...")
        
        # Define paths for input images
        image_path = os.path.join(input_dir, "image.png")
        watermark_path = os.path.join(input_dir, "watermark.png")
        
        # Load images
        image = cv2.imread(image_path)  # Load image in color
        watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)

        if image is None or watermark is None:
            logging.error("Could not load image or watermark. Please ensure the files exist in the input directory.")
            return

        # Convert image to grayscale for watermark processing
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize watermark to match the image dimensions
        logging.info("Resizing watermark to match image dimensions...")
        watermark = cv2.resize(watermark, (grayscale_image.shape[1], grayscale_image.shape[0]))

        # Pad image and watermark to ensure proper block processing
        padded_image, padded_watermark = self._pad_to_block_size(grayscale_image, watermark)

        # Split the image and watermark into blocks
        blocks = self._split_into_blocks(padded_image, padded_watermark)

        # Process each block to embed the watermark
        embedded_blocks = []
        logging.info("Processing blocks and embedding watermark...")
        for block in tqdm(blocks, desc="Embedding watermark"):
            embedded_blocks.append(self._process_block(block))

        # Reconstruct the watermarked image from the blocks
        watermarked_image_padded = self._reconstruct_from_blocks(embedded_blocks, padded_image.shape)

        # Remove padding to restore original dimensions
        watermarked_image = watermarked_image_padded[:grayscale_image.shape[0], :grayscale_image.shape[1]]

        # Generate a color version with yellow-tinted watermark
        watermarked_image_colored = self._add_yellow_tint(image, watermarked_image, watermark)

        # Generate a grayscale watermark version (rest of the image stays the same)
        watermarked_image_grayscale = self._combine_with_grayscale_watermark(image, watermarked_image)

        # Save results to the output directory
        os.makedirs(output_dir, exist_ok=True)
        grayscale_path = os.path.join(output_dir, "watermarked_image_grayscale.png")
        colored_path = os.path.join(output_dir, "watermarked_image_colored.png")

        logging.info("Saving watermarked images...")
        cv2.imwrite(grayscale_path, watermarked_image_grayscale)
        cv2.imwrite(colored_path, watermarked_image_colored)

        logging.info(f"Watermarked images saved:\n- Grayscale Combined: {grayscale_path}\n- Colored with Tint: {colored_path}")

    def _pad_to_block_size(self, image: np.ndarray, watermark: np.ndarray) -> tuple:
        """Pad image and watermark to be multiples of the block size."""
        block_size = self.config['block_size']
        pad_h = (block_size - image.shape[0] % block_size) % block_size
        pad_w = (block_size - image.shape[1] % block_size) % block_size
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant')
        padded_watermark = np.pad(watermark, ((0, pad_h), (0, pad_w)), mode='constant')
        return padded_image, padded_watermark

    def _split_into_blocks(self, image: np.ndarray, watermark: np.ndarray) -> list:
        """Split the image and watermark into blocks."""
        block_size = self.config['block_size']
        blocks = []

        for i in range(0, image.shape[0], block_size):
            for j in range(0, image.shape[1], block_size):
                image_block = image[i:i + block_size, j:j + block_size]
                watermark_block = watermark[i:i + block_size, j:j + block_size]
                blocks.append((image_block, watermark_block, self.config['alpha']))

        return blocks

    def _process_block(self, block_data: tuple) -> np.ndarray:
        """Embed watermark in an image block using DWT-SVD."""
        block, watermark_block, alpha = block_data

        # Apply DWT to the image block
        coeffs = pywt.wavedec2(block, self.config['wavelet'], level=self.config['level'])
        LL, (LH, HL, HH) = coeffs

        # Apply SVD to LL and the watermark block
        U_i, S_i, V_i = svd(LL, full_matrices=False)
        U_w, S_w, V_w = svd(watermark_block, full_matrices=False)

        # Embed the watermark into the singular values
        min_len = min(S_i.shape[0], S_w.shape[0])
        S_new = S_i[:min_len] + alpha * S_w[:min_len]

        # Reconstruct the LL component and the full block
        LL_modified = np.dot(U_i[:, :min_len], np.dot(np.diag(S_new), V_i[:min_len, :]))
        coeffs_modified = (LL_modified, (LH, HL, HH))
        return pywt.waverec2(coeffs_modified, self.config['wavelet'])

    def _reconstruct_from_blocks(self, blocks: list, image_shape: tuple) -> np.ndarray:
        """Reconstruct the image from blocks."""
        block_size = self.config['block_size']
        reconstructed_image = np.zeros(image_shape, dtype=np.float64)

        block_idx = 0
        for i in range(0, image_shape[0], block_size):
            for j in range(0, image_shape[1], block_size):
                reconstructed_image[i:i + block_size, j:j + block_size] = blocks[block_idx]
                block_idx += 1

        return reconstructed_image

    def _add_yellow_tint(self, original_image: np.ndarray, watermarked_image: np.ndarray, watermark: np.ndarray) -> np.ndarray:
        """Add yellow tint only to the watermark regions in the watermarked image."""
        watermark_resized = cv2.resize(watermark, (watermarked_image.shape[1], watermarked_image.shape[0]))
        _, watermark_mask = cv2.threshold(watermark_resized, 128, 255, cv2.THRESH_BINARY)

        # Copy the original image to keep its color
        watermarked_image_colored = original_image.copy()

        # Apply yellow tint to the watermark regions
        watermarked_image_colored[watermark_mask > 0] = [0, 255, 255]  # Yellow in BGR format

        return watermarked_image_colored

    def _combine_with_grayscale_watermark(self, original_image: np.ndarray, watermarked_image: np.ndarray) -> np.ndarray:
        """Combine the grayscale watermarked image with the original color."""
        watermarked_image_resized = cv2.resize(watermarked_image, (original_image.shape[1], original_image.shape[0]))
        combined_image = original_image.copy()

        # Replace the blue, green, and red channels with the watermarked grayscale image
        combined_image[:, :, 0] = watermarked_image_resized  # Blue channel
        combined_image[:, :, 1] = watermarked_image_resized  # Green channel
        combined_image[:, :, 2] = watermarked_image_resized  # Red channel
        
        return combined_image

# Configuration
config = {
    'block_size': 8,
    'alpha': 0.1,
    'wavelet': 'haar',
    'level': 1
}

# Example Usage
if __name__ == "__main__":
    input_dir = r"C:\Users\raman\Desktop\cvfa2\Input_Images"
    output_dir = r"C:\Users\raman\Desktop\cvfa2\Output_images"

    embedder = WatermarkEmbedder(config)
    embedder.embed_watermark(input_dir=input_dir, output_dir=output_dir)