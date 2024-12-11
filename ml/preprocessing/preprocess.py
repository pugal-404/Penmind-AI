import cv2
import numpy as np
from PIL import Image
import io
from scipy import ndimage
import tensorflow as tf
import base64
import logging
import os
import pywt
import random
import yaml
import argparse
import codecs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path="config/config.yaml"):
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the configuration file
    
    Returns:
        dict: Loaded configuration
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
        return cfg
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def enhance_contrast(image):
    """
    Enhance contrast using CLAHE with less aggressive settings.
    """
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))  # Reduced clip limit
    return clahe.apply(image)

def apply_bilateral_filter(image):
    """
    Apply a bilateral filter to the image to reduce noise while preserving edges.
    """
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)


def adaptive_threshold(image):
    """
    Apply adaptive thresholding with less aggressive parameters to retain more details.
    """
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 19, 5)  # Increased block size, adjusted C

def preprocess_image(image_path,target_size =None):
    """
    Adjusted preprocessing to make the output less like a photocopy and more readable.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    image = enhance_contrast(image)
    image = adaptive_threshold(image)

    # Use a smaller kernel for morphological opening to reduce its impact
    kernel = np.ones((1,1), np.uint8)  
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Normalize the image to have pixel values between 0 and 1
    image = image.astype('float32') / 255.0

    return image

def augment_image(image):
    angle = np.random.uniform(-15, 15)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    scale = np.random.uniform(0.8, 1.2)
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    tx, ty = np.random.uniform(-5, 5, 2)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    shear = np.random.uniform(-0.1, 0.1)
    M = np.float32([[1, shear, 0], [0, 1, 0]])
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    image = elastic_transform(image, image.shape[1] * 2, image.shape[1] * 0.08, image.shape[1] * 0.08)
    
    return image

def elastic_transform(image, alpha, sigma, alpha_affine):
    """
    Apply elastic deformation to an image as described in [Simard2003].
    """
    random_state = np.random.RandomState(None)
    shape = image.shape
    dx = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))
    
    return ndimage.map_coordinates(image, indices, order=1, mode='reflect')

def segment_lines(image):
    """
    Segment the image into lines using projection profile method.
    
    Args:
        image (numpy.ndarray): Input image
    
    Returns:
        list: List of line images
    """
    horizontal_projection = np.sum(image, axis=1)
    line_boundaries = np.where(np.diff(horizontal_projection > 0))[0]
    lines = []
    for i in range(0, len(line_boundaries), 2):
        if i + 1 < len(line_boundaries):
            lines.append(image[line_boundaries[i]:line_boundaries[i+1], :])
    return lines

def segment_words(line_image):
    """
    Segment a line image into words using connected component analysis.
    
    Args:
        line_image (numpy.ndarray): Input line image
    
    Returns:
        list: List of word images
    """
    _, labels, stats, _ = cv2.connectedComponentsWithStats(line_image, connectivity=8)
    words = []
    for i in range(1, np.max(labels) + 1):
        x, y, w, h, area = stats[i]
        if area > 50:  # Minimum area threshold to avoid noise
            words.append(line_image[y:y+h, x:x+w])
    return words

def generate_synthetic_data(num_samples, target_size=(128, 512), characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:,.<>?`~∫∑∏≠≤≥∈∉"):
    """
    Generate synthetic handwriting data with elastic distortions.
    
    Args:
        num_samples (int): Number of synthetic samples to generate
        target_size (tuple): Target size of the generated images
        characters (str): Character set to use for generating text
    
    Returns:
        tuple: Numpy arrays of synthetic images and labels
    """
    synthetic_images = []
    synthetic_labels = []
    
    for _ in range(num_samples):
        label = ''.join(np.random.choice(list(characters), size=np.random.randint(5, 15)))
        
        image = np.ones(target_size, dtype=np.uint8) * 255
        
        font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
        font_scale = np.random.uniform(0.5, 1.0)
        thickness = np.random.randint(1, 3)
        text_color = (0, 0, 0)  # Black text
        
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = (target_size[1] - text_size[0]) // 2
        text_y = (target_size[0] + text_size[1]) // 2
        
        cv2.putText(image, label, (text_x, text_y), font, font_scale, text_color, thickness)
        
        image = elastic_transform(image, alpha=random.uniform(30, 60), sigma=random.uniform(3, 6), alpha_affine=random.uniform(3, 6))
        
        processed_image = preprocess_image(image, target_size)
        
        if processed_image is not None:
            synthetic_images.append(processed_image)
            synthetic_labels.append(label)
    
    return np.array(synthetic_images), np.array(synthetic_labels)

def load_images_from_directory(directory):
    images = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".png"):
                image_path = os.path.join(subdir, file)
                processed_image = preprocess_image(image_path)
                if processed_image is not None:
                    images.append(processed_image)
    return images

def process_dataset(input_directory, output_directory, target_size=(128, 512)):
    """
    Process entire dataset, preprocessing and saving images.
    """
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    processed_count = 0
    failed_count = 0
    
    # Walk through all subdirectories
    for subdir, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                try:
                    # Construct full input path
                    input_path = os.path.join(subdir, file)
                    
                    # Preprocess the image
                    processed_image = preprocess_image(input_path, target_size)
                    
                    if processed_image is not None:
                        # Construct output path
                        relative_path = os.path.relpath(subdir, input_directory)
                        output_subdir = os.path.join(output_directory, relative_path)
                        os.makedirs(output_subdir, exist_ok=True)
                        
                        output_path = os.path.join(output_subdir, file)
                        
                        # Save processed image
                        if processed_image.ndim == 2:  # Checks if the image is grayscale
                            processed_uint8 = (processed_image * 255).astype(np.uint8)
                        else:  # Handles images with channels
                            processed_uint8 = (processed_image[:,:,0] * 255).astype(np.uint8)
                        cv2.imwrite(output_path, processed_uint8)
                        
                        processed_count += 1
                    else:
                        failed_count += 1
                        logger.warning(f"Failed to process: {input_path}")
                
                except Exception as e:
                    failed_count += 1
                    logger.error(f"Error processing {input_path}: {e}")
    
    return processed_count, failed_count

def main():
    """
    Main function to handle command-line arguments and execute preprocessing.
    """
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Handwriting Image Preprocessing Tool")
    
    # Add arguments
    parser.add_argument('--config', default='config/config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--input', 
                        help='Input directory containing images')
    parser.add_argument('--output', 
                        help='Output directory for processed images')
    parser.add_argument('--generate-synthetic', type=int, default=0, 
                        help='Number of synthetic images to generate')
    parser.add_argument('--augment', action='store_true', 
                        help='Apply data augmentation')
    parser.add_argument('--verbose', action='store_true', 
                        help='Enable verbose logging')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration: {config}")
    
    # Determine input and output directories
    input_directory = args.input or config.get('paths', {}).get('dataset')
    output_directory = args.output or config.get('paths', {}).get('preprocessed_dataset')
    
    logger.info(f"Input directory: {input_directory}")
    logger.info(f"Output directory: {output_directory}")
    
    if not input_directory or not output_directory:
        logger.error("Input or output directory not specified")
        logger.error("Please provide directories via command-line arguments or config file")
        return
    
    
    # Process dataset
    logger.info(f"Processing images from {input_directory}")
    processed_count, failed_count = process_dataset(input_directory, output_directory)
    
    logger.info(f"Processed {processed_count} images")
    logger.info(f"Failed to process {failed_count} images")
    
    # Generate synthetic data if requested
    if args.generate_synthetic > 0:
        logger.info(f"Generating {args.generate_synthetic} synthetic images")
        synthetic_images, synthetic_labels = generate_synthetic_data(args.generate_synthetic)
        logger.info(f"Generated {len(synthetic_images)} synthetic images")

    # Optional data augmentation
    if args.augment:
        logger.info("Applying data augmentation.")
        augmented_count = augment_image(output_directory)
        logger.info(f"Augmented {augmented_count} images.")

if __name__ == "__main__":
    main()