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

logger = logging.getLogger(__name__)

def wavelet_denoise(image, method='BayesShrink', mode='soft'):
    """
    Apply wavelet denoising to reduce noise while preserving features.
    
    Args:
        image (numpy.ndarray): Input image
        method (str): Thresholding method ('BayesShrink' or 'VisuShrink')
        mode (str): Thresholding mode ('soft' or 'hard')
    
    Returns:
        numpy.ndarray: Denoised image
    """
    coeffs = pywt.wavedec2(image, 'haar', level=2)
    
    if method == 'BayesShrink':
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    else:  # VisuShrink
        sigma = 0.1
    
    threshold = sigma * np.sqrt(2 * np.log(image.size))
    new_coeffs = [pywt.threshold(c, threshold, mode=mode) for c in coeffs]
    
    denoised_image = pywt.waverec2(new_coeffs, 'haar')
    
    return denoised_image

def adaptive_binarization(image, block_size=11, C=2):
    """
    Apply adaptive binarization to handle diverse lighting conditions.
    
    Args:
        image (numpy.ndarray): Input grayscale image
        block_size (int): Size of the local neighborhood for thresholding
        C (int): Constant subtracted from the mean
    
    Returns:
        numpy.ndarray: Binarized image
    """
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)

def deep_skew_correction(image):
    """
    Apply deep learning-based skew correction.
    
    Args:
        image (numpy.ndarray): Input image
    
    Returns:
        numpy.ndarray: Skew-corrected image
    """
    # TODO: Implement or integrate a deep learning model for skew correction
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    if lines is not None:
        angle = np.median([line[0][1] for line in lines])
        if angle > np.pi / 4:
            angle = angle - np.pi / 2
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle * 180 / np.pi, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return image

def preprocess_image(image_data, target_size=(128, 512)):
    try:
        if isinstance(image_data, str):
            image_data = base64.b64decode(image_data)
        
        if isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        elif isinstance(image_data, np.ndarray):
            image = Image.fromarray(image_data)
        elif isinstance(image_data, Image.Image):
            image = image_data
        else:
            raise ValueError("Unsupported image data type")

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = image.resize(target_size, Image.LANCZOS)
        
        image_array = np.array(image)
        
        if image_array.ndim == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        image_array = wavelet_denoise(image_array)
        image_array = adaptive_binarization(image_array)
        image_array = deep_skew_correction(image_array)
        
        image_array = image_array.astype(np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=-1)
        
        return image_array
    
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        return None

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

def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """
    Apply elastic transform to the image.
    
    Args:
        image (numpy.ndarray): Input image
        alpha (float): Scale factor for deformation
        sigma (float): Smoothing factor
        alpha_affine (float): Range of affine transform
        random_state (numpy.random.RandomState): Random state for reproducibility
    
    Returns:
        numpy.ndarray: Elastically transformed image
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = ndimage.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

    return ndimage.map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

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

if __name__ == "__main__":
    # Test the preprocessing function
    test_image = np.random.rand(100, 200, 3) * 255
    test_image = test_image.astype(np.uint8)
    
    preprocessed = preprocess_image(test_image)
    if preprocessed is not None:
        print(f"Preprocessed image shape: {preprocessed.shape}")
        print(f"Preprocessed image dtype: {preprocessed.dtype}")
        print(f"Preprocessed image min value: {preprocessed.min()}")
        print(f"Preprocessed image max value: {preprocessed.max()}")
    
    # Test with base64 encoded image
    with open("path/to/test_image.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    preprocessed = preprocess_image(encoded_string)
    if preprocessed is not None:
        print(f"Preprocessed base64 image shape: {preprocessed.shape}")
    
    # Test the augmentation function
    augmented = augment_image(test_image)
    print(f"Augmented image shape: {augmented.shape}")

    # Generate synthetic data
    synthetic_images, synthetic_labels = generate_synthetic_data(100)
    print(f"Generated {len(synthetic_images)} synthetic images with {len(synthetic_labels)} labels")

