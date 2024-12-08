import cv2
import numpy as np
from PIL import Image
import io
from scipy import ndimage
import tensorflow as tf
import base64
import logging
import os

logger = logging.getLogger(__name__)

def preprocess_image(image_data, target_size=(128, 512)):
    try:
        # Handle different input types
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

        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize
        image = image.resize(target_size, Image.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Convert to grayscale
        if image_array.ndim == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Noise reduction using Non-Local Means denoising
        image_array = cv2.fastNlMeansDenoising(image_array, None, 10, 7, 21)
        
        # Contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        image_array = clahe.apply(image_array)
        
        # Binarization using Sauvola's method
        window_size = 25
        k = 0.2
        R = 128
        thresh_sauvola = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        mean = cv2.boxFilter(image_array, -1, (window_size, window_size))
        mean_square = cv2.boxFilter(image_array**2, -1, (window_size, window_size))
        variance = mean_square - mean**2
        threshold = mean * (1 + k * ((variance / R)**0.5 - 1))
        image_array = np.where(image_array > threshold, 255, 0).astype(np.uint8)
        
        # Skew correction using Hough Transform
        edges = cv2.Canny(image_array, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
        if lines is not None:
            angle = np.median([line[0][1] for line in lines])
            if angle > np.pi / 4:
                angle = angle - np.pi / 2
            (h, w) = image_array.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle * 180 / np.pi, 1.0)
            image_array = cv2.warpAffine(image_array, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # Normalize pixel values
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add channel dimension
        image_array = np.expand_dims(image_array, axis=-1)
        
        return image_array
    
    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        return None

def segment_lines(image):
    # Implement line segmentation using projection profile
    horizontal_projection = np.sum(image, axis=1)
    line_boundaries = np.where(np.diff(horizontal_projection > 0))[0]
    lines = []
    for i in range(0, len(line_boundaries), 2):
        if i + 1 < len(line_boundaries):
            lines.append(image[line_boundaries[i]:line_boundaries[i+1], :])
    return lines

def segment_words(line_image):
    # Implement word segmentation using connected component analysis
    _, labels, stats, _ = cv2.connectedComponentsWithStats(line_image, connectivity=8)
    words = []
    for i in range(1, np.max(labels) + 1):
        x, y, w, h, area = stats[i]
        if area > 50:  # Minimum area threshold to avoid noise
            words.append(line_image[y:y+h, x:x+w])
    return words

def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def augment_image(image):
    # Random rotation
    angle = np.random.uniform(-15, 15)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Random scaling
    scale = np.random.uniform(0.8, 1.2)
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    
    # Random translation
    tx = np.random.uniform(-5, 5)
    ty = np.random.uniform(-5, 5)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Random shear
    shear = np.random.uniform(-0.1, 0.1)
    M = np.float32([[1, shear, 0], [0, 1, 0]])
    image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Elastic distortion
    image = elastic_transform(image, image.shape[1] * 2, image.shape[1] * 0.08, image.shape[1] * 0.08)
    
    return image

def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    
    # Random affine
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

def preprocess_dataset(dataset_path, target_size=(128, 512)):
    processed_images = []
    labels = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                label = os.path.basename(root)  # Assuming the folder name is the label
                
                image = Image.open(image_path)
                processed_image = preprocess_image(image, target_size)
                
                if processed_image is not None:
                    processed_images.append(processed_image)
                    labels.append(label)
    
    return np.array(processed_images), np.array(labels)

def generate_synthetic_data(num_samples, target_size=(128, 512)):
    synthetic_images = []
    synthetic_labels = []
    
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:,.<>?`~∫∑∏≠≤≥∈∉"
    
    for _ in range(num_samples):
        image = np.random.rand(*target_size) * 255
        image = image.astype(np.uint8)
        label = ''.join(np.random.choice(list(characters), size=np.random.randint(5, 15)))
        
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
    import base64
    with open("path/to/test_image.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    preprocessed = preprocess_image(encoded_string)
    if preprocessed is not None:
        print(f"Preprocessed base64 image shape: {preprocessed.shape}")
    
    # Test the augmentation function
    augmented = augment_image(test_image)
    print(f"Augmented image shape: {augmented.shape}")

    # Test dataset preprocessing
    dataset_path = "path/to/IAM/dataset"
    processed_images, labels = preprocess_dataset(dataset_path)
    print(f"Processed {len(processed_images)} images with {len(labels)} labels")

    # Generate synthetic data
    synthetic_images, synthetic_labels = generate_synthetic_data(100)
    print(f"Generated {len(synthetic_images)} synthetic images with {len(synthetic_labels)} labels")

