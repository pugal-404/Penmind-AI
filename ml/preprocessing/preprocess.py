import cv2
import numpy as np
from PIL import Image
import io

def preprocess_image(image, target_size=(64, 128)):
    try:
        # Convert to PIL Image if input is bytes or numpy array
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Resize
        image = image.resize(target_size, Image.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Apply adaptive thresholding
        image_array = cv2.adaptiveThreshold(
            image_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Normalize pixel values
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add channel dimension
        image_array = np.expand_dims(image_array, axis=-1)
        
        return image_array
    
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None

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
    
    return image

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
    
    # Test the augmentation function
    augmented = augment_image(test_image)
    print(f"Augmented image shape: {augmented.shape}")