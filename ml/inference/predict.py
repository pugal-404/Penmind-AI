import tensorflow as tf
import numpy as np
from ml.preprocessing.preprocess import preprocess_image
from ml.models.cnn_lstm_model import ctc_loss
import yaml
import os
import logging
import cv2
import onnxruntime as rt
from PIL import Image
import io
import base64

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)

config = load_config()

# Initialize ONNX Runtime session for GPU acceleration
onnx_model_path = os.path.join(config['paths']['model_versions'], 'model_latest.onnx')
session = rt.InferenceSession(onnx_model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def load_model(model_path):
    """Load the TensorFlow model."""
    return tf.keras.models.load_model(model_path, custom_objects={'ctc_loss': ctc_loss})

def predict_with_confidence(image, confidence_threshold=0.7):
    """
    Predict text from image with confidence scores.
    
    Args:
        image (numpy.ndarray): Input image
        confidence_threshold (float): Threshold for confidence scores
    
    Returns:
        tuple: Predicted text and confidence score
    """
    preprocessed_image = preprocess_image(image, target_size=tuple(config['model']['input_shape'][:2]))
    
    if preprocessed_image is None:
        logger.error("Failed to preprocess the image")
        return "Error: Unable to preprocess the image.", 0.0
    
    # Ensure the image has the correct shape for the model
    if preprocessed_image.shape != tuple(config['model']['input_shape']):
        preprocessed_image = tf.image.resize(preprocessed_image, config['model']['input_shape'][:2])
        preprocessed_image = tf.expand_dims(preprocessed_image, axis=0)
    else:
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    
    # Run inference using ONNX Runtime
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    prediction = session.run([output_name], {input_name: preprocessed_image.astype(np.float32)})[0]
    
    # Decode the prediction
    decoded_text = decode_prediction(prediction[0], config['model']['character_set'])
    confidence = np.max(prediction)
    
    if confidence < confidence_threshold:
        logger.warning(f"Low confidence prediction: {confidence:.2f}")
        return f"Low confidence: {decoded_text}", confidence
    
    return decoded_text, confidence

def decode_prediction(prediction, character_set):
    """
    Decode the model's prediction to text.
    
    Args:
        prediction (numpy.ndarray): Model's output prediction
        character_set (str): Set of characters used for prediction
    
    Returns:
        str: Decoded text
    """
    # Implement beam search decoding
    input_len = np.ones(prediction.shape[0]) * prediction.shape[1]
    results = tf.keras.backend.ctc_decode(prediction, input_length=input_len, greedy=False, beam_width=100, top_paths=1)
    
    # Extract the best path
    best_path = results[0][0]
    
    # Convert to text
    text = ''.join([character_set[index] for index in best_path[0] if index != -1 and index < len(character_set)])
    
    return text

def real_time_inference(camera_input):
    """
    Perform real-time inference on camera input.
    
    Args:
        camera_input (numpy.ndarray): Input image from camera
    
    Returns:
        tuple: Predicted text and confidence score
    """
    return predict_with_confidence(camera_input)

def batch_process(image_list):
    """
    Process a batch of images for prediction.
    
    Args:
        image_list (list): List of input images
    
    Returns:
        list: List of tuples containing predicted text and confidence scores
    """
    results = []
    for image in image_list:
        text, confidence = predict_with_confidence(image)
        results.append((text, confidence))
    return results

def handle_ambiguous_prediction(text, confidence, threshold=0.5):
    """
    Handle ambiguous predictions by flagging them for user review.
    
    Args:
        text (str): Predicted text
        confidence (float): Confidence score
        threshold (float): Confidence threshold for flagging ambiguous predictions
    
    Returns:
        tuple: Flagged text and boolean indicating if review is needed
    """
    if confidence < threshold:
        return f"[REVIEW NEEDED] {text}", True
    return text, False

def process_base64_image(base64_string):
    """
    Process a base64 encoded image.
    
    Args:
        base64_string (str): Base64 encoded image string
    
    Returns:
        numpy.ndarray: Decoded image as a numpy array
    """
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return np.array(image)

if __name__ == "__main__":
    # Test the prediction function
    test_image = cv2.imread("path/to/test_image.jpg")
    predicted_text, confidence = predict_with_confidence(test_image)
    logger.info(f"Predicted text: {predicted_text}")
    logger.info(f"Confidence: {confidence:.2f}")
    
    # Test real-time inference (simulated with a single frame)
    camera_frame = cv2.imread("path/to/camera_frame.jpg")
    real_time_text, real_time_confidence = real_time_inference(camera_frame)
    logger.info(f"Real-time prediction: {real_time_text}")
    logger.info(f"Real-time confidence: {real_time_confidence:.2f}")
    
    # Test batch processing
    batch_images = [cv2.imread(f"path/to/batch_image_{i}.jpg") for i in range(5)]
    batch_results = batch_process(batch_images)
    for i, (text, conf) in enumerate(batch_results):
        logger.info(f"Batch image {i+1}: Text = {text}, Confidence = {conf:.2f}")
    
    # Test ambiguous prediction handling
    ambiguous_text, needs_review = handle_ambiguous_prediction("Ambiguous text", 0.4)
    logger.info(f"Ambiguous prediction: {ambiguous_text}")
    logger.info(f"Needs review: {needs_review}")
    
    # Test base64 image processing
    with open("path/to/base64_image.txt", "r") as f:
        base64_string = f.read().strip()
    base64_image = process_base64_image(base64_string)
    base64_text, base64_confidence = predict_with_confidence(base64_image)
    logger.info(f"Base64 image prediction: {base64_text}")
    logger.info(f"Base64 image confidence: {base64_confidence:.2f}")

