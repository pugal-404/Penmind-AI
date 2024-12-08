from ml.models.cnn_lstm_model import ctc_loss
import tensorflow as tf
import numpy as np
from ml.preprocessing.preprocess import preprocess_image
import yaml
import os
import logging

logger = logging.getLogger(__name__)

def load_model(model_path):
    return tf.keras.models.load_model(model_path, custom_objects={'ctc_loss': ctc_loss})

def predict(model, image, config):
    logger.info(f"Input image: shape={image.shape}, dtype={image.dtype}")
    
    preprocessed_image = preprocess_image(image, target_size=tuple(config['model']['input_shape'][:2]))
    if preprocessed_image is None:
        logger.error("Failed to preprocess the image")
        return "Error: Unable to preprocess the image."
    
    logger.info(f"Preprocessed image: shape={preprocessed_image.shape}, dtype={preprocessed_image.dtype}")
    
    # Ensure the image has the correct shape for the model
    if preprocessed_image.shape != tuple(config['model']['input_shape']):
        logger.warning(f"Resizing image to match model input shape: {tuple(config['model']['input_shape'])}")
        preprocessed_image = tf.image.resize(preprocessed_image, config['model']['input_shape'][:2])
        preprocessed_image = tf.expand_dims(preprocessed_image, axis=0)
    else:
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    
    logger.info(f"Model input: shape={preprocessed_image.shape}, dtype={preprocessed_image.dtype}")
    
    prediction = model.predict(preprocessed_image)
    logger.info(f"Model output: shape={prediction.shape}, dtype={prediction.dtype}")
    
    recognized_text = decode_prediction(prediction[0], config['model']['character_set'])
    
    return recognized_text

def decode_prediction(prediction, character_set):
    # Implement beam search decoding
    input_len = np.ones(prediction.shape[0]) * prediction.shape[1]
    results = tf.keras.backend.ctc_decode(prediction, input_length=input_len, greedy=False, beam_width=100, top_paths=1)
    
    # Extract the best path
    best_path = results[0][0]
    
    # Convert to text
    text = ''.join([character_set[index] for index in best_path[0] if index != -1 and index < len(character_set)])
    
    # Split text into lines
    lines = text.split('\n')
    
    return lines

# Load configuration
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
config_path = os.path.join(project_root, 'config', 'config.yaml')
with open(config_path, "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

