import io
from PIL import Image
import numpy as np
from ml.models.cnn_lstm_model import create_advanced_cnn_lstm_model
from ml.preprocessing.preprocess import preprocess_image
import yaml
from app.utils.logger import logger
import tensorflow as tf

# Step 1: Load configuration
with open("config/config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Step 2: Create and load the model
model = create_advanced_cnn_lstm_model(
    tuple(config["model"]["input_shape"]), 
    config["model"]["num_classes"]
)
model.load_weights(config["paths"]["model_weights"])

# Step 3: Define the recognition function
def recognize_text(image_bytes):
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
        
        # Decode prediction
        recognized_text = decode_prediction(prediction[0])
        
        return recognized_text
    except Exception as e:
        logger.error(f"Error in recognize_text: {str(e)}")
        raise

# Step 4: Define the decoding function
def decode_prediction(prediction):
    characters = config["model"]["character_set"]
    return ''.join([characters[np.argmax(pred)] for pred in prediction])

# Step 5: TensorFlow setup for GPU (if available)
def setup_gpu():
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU setup complete. Found {len(gpus)} GPU(s).")
        else:
            logger.info("No GPUs found. Using CPU.")
    except Exception as e:
        logger.error(f"Error setting up GPU: {str(e)}")

# Call GPU setup
setup_gpu()