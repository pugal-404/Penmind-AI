import tensorflow as tf
import numpy as np
from ml.preprocessing.preprocess import preprocess_image
from ml.training.train import generate_character_set

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict(model, image, character_set):
    preprocessed_image = preprocess_image(image)
    if preprocessed_image is None:
        return "Error: Unable to preprocess the image."
    
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
    
    prediction = model.predict(preprocessed_image)
    recognized_text = decode_prediction(prediction[0], character_set)
    
    return recognized_text

def decode_prediction(prediction, character_set):
    return ''.join([character_set[np.argmax(pred)] for pred in prediction])

if __name__ == "__main__":
    model_path = 'path/to/your/saved/model.h5'
    model = load_model(model_path)
    
    character_set = generate_character_set()
    
    # Test with a sample image
    sample_image_path = 'path/to/sample_image.png'
    with open(sample_image_path, 'rb') as image_file:
        sample_image = image_file.read()
    
    recognized_text = predict(model, sample_image, character_set)
    print(f"Recognized text: {recognized_text}")

# Example of error handling
try:
    invalid_image_path = 'path/to/invalid_image.txt'
    with open(invalid_image_path, 'rb') as invalid_file:
        invalid_image = invalid_file.read()
    
    result = predict(model, invalid_image, character_set)
    print(f"Result for invalid image: {result}")
except Exception as e:
    print(f"Error occurred: {str(e)}")

print("Inference script completed.")

