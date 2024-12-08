import io
from PIL import Image
import numpy as np
import os
import sys
import yaml
import tensorflow as tf
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, BertTokenizer, BertForMaskedLM
import re
import base64
import traceback
from tenacity import retry, stop_after_attempt, wait_exponential

# Add the project root to PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from ml.preprocessing.preprocess import preprocess_image, segment_lines, segment_words
from app.utils.logger import get_logger

# Load configuration
config_path = os.path.join(project_root, "config", "config.yaml")
with open(config_path, "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

# Logger initialization
logger = get_logger(__name__)

# Global variables for models
processor = None
model_cache = {}
bert_tokenizer = None
bert_model = None

def get_model(model_type):
    if model_type == 'base':
        model_key = 'handwritten_base'
    elif model_type == 'small':
        model_key = 'handwritten_small'
    elif model_type == 'math':
        model_key = 'math'
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    if model_key not in model_cache:
        model_name = config["huggingface"]["models"][model_key]
        model_cache[model_key] = load_model_with_retry(model_name)
    return model_cache[model_key]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def load_model_with_retry(model_name):
    logger.info(f"Attempting to load model: {model_name}")
    try:
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
        logger.info(f"Successfully loaded model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def initialize_models():
    global processor, bert_tokenizer, bert_model

    try:
        logger.info("Starting model initialization")
        processor = TrOCRProcessor.from_pretrained(config["huggingface"]["models"]["handwritten_base"])
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error initializing models: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def recognize_trocr(image, model_type='base'):
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')

        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        
        model = get_model(model_type)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return recognized_text
    except Exception as e:
        logger.error(f"Error in recognize_trocr: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def recognize_text(image_data, model_type='ensemble'):
    try:
        image = convert_to_pil_image(image_data)
        preprocessed_image = preprocess_image(image)
        
        if preprocessed_image is None:
            raise ValueError("Failed to preprocess the image")

        lines = segment_lines(preprocessed_image)
        recognized_lines = []

        for line in lines:
            words = segment_words(line)
            line_text = []
            for word in words:
                word_image = Image.fromarray((word * 255).astype(np.uint8))
                if model_type == 'ensemble':
                    results = []
                    for model in ['base', 'small', 'math']:
                        results.append(recognize_trocr(word_image, model))
                    word_text = ensemble_decision(results)
                else:
                    word_text = recognize_trocr(word_image, model_type)
                
                line_text.append(word_text)
            
            line_text = ' '.join(line_text)
            line_text = post_process_text(line_text)
            recognized_lines.append(line_text)

        return '\n'.join(recognized_lines)
    except Exception as e:
        logger.error(f"Error in recognize_text: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def convert_to_pil_image(image_data):
    if isinstance(image_data, bytes):
        return Image.open(io.BytesIO(image_data))
    elif isinstance(image_data, str):
        image_data = base64.b64decode(image_data)
        return Image.open(io.BytesIO(image_data))
    elif isinstance(image_data, np.ndarray):
        return Image.fromarray(image_data)
    elif isinstance(image_data, Image.Image):
        return image_data
    else:
        raise ValueError("Unsupported image data type")

def ensemble_decision(results):
    # Implement a more sophisticated ensemble decision
    # For now, we'll use the result with the highest confidence (longest text)
    return max(results, key=len)

def post_process_text(text):
    text = normalize_case(text)
    text = handle_special_characters(text)
    text = apply_grammar_rules(text)
    text = correct_spelling(text)
    return text

def normalize_case(text):
    # Convert to sentence case
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return ' '.join(sentence.capitalize() for sentence in sentences)

def handle_special_characters(text):
    special_chars = {
        '∫': '\\int',
        '∑': '\\sum',
        '∏': '\\prod',
        '≠': '\\neq',
        '≤': '\\leq',
        '≥': '\\geq',
        '∈': '\\in',
        '∉': '\\notin',
    }
    for char, replacement in special_chars.items():
        text = text.replace(char, replacement)
    return text

def apply_grammar_rules(text):
    # Implement basic grammar rules
    text = re.sub(r'\s+([.,!?])', r'\1', text)  # Remove spaces before punctuation
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between lowercase and uppercase letters
    return text

def correct_spelling(text):
    # Use BERT for spelling correction
    tokens = bert_tokenizer.tokenize(text)
    for i, token in enumerate(tokens):
        if token not in bert_tokenizer.vocab:
            masked_text = ' '.join(tokens[:i] + ['[MASK]'] + tokens[i+1:])
            input_ids = bert_tokenizer.encode(masked_text, return_tensors='pt')
            with torch.no_grad():
                outputs = bert_model(input_ids)
            predicted_token = bert_tokenizer.convert_ids_to_tokens(outputs.logits[0, i].argmax().item())
            tokens[i] = predicted_token
    return bert_tokenizer.convert_tokens_to_string(tokens)

# TensorFlow setup for GPU (if available)
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
        logger.error(traceback.format_exc())

# Call GPU setup
setup_gpu()

initialize_models()