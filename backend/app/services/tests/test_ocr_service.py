import pytest
from app.services.ocr_service import recognize_text, handle_special_characters
from PIL import Image
import io
import numpy as np

def create_test_image(text):
    image = Image.new('RGB', (100, 30), color='white')
    return image

@pytest.mark.parametrize("model_type", ['cnn_lstm', 'trocr_base', 'trocr_small', 'trocr_math'])
def test_recognize_text(model_type):
    test_image = create_test_image("Hello")
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    result = recognize_text(img_byte_arr, model_type)
    assert isinstance(result, str)
    assert len(result) > 0

def test_handle_special_characters():
    input_text = "∫x dx + ∑i=1^n ai ≠ 0"
    expected_output = "\\int x dx + \\sum i=1^n ai \\neq 0"
    assert handle_special_characters(input_text) == expected_output

def test_recognize_text_invalid_model():
    test_image = create_test_image("Hello")
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    with pytest.raises(ValueError):
        recognize_text(img_byte_arr, 'invalid_model')

