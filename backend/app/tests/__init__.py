import pytest
from fastapi.testclient import TestClient
from backend.main import app
import io
from PIL import Image
import numpy as np

client = TestClient(app)

def create_test_image():
    img = Image.new('L', (64, 128), color=255)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def test_recognize_handwriting():
    test_image = create_test_image()
    response = client.post("/recognize", files={"file": ("test_image.png", test_image, "image/png")})
    assert response.status_code == 200
    assert "text" in response.json()

def test_recognize_handwriting_invalid_file():
    response = client.post("/recognize", files={"file": ("test.txt", b"invalid content", "text/plain")})
    assert response.status_code == 422  # Unprocessable Entity

def test_cors_headers():
    response = client.options("/recognize", headers={"Origin": "http://localhost:3000"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"

if __name__ == "__main__":
    pytest.main([__file__])

