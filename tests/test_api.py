# tests/test_api.py
import unittest
from fastapi.testclient import TestClient
from backend.main import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_recognize_endpoint(self):
        with open("path/to/test/image.png", "rb") as f:
            response = self.client.post("/recognize", files={"file": f})
        self.assertEqual(response.status_code, 200)
        self.assertIn("text", response.json())

if __name__ == "__main__":
    unittest.main()