import unittest
import numpy as np
from PIL import Image
import io
from ml.preprocessing.preprocess import preprocess_image, augment_image

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Create a sample image for testing
        self.sample_image = Image.new('RGB', (100, 50), color='white')
        self.sample_image_array = np.array(self.sample_image)

    def test_preprocess_image_valid(self):
        preprocessed = preprocess_image(self.sample_image)
        self.assertIsNotNone(preprocessed)
        self.assertEqual(preprocessed.shape, (64, 128, 1))
        self.assertTrue(np.all(preprocessed >= 0) and np.all(preprocessed <= 1))

    def test_preprocess_image_numpy_input(self):
        preprocessed = preprocess_image(self.sample_image_array)
        self.assertIsNotNone(preprocessed)
        self.assertEqual(preprocessed.shape, (64, 128, 1))

    def test_preprocess_image_bytes_input(self):
        img_byte_arr = io.BytesIO()
        self.sample_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        preprocessed = preprocess_image(img_byte_arr)
        self.assertIsNotNone(preprocessed)
        self.assertEqual(preprocessed.shape, (64, 128, 1))

    def test_preprocess_image_invalid_input(self):
        invalid_input = "not an image"
        preprocessed = preprocess_image(invalid_input)
        self.assertIsNone(preprocessed)

    def test_preprocess_image_corrupted_input(self):
        corrupted_bytes = b'corrupted image data'
        preprocessed = preprocess_image(corrupted_bytes)
        self.assertIsNone(preprocessed)

    def test_augment_image(self):
        augmented = augment_image(self.sample_image_array)
        self.assertEqual(augmented.shape, self.sample_image_array.shape)
        self.assertFalse(np.array_equal(augmented, self.sample_image_array))

if __name__ == '__main__':
    unittest.main()

