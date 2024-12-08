from scipy import io
import unittest
import numpy as np
from PIL import Image
from ml.preprocessing.preprocess import preprocess_image, augment_image

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.test_image = np.random.rand(100, 200, 3) * 255
        self.test_image = self.test_image.astype(np.uint8)

    def test_preprocess_image(self):
        preprocessed = preprocess_image(self.test_image)
        self.assertIsNotNone(preprocessed)
        self.assertEqual(preprocessed.shape, (64, 128, 1))
        self.assertEqual(preprocessed.dtype, np.float32)
        self.assertTrue(0 <= preprocessed.min() <= preprocessed.max() <= 1)

    def test_preprocess_image_with_pil(self):
        pil_image = Image.fromarray(self.test_image)
        preprocessed = preprocess_image(pil_image)
        self.assertIsNotNone(preprocessed)
        self.assertEqual(preprocessed.shape, (64, 128, 1))

    def test_preprocess_image_with_bytes(self):
        byte_io = io.BytesIO()
        Image.fromarray(self.test_image).save(byte_io, format='PNG')
        byte_io.seek(0)
        preprocessed = preprocess_image(byte_io.getvalue())
        self.assertIsNotNone(preprocessed)
        self.assertEqual(preprocessed.shape, (64, 128, 1))

    def test_augment_image(self):
        augmented = augment_image(self.test_image)
        self.assertEqual(augmented.shape, self.test_image.shape)
        self.assertFalse(np.array_equal(augmented, self.test_image))

if __name__ == '__main__':
    unittest.main()

