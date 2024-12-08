import unittest
import numpy as np
import tensorflow as tf
from ml.models.cnn_lstm_model import create_advanced_cnn_lstm_model, ctc_loss

class TestModel(unittest.TestCase):
    def setUp(self):
        self.input_shape = (64, 128, 1)
        self.num_classes = 95
        self.model = create_advanced_cnn_lstm_model(self.input_shape, self.num_classes)

    def test_model_creation(self):
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.input_shape, (None, 64, 128, 1))
        self.assertEqual(self.model.output_shape, (None, self.num_classes))

    def test_model_compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=ctc_loss,
            metrics=['accuracy']
        )
        self.assertTrue(self.model.optimizer)
        self.assertTrue(self.model.loss)

    def test_model_prediction(self):
        test_input = np.random.rand(1, 64, 128, 1)
        prediction = self.model.predict(test_input)
        self.assertEqual(prediction.shape, (1, self.num_classes))

    def test_ctc_loss(self):
        y_true = tf.constant([[1, 2, 3, -1, -1]], dtype=tf.float32)
        y_pred = tf.random.normal((1, 10, 95))
        loss = ctc_loss(y_true, y_pred)
        self.assertIsNotNone(loss)
        self.assertTrue(tf.is_tensor(loss))

if __name__ == '__main__':
    unittest.main()