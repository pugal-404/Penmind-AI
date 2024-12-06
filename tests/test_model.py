import unittest
import numpy as np
import tensorflow as tf
from ml.models.cnn_lstm_model import create_cnn_lstm_model

class TestModel(unittest.TestCase):
    def setUp(self):
        self.input_shape = (64, 128, 1)
        self.num_classes = 62
        self.model = create_cnn_lstm_model(self.input_shape, self.num_classes)

    def test_model_creation(self):
        self.assertIsInstance(self.model, tf.keras.Model)
        self.assertEqual(self.model.input_shape[1:], self.input_shape)
        self.assertEqual(self.model.output_shape[1], self.num_classes)

    def test_model_compile(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.assertIsNotNone(self.model.optimizer)
        self.assertIsNotNone(self.model.loss)
        self.assertIsNotNone(self.model.metrics)

    def test_model_prediction(self):
        sample_input = np.random.rand(1, *self.input_shape)
        prediction = self.model.predict(sample_input)
        self.assertEqual(prediction.shape, (1, self.num_classes))
        self.assertAlmostEqual(np.sum(prediction), 1.0, places=6)

    def test_model_training(self):
        x_train = np.random.rand(100, *self.input_shape)
        y_train = np.random.randint(0, self.num_classes, (100, 1))
        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(x_train, y_train, epochs=5, validation_split=0.2, verbose=0)

        self.assertIn('loss', history.history)
        self.assertIn('accuracy', history.history)
        self.assertIn('val_loss', history.history)
        self.assertIn('val_accuracy', history.history)

    def test_model_overfit_single_batch(self):
        x_single = np.random.rand(1, *self.input_shape)
        y_single = np.random.randint(0, self.num_classes, (1, 1))
        y_single = tf.keras.utils.to_categorical(y_single, self.num_classes)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(x_single, y_single, epochs=100, verbose=0)

        self.assertGreater(history.history['accuracy'][-1], 0.9)

    def test_model_generalization(self):
        x_train = np.random.rand(1000, *self.input_shape)
        y_train = np.random.randint(0, self.num_classes, (1000, 1))
        y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)

        x_val = np.random.rand(200, *self.input_shape)
        y_val = np.random.randint(0, self.num_classes, (200, 1))
        y_val = tf.keras.utils.to_categorical(y_val, self.num_classes)

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val), verbose=0)

        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]

        self.assertLess(abs(train_acc - val_acc), 0.2)

if __name__ == '__main__':
    unittest.main()

