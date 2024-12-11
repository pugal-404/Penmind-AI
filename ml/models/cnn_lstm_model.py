import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch

class VitCNNLSTMModel:
    def __init__(self, input_shape=(64, 256, 1), num_classes=128, use_transformer=True):
        """
        Initialize the hybrid ViT-CNN-LSTM model.
        
        Args:
            input_shape (tuple): Shape of input images
            num_classes (int): Number of output classes
            use_transformer (bool): Whether to use additional transformer layers
        """
        # Hugging Face ViT configuration
        self.vit_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        self.vit_model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
        
        # Freeze ViT base model weights
        for param in self.vit_model.parameters():
            param.requires_grad = False
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_transformer = use_transformer
        
    def attention_mechanism(self, query, key, value):
        """
        Implement an attention mechanism.
        """
        score = tf.matmul(query, key, transpose_b=True)
        distribution = tf.nn.softmax(score)
        return tf.matmul(distribution, value)
    
    def _preprocess_with_vit(self, images):
        """
        Preprocess images using ViT image processor.
        
        Args:
            images (np.ndarray): Input images
        
        Returns:
            torch.Tensor: Processed images
        """
        # Convert TensorFlow/NumPy images to PIL format
        processed_images = []
        for image in images:
            # Ensure the image is in the right format and range
            image = (image * 255).astype(np.uint8)
            pil_image = tf.keras.preprocessing.image.array_to_img(image)
            processed_images.append(pil_image)
        
        # Use ViT processor to prepare images
        inputs = self.vit_processor(processed_images, return_tensors="pt")
        return inputs
    
    def extract_vit_features(self, images):
        """
        Extract features from ViT model.
        
        Args:
            images (np.ndarray): Input images
        
        Returns:
            np.ndarray: Extracted features
        """
        # Preprocess images
        inputs = self._preprocess_with_vit(images)
        
        # Extract features using ViT
        with torch.no_grad():
            outputs = self.vit_model(**inputs)
        
        # Convert to NumPy and extract features
        features = outputs.logits.numpy()
        return features
    
    def create_model(self):
        """
        Create the hybrid ViT-CNN-LSTM model.
        
        Returns:
            tf.keras.Model: Constructed model
        """
        inputs = layers.Input(shape=self.input_shape)
        
        # CNN layers
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Prepare the feature maps for the LSTM layers
        new_shape = ((self.input_shape[0] // 8), (self.input_shape[1] // 8) * 128)
        x = layers.Reshape(target_shape=new_shape)(x)
        
        if self.use_transformer:
            # Transformer layers
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
            x = layers.Dropout(0.1)(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.Dense(128)(x)
            x = layers.Dropout(0.1)(x)
        else:
            # Bidirectional LSTM layers with attention
            lstm_out = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
            lstm_out = layers.Dropout(0.5)(lstm_out)
            
            # Self-attention mechanism
            query = layers.Dense(128)(lstm_out)
            key = layers.Dense(128)(lstm_out)
            value = layers.Dense(128)(lstm_out)
            context_vector = layers.Lambda(lambda x: self.attention_mechanism(x[0], x[1], x[2]))([query, key, value])
            
            # Combine context vector with LSTM output
            x = layers.Concatenate()([lstm_out, context_vector])
            x = layers.Bidirectional(layers.LSTM(128))(x)
            x = layers.Dropout(0.5)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create the model
        model = models.Model(inputs=inputs, outputs=outputs)
        return model
    
    def ctc_loss(self, y_true, y_pred):
        """
        Implement CTC loss function.
        """
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    
    def compile_model(self, model):
        """
        Compile the model with appropriate optimizer and loss.
        
        Args:
            model (tf.keras.Model): Model to compile
        
        Returns:
            tf.keras.Model: Compiled model
        """
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=self.ctc_loss,
            metrics=['accuracy']
        )
        return model

def main():
    # Configuration
    input_shape = (64, 256, 1)  # Updated input shape
    num_classes = 128  # Increased to accommodate more characters and symbols
    
    # Create ViT-CNN-LSTM model
    vit_cnn_lstm = VitCNNLSTMModel(input_shape, num_classes, use_transformer=True)
    
    # Create model
    model = vit_cnn_lstm.create_model()
    
    # Compile model
    model = vit_cnn_lstm.compile_model(model)
    
    # Print model summary
    model.summary()

if __name__ == "__main__":
    main()