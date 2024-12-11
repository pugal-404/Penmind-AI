import os
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from transformers import TFAutoModel, AutoConfig
from huggingface_hub import HfApi, hf_hub_download

# Set environment variable for HuggingFace API key securely
api_token = os.getenv("HUGGINGFACE_API_KEY")

def create_advanced_cnn_lstm_model(input_shape, num_classes, use_vit=True):
    inputs = Input(shape=input_shape)

    if use_vit:
        try:
            # Load the Vision Transformer model with appropriate config
            config = AutoConfig.from_pretrained('google/vit-base-patch16-224')
            vit_model = TFAutoModel.from_pretrained('google/vit-base-patch16-224', config=config, use_auth_token=api_token)
            # Resize and preprocess the input appropriately for the ViT model
            x = layers.Resizing(224, 224)(inputs)
            x = layers.Lambda(lambda x: tf.cast(x, tf.float32))(x)
            vit_outputs = vit_model(x, return_dict=True)
            x = layers.GlobalAveragePooling2D()(vit_outputs.last_hidden_state)
            x = layers.Dense(128, activation='relu')(x)
        except Exception as e:
            print(f"Vision Transformer loading error: {e}")
            x = fallback_cnn_model(inputs)
    else:
        x = fallback_cnn_model(inputs)

    # LSTM and output layers
    x = layers.RepeatVector(3)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model

def fallback_cnn_model(inputs):
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return layers.Flatten()(x)

if __name__ == "__main__":
    input_shape = (128, 128, 3)
    num_classes = 128
    model = create_advanced_cnn_lstm_model(input_shape, num_classes, use_vit=True)
    model.summary()
