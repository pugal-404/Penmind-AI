import tensorflow as tf
from tensorflow import layers, models

def create_advanced_cnn_lstm_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # CNN layers with residual connections
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Residual block 1
    residual = x
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Residual block 2
    residual = x
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, residual])
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Prepare the feature maps for the LSTM layers
    new_shape = ((input_shape[0] // 8), (input_shape[1] // 8) * 128)
    x = layers.Reshape(target_shape=new_shape)(x)
    
    # Bidirectional LSTM layers
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
    x = layers.Dropout(0.25)(x)
    
    # Attention mechanism
    attention = layers.Dense(1, activation='tanh')(x)
    attention = layers.Flatten()(attention)
    attention = layers.Activation('softmax')(attention)
    attention = layers.RepeatVector(256)(attention)
    attention = layers.Permute([2, 1])(attention)
    
    sent_representation = layers.Multiply()([x, attention])
    sent_representation = layers.Lambda(lambda xin: tf.keras.backend.sum(xin, axis=-2))(sent_representation)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(sent_representation)
    
    # Create and compile the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

if __name__ == "__main__":
    input_shape = (64, 128, 1)  # Example input shape
    num_classes = 95  # 26 lowercase, 26 uppercase, 10 digits, 33 special characters
    
    model = create_advanced_cnn_lstm_model(input_shape, num_classes)
    model.summary()
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )