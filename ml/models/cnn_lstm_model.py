import tensorflow as tf
from tensorflow.keras import layers, models

def attention_mechanism(query, key, value):
    """
    Implement an attention mechanism.
    
    Args:
        query (tf.Tensor): Query tensor
        key (tf.Tensor): Key tensor
        value (tf.Tensor): Value tensor
    
    Returns:
        tf.Tensor: Context vector after applying attention
    """
    score = tf.matmul(query, key, transpose_b=True)
    distribution = tf.nn.softmax(score)
    return tf.matmul(distribution, value)

def create_advanced_cnn_lstm_model(input_shape, num_classes, use_transformer=False):
    """
    Create an advanced CNN-LSTM model with optional Transformer layers.
    
    Args:
        input_shape (tuple): Shape of the input image
        num_classes (int): Number of output classes
        use_transformer (bool): Whether to use Transformer layers
    
    Returns:
        tf.keras.Model: The constructed model
    """
    inputs = layers.Input(shape=input_shape)
    
    # CNN layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Prepare the feature maps for the LSTM layers
    new_shape = ((input_shape[0] // 8), (input_shape[1] // 8) * 128)
    x = layers.Reshape(target_shape=new_shape)(x)
    
    if use_transformer:
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
        context_vector = layers.Lambda(lambda x: attention_mechanism(x[0], x[1], x[2]))([query, key, value])
        
        # Combine context vector with LSTM output
        x = layers.Concatenate()([lstm_out, context_vector])
        x = layers.Bidirectional(layers.LSTM(128))(x)
        x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

def ctc_loss(y_true, y_pred):
    """
    Implement CTC loss function.
    
    Args:
        y_true (tf.Tensor): True labels
        y_pred (tf.Tensor): Predicted labels
    
    Returns:
        tf.Tensor: CTC loss value
    """
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

if __name__ == "__main__":
    input_shape = (64, 256, 1)  # Updated input shape
    num_classes = 128  # Increased to accommodate more characters and symbols
    
    model = create_advanced_cnn_lstm_model(input_shape, num_classes)
    model.summary()
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=ctc_loss,
        metrics=['accuracy']
    )

