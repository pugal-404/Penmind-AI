import tensorflow as tf
import numpy as np
import os
from ml.models.cnn_lstm_model import create_advanced_cnn_lstm_model
from ml.preprocessing.preprocess import preprocess_image, augment_image
from tensorflow import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard

def load_dataset(data_dir, target_size=(64, 128)):
    """
    Load and preprocess the dataset.
    
    Args:
    data_dir (str): Directory containing the dataset.
    target_size (tuple): Target size for the preprocessed images.
    
    Returns:
    tuple: (x_train, y_train), (x_val, y_val), character_set
    """
    character_set = []
    images = []
    labels = []
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                label = os.path.basename(root)
                if label not in character_set:
                    character_set.append(label)
                
                img_path = os.path.join(root, file)
                img = preprocess_image(img_path, target_size)
                
                if img is not None:
                    images.append(img)
                    labels.append(character_set.index(label))
    
    x = np.array(images)
    y = np.array(labels)
    
    # Split the data into training and validation sets
    split = int(0.8 * len(x))
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]
    
    return (x_train, y_train), (x_val, y_val), character_set

def data_generator(x, y, batch_size, character_set, augment=True):
    num_samples = x.shape[0]
    num_classes = len(character_set)
    
    while True:
        indices = np.random.permutation(num_samples)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_x = x[batch_indices]
            batch_y = y[batch_indices]
            
            if augment:
                batch_x = np.array([augment_image(img) for img in batch_x])
            
            batch_y_onehot = tf.keras.utils.to_categorical(batch_y, num_classes)
            
            yield batch_x, batch_y_onehot

def train_model(model, train_data, val_data, character_set, epochs=50, batch_size=32):
    x_train, y_train = train_data
    x_val, y_val = val_data
    
    # Define callbacks
    checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)
    
    # Create data generators
    train_gen = data_generator(x_train, y_train, batch_size, character_set, augment=True)
    val_gen = data_generator(x_val, y_val, batch_size, character_set, augment=False)
    
    # Train the model
    history = model.fit(
        train_gen,
        steps_per_epoch=len(x_train) // batch_size,
        validation_data=val_gen,
        validation_steps=len(x_val) // batch_size,
        epochs=epochs,
        callbacks=[checkpoint, reduce_lr, early_stopping, tensorboard]
    )
    
    return history

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Load and preprocess data
    data_dir = 'path/to/your/dataset'  # Replace with the actual path to your dataset
    (x_train, y_train), (x_val, y_val), character_set = load_dataset(data_dir)
    
    # Create and compile the model
    input_shape = (64, 128, 1)
    num_classes = len(character_set)
    model = create_advanced_cnn_lstm_model(input_shape, num_classes)
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model
    history = train_model(model, (x_train, y_train), (x_val, y_val), character_set)
    
    # Save the final model
    model.save('handwriting_recognition_model.h5')
    
    print("Training completed. Model saved as 'handwriting_recognition_model.h5'")
