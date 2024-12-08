import tensorflow as tf
import numpy as np
import os
from ml.models.cnn_lstm_model import create_advanced_cnn_lstm_model, ctc_loss
from ml.preprocessing.preprocess import preprocess_image, augment_image
import yaml
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

def load_config():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

def load_dataset(data_dir, config):
    images = []
    labels = []
    character_set = config['model']['character_set']
    
    for char in character_set:
        char_dir = os.path.join(data_dir, char)
        if os.path.isdir(char_dir):
            for file in os.listdir(char_dir):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(char_dir, file)
                    img = preprocess_image(img_path, target_size=tuple(config['model']['input_shape'][:2]))
                    
                    if img is not None:
                        images.append(img)
                        labels.append(character_set.index(char))
    
    return np.array(images), np.array(labels)

def data_generator(x, y, batch_size, config, augment=True):
    num_samples = x.shape[0]
    num_classes = len(config['model']['character_set'])
    
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

def train_model(config):
    # Load and preprocess data
    x, y = load_dataset(config['paths']['dataset'], config)
    
    # Split the data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    
    # Create and compile the model
    model = create_advanced_cnn_lstm_model(tuple(config['model']['input_shape']), len(config['model']['character_set']))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
        loss=ctc_loss,
        metrics=['accuracy']
    )
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(config['paths']['best_model'], save_best_only=True, monitor='val_accuracy'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir=config['paths']['logs'])
    ]
    
    # Create data generators
    train_gen = data_generator(x_train, y_train, config['training']['batch_size'], config, augment=config['training']['data_augmentation'])
    val_gen = data_generator(x_val, y_val, config['training']['batch_size'], config, augment=False)
    
    # Train the model
    history = model.fit(
        train_gen,
        steps_per_epoch=len(x_train) // config['training']['batch_size'],
        validation_data=val_gen,
        validation_steps=len(x_val) // config['training']['batch_size'],
        epochs=config['training']['epochs'],
        callbacks=callbacks
    )
    
    # Save the final model
    model.save(config['paths']['model_save'])
    
    return history

if __name__ == "__main__":
    config = load_config()
    history = train_model(config)
    
    logger.info("Training completed. Model saved.")

    # Plot training history
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(config['paths']['logs'], 'training_history.png'))
    plt.close()

    logger.info("Training history plot saved.")
