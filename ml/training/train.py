import tensorflow as tf
import numpy as np
import random
import os
from ml.models.cnn_lstm_model import create_advanced_cnn_lstm_model, ctc_loss
from ml.preprocessing.preprocess import preprocess_image, augment_image, elastic_transform
import yaml
from sklearn.model_selection import train_test_split
import logging
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import Mean
import editdistance
import torch.onnx

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

def load_dataset(data_dir, config):
    """Load and preprocess the dataset."""
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
    """Generate batches of data with optional augmentation."""
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
                batch_x = np.array([elastic_transform(img, alpha=random.uniform(30, 60), sigma=random.uniform(3, 6), alpha_affine=random.uniform(3, 6)) for img in batch_x])
            
            batch_y_onehot = tf.keras.utils.to_categorical(batch_y, num_classes)
            
            yield batch_x, batch_y_onehot

def calculate_cer(y_true, y_pred):
    """Calculate Character Error Rate (CER)."""
    total_cer = 0
    for true, pred in zip(y_true, y_pred):
        total_cer += editdistance.eval(true, pred) / len(true)
    return total_cer / len(y_true)

def calculate_wer(y_true, y_pred):
    """Calculate Word Error Rate (WER)."""
    total_wer = 0
    for true, pred in zip(y_true, y_pred):
        true_words = true.split()
        pred_words = pred.split()
        total_wer += editdistance.eval(true_words, pred_words) / len(true_words)
    return total_wer / len(y_true)

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, character_set):
        super().__init__()
        self.validation_data = validation_data
        self.character_set = character_set
        self.cer_metric = Mean()
        self.wer_metric = Mean()

    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        predictions = self.model.predict(x_val)
        decoded_predictions = self.decode_predictions(predictions)
        decoded_true = self.decode_labels(y_val)
        
        cer = calculate_cer(decoded_true, decoded_predictions)
        wer = calculate_wer(decoded_true, decoded_predictions)
        
        self.cer_metric.update_state(cer)
        self.wer_metric.update_state(wer)
        
        logs['val_cer'] = self.cer_metric.result().numpy()
        logs['val_wer'] = self.wer_metric.result().numpy()
        
        logger.info(f"Epoch {epoch+1}: CER = {logs['val_cer']:.4f}, WER = {logs['val_wer']:.4f}")

    def decode_predictions(self, predictions):
        # Implement CTC decoding here
        pass

    def decode_labels(self, labels):
        # Convert one-hot encoded labels to text
        pass

def train_model(config):
    """Train the handwriting recognition model."""
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
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(
        config['paths']['best_model'],
        save_best_only=True,
        monitor='val_accuracy'
    )
    tensorboard = TensorBoard(log_dir=config['paths']['logs'])
    metrics_callback = MetricsCallback((x_val, y_val), config['model']['character_set'])
    
    callbacks = [reduce_lr, early_stopping, model_checkpoint, tensorboard, metrics_callback]
    
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

def retrain_model(new_data, config):
    """Retrain the model with new data (continuous learning)."""
    # Load the existing model
    model = tf.keras.models.load_model(config['paths']['model_save'], custom_objects={'ctc_loss': ctc_loss})
    
    # Preprocess new data
    x_new, y_new = preprocess_new_data(new_data)
    
    # Combine new data with a subset of existing data
    x_existing, y_existing = load_dataset(config['paths']['dataset'], config)
    x_combined = np.concatenate([x_existing, x_new])
    y_combined = np.concatenate([y_existing, y_new])
    
    # Retrain the model
    history = train_model(config)
    
    # Save the retrained model with a new version
    version = get_next_model_version(config['paths']['model_versions'])
    model.save(os.path.join(config['paths']['model_versions'], f'model_v{version}.h5'))
    
    return history

def get_next_model_version(versions_dir):
    """Get the next model version number."""
    existing_versions = [int(f.split('_v')[1].split('.')[0]) for f in os.listdir(versions_dir) if f.startswith('model_v')]
    return max(existing_versions) + 1 if existing_versions else 1

def preprocess_new_data(new_data):
    """Preprocess new data for retraining."""
    # Implement preprocessing for new data
    pass

def save_model_as_onnx(model, save_path="ml/models/versions/model_latest.onnx"):
    dummy_input = torch.randn(1, 1, 128, 512)  # Adjust the input shape for your model
    torch.onnx.export(model, dummy_input, save_path, export_params=True)
    print(f"Model saved as ONNX at {save_path}")
    
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

