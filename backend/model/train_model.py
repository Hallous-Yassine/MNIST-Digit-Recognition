"""
train_model.py (Enhanced Version - Fixed with Data Validation)
---------------------------------------------------------------
Enhanced CNN for MNIST with data augmentation, improved architecture,
and better training strategies for higher accuracy.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, 
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys

# ==============================================================
# 🔧 Enhanced Configuration
# ==============================================================
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, 'mnist_cnn_enhanced.h5')
EPOCHS = 40
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1

# ==============================================================
# 📥 Load & Preprocess Data with Validation
# ==============================================================
def load_data(data_dir):
    """Load pre-saved MNIST data from .npy files with validation."""
    try:
        x_train = np.load(os.path.join(data_dir, 'x_train.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        x_test = np.load(os.path.join(data_dir, 'x_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        # Validate data
        print(f"📊 Dataset loaded:")
        print(f"   Training: {x_train.shape}, Labels: {y_train.shape}")
        print(f"   Test: {x_test.shape}, Labels: {y_test.shape}")
        print(f"   Label range: {y_train.min()} to {y_train.max()}")
        print(f"   Unique labels: {np.unique(y_train)}")
        
        # Check if labels are valid for MNIST (0-9)
        if y_train.max() > 9 or y_train.min() < 0:
            print(f"\n⚠️  WARNING: Labels outside 0-9 range detected!")
            print(f"   This appears to be EMNIST or corrupted data.")
            print(f"   Please re-run download_dataset.py to get clean MNIST data.\n")
            
            response = input("Continue anyway with filtered data? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
            
            # Filter to only digits 0-9
            train_mask = (y_train >= 0) & (y_train <= 9)
            test_mask = (y_test >= 0) & (y_test <= 9)
            
            x_train = x_train[train_mask]
            y_train = y_train[train_mask]
            x_test = x_test[test_mask]
            y_test = y_test[test_mask]
            
            print(f"✅ Filtered to digits 0-9:")
            print(f"   Training samples: {len(x_train)}")
            print(f"   Test samples: {len(x_test)}\n")
        
        return (x_train, y_train), (x_test, y_test)
        
    except FileNotFoundError:
        print(f"❌ ERROR: Data files not found in {data_dir}")
        print("Please run download_dataset.py first.")
        sys.exit(1)


def preprocess_data(x, y):
    """Normalize images and one-hot encode labels."""
    x = x.astype('float32') / 255.0
    
    # Ensure images have channel dimension
    if len(x.shape) == 3:
        x = np.expand_dims(x, -1)
    
    # Validate labels before one-hot encoding
    if y.max() >= 10 or y.min() < 0:
        raise ValueError(f"Labels must be 0-9, found range {y.min()}-{y.max()}")
    
    y = to_categorical(y, 10)
    return x, y


# ==============================================================
# 🎨 Data Augmentation
# ==============================================================
def create_data_generator():
    """Create ImageDataGenerator with augmentation for training."""
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1
    )


# ==============================================================
# 🧠 Enhanced CNN Model
# ==============================================================
def build_enhanced_cnn():
    """
    Enhanced CNN architecture with:
    - Batch Normalization for stable training
    - Deeper network for better feature extraction
    - Global Average Pooling to reduce parameters
    """
    model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Classification head
        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # Output
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ==============================================================
# 🚀 Train & Evaluate
# ==============================================================
def train_and_evaluate():
    # Load and preprocess with validation
    (x_train, y_train), (x_test, y_test) = load_data(DATA_DIR)
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # Split validation from training
    val_size = int(len(x_train) * VALIDATION_SPLIT)
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:]

    print(f"\n📊 Final dataset split:")
    print(f"   Training: {len(x_train)} samples")
    print(f"   Validation: {len(x_val)} samples")
    print(f"   Test: {len(x_test)} samples\n")

    # Create model
    model = build_enhanced_cnn()
    model.summary()

    # Data augmentation
    datagen = create_data_generator()
    datagen.fit(x_train)

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy', 
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH, 
            monitor='val_accuracy', 
            save_best_only=True, 
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.5, 
            patience=4,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print("\n🚀 Starting enhanced training with data augmentation...\n")
    
    # Train with augmented data
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        verbose=1
    )

    # Reload best model
    try:
        best_model = tf.keras.models.load_model(MODEL_PATH)
        print(f"\n✅ Successfully reloaded best model from: {MODEL_PATH}")
    except Exception as e:
        print(f"\n❌ ERROR: Could not reload best model: {e}")
        return

    # Final evaluation
    test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
    print(f"\n" + "="*60)
    print(f"🎯 FINAL RESULTS")
    print(f"="*60)
    print(f"✨ Test Accuracy: {test_acc * 100:.2f}%")
    print(f"📊 Test Loss: {test_loss:.4f}")
    
    if 'val_accuracy' in history.history:
        best_val_acc = max(history.history['val_accuracy'])
        print(f"🏆 Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    
    print(f"💾 Model saved at: {MODEL_PATH}")
    print(f"="*60 + "\n")


# ==============================================================
# 🏁 Main
# ==============================================================
if __name__ == "__main__":
    train_and_evaluate()