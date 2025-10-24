"""
download_dataset.py
-------------------
Download clean MNIST dataset (digits 0-9 only)
Creates data/ directory with .npy files
"""

import os
import numpy as np
import tensorflow as tf

# Setup paths
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)

print("📥 Downloading MNIST dataset...")
print("="*60)

# Load MNIST from Keras
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Validate data
print(f"✅ Download complete!")
print(f"\n📊 Dataset Information:")
print(f"   Training images: {x_train.shape}")
print(f"   Training labels: {y_train.shape}")
print(f"   Test images: {x_test.shape}")
print(f"   Test labels: {y_test.shape}")
print(f"\n   Label range: {y_train.min()} to {y_train.max()}")
print(f"   Unique labels: {sorted(np.unique(y_train).tolist())}")

# Verify it's MNIST (should only have 0-9)
if y_train.max() > 9 or y_test.max() > 9:
    print("\n⚠️  WARNING: Unexpected labels detected!")
    print("   Expected: 0-9 (MNIST digits)")
    print(f"   Found: {np.unique(y_train)}")
else:
    print("   ✓ Valid MNIST dataset (digits 0-9)")

# Save to .npy files
np.save(os.path.join(data_dir, 'x_train.npy'), x_train)
np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
np.save(os.path.join(data_dir, 'x_test.npy'), x_test)
np.save(os.path.join(data_dir, 'y_test.npy'), y_test)

print(f"\n💾 Data saved to: {data_dir}/")
print(f"   - x_train.npy ({x_train.nbytes / 1024 / 1024:.1f} MB)")
print(f"   - y_train.npy ({y_train.nbytes / 1024:.1f} KB)")
print(f"   - x_test.npy ({x_test.nbytes / 1024 / 1024:.1f} MB)")
print(f"   - y_test.npy ({y_test.nbytes / 1024:.1f} KB)")

print("\n" + "="*60)
print("✅ MNIST dataset ready for training!")
print("   Run: python model/train_model.py")
print("="*60)