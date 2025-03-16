import tensorflow as tf
import os

# Print TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Print if TensorFlow was built with CUDA
print(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")

# List all available devices
print("\nAvailable devices:")
for device in tf.config.list_physical_devices():
    print(device)

# Check specifically for GPUs
print("\nAvailable GPUs:")
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print(gpu)

if not gpus:
    print("No GPU found. TensorFlow is running on CPU only.")
else:
    print(f"Found {len(gpus)} GPU(s). TensorFlow can use GPU acceleration.")

# Print memory info if GPU is available
if gpus:
    try:
        # Get GPU memory info
        print("\nGPU Memory Info:")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth enabled for {gpu}")
    except RuntimeError as e:
        print(f"GPU memory configuration error: {e}") 