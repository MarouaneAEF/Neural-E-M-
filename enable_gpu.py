"""
Helper module to enable Apple GPU acceleration for TensorFlow
"""
import tensorflow as tf
import os
import time

def enable_gpu():
    """Enable GPU acceleration for Apple Silicon"""
    # Set environment variables
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations which can conflict with MPS
    
    # Check for available GPUs
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        try:
            # Configure memory growth to avoid allocating all GPU memory at once
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU enabled: {physical_devices}")
            
            # Use TensorFlow's device placement logging to confirm operations run on GPU
            tf.debugging.set_log_device_placement(True)
            
            # Create a test tensor to verify GPU availability
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c = tf.matmul(a, b)
                print(f"Test tensor multiplication result shape: {c.shape}")
            
            print("GPU acceleration is active and working correctly")
            
            # Disable device placement logging for normal operation
            tf.debugging.set_log_device_placement(False)
            return True
        except RuntimeError as e:
            print(f"GPU acceleration error: {e}")
            print("Falling back to CPU")
            return False
    else:
        print("No GPU found, using CPU only")
        return False 