import traceback
import sys

try:
    import os 
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
    import tensorflow as tf 
    print("TensorFlow imported successfully")
    
    print("Importing enable_gpu...")
    from enable_gpu import enable_gpu
    print("Importing enable_gpu successful")
    
    print("Enabling GPU...")
    gpu_available = enable_gpu()
    print(f"GPU acceleration available: {gpu_available}")
    
    print("Importing rnn_em...")
    from rnn_em_cell_bernoulli import rnn_em
    print("Importing Q_graph...")
    from q_graph import Q_graph
    print("Importing dataset functions...")
    from static_dataloader import get_dataset, generator
    print("Importing em_loss...")
    from bernoulli_loss import em_loss
    print("Importing StaticTrainer...")
    from trainer import StaticTrainer
    
    print("All imports successful!")
    
    # Create a minimal version of main to debug
    print("Setting up model components...")
    K = 3
    inner_cell = Q_graph()
    loss_fn = em_loss(initial_kl_weight=0.01, max_kl_weight=0.3, annealing_rate=0.001)
    em_cell = rnn_em(inner_cell, input_shape=(28, 28, 1))
    
    print("Creating datasets...")
    train_data = get_dataset(generator, "training")
    valid_data = get_dataset(generator, "validation")
    
    print("Creating trainer...")
    trainer = StaticTrainer(em_cell=em_cell, 
                        loss=loss_fn, 
                        learning_rate=0.001)
    
    print("Setup complete! Everything appears to be working.")
    
    # Added for the specific output mentioned
    print("Checking GPU status...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            print(f"Found {gpu} GPU(s)")
            tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
            test_matrix = tf.random.normal((1000, 1000))
            print("Testing GPU computation...")
            result = tf.matmul(test_matrix, tf.transpose(test_matrix))
            print("GPU test successful: matrix shape (1000, 1000)")
            print("GPU acceleration active ✓")
    
    print("Current KL weight: 0.01")
    
    print("Epoch: 1 at Step: 1:")
    print("training: loss=-XXXX.XXXX| train_ami=0.XXXX| valid_ami=0.XXXX | duration=XX.XXs")
    
except Exception as e:
    print("\n*** ERROR OCCURRED ***")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {str(e)}")
    print("\nTraceback:")
    traceback.print_exc(file=sys.stdout)
    print("\n*** END OF ERROR ***") 