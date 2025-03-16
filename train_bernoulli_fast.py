import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow.keras.metrics import Mean
import numpy as np
import matplotlib.pyplot as plt 
import datetime
import time

# Configure TensorFlow for Apple M3 GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
try:
    # More detailed GPU configuration
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"Found {len(physical_devices)} GPU(s)")
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled")
        
        # Test GPU with a simple operation
        with tf.device('/GPU:0'):
            print("Testing GPU computation...")
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            print(f"GPU test successful: matrix shape {c.shape}")
        
        print("GPU acceleration active ✓")
    else:
        print("No GPU found, using CPU instead")
except Exception as e:
    print(f"Error configuring GPU: {e}")
    print("Falling back to CPU")

from rnn_em_cell_bernoulli import rnn_em
from q_graph import Q_graph
from static_dataloader import get_dataset, generator, BATCH_SIZE
from util import bitflip_noisy_static, ami_score
from bernoulli_loss import em_loss

# FAST DEV VERSION SETTINGS
print("🚀 Running FAST development version with minimal settings")
K = 3 
# Simple fixed learning rate for fast dev
lr = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
max_epoch = 5  # Reduced epochs
inner_cell = Q_graph()
loss_fn = em_loss(initial_kl_weight=0.05, max_kl_weight=0.2, annealing_rate=0.01)  # Faster annealing
rnn_cell = rnn_em(inner_cell, input_shape=(28, 28, 1))

# Create folders for monitoring
os.makedirs('./logs', exist_ok=True)
os.makedirs('./plots', exist_ok=True)

# Super minimal checkpoint setup
checkpoint_dir = './ckpt/static_fast'
checkpoint = tf.train.Checkpoint(step = tf.Variable(0),
                                 ami = tf.Variable(-1e10),
                                 optimizer=optimizer,
                                 model=rnn_cell.model)

checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                directory=checkpoint_dir,
                                                max_to_keep=1)

# Very minimal TensorBoard logging 
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = './logs/fast_' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

print("Loading dataset...")
# Take only 20% of the data for fast development
train_data = get_dataset(generator, "training").take(50)
valid_data = get_dataset(generator, "validation").take(10)
print("Dataset loaded!")

# Very minimalist visualization
def visualize_clusters(gamma, features):
    """Quick visualization of current clusters"""
    cluster_assignments = tf.argmax(gamma, axis=1)
    sample_img = features[0, :, :, 0].numpy()
    sample_assignment = cluster_assignments[0, :, :, 0].numpy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(sample_img, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    cmap = plt.cm.get_cmap('viridis', K)
    ax2.imshow(sample_assignment, cmap=cmap, vmin=0, vmax=K-1)
    ax2.set_title('Cluster Assignments')
    ax2.axis('off')
    
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax2)
    plt.tight_layout()
    plt.savefig(f'./plots/clusters_fast.png')
    plt.close()
    print("✅ Visualization saved to ./plots/clusters_fast.png")

@tf.function
def train_step(features, n_iterations=10):  # Extremely reduced iterations
    # Explicitly place operations on GPU if available
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    device = '/GPU:0' if gpu_available else '/CPU:0'
    
    with tf.device(device):
        features_corrupted = bitflip_noisy_static(features)
        hidden_state = rnn_cell.initial_state(BATCH_SIZE, K)
        
        for i in range(n_iterations):
            with tf.GradientTape(persistent=True) as tape:
                inputs = (features_corrupted, features) 
                # E-step : computing gammas
                hidden_state  = rnn_cell(inputs, hidden_state)
                rnn_state, preds, gamma = hidden_state
                loss_rnn_em  = loss_fn(preds, features, gamma) 
            # M-step : maximizing EM loss interpolated with kl loss 
            gradients = tape.gradient(loss_rnn_em, rnn_cell.model.trainable_weights)
            # Apply gradient clipping to prevent exploding gradients
            gradients = [tf.clip_by_norm(g, 3.0) if g is not None else g for g in gradients]
            optimizer.apply_gradients(zip(gradients, rnn_cell.model.trainable_weights))
    
    return loss_rnn_em, gamma

def validation(dataset):
    # Very minimal validation
    ami_values = []
    hidden_state = rnn_cell.initial_state(BATCH_SIZE, K)
    
    # Just take 5 batches for validation
    for features, groups in dataset.take(5):
        features_corrupted = bitflip_noisy_static(features)
        inputs = (features_corrupted, features)
        hidden_state = rnn_cell(inputs, hidden_state)
        _, _, gamma = hidden_state
        ami_valid = ami_score(gamma, groups)
        ami_values.append(ami_valid)

    return tf.reduce_mean(ami_values)

# # # # # # # # # # # #
#  FAST Training loop #
# # # # # # # # # # # #
print("⏳ Starting fast training...")
n_iterations = 3  # Minimal number of epochs

for epoch in range(n_iterations):
    train_loss_mean = Mean()
    now = time.perf_counter()
    
    print(f"Epoch {epoch+1}/{n_iterations}")
    
    for step, (features, groups) in enumerate(train_data):
        checkpoint.step.assign_add(1)
        loss_rnn_em, gamma = train_step(features)  
        train_loss_mean(loss_rnn_em)
        
        # Print progress every 10 steps
        if step % 10 == 0:
            print(f"  Step {step+1}: loss={loss_rnn_em:.4f}")
        
        # Only validate once per epoch
        if step == 0 and epoch > 0:
            # Quick validation
            vami_score = validation(valid_data)
            duration = time.perf_counter() - now
            train_string = f"Validation AMI: {vami_score.numpy():.4f} | Duration: {duration:.2f}s"
            print(train_string)
            
            # Save visualization
            visualize_clusters(gamma, features)
            
            # Reset timer
            now = time.perf_counter()
    
    # End of epoch summary
    epoch_loss = train_loss_mean.result()
    vami_score = validation(valid_data)
    print(f"Epoch {epoch+1} summary - Loss: {epoch_loss:.4f}, AMI: {vami_score.numpy():.4f}")
    
    # Save model at the end of each epoch
    checkpoint_manager.save()
    print(f"Model saved at step {checkpoint.step.numpy()}")

print("🎉 Fast training completed!")
# Final visualization
for features, groups in valid_data.take(1):
    hidden_state = rnn_cell.initial_state(BATCH_SIZE, K)
    inputs = (features, features)  # No corruption
    hidden_state = rnn_cell(inputs, hidden_state)
    _, _, gamma = hidden_state
    visualize_clusters(gamma, features)
    break

print("✨ All done! Check ./plots/clusters_fast.png for visualization.") 