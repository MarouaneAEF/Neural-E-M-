import os
import gc
os.nice(10)  # Lower process priority: GPU gets full power, OS stays responsive
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.metrics import Mean
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

# Thread limits must be set before any TF operation initializes the runtime
tf.config.threading.set_inter_op_parallelism_threads(2)   # orchestration threads
tf.config.threading.set_intra_op_parallelism_threads(4)   # within-op threads (4 P-cores)

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
        
        print("GPU acceleration active ")
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

K = 3 
# Set up learning rate schedule instead of constant learning rate
initial_lr = 0.001
lr = initial_lr  # Keep a reference to the initial value for fallback

# Learning rate schedule: start high and decrease over time
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True)

# Use the learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
max_epoch = 100 
inner_cell = Q_graph()
loss_fn = em_loss(initial_kl_weight=0.01, max_kl_weight=0.3, annealing_rate=0.001)
rnn_cell = rnn_em(inner_cell, input_shape=(28, 28, 1))

# Create folders for monitoring
os.makedirs('./logs', exist_ok=True)
os.makedirs('./plots', exist_ok=True)

# setting checkpoint
checkpoint_dir = './ckpt/static'
checkpoint = tf.train.Checkpoint(step = tf.Variable(0, dtype=tf.int64),
                                 ami = tf.Variable(-1e10),
                                 optimizer=optimizer,
                                 model=rnn_cell.model)

checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                directory=checkpoint_dir,
                                                max_to_keep=3)

# Set up TensorBoard logging
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = './logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

train_data = get_dataset(generator, "training")
valid_data = get_dataset(generator, "validation")

# Function to visualize cluster assignments
def visualize_clusters(gamma, features, epoch, step):
    """Visualize cluster assignments and save to file"""
    # Get the most probable cluster assignment for each pixel
    cluster_assignments = tf.argmax(gamma, axis=1)  # shape: [batch_size, H, W, 1]
    
    # Take the first image in the batch for visualization
    sample_img = features[0, 0, :, :, 0].numpy()
    sample_assignment = cluster_assignments[0, :, :, 0].numpy()
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot the original image
    ax1.imshow(sample_img, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    # Plot the cluster assignments with different colors for each cluster
    cmap = plt.cm.get_cmap('viridis', K)
    ax2.imshow(sample_assignment, cmap=cmap, vmin=0, vmax=K-1)
    ax2.set_title('Cluster Assignments')
    ax2.axis('off')
    
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax2)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'./plots_pe/clusters_epoch{epoch}_step{step}.png')
    plt.close()

# Fixed number of EM iterations: avoids Python-level retracing from variable n_iterations
N_EM_ITERATIONS = 15

@tf.function(input_signature=[
    tf.TensorSpec(shape=(BATCH_SIZE, 1, 28, 28, 1), dtype=tf.float32)
])
def train_step(features):
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    device = '/GPU:0' if gpu_available else '/CPU:0'

    with tf.device(device):
        features_corrupted = bitflip_noisy_static(features)
        hidden_state = rnn_cell.initial_state(BATCH_SIZE, K)
        current_lr = lr_schedule(optimizer.iterations)

        for i in range(N_EM_ITERATIONS):
            # Non-persistent tape: resources freed automatically after one .gradient() call
            with tf.GradientTape() as tape:
                inputs = (features_corrupted, features)
                hidden_state = rnn_cell(inputs, hidden_state)
                rnn_state, preds, gamma = hidden_state
                loss_rnn_em = loss_fn(preds, features, gamma)
            gradients = tape.gradient(loss_rnn_em, rnn_cell.model.trainable_weights)
            gradients = [tf.clip_by_norm(g, 3.0) if g is not None else g for g in gradients]
            optimizer.apply_gradients(zip(gradients, rnn_cell.model.trainable_weights))

    return loss_rnn_em, gamma, current_lr


def validation(dataset):
    # Explicitly place operations on GPU if available
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    device = '/GPU:0' if gpu_available else '/CPU:0'
    
    ami_values = []
    with tf.device(device):
        for features, groups in dataset:
            # Reset hidden state per batch to avoid state leakage across batches
            hidden_state = rnn_cell.initial_state(BATCH_SIZE, K)
            features_corrupted = bitflip_noisy_static(features)
            inputs = (features_corrupted, features)
            hidden_state = rnn_cell(inputs, hidden_state)
            _, _, gamma = hidden_state
            ami_valid = ami_score(gamma, groups)
            ami_values.append(ami_valid)

    return tf.reduce_mean(ami_values)
        

# # # # # # # # # # # #
#   Trainning loop    #
# # # # # # # # # # # #
n_iterations = 100

# Initialize variables to track best performance
best_ami = -1
patience = 0
max_patience = 5  # Early stopping after 5 epochs without improvement

# Reduce validation frequency
validation_frequency = 50  # Changed from 25 to 50
visualization_frequency = 200  # Changed from 100 to 200
# Take fewer samples for validation to speed up
validation_samples = 15  # Changed from 30 to 15

for epoch in range(n_iterations):
    train_ami_mean = Mean()
    train_loss_mean = Mean()
    now = time.perf_counter()
    for step, (features, groups) in enumerate(train_data):
        checkpoint.step.assign_add(1)
        
        loss_rnn_em, gamma, current_lr = train_step(features)
        train_loss_mean(loss_rnn_em)
        ami_train = ami_score(gamma, groups)
        train_ami_mean(ami_train)         
        
        # Log metrics to TensorBoard less frequently
        if step % 10 == 0:  # Only log every 10 steps
            with summary_writer.as_default():
                tf.summary.scalar('train_loss', loss_rnn_em, step=checkpoint.step)
                tf.summary.scalar('train_ami', ami_train, step=checkpoint.step)
                tf.summary.scalar('learning_rate', current_lr, step=checkpoint.step)
        
        if step % validation_frequency == 0:
            print(f"Epoch: {epoch + 1} at Step: {step + 1}:")
            tloss = train_loss_mean.result()
            tami_score = train_ami_mean.result()
            train_loss_mean.reset_state() 
            train_ami_mean.reset_state()
            
            # Less frequent validation
            vami_score = validation(valid_data.take(validation_samples))
            gc.collect()  # Free CPU tensors accumulated during validation

            # Log validation metrics
            with summary_writer.as_default():
                tf.summary.scalar('validation_ami', vami_score, step=checkpoint.step)
            
            duration = time.perf_counter() - now
            train_string = f"training for one batch : loss={tloss.numpy():.4f}| t_ami_score={tami_score:.4f} | v_ami_score={vami_score.numpy():.4f} | lr={current_lr:.6f} | duration={duration:.2f}s"
            print(train_string)
            
            # Visualize clusters occasionally with reduced frequency
            if step % visualization_frequency == 0:
                try:
                    visualize_clusters(gamma, features, epoch, step)
                except Exception as e:
                    print(f"Error visualizing clusters: {e}")
            
            # Check if we should save the model
            if vami_score.numpy() > checkpoint.ami:
                checkpoint.ami = vami_score
                checkpoint_manager.save()
                # Reset patience when we improve
                patience = 0
                
                # Update best AMI
                if vami_score.numpy() > best_ami:
                    best_ami = vami_score.numpy()
                    print(f"New best AMI: {best_ami:.4f}")
            else:
                # Increment patience when we don't improve
                patience += 1
                
            now = time.perf_counter()
    
    # Early stopping check
    if patience >= max_patience:
        print(f"Early stopping triggered after {epoch + 1} epochs without improvement")
        break
        
print(f"Training completed. Best AMI score: {best_ami:.4f}")
