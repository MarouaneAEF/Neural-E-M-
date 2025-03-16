import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf 
import time
import matplotlib.pyplot as plt
import datetime

# Import and enable GPU acceleration
from enable_gpu import enable_gpu
gpu_available = enable_gpu()
print(f"GPU acceleration available: {gpu_available}")

from rnn_em_cell_bernoulli import rnn_em
from q_graph import Q_graph
from static_dataloader import get_dataset, generator
from bernoulli_loss import em_loss
from trainer import StaticTrainer

# Create directories for logs and visualizations
os.makedirs('./logs', exist_ok=True)
os.makedirs('./plots', exist_ok=True)

# Set up TensorBoard logging
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = './logs/main_' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

K = 3 
# Use a learning rate schedule instead of a fixed rate
initial_lr = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True)

inner_cell = Q_graph()
# Use KL annealing in the loss function
loss_fn = em_loss(initial_kl_weight=0.01, max_kl_weight=0.3, annealing_rate=0.001)
em_cell = rnn_em(inner_cell, input_shape=(28, 28, 1))


train_data = get_dataset(generator, "training")
valid_data = get_dataset(generator, "validation")

# Pass the learning rate schedule to the trainer
trainer = StaticTrainer(em_cell=em_cell, 
                        loss=loss_fn, 
                        learning_rate=lr_schedule)
n_epochs = 120

# Add early stopping
patience = 0
max_patience = 10
best_ami = -1

start = time.perf_counter()

for epoch in range(n_epochs):
    # Check for early stopping
    if patience >= max_patience:
        print(f"Early stopping triggered after {epoch} epochs without improvement")
        break
        
    # Get the current AMI score before training
    prev_ami = trainer.checkpoint.ami.numpy()
    
    # Train for one epoch
    trainer.train(train_data, valid_data, epoch)
    
    # Check if AMI improved
    current_ami = trainer.checkpoint.ami.numpy()
    
    # Log AMI to TensorBoard
    with summary_writer.as_default():
        tf.summary.scalar('validation_ami', current_ami, step=epoch)
    
    if current_ami <= prev_ami:
        patience += 1
        print(f"No improvement in AMI. Patience: {patience}/{max_patience}")
    else:
        patience = 0
        # Update best AMI
        if current_ami > best_ami:
            best_ami = current_ami
            print(f"New best AMI: {best_ami:.4f}")

duration = time.perf_counter() - start

print(f"Training duration: {duration:.2f}s")
print(f"Best AMI score: {best_ami:.4f}")

# Create a final visualization of the clustering results
test_features, test_groups = next(iter(valid_data))
hidden_state = em_cell.initial_state(test_features.shape[0], K)
inputs = (test_features, test_features)  # No corruption for visualization
hidden_state = em_cell(inputs, hidden_state)
_, _, gamma = hidden_state

# Get the most probable cluster assignment for each pixel
cluster_assignments = tf.argmax(gamma, axis=1)

# Visualize a few examples
num_examples = min(5, test_features.shape[0])
fig, axes = plt.subplots(2, num_examples, figsize=(3*num_examples, 6))

for i in range(num_examples):
    # Original image
    axes[0, i].imshow(test_features[i, :, :, 0], cmap='gray')
    axes[0, i].set_title(f"Original {i+1}")
    axes[0, i].axis('off')
    
    # Cluster assignments
    cmap = plt.cm.get_cmap('viridis', K)
    cluster_img = axes[1, i].imshow(cluster_assignments[i, :, :, 0], cmap=cmap, vmin=0, vmax=K-1)
    axes[1, i].set_title(f"Clusters {i+1}")
    axes[1, i].axis('off')

# Add a colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(cluster_img, cax=cbar_ax)
cbar.set_ticks(range(K))
cbar.set_ticklabels([f'Cluster {i+1}' for i in range(K)])

plt.tight_layout()
plt.savefig('./plots/final_clusters.png')
plt.close()

print(f"Final visualization saved to ./plots/final_clusters.png")