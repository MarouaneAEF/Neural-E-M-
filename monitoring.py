"""
Helper module for monitoring Neural EM convergence and visualizing results
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from matplotlib.animation import FuncAnimation

def visualize_clusters(gamma, features, output_dir='./plots', name=None):
    """
    Visualize cluster assignments for a batch of data
    
    Args:
        gamma: Cluster responsibility matrix [batch_size, K, H, W, 1]
        features: Input images [batch_size, H, W, 1]
        output_dir: Directory to save visualizations
        name: Optional filename suffix
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the number of clusters
    K = gamma.shape[1]
    
    # Get the most probable cluster assignment for each pixel
    cluster_assignments = tf.argmax(gamma, axis=1)  # shape: [batch_size, H, W, 1]
    
    # Take the first few images in the batch for visualization
    num_samples = min(3, features.shape[0])
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    for i in range(num_samples):
        # Original image
        if num_samples == 1:
            ax1 = axes[0]
            ax2 = axes[1]
        else:
            ax1 = axes[0, i]
            ax2 = axes[1, i]
            
        ax1.imshow(features[i, :, :, 0], cmap='gray')
        ax1.set_title(f'Original {i+1}')
        ax1.axis('off')
        
        # Cluster assignments
        cmap = plt.cm.get_cmap('viridis', K)
        cluster_img = ax2.imshow(cluster_assignments[i, :, :, 0], cmap=cmap, vmin=0, vmax=K-1)
        ax2.set_title(f'Clusters {i+1}')
        ax2.axis('off')
    
    # Add color bar
    if num_samples > 1:
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        cbar = fig.colorbar(cluster_img, cax=cbar_ax)
        cbar.set_ticks(range(K))
        cbar.set_ticklabels([f'C{i}' for i in range(K)])
    
    plt.tight_layout()
    
    # Save figure
    timestamp = int(time.time())
    filename = f'clusters_{timestamp}_{name}.png' if name else f'clusters_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()
    
    return filepath

def visualize_cluster_means(predictions, K, output_dir='./plots', name=None):
    """
    Visualize the learned cluster means (Bernoulli parameters)
    
    Args:
        predictions: Cluster means/Bernoulli parameters [batch_size, K, H, W, 1]
        K: Number of clusters
        output_dir: Directory to save visualizations
        name: Optional filename suffix
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Take one example from the batch
    preds = predictions[0]  # shape: [K, H, W, 1]
    
    # Create figure
    fig, axes = plt.subplots(1, K, figsize=(4*K, 4))
    
    # Display each cluster's mean
    for k in range(K):
        if K == 1:
            ax = axes
        else:
            ax = axes[k]
            
        ax.imshow(preds[k, :, :, 0], cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'Cluster {k+1} Prototype')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    timestamp = int(time.time())
    filename = f'cluster_means_{timestamp}_{name}.png' if name else f'cluster_means_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150)
    plt.close()
    
    return filepath

def create_cluster_evolution_animation(cluster_images, output_path='./plots/cluster_evolution.gif'):
    """
    Create an animation showing how cluster assignments evolve over time
    
    Args:
        cluster_images: List of (timestep, image array) tuples
        output_path: Path to save the animation
    """
    # Sort by timestep
    cluster_images.sort(key=lambda x: x[0])
    
    # Extract just the images
    images = [img for _, img in cluster_images]
    
    if not images:
        print("No images to animate")
        return
        
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # First frame
    im = ax.imshow(images[0], animated=True)
    ax.axis('off')
    
    def update(frame):
        im.set_array(images[frame])
        ax.set_title(f'Iteration {frame+1}/{len(images)}')
        return [im]
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(images), interval=200, blit=True)
    
    # Save as GIF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani.save(output_path, writer='pillow', fps=4)
    plt.close()
    
    return output_path

def plot_loss_curves(train_losses, val_ami_scores, output_dir='./plots'):
    """Plot training loss and validation AMI scores over time"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Plot AMI scores
    plt.subplot(1, 2, 2)
    eval_steps = np.arange(0, len(train_losses), len(train_losses)//len(val_ami_scores))[:len(val_ami_scores)]
    plt.plot(eval_steps, val_ami_scores)
    plt.title('Validation AMI Score')
    plt.xlabel('Steps')
    plt.ylabel('AMI Score')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    
    return filepath 