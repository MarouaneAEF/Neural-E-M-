import os
import gc
os.nice(10)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.metrics import Mean
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time

# Thread limits must be set before any TF operation initializes the runtime
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"Found {len(physical_devices)} GPU(s)")
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        with tf.device('/GPU:0'):
            c = tf.matmul(tf.random.normal([1000, 1000]), tf.random.normal([1000, 1000]))
            print(f"GPU test successful: {c.shape}")
        print("GPU acceleration active")
    else:
        print("No GPU found, using CPU instead")
except Exception as e:
    print(f"Error configuring GPU: {e}")

from rnn_em_cell_bernoulli import rnn_em
from q_graph import Q_graph
from sequential_dataloader import get_dataset, generator, BATCH_SIZE, SEQUENCE_LENGTH
from util import bitflip_noisy_static, ami_score
from bernoulli_loss import em_loss

K = 3
LSTM_UNITS = 64   # must match Q_graph LSTM size

initial_lr = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=initial_lr,
    decay_steps=1000,
    decay_rate=0.9,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
inner_cell = Q_graph()
loss_fn = em_loss(initial_kl_weight=0.01, max_kl_weight=0.3, annealing_rate=0.001)
rnn_cell = rnn_em(inner_cell, input_shape=(28, 28, 1))

os.makedirs('./logs', exist_ok=True)
os.makedirs('./plots_dynamic', exist_ok=True)

checkpoint_dir = './ckpt/dynamic'
checkpoint = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64),
                                 ami=tf.Variable(-1e10),
                                 optimizer=optimizer,
                                 model=rnn_cell.model)
checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                directory=checkpoint_dir,
                                                max_to_keep=3)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = './logs/dynamic_' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)

train_data = get_dataset(generator, "training")
valid_data = get_dataset(generator, "validation")


def visualize_clusters(gamma, features, epoch, step):
    # gamma : (B, K, H, W, 1)   features : (B, T, H, W, 1)
    cluster_assignments = tf.argmax(gamma, axis=1)          # (B, H, W, 1)
    sample_img        = features[0, 0, :, :, 0].numpy()    # first frame, first seq
    sample_assignment = cluster_assignments[0, :, :, 0].numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(sample_img, cmap='gray')
    ax1.set_title('Frame 0 — Original')
    ax1.axis('off')
    cmap = plt.cm.get_cmap('viridis', K)
    ax2.imshow(sample_assignment, cmap=cmap, vmin=0, vmax=K - 1)
    ax2.set_title('Cluster Assignments')
    ax2.axis('off')
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax2)
    plt.tight_layout()
    plt.savefig(f'./plots_dynamic/clusters_epoch{epoch}_step{step}.png')
    plt.close()


N_EM_ITERATIONS = 15

# Hidden state is passed as explicit TF tensors with fixed shapes so that
# @tf.function traces once and never retraces across frames or sequences.
# Shapes: lstm_h/c = (B*K, LSTM_UNITS), preds/gamma = (B, K, 28, 28, 1)
@tf.function(input_signature=[
    tf.TensorSpec(shape=(BATCH_SIZE, 1, 28, 28, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(BATCH_SIZE * K, LSTM_UNITS), dtype=tf.float32),
    tf.TensorSpec(shape=(BATCH_SIZE * K, LSTM_UNITS), dtype=tf.float32),
    tf.TensorSpec(shape=(BATCH_SIZE, K, 28, 28, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(BATCH_SIZE, K, 28, 28, 1), dtype=tf.float32),
])
def train_step_frame(frame, lstm_h, lstm_c, preds, gamma):
    """One gradient step on a single frame; LSTM state passed explicitly."""
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    device = '/GPU:0' if gpu_available else '/CPU:0'

    with tf.device(device):
        frame_noisy = bitflip_noisy_static(frame)
        hidden_state = ([lstm_h, lstm_c], preds, gamma)
        current_lr = lr_schedule(optimizer.iterations)

        for _ in range(N_EM_ITERATIONS):
            with tf.GradientTape() as tape:
                hidden_state = rnn_cell((frame_noisy, frame), hidden_state)
                rnn_state_new, preds_new, gamma_new = hidden_state
                loss = loss_fn(preds_new, frame, gamma_new)
            grads = tape.gradient(loss, rnn_cell.model.trainable_weights)
            grads = [tf.clip_by_norm(g, 3.0) if g is not None else g for g in grads]
            optimizer.apply_gradients(zip(grads, rnn_cell.model.trainable_weights))

    # Return updated state components so Python can thread them to the next frame
    return loss, rnn_state_new[0], rnn_state_new[1], preds_new, gamma_new, current_lr


def make_initial_state():
    """Return zero LSTM state + random initial preds/gamma for a new sequence."""
    lstm_h = tf.zeros([BATCH_SIZE * K, LSTM_UNITS])
    lstm_c = tf.zeros([BATCH_SIZE * K, LSTM_UNITS])
    _, preds, gamma = rnn_cell.initial_state(BATCH_SIZE, K)
    return lstm_h, lstm_c, preds, gamma


def validation(dataset):
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    device = '/GPU:0' if gpu_available else '/CPU:0'
    ami_values = []
    with tf.device(device):
        for features, groups in dataset:
            lstm_h, lstm_c, preds, gamma = make_initial_state()
            for t in range(SEQUENCE_LENGTH):
                frame_t = features[:, t:t + 1, :, :, :]
                frame_noisy = bitflip_noisy_static(frame_t)
                hidden_state = ([lstm_h, lstm_c], preds, gamma)
                hidden_state = rnn_cell((frame_noisy, frame_t), hidden_state)
                (lstm_h, lstm_c), preds, gamma = hidden_state[0], hidden_state[1], hidden_state[2]
            ami_val = ami_score(gamma, groups[:, -1:, :, :, :])
            ami_values.append(ami_val)
    return tf.reduce_mean(ami_values)


# Training loop
n_iterations = 100
best_ami = -1
patience = 0
max_patience = 5
validation_frequency = 50
visualization_frequency = 200
validation_samples = 15

for epoch in range(n_iterations):
    train_ami_mean = Mean()
    train_loss_mean = Mean()
    now = time.perf_counter()

    for step, (features, groups) in enumerate(train_data):
        checkpoint.step.assign_add(1)

        # Reset state at the start of each new sequence
        lstm_h, lstm_c, preds, gamma = make_initial_state()

        # Temporal loop: hidden state flows from frame t to frame t+1
        for t in range(SEQUENCE_LENGTH):
            frame_t = features[:, t:t + 1, :, :, :]
            loss, lstm_h, lstm_c, preds, gamma, current_lr = train_step_frame(
                frame_t, lstm_h, lstm_c, preds, gamma
            )

        # Metrics evaluated on the last frame
        train_loss_mean(loss)
        ami_train = ami_score(gamma, groups[:, -1:, :, :, :])
        train_ami_mean(ami_train)

        if step % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar('train_loss', loss, step=checkpoint.step)
                tf.summary.scalar('train_ami', ami_train, step=checkpoint.step)
                tf.summary.scalar('learning_rate', current_lr, step=checkpoint.step)

        if step % validation_frequency == 0:
            print(f"Epoch: {epoch + 1} at Step: {step + 1}:")
            tloss = train_loss_mean.result()
            tami_score = train_ami_mean.result()
            train_loss_mean.reset_state()
            train_ami_mean.reset_state()

            vami_score = validation(valid_data.take(validation_samples))
            gc.collect()

            with summary_writer.as_default():
                tf.summary.scalar('validation_ami', vami_score, step=checkpoint.step)

            duration = time.perf_counter() - now
            print(f"loss={tloss.numpy():.4f} | t_ami={tami_score:.4f} | v_ami={vami_score.numpy():.4f} | lr={current_lr:.6f} | duration={duration:.2f}s")

            if step % visualization_frequency == 0:
                try:
                    visualize_clusters(gamma, features, epoch, step)
                except Exception as e:
                    print(f"Error visualizing clusters: {e}")

            if vami_score.numpy() > checkpoint.ami:
                checkpoint.ami = vami_score
                checkpoint_manager.save()
                patience = 0
                if vami_score.numpy() > best_ami:
                    best_ami = vami_score.numpy()
                    print(f"New best AMI: {best_ami:.4f}")
            else:
                patience += 1

            now = time.perf_counter()

    if patience >= max_patience:
        print(f"Early stopping triggered after {epoch + 1} epochs")
        break

print(f"Training completed. Best AMI score: {best_ami:.4f}")
