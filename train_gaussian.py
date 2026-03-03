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

# Thread limits before TF runtime initialisation
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(4)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        with tf.device('/GPU:0'):
            _ = tf.matmul(tf.random.normal([100, 100]), tf.random.normal([100, 100]))
        print(f"GPU acceleration active: {physical_devices[0].name}")
    else:
        print("No GPU found, using CPU")
except Exception as e:
    print(f"GPU config error: {e}")

from rnn_em_cell_gaussian import rnn_em_gaussian
from q_graph import Q_graph
from static_dataloader import get_dataset, generator, BATCH_SIZE
from util import bitflip_noisy_static, ami_score
from gaussian_loss import gaussian_em_loss

# ------------------------------------------------------------------
# Hyperparameters
# ------------------------------------------------------------------
K          = 3
IMAGE_SIZE = 28    # shapes.h5

# sigma controls E-step softness:
#   0.25  is a good default for [0,1] normalised shapes images.
#   Reduce to 0.1 if assignments are too soft after ~20 epochs.
SIGMA = 0.25

# decay_steps scaled so LR decays ~10% every ~5 epochs
# (shapes has ~875 batches/epoch × 15 EM iters = 13125 grad steps/epoch)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=50_000,
    decay_rate=0.9,
    staircase=True)

optimizer  = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
inner_cell = Q_graph(image_size=IMAGE_SIZE)
loss_fn    = gaussian_em_loss(sigma=SIGMA)
rnn_cell   = rnn_em_gaussian(inner_cell,
                              input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
                              sigma=SIGMA)

# ------------------------------------------------------------------
# Logging / checkpoints
# ------------------------------------------------------------------
os.makedirs('./logs',           exist_ok=True)
os.makedirs('./plots_gaussian', exist_ok=True)

checkpoint = tf.train.Checkpoint(
    step      = tf.Variable(0, dtype=tf.int64),
    ami       = tf.Variable(-1e10, dtype=tf.float32),
    optimizer = optimizer,
    model     = rnn_cell.model)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, './ckpt/gaussian', max_to_keep=3)

current_time   = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
summary_writer = tf.summary.create_file_writer(f'./logs/gaussian_{current_time}')

train_data = get_dataset(generator, 'training')
valid_data = get_dataset(generator, 'validation')


# ------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------

def visualize_clusters(gamma, features, epoch, step):
    """Save cluster-assignment map alongside the original image."""
    cluster_assignments = tf.argmax(gamma, axis=1)    # (B, H, W, 1)
    sample_img          = features[0, 0, :, :, 0].numpy()
    sample_assignment   = cluster_assignments[0, :, :, 0].numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(sample_img, cmap='gray')
    ax1.set_title('Original image')
    ax1.axis('off')
    cmap = plt.cm.get_cmap('viridis', K)
    ax2.imshow(sample_assignment, cmap=cmap, vmin=0, vmax=K - 1)
    ax2.set_title('Cluster assignments')
    ax2.axis('off')
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax2)
    plt.tight_layout()
    plt.savefig(f'./plots_gaussian/clusters_epoch{epoch}_step{step}.png')
    plt.close()


# ------------------------------------------------------------------
# Train step  (@tf.function, fixed input signature -> zero retracing)
# ------------------------------------------------------------------

N_EM_ITERATIONS = 15

@tf.function(input_signature=[
    tf.TensorSpec(shape=(BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=tf.float32)
])
def train_step(features):
    features_noisy = bitflip_noisy_static(features)
    hidden_state   = rnn_cell.initial_state(BATCH_SIZE, K)
    current_lr     = lr_schedule(optimizer.iterations)

    for _ in range(N_EM_ITERATIONS):
        with tf.GradientTape() as tape:
            hidden_state = rnn_cell((features_noisy, features), hidden_state)
            _, preds, gamma = hidden_state
            loss = loss_fn(preds, features, gamma)
        grads = tape.gradient(loss, rnn_cell.model.trainable_weights)
        grads = [tf.clip_by_norm(g, 3.0) if g is not None else g for g in grads]
        optimizer.apply_gradients(zip(grads, rnn_cell.model.trainable_weights))

    return loss, gamma, current_lr


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

def validation(dataset):
    ami_values = []
    for features, groups in dataset:
        hidden_state   = rnn_cell.initial_state(BATCH_SIZE, K)
        features_noisy = bitflip_noisy_static(features)
        hidden_state   = rnn_cell((features_noisy, features), hidden_state)
        _, _, gamma    = hidden_state
        ami_values.append(ami_score(gamma, groups))
    return tf.reduce_mean(ami_values)


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
n_iterations            = 100
best_ami                = -1.0
patience                = 0
max_patience            = 15   # plus permissif : v_ami est bruyant (15 batches)
warmup_steps            = 100  # pas d'early stopping avant ce seuil
validation_frequency    = 50
visualization_frequency = 200
validation_samples      = 30   # plus de batches = estimation plus stable

for epoch in range(n_iterations):
    train_ami_mean  = Mean()
    train_loss_mean = Mean()
    now = time.perf_counter()

    for step, (features, groups) in enumerate(train_data):
        checkpoint.step.assign_add(1)

        loss, gamma, current_lr = train_step(features)
        train_loss_mean(loss)
        ami_train = ami_score(gamma, groups)
        train_ami_mean(ami_train)

        if step % 10 == 0:
            with summary_writer.as_default():
                tf.summary.scalar('train_loss',    loss,      step=checkpoint.step)
                tf.summary.scalar('train_ami',     ami_train, step=checkpoint.step)
                tf.summary.scalar('learning_rate', current_lr, step=checkpoint.step)

        if step % validation_frequency == 0:
            tloss      = train_loss_mean.result()
            tami_score = train_ami_mean.result()
            train_loss_mean.reset_state()
            train_ami_mean.reset_state()

            vami_score = validation(valid_data.take(validation_samples))
            gc.collect()

            with summary_writer.as_default():
                tf.summary.scalar('validation_ami', vami_score, step=checkpoint.step)

            duration = time.perf_counter() - now
            print(f'Epoch {epoch+1:3d} | Step {step+1:4d} | '
                  f'loss={tloss.numpy():.4f} | '
                  f't_ami={tami_score:.4f} | '
                  f'v_ami={vami_score.numpy():.4f} | '
                  f'lr={current_lr:.6f} | '
                  f'{duration:.1f}s')

            if step % visualization_frequency == 0:
                try:
                    visualize_clusters(gamma, features, epoch, step)
                except Exception as e:
                    print(f'Visualisation skipped: {e}')

            if vami_score.numpy() > checkpoint.ami:
                checkpoint.ami.assign(vami_score)
                checkpoint_manager.save()
                patience = 0
                if vami_score.numpy() > best_ami:
                    best_ami = vami_score.numpy()
                    print(f'  New best AMI: {best_ami:.4f}  (sigma={SIGMA})')
            elif int(checkpoint.step) > warmup_steps:
                # Patience ne s'incrémente qu'après le warmup
                # pour éviter qu'un v_ami initial chanceux bloque l'entraînement
                patience += 1

            now = time.perf_counter()

            if patience >= max_patience:
                print(f'Early stopping at step {int(checkpoint.step)} '
                      f'(epoch {epoch + 1})')
                break

    else:
        # La boucle step s'est terminée normalement : continuer l'epoch suivante
        continue
    # La boucle step a été interrompue par break : sortir aussi de la boucle epoch
    break

print(f'Training complete. Best AMI: {best_ami:.4f}')
