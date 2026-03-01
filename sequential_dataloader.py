import h5py
import os
import numpy as np
# shut INFO and WARNING messages up
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

NEM_DATA = os.environ.get("filename", "./data/flying_mnist_hard_3digits.h5")
BATCH_SIZE = 8
SEQUENCE_LENGTH = 21
FEATURE_SHAPE = (28, 28, 1)


class generator(object):
    config = {
    "usage" : "training",
    "batch_size": BATCH_SIZE,
    "sequence_length" : SEQUENCE_LENGTH,
    "filename" : NEM_DATA,
    "out_list" : ("features", "groups"),
    }

    def __call__(self):
        with h5py.File(self.config["filename"], 'r') as hdf5:
            num_samples = hdf5[self.config["usage"]]["features"].shape[1]
            num_batches = num_samples // self.config["batch_size"]
            for i in range(0, num_batches):
                start = i * self.config["batch_size"]
                end = (i + 1) * self.config["batch_size"]
                # HDF5 shape: (T, N, H, W, 1) -> transpose to (B, T, H, W, 1)
                features = hdf5[self.config["usage"]]["features"][:self.config["sequence_length"], start:end]
                groups   = hdf5[self.config["usage"]]["groups"][:self.config["sequence_length"], start:end]
                features = np.transpose(features, axes=[1, 0, 2, 3, 4])
                groups   = np.transpose(groups,   axes=[1, 0, 2, 3, 4])
                yield features, groups


def normalize_data(data, groups):
    min_val = tf.reduce_min(data, keepdims=True)
    max_val = tf.reduce_max(data, keepdims=True)
    data_norm = (data - min_val) / (max_val - min_val + 1e-8)
    return data_norm, groups


def get_dataset(generator, usage):
    config = generator.config
    if usage == "training":
        generator.config["usage"] = usage
    elif usage == "validation":
        generator.config["usage"] = usage
    elif usage == "test":
        generator.config["usage"] = usage
    else:
        raise ValueError(f"Invalid usage: {usage}")

    generator = generator()
    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
            (config["batch_size"], config["sequence_length"]) + FEATURE_SHAPE,
            (config["batch_size"], config["sequence_length"]) + FEATURE_SHAPE,
        )
    )

    dataset = dataset.map(normalize_data, num_parallel_calls=tf.data.AUTOTUNE)

    assert dataset.element_spec[0].shape == (config["batch_size"], config["sequence_length"]) + FEATURE_SHAPE
    assert dataset.element_spec[1].shape == (config["batch_size"], config["sequence_length"]) + FEATURE_SHAPE

    return dataset.prefetch(tf.data.AUTOTUNE).cache()
