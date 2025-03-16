import h5py
import os
# shut INFO and WARNING messages up 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf 
import numpy as np
NEM_DATA = os.environ.get("filename", "./data/shapes.h5")
# Reduce batch size for better performance on M3
BATCH_SIZE = 8  # Reduced from 16 to 8
FEATURE_SHAPE = (28, 28, 1)






class generator(object):
    # class wise configuration : commun for all batches and generator should not take 
    # any argument 
    config = {
    "usage" : "training",
    "batch_size": BATCH_SIZE,
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
                features = (hdf5[self.config["usage"]]["features"][:, start:end,:,:,:])
                groups =  (hdf5[self.config["usage"]]["groups"][:, start:end,:,:,:])
               
                yield np.transpose(features, axes=[1,0,2,3,4]), np.transpose(groups, axes=[1,0,2,3,4])


def normalize_data(data,groups):
    # perform normalization here
    data_norm = (data - tf.reduce_min(data, keepdims=True)) / (tf.reduce_max(data, keepdims=True) - tf.reduce_min(data, keepdims=True))
    data_norm = tf.reshape(data_norm, shape=tf.shape(data))
    
    
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
                (config["batch_size"], 1) + FEATURE_SHAPE,
                (config["batch_size"], 1) +  FEATURE_SHAPE,
        )
    )

    # dataset.map(normalize_data)
    #TODO map for data normalization
    # dataset.map(normalize_data)
    
    assert dataset.element_spec[0].shape ==  (config["batch_size"], 1) + FEATURE_SHAPE
    assert dataset.element_spec[1].shape == (config["batch_size"], 1) + FEATURE_SHAPE

    return dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()


           