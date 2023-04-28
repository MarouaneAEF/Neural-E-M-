import h5py
import os
# shut INFO and WARNING messages up 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf 

NEM_DATA = os.environ.get("filename", "./data/flying_mnist_hard_3digits.h5")
BATCH_SIZE = 64
SEQUENCE_LENGHT = 20 + 1 
FEATURE_SHAPE = (1, 24, 24, 1)






class generator(object):
    # class wise configuration : commun for all batches and generator should not take 
    # any argument 
    config = {
    "usage" : "training",
    "batch_size": BATCH_SIZE,
    "sequence_length" : SEQUENCE_LENGHT ,
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
                features = (hdf5[self.config["usage"]]["features"]
                            [:self.config["sequence_length"], start:end]
                            [:,:,None])
                groups =  (hdf5[self.config["usage"]]["groups"]
                            [:self.config["sequence_length"], start:end]
                            [:,:,None])
                yield features, groups




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
    print("sequence length", config["sequence_length"])
    generator = generator()
    dataset = tf.data.Dataset.from_generator(
        generator=generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=(
                (config["sequence_length"], config["batch_size"]) + FEATURE_SHAPE,
                (config["sequence_length"], config["batch_size"]) +  FEATURE_SHAPE,
        )
    )

    print(dataset.element_spec[0].shape)
    assert dataset.element_spec[0].shape == (config["sequence_length"], config["batch_size"]) +  FEATURE_SHAPE
    assert dataset.element_spec[1].shape == (config["sequence_length"], config["batch_size"]) + FEATURE_SHAPE

    return dataset.prefetch(tf.data.experimental.AUTOTUNE).cache()


           