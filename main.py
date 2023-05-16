from rnn_em_cell_bernoulli import rnn_em
from q_graph import Q_graph
from static_dataloader import get_dataset, generator
from bernoulli_loss import em_loss
from trainer import StaticTrainer
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf 
import time
K = 3 
lr = 0.001 
inner_cell =  Q_graph()
loss_fn = em_loss()
em_cell = rnn_em(inner_cell, input_shape=(28, 28, 1))



train_data = get_dataset(generator, "training")
valid_data = get_dataset(generator, "validation")

trainer = StaticTrainer(em_cell=em_cell, 
                        loss=loss_fn, 
                        learning_rate=1e-4)
n_epochs = 120

start = time.perf_counter()

for epoch in range(n_epochs):

    trainer.train(train_data, valid_data, epoch)

duration = time.perf_counter() - start

print(f"training duration : {duration:.2f}s")