from rnn_em_cell_bernoulli import rnn_em
from q_graph import Q_graph
from static_dataloader import get_dataset, generator, BATCH_SIZE
from bernoulli_loss import em_loss

from trainer import StaticTrainer

K = 3 
lr = 1e-4
max_epoch = 100 
inner_cell =  Q_graph()
loss_fn = em_loss()
rnn_cell = rnn_em(inner_cell, input_shape=(28, 28, 1))


train_data = get_dataset(generator, "training")
valid_data = get_dataset(generator, "validation")

trainer = StaticTrainer(em_cell=rnn_cell, 
                        loss=loss_fn, 
                        learning_rate=1e-4)

# for epoch in range(50):

#     trainer.train(train_data, valid_data, epoch)