import sys

from torch.utils import data
sys.path.insert(0,'../..')

import libs
import libs_dataset
import numpy
import torch

import models.model_0.model as Model


features_count = 128

dataset_in = libs_dataset.DatasetMnist()

dataset = libs_dataset.DatasetAnomaly(dataset_in)
 
model = Model.Create(dataset.input_shape, features_count)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size = 128

for epoch in range(5):

    batches_count = dataset.get_training_count()//batch_size

    for i in range(batches_count):
        x, y = dataset.get_training_batch(batch_size)

        logits = model.forward(x)


        loss = torch.nn.functional.cross_entropy(logits, torch.arange(x.shape[0]).to(x.device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(epoch, i, batches_count, loss)

    model.save("models/model_0/trained/")

