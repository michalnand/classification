import sys
sys.path.insert(0,'../..')

import numpy
import torch
import sklearn.manifold
import matplotlib.pyplot as plt

import libs
import libs_dataset

import models.model_0.model     as Model


dataset = libs_dataset.DatasetMnist()

model = Model.Create(dataset.input_shape, 128)
model.load("./models/model_0/trained/")
model.eval()

x, y = dataset.get_testing_batch(batch_size=1024)

labels = torch.argmax(y, dim=1).detach().to("cpu").numpy()

 
print("computing features")
#features = x.view(x.shape[0], -1) 
features = model.eval_features(x)
features = features.detach().to("cpu").numpy()


print("training t-sne")
features_embedded = sklearn.manifold.TSNE(n_components=2).fit_transform(features)

print(features_embedded.shape)


plt.scatter(features_embedded[:, 0], features_embedded[:, 1], c=labels, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()