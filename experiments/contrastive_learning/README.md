# contrastive learning, MNIST test

model is training to make closer features for same class images, and distance features for different class images

![](images/cl.png)



model is simple CNN network, producing 128 features

```python
nn.Conv2d(input_shape[0], 16, kernel_size = 3, stride = 2, padding = 1),
nn.ReLU(),  
nn.Conv2d(16, 32, kernel_size = 3, stride = 2, padding = 1),
nn.ReLU(),
nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
nn.ReLU(), 
nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 0),
nn.ReLU(), 

nn.Flatten(),
nn.Linear(5*5*64, 128),
nn.ReLU()
```

### class IDs
- 0 : input classes are the same,  target = 0
- 1 : input classes are different, target = 1

### model output is simple Euclidean distance : 

```python
fa = model(xa)
fb = model(xb)
  
predicted = ((fa - fb)**2).mean(dim=1)
```

### loss function
```python
l1 = (1.0 - target)*predicted
l2 = target*max(1.0 - predicted, 0)

loss  = (l1 + l2).mean()
```

### results
arround 99.5% accuracy on testing set, training for 10 epochs

```
accuracy   = 99.582 [%]
hit_count  = 10006
miss_count = 42

class_accuracy = 99.34%   99.782%   


confusion_matrix = 
        4969          10
          32        5037
```