
### DeepInfoMax

As [mentioned here](https://www.microsoft.com/en-us/research/blog/deep-infomax-learning-good-representations-through-mutual-information-maximization/):

> DIM is based on two learning principles: mutual information maximization in the vein of the infomax optimization principle and self-supervision, an important unsupervised learning method that relies on intrinsic properties of the data to provide its own annotation.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```


```python
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(3, 64, kernel_size=4, stride=1)
        self.c1 = nn.Conv2d(64, 128, kernel_size=4, stride=1)
        self.c2 = nn.Conv2d(128, 256, kernel_size=4, stride=1)
        self.c3 = nn.Conv2d(256, 512, kernel_size=4, stride=1)
        self.l1 = nn.Linear(512*20*20, 64)

        self.b1 = nn.BatchNorm2d(128)
        self.b2 = nn.BatchNorm2d(256)
        self.b3 = nn.BatchNorm2d(512)

    def forward(self, x):
        h = F.relu(self.c0(x))
        features = F.relu(self.b1(self.c1(h)))
        h = F.relu(self.b2(self.c2(features)))
        h = F.relu(self.b3(self.c3(h)))
        encoded = self.l1(h.view(x.shape[0], -1))
        return encoded, features
```


```python
class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(128, 64, kernel_size=3)
        self.c1 = nn.Conv2d(64, 32, kernel_size=3)
        self.l0 = nn.Linear(32 * 22 * 22 + 64, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = h.view(y.shape[0], -1)
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)
```


```python
class LocalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(192, 512, kernel_size=1)
        self.c1 = nn.Conv2d(512, 512, kernel_size=1)
        self.c2 = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)
```


```python
class PriorDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(64, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))
```


```python
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64, 15)
        self.bn1 = nn.BatchNorm1d(15)
        self.l2 = nn.Linear(15, 10)
        self.bn2 = nn.BatchNorm1d(10)
        self.l3 = nn.Linear(10, 10)
        self.bn3 = nn.BatchNorm1d(10)

    def forward(self, x):
        encoded, _ = x[0], x[1]
        clazz = F.relu(self.bn1(self.l1(encoded)))
        clazz = F.relu(self.bn2(self.l2(clazz)))
        clazz = F.softmax(self.bn3(self.l3(clazz)), dim=1)
        return clazz
```

### Encoder takes as input an image of size `(32,32)` and returns features of size `(128,26,26)` and encoded output of length `(64)`


```python
x = torch.randn(1,3,32,32)
with torch.no_grad():
    encoded, features = Encoder()(x)
encoded.size(), features.size()
```




    (torch.Size([1, 64]), torch.Size([1, 128, 26, 26]))



### Global discriminator learns to discriminate whether or not `encoded` and `features` come from the same image. Note that `encoded` and `features` are concatenated in the linear layer.


```python
with torch.no_grad():
    out = GlobalDiscriminator()(encoded, features)
out.size()
```




    torch.Size([1, 1])



### Local discriminator does the same thing but for each individual cell. Note that `encoded` and `features` are concatenated at the start at the convolutional layer.


```python
encoded_expanded = encoded.unsqueeze(2).unsqueeze(3).expand(-1,-1,26,26)
x = torch.cat((features, encoded_expanded), dim=1)

with torch.no_grad():
    out = LocalDiscriminator()(x)
out.size()
```




    torch.Size([1, 1, 26, 26])



### Prior discriminator simply learns to predict whether or not `encoded` comes from a uniform distribution


```python
with torch.no_grad():
    out = PriorDiscriminator()(encoded)
out.size()
```




    torch.Size([1, 1])



In the [official implementation](https://github.com/DuaneNielsen/DeepInfomaxPytorch/blob/master/train.py#L25-L49),

* `y` is `encoded`
* `M` is `features`
* `M_prime` is `features` from another image

The objective is to maximize the log likelihood for (`y`, `M`) and minimize that for (`y`, `M_prime`).

### How is `y_prime` created?

In every batch, the sequence of the images [is changed](https://github.com/DuaneNielsen/DeepInfomaxPytorch/blob/master/train.py#L91-L92) (in a non-random way). This is different from the random way the sequence was changed in MINE.

```python
            y, M = encoder(x)
            # rotate images to create pairs for comparison
            M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
```
