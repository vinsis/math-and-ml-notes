Code samples are taken from [here](https://github.com/wilson1yan/cs294-158-ssl/blob/master/deepul_helper/tasks/simclr.py) and [here](https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py#L15-L22).

The SimCLR framework has four major components:

### 1. A stochastic data augmentation module:

Taken from [here](https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py#L15-L22):

```python
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
```

### 2. A neural network _base encoder `f(·)`_ that extracts representation vectors from augmented data examples:


```python
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
```


```python
backbone = models.resnet18(pretrained=False, num_classes=50)
[name for (name,_) in backbone.named_children()]
```




    ['conv1',
     'bn1',
     'relu',
     'maxpool',
     'layer1',
     'layer2',
     'layer3',
     'layer4',
     'avgpool',
     'fc']




```python
backbone.fc
```




    Linear(in_features=512, out_features=50, bias=True)




```python
backbone(torch.randn(4,3,224,224)).size()
```




    torch.Size([4, 50])



### 3. A small neural network _projection head `g(·)`_ that mapsrepresentations to the space where contrastive loss is applied

We use a MLP with one hidden layer to obtain:

$$ z_i = g(h_i) = W_2 (\sigma (W_1(h_i))) $$

We find it beneficial to define the contrastive loss on $z_i’s$ rather than $h_i’s$


[Source](https://github.com/wilson1yan/cs294-158-ssl/blob/master/deepul_helper/tasks/simclr.py#L30-L36)

```python
        self.proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.projection_dim, bias=False),
            BatchNorm1d(self.projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.projection_dim, self.projection_dim, bias=False),
            BatchNorm1d(self.projection_dim, center=False)
        )
```

### 4. A contrastive loss function defined for a contrastive prediction task

> We randomly sample a minibatch of `N` examples and define the contrastive prediction task on pairs of augmented examples derived from the minibatch, resulting in `2N` data points. We do not sample negative examples explicitly.  Instead, given a positive pair, we treatthe other `2(N−1)` augmented examples within a minibatch as negative examples.

No wonder you need such a huge batch size to train.

> To keep it simple, we do not train the model with a memory bank. Instead, we vary the training batch size `N` from `256` to `8192`.  A batch size of `8192` gives us `16382` negative examples per positive pair from both augmentation views. 

Define $l(i,j)$ as:

$$ l(i,j) = -log \frac{exp(sim(i,j)/\tau)}{\sum_{1,k\ne i}^{2N} exp(sim(i,k/\tau))} $$

Then the loss is defined as:

$$ \frac{1}{2N} \sum_{k=1}^{N} [ l(2k-1,2k) + l(2k,2k-1) ] $$

where adjacent images at indices `2k` and `2k-1` are augmentations of the same image.

> Training with large batch size may be unstable when using standard SGD/Momentum with linear learning rate scaling. To stabilize the training, we use the LARS optimizer for all batch sizes.  We train our model with CloudTPUs, using 32 to 128 cores depending on the batch size.


```python
class SimCLR(torch.nn.Module):
    def __init__(self, base_encoder, output_dim=128):
        super(self, SimCLR).__init__()
        self.temperature = 0.5
        self.output_dim = output_dim
        
        latent_dim = base_encoder.fc.out_features
        
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, self.output_dim, bias=False),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim, bias=False),
            nn.BatchNorm1d(self.output_dim, center=False)
        )
        
    def forward(self, images):
        N = images[0].shape[0]
        xi, xj = images
        hi, hj = self.base_encoder(xi), self.base_encoder(xj) # (N, latent_dim)
        zi, zj = self.proj(hi), self.proj(hj) # (N, output_dim)
        zi, zj = F.normalize(zi, dim=-1), F.normalize(zj, dim=-1)
        
        # Each training example has 2N - 2 negative samples
        # Thus we have 2N * (2N-2) negative samples and 4N positive samples
        all_features = torch.cat([zi,zj], dim=0) # (2N, output_dim)
        sim_mat = (all_features @ all_features.T) / self.temperature # (2N,2N)
        # set all diagonal entries to -inf
        sim_mat[torch.arange(0,2*N), torch.arange(0,2*N)] = torch.tensor(-float('inf'))
        # image i should match with image N+i
        # image N+i should match with image i
        labels = torch.cat( [N + torch.arange(N), torch.arange(N)] ).long()
        loss = F.cross_entropy(sim_mat, labels, reduction='mean')
        return loss
```

> We conjecture that one serious issue when using only random cropping as data augmentation is that most patches from an image share a similar color distribution. Figure 6 shows that color histograms alone suffice to distinguish images. Neural nets may exploit this shortcut to solve the predictive task. Therefore, it is critical to compose cropping with color distortionin order to learn generalizable features.

### Contrastive learning needs stronger data augmentation than supervised learning

> ## A nonlinear projection head improves the representation quality of the layer before it


```python

```
