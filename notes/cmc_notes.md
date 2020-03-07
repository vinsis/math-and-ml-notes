## Contrastive Multiview Coding

Code snippets are used from [the official implementation](https://github.com/HobbitLong/CMC).

### Key idea
Different transformations of image (changing color space, segmentation, depth view etc) still have the same semantic content. Hence their representations should also be similar. Thus:

> Given a pair of sensory views, a deep representation is learnt by bringing views of the same scene together in embedding space, while pushing views of different scenes apart.
> 

---

### Contrastive objective vs cross-view prediction

Cross-view prediction is the standard encoder decoder architecture where the loss is measured pixel-wise between the constructed output and the input. Pixel-wise loss doesn't care about which pixels are important and which pixels are not.

In constrastive objective two different inputs representing the same semantic content create two representations. The loss is measured between the two representations. This way the model has a change to learn which information to keep and which to discard while _encoding_ an image. Thus the learned representation is better as it ignores all the noise and retains all the important information.

---

### Contrastive learning with two views

`V1` is a dataset of images with one kind of transformation (or view). `V2` is a dataset of the same images but seen in a different view. One view is sampled from `V1` and one view is sampled from `V2`. If both the views belong to the same image, we want a critic h<sub>θ</sub>(.) to give a high value. If they don't, the critic will give a low value. Here is how the visual looks like:

![](https://github.com/vinsis/math-and-ml-notes/raw/master/images/cmc_gif.gif)

The loss function is constructed like so:

![](https://github.com/vinsis/math-and-ml-notes/raw/master/images/L_v1_v2.png)

---

### Implementing the critic

> To extract compact latent representations of v<sub>1</sub> and v<sub>2</sub>, we employ two encoders fθ<sub>1</sub>(·) and fθ<sub>2</sub>(·) with parameters θ<sub>1</sub> and θ<sub>2</sub> respectively. The latent representions are extracted as z1=f<sub>θ1</sub>(v<sub>1</sub>), z2=f<sub>θ2</sub>(v<sub>2</sub>). On top of these features, the score is computed as the exponential of a bivariate function of z<sub>1</sub> and z<sub>1</sub>, e.g., a bilinear function parameterized by W<sub>12</sub>.
> 

We make the loss between the views symmetric:

L(V<sub>1</sub>, V<sub>2</sub>) = L<sub>Contrast</sub><sup>V<sub>1</sub>V<sub>2</sub></sup> + L<sub>Contrast</sub><sup>V<sub>2</sub>V<sub>1</sub></sup>

> ... we use the representation z<sub>1</sub>, z<sub>2</sub>, or the concatenation of both, [z<sub>1</sub>,z<sub>2</sub>], depending on our paradigm
> 

### An example of a critic

The critic takes images from two views: L space and AB space. It has f<sub>θ1</sub> = `l_to_ab` and f<sub>θ2</sub> = `ab_to_l`. I find these names misleading since `self.l_to_ab` does not map `l` to `ab`. It maps `l` to a vector of dimension = `feat_dim`. The same applies to `ab_to_l`. 

[Source](https://github.com/HobbitLong/CMC/blob/58d06e9a82f7fea2e4af0a251726e9c6bf67c7c9/models/alexnet.py#L7)

```python
class alexnet(nn.Module):
    def __init__(self, feat_dim=128):
        super(alexnet, self).__init__()

        self.l_to_ab = alexnet_half(in_channel=1, feat_dim=feat_dim)
        self.ab_to_l = alexnet_half(in_channel=2, feat_dim=feat_dim)

    def forward(self, x, layer=8):
        l, ab = torch.split(x, [1, 2], dim=1)
        feat_l = self.l_to_ab(l, layer)
        feat_ab = self.ab_to_l(ab, layer)
        return feat_l, feat_ab
```

---

### Connection with mutual information

It can be shown that the optical critic is proportional to the density ratio between p(z<sub>1</sub>, z<sub>2</sub>) and p(z<sub>1</sub>)p(z<sub>2</sub>).

It can also be shown that 
I(z<sub>1</sub>; z<sub>2</sub>) >= log(k) - L<sub>Contrast</sub>

where k `is the number of negative pairs in sample set`.

> Hence minimizing the objective `L` maximizes the lower bound on the mutual information I(z<sub>1</sub>; z<sub>2</sub>), which is bounded above by I(v<sub>1</sub>; v<sub>2</sub>) by the data processing inequality. The dependency on `k` also suggests that using more negative samples can lead to an improved representation; we show that this is indeed the case.
> 
---

### Contrastive learning with more than two views

There are two ways to do so:

#### Core graph view
Given `M` views V<sub>1</sub>, ..., V<sub>M</sub>, we can choose to optimize over one view only. What this means is that the model will learn best how to learn representations of image in that particular view.

If we want to optimize over the first view, the loss function is defined as:

L(V<sub>1</sub>) = Σ<sub>j</sub> L(V<sub>1</sub>, V<sub>j</sub>)

A more general equation is:

L(V<sub>i</sub>) = Σ<sub>j</sub> L(V<sub>i</sub>, V<sub>j</sub>)

#### Full graph view
Here you optimize over all views by choosing all possible `(i,j)` pairs for creating a loss function. There are <sup>M</sup>C<sub>2 </sub> ways to do so.

> Both these formulations have the effect that information is prioritized in proportion to the numberof views that share that information. This can be seen in the information diagrams visualized below:
> 

![](https://github.com/vinsis/math-and-ml-notes/raw/master/images/learning_from_many_views.png)

This in the core graph view, the mutual information between say V<sub>2</sub> and V<sub>3</sub> is discarded but not in the case of full graph view.

> Under both the core view and full graph objectives, a factor,like “presence of dog”, that is common to all views will be preferred over a factor that affects fewerviews, such as “depth sensor noise”.
> 

---

### Approximating the softmax distribution with noise-contrastive estimation

Let's revisit the function below:

![](https://github.com/vinsis/math-and-ml-notes/raw/master/images/L_v1_v2.png)

If `k` in the above formula is large, computing the full softmax loss will be expensive. 

This problem is solved by using noise contrastive estimation trick. Assume that the `m` negative samples are distributed uniformly i.e. p<sub>n</sub> is uniform. Then we have: 

P(D=1|v<sub>2</sub>; v<sub>1</sub><sup>i</sup> ) = p<sub>d</sub>(v<sub>2</sub> | v<sub>1</sub><sup>i</sup>) / \[ p<sub>d</sub>(v<sub>2</sub> | v<sub>1</sub><sup>i</sup>) + m*p<sub>n</sub>(v<sub>2</sub> | v<sub>1</sub><sup>i</sup>) \]

The distribution of positive samples p<sub>d</sub> is unknown. p<sub>d</sub> is approximated by an unnormalized density h<sub>θ</sub>(.).

In this paper, h<sub>θ</sub>(v<sub>1</sub><sup>i</sup>, v<sub>2</sub><sup>i</sup>) = <v<sub>1</sub><sup>i</sup>, v<sub>2</sub><sup>i</sup>> where <.,.> stands for dot product.

---

### Implementation of the loss function

This section mostly explains the implementation of the NCE Loss [here](https://github.com/HobbitLong/CMC/tree/58d06e9a82f7fea2e4af0a251726e9c6bf67c7c9/NCE).

[This file](https://github.com/HobbitLong/CMC/blob/58d06e9a82f7fea2e4af0a251726e9c6bf67c7c9/NCE/alias_multinomial.py) simply uses [a simple trick](https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/) to allow faster sampling. I won't go into the details here since it is not relevant to the central idea of the paper. In short it creates a class called `AliasMethod` which is used in lieu of multinomial sampling:

`self.multinomial = AliasMethod(self.unigrams)`

The implementation converts RGB image into LAB and splits it into two views: `L` and `AB`.

#### Storing representations in memory bank

> We maintain a memory bank to store latent features for each training sample. Therefore, we can efficiently retrieve `m` noise samples from the memory bank to pair with each positive sample without recomputing their features. The memory bank is dynamically updated with features computed on the fly.

These representations are stored in the same file that calculates the NCELoss: `NCEAverage.py`. This is done using the `register_buffer` property of PyTorch `nn.module`s:

```python
self.register_buffer('memory_l', torch.rand(outputSize, inputSize)
self.register_buffer('memory_ab', torch.rand(outputSize, inputSize)
```

Here `outputSize` is the size of the dataset and `inputSize` is the size of representations (128 in case of Alexnet).

We want these representations to have a unit size on average. The way these are initialized is by uniform sampling from the interval `[-a,a]` such that the expected value of L2 norm of vector with size `inputSize` is 1. In other words,

Σ<sub>i</sub> E[x<sub>i</sub><sup>2</sup>] = 1
which means
Σ<sub>i</sub> Var[x<sub>i</sub>] + (E[x<sub>i</sub>])<sup>2</sup> = 1

Solving this gives us:
a = `1. / math.sqrt(inputSize / 3)`.

Thus the actual initilization of memory bank looks like:

```python
stdv = 1. / math.sqrt(inputSize / 3)
self.register_buffer('memory_l', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
self.register_buffer('memory_ab', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))
```

The following code below then does the following:

1. randomly sample negative samples and get the values from the memory bank
2. copy the values of the positive samples in the first index
3. calculate dot product using batch matrix multiplication (`bmm`)

```python

# score computation
if idx is None:
    idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
    idx.select(1, 0).copy_(y.data)
# sample
weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
weight_l = weight_l.view(batchSize, K + 1, inputSize)
out_ab = torch.bmm(weight_l, ab.view(batchSize, inputSize, 1))
# sample
weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
weight_ab = weight_ab.view(batchSize, K + 1, inputSize)
out_l = torch.bmm(weight_ab, l.view(batchSize, inputSize, 1))
```

Finally the memory bank is updated using momentum [here](https://github.com/HobbitLong/CMC/blob/58d06e9a82f7fea2e4af0a251726e9c6bf67c7c9/NCE/NCEAverage.py#L70-L83).

```python
l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
l_pos.mul_(momentum)
l_pos.add_(torch.mul(l, 1 - momentum))
l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
updated_l = l_pos.div(l_norm)
self.memory_l.index_copy_(0, y, updated_l)
```
