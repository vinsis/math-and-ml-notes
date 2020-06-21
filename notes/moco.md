## Momentum Contrast for Unsupervised Visual Representation Learning

- [Official implementation](https://github.com/facebookresearch/moco). I will use some code snippets from here.

### Background
> (Contrastive loss based) methods can  be  thought  of  as  building dynamic dictionaries. The “keys” (tokens) in the dictionary are sampled from  data (e.g., images or patches) and are represented by an encoder network. Unsupervised learning trains encoders to perform dictionary look-up:  an encoded “query” should be similar to its matching key and dissimilar to others.

> From this perspective, we hypothesize that it is desirable to  build  dictionaries  that  are: (i) large and (ii) consistent as they evolve during training.

### Two main lines of performing unsupervised/self-unsupervised learning:

1. Loss functions: A model learns to predict a target. A target could be fixed (`reconstructing the input pixels using L1/L2 losses`) or moving.

- __Contrastive losses__: `Instead of matching an input to a fixed target, in contrastive loss formulations the target can vary on-the-fly during training and can be defined in terms of the data representation computed by a network.`

- __Adversarial losses__: They `measure the difference between probability distributions.`

2. Pretext tasks: `The term “pretext” implies that the task being solved is not of genuine interest, but is solved only for the true purpose of learning a good data representation.` Some examples are `denoising auto-encoders, context auto-encoders, or cross-channel auto-encoders (colorization).`

> __Contrastive  learning vs.  pretext  tasks__: Various  pretext tasks can be based on some form of contrastive loss functions. The instance discrimination method is related to the exemplar-based task and NCE. The pretext task in contrastive predictive coding (CPC) is a form of context auto-encoding, and in contrastive multi view coding (CMC) it is related to colorization.

---

### Key idea

You are given two neural networks (`encoder` aka `f_q` and `momentum encoder` aka `f_k`) as shown below:

![](https://user-images.githubusercontent.com/11435359/71603927-0ca98d00-2b14-11ea-9fd8-10d984a2de45.png)

The query `q` matches exactly one of the keys (chosen to be `k0`). `encoder` is learnt through backprop. `momentum encoder` then copies the parameters of `encoder` but uses a moving average:

```python
f_k.params = momentum * f_k.params + (1-momentum)*f_q.params
```

### Code and some minor details

`f_q` and `f_k` are build from the same encoder class (a ResNet) with `output_dim`=128:

```python
self.encoder_q = base_encoder(num_classes=output_dim)
self.encoder_k = base_encoder(num_classes=output_dim)
```

The initial parameters of the two encoders are [set to be the same](https://github.com/facebookresearch/moco/blob/master/moco/builder.py#L34-L36):
```python
for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
    param_k.data.copy_(param_q.data)  # initialize
    param_k.requires_grad = False  # not update by gradient
```

The parameters update with has pseudocode

```python
f_k.params = momentum * f_k.params + (1-momentum)*f_q.params
```

is [implemented](https://github.com/facebookresearch/moco/blob/master/moco/builder.py#L45-L50) like so:

```python
for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
        param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
```

Here we have `K: queue size; number of negative keys (default: 65536)`.

The queue is [defined](https://github.com/facebookresearch/moco/blob/master/moco/builder.py#L39-L40) as shown below. Each column of the matrix is a negative key (embedding). The embeddings are all normalized.

```python
self.register_buffer("queue", torch.randn(dim, K))
self.queue = nn.functional.normalize(self.queue, dim=0)
```

#### Tackling issues with batch normalization by shuffling

__I don't fully understand what the problem is and what the solution is. I will update this section once I understand.__

> ...  we found that using BN prevents the model from  learning  good  representations.

> We resolve this problem by shuffling BN. We train with multiple GPUs and perform BN on the samples independently for each GPU (as done in common practice). For the key encoder `f_k`, we shuffle the sample order in the current mini-batch before distributing it among GPUs (and shuffle back  after  encoding);  the  sample  order  of  the  mini-batch for  the  query  encoder `f_q` is  not  altered. This ensures  the batch statistics used to compute a query and its positive key come from two different subsets. This effectively tackles the cheating issue and allows training to benefit from BN.

### Forward pass

In forward pass, you provide a batch of query and key images: `im_q` and `im_k`.

[Computation of query features](https://github.com/facebookresearch/moco/blob/master/moco/builder.py#L124-L126) is straight-forward:

```python
# compute query features
q = self.encoder_q(im_q)  # queries: NxC
q = nn.functional.normalize(q, dim=1)
```

Then the key encoder is updated and keys are calculated like so:

```python
self._momentum_update_key_encoder()
k = self.encoder_k(im_k)  # keys: NxC
```

Note that `k` are the positive samples for `q`. The negative samples come from the queue.

```python
l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
logits = torch.cat([l_pos, l_neg], dim=1)
logits /= self.T # self.T is the temperature (default value 0.07)
```

Finally the keys in the queue are updated:

```python
batch_size = k.shape[0]
ptr = int(self.queue_ptr)
self.queue[:, ptr:ptr + batch_size] = keys.T
ptr = (ptr + batch_size) % self.K  # move pointer
```

Note that I skipped (for now) the shuffling and de-shuffling part since I don't clearly understand it.
