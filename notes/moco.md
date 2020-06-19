## Momentum Contrast for Unsupervised Visual Representation Learning

- [Official implementation](https://github.com/facebookresearch/moco). I will use some code snippets from here.

### Key idea
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

The query `q` matches exactly one of the keys (chosen to be `k0`). `encoder` is learnt through backprop. `momentum encoder` then copies the parameters of `encoder` but used a moving average:

```python
f_k.params = momentum * f_k.params + (1-momentum)*f_q.params
```

### Code and some minor details
