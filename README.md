Links to some important research papers or links. I plan to add notes as I go through each topic one by one.

### Disentangled representations
* [Quick overview by Google](https://ai.googleblog.com/2019/04/evaluating-unsupervised-learning-of.html)
* [Disentangling Disentanglement in Variational Autoencoders](https://arxiv.org/abs/1812.02833)
* [InfoGAN-CR: Disentangling Generative Adversarial Networks with Contrastive Regularizers](https://arxiv.org/abs/1906.06034)

__Contrastive Coding__:
* [Data-Efficient Image Recognition with Contrastive Predictive Coding](https://arxiv.org/abs/1905.09272)
* [Contrastive Multiview Coding](https://arxiv.org/abs/1906.05849)
* Google disspelling a lot of misconceptions about disentangled representations: [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://arxiv.org/abs/1811.12359)

### Memorization in neural networks
* [Blog post by BAIR](https://bair.berkeley.edu/blog/2019/08/13/memorization/)
* [Identity Crisis: Paper](https://arxiv.org/abs/1902.04698)

### Information theory based (unsupervised) learning
* [Invariant Information Clustering](https://arxiv.org/abs/1807.06653)

From the paper:
> Consider  now  a  pair  of  such  cluster  assignment  variables z and z′ for two inputs xand x′respectively. Their conditional  joint distribution is given by P(z=c, z′=c′|x,x′) = Φc(x)·Φc′(x′).This equation states that z and z′ are independent when conditioned on specific inputs x and x′; however, in general they are not independent after marginalization over a dataset of input pairs(xi,x′i), i= 1, . . . , n. 

I think this happens because information is lost when summing up observations. Correlation comes into picture in presence of uncertainty, which comes up when less information is available.

#### IIC Loss:

The official implementation is [here](https://github.com/xu-ji/IIC/blob/master/code/utils/cluster/IID_losses.py#L6-L33). Below are two (quite similar) implementations which are simplified.

```python
def IIC(z, zt, C=10):
    log = torch.log
    EPS = 1e-5
    P = (z.unsqueeze(2) * zt.unsqueeze(1)).sum(dim=0)
    P = (P + P.t())/2 / P.sum()
    P[ (P < EPS).data ] = EPS
    Pi = P.sum(dim=1).view(C,1).expand(C,C)
    Pj = P.sum(dim=0).view(1,C).expand(C,C)
    return (P * ( log(Pi) + log(Pj) - log(P) )).sum()

def IICv2(z, zt, C=10):
    log = torch.log
    EPS = 1e-5
    P = (z.unsqueeze(2) * zt.unsqueeze(1)).sum(dim=0)
    P = (P + P.t())/2 / P.sum()
    P[ (P < EPS).data ] = EPS
    Pi = P.sum(dim=1).view(C,1).expand(C,C)
    Pj = Pi.T
    return (P * ( log(Pi) + log(Pj) - log(P) )).sum()
```


> __Why  degenerate  solutions  are  avoided.__ Mutual information expands to I(z, z′) = H(z)−H(z|z′). Hence, maximizing this quantity trades-off minimizing the conditional cluster assignment entropy H(z|z′) and maximising individual cluster assignments  entropy H(z). The  smallest  value  of H(z|z′) is 0, obtained when the cluster assignments are exactly predictable from each other. The largest value of H(z) is lnC, obtained when all clusters are equally likely to be picked. This occurs when the data is assigned evenly between the clusters, equalizing their mass. Therefore the loss is not minimised if all samplesare assigned to a single cluster (i.e. output class is identicalfor all samples).

> __Meaning of mutual information.__  Firstly, due to the soft clustering, entropy alone could be maximised trivially by setting all prediction vectors Φ(x) to uniform distributions, resulting in no clustering. This is corrected by the conditional entropy component, which encourages deterministic one-hot predictions. For example, even for the degenerate case of identical pairs x=x′, the IIC objective encourages a deterministic clustering function (i.e.Φ(x) is a one-hot vector) as this results in null conditional entropy H(z|z′) = 0. Secondly, the objective of IIC is to find what is common between two data points that share redundancy,such as different images of the same object, explicitly encouraging  distillation of the common part while  ignoring the rest, i.e. instance details specific to one of the samples.This would not be possible without pairing samples.

> __Image clustering.__ IIC requires a source of paired samples (x,x′), which are often unavailable in unsupervised image clustering applications. In this case, we propose to use generated image pairs, consisting of image x and its randomly perturbed version x′=gx. The objective eq. (1) can thus be written as:maxΦ I(Φ(x),Φ(gx))


* [Deep Infomax](https://arxiv.org/abs/1808.06670)
* [Learning Representations by Maximizing Mutual Information Across Views](https://arxiv.org/abs/1906.00910)
* How Google invalidated most of the above research: [On Mutual Information Maximization for Representation Learning](https://arxiv.org/abs/1907.13625)

### Links between ResNets and ODEs
* [Beyond Finite Layer Neural Networks](https://arxiv.org/pdf/1710.10121.pdf)
* [Neural Ordinary Differential Equations](https://arxiv.org/pdf/1806.07366.pdf)
* [Invertible ResNets](https://arxiv.org/pdf/1811.00995.pdf)

### Normalizing Flows
* [Detailed hands-on introduction](https://github.com/acids-ircam/pytorch_flows)
* [PyTorch implementations of density estimation algorithms](https://github.com/kamenbliznashki/normalizing_flows)

### Transformers
* [Attention is all you need](https://arxiv.org/abs/1706.03762)
* [Links to papers and code on various transformers by HuggingFace](https://github.com/huggingface/transformers)

### Geometric deep learning
* [Good list of resources](http://geometricdeeplearning.com/)

### Probabilistic programming
* [An introduction to probabilistic programming](https://arxiv.org/abs/1809.10756)

### Miscellaneous
* [Lottery ticket hypothesis](http://news.mit.edu/2019/smarter-training-neural-networks-0506)
* [Zero-shot knowledge transfer](https://arxiv.org/abs/1905.09768)
* [SpecNet](https://arxiv.org/abs/1905.10915)

