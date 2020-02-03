## β-VAE

### Key idea

A variational autoencoder tries to learn the distribution of a set of latent variables `z` which encode an image `x`: q<sub>ϕ</sub>(z | x). To encourage disentanglement we try to constrain `q` to be close to a multivariate unit Gaussian distribution `N(0, I)` denoted by `p(z)`. We use KL divergence as a loss function for this:

minimize D<sub>KL</sub>(q(z|x), p(z))

We can think of `z` as a union of two disjoint subsets `v` and `w` where
* `v` is a set of latent variables which are disentangled
* `w` is a set of latent variables which may be entangled

`v`s are assumed to be conditionally independent: p(v|x) = Π<sub>i</sub>p(v<sub>i</sub> | x)

Given a distribution over latent variables, we want to maximize the log-likelihood of images in our dataset: E<sub>q(z|x)</sub>[log p<sub>θ</sub>(x|z)]

Thus we want to maximize:

- E<sub>q(z|x)</sub>[log p<sub>θ</sub>(x|z)] - β D<sub>KL</sub>(q<sub>ϕ</sub>(z|x), p(z))

> Varying β changes the degree of applied learning pressure during training, thus encouraging different learnt representations.β-VAE where β = 1 corresponds to the original VAE formulation of (Kingma & Welling, 2014).

> Since the data `x` is generated using at least some conditionally independent ground truth factors `v`, and the D<sub>KL</sub> term of the β-VAE objective function encourages conditional independence  in q<sub>φ</sub>(z|x),  we  hypothesize  that  higher  values  of β should  encourage  learning  a disentangled representation of `v`.

---

### How is the disentanglement metric evaluated?

Let's say we want to evaluate the effectiveness of disentangled factor `k` (could be scale, color etc)

* Sample two sets of latent representations v<sub>1</sub> and v<sub>2</sub>. Enforce v<sub>1</sub>[k] = v<sub>2</sub>[k]
* Create images `x` and `y` from v<sub>1</sub> and v<sub>2</sub>.
* Get latent representations of `x` and `y`: z<sub>1</sub> and z<sub>2</sub>.
* Calculate the absolute difference |z<sub>1</sub> - z<sub>2</sub>|.
* Train a linear classifier to classify `k`.

If the encoder learnt disentanglement effectively, z[k] will have low variance compared to z[i≠k] and the linear classifier should learn easily how to classify.

> The accuracy of this classifier over multiple batches is used as our disentanglement metric score.We choose a linear classifier with low VC-dimension in order to ensure it has no capacity to perform non-linear disentangling by itself.

---

### Questions

* How do we know which index corresponds to a given disentanglement factor?
* I believe the only way is to change each index and look at the results. Something similar is done [here](https://github.com/1Konny/Beta-VAE/blob/master/solver.py#L346) where the index is termed `loc=1`.

By changing the value at `loc`, we get a new `z` which is passed to the decoder and the resulting output image is collected:

[Source](https://github.com/1Konny/Beta-VAE/blob/master/solver.py#L398-L402)
```
for val in interpolation:
    z[:, row] = val
    sample = F.sigmoid(decoder(z)).data
    samples.append(sample)
    gifs.append(sample)
```

---

### Other notes from the paper

> The most informative latent units z<sub>m</sub> of β-VAE have the highest KL divergence from the unit Gaussian prior `(p(z) = N(0,I))`, while the uninformative latents have KL divergence close to zero. (on 2D shapes dataset with β=4)

> We found that larger latent `z` layer sizes `m` require higher constraint pressures (higher β values). Furthermore, the relationship of β for a given m is characterized by an inverted U curve. Whenβis too low or too high the model learns an entangled latent representation due to either too much or too little capacity in the latent `z` bottleneck.

> We also note that VAE reconstruction quality is a poor indicator of learnt disentanglement. Good disentangled representations often lead to blurry reconstructions due to the restricted capacity of the latent information channel `z`, while entangled representations often result in the sharpest reconstructions.
