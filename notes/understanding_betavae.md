## Understanding disentangling in β-VAE

### Overview
This paper puts forth explanations about why β-VAE learns disentangled representations. It looks closely at how the constraint of latent posterior being as close as possible to unit Gaussian affects learning of latent representations.

#### Points close in data space are forced to get closer in latent space as β is increased
Recall that KL(p(x;μ<sub>1</sub>, σ<sub>1</sub>), q(x;μ<sub>2</sub>, σ<sub>2</sub>)) = log(σ<sub>2</sub>/σ<sub>1</sub>) + (σ<sub>1</sub><sup>2</sup> + (μ<sub>1</sub> - μ<sub>2</sub>)<sup>2</sup> / (2σ<sub>2</sub><sup>2</sup>)) - 1/2

In order to decrease KL(p,q), we can either bring μ<sub>2</sub> close to μ<sub>1</sub> or increase the variance σ<sub>2</sub>. Increasing the variance means increasing the overlap between two distributions as shown in this figure:

![](https://github.com/vinsis/math-and-ml-notes/raw/master/images/understanding_betavae.png)

> However, a greater degree of overlap between posterior distributions will tend to result in a cost in terms of log likelihood due to their reduced average discriminability. A sample drawn from the posterior given one data point may have a higher probability under the posterior of a different data point, an increasingly frequent occurrence as overlap between the distributions is increased.

> Nonetheless, under a constraint of maximizing such overlap, the smallest cost in the log likelihood can be achieved by arranging nearby points in data space close together in the latent space. By doings o, when samples from a given posterior `q(z2|x2)` are more likely under another data point such as `x1`, the log likelihood E<sub>q(z2|x2)</sub>[logp(x2|z2)] cost will be smaller if `x1` is close to `x2` in data space.

#### Forcing independence between latent dimensions forces the model to align dimensions with components that make different contributions to reconstruction

Let us define β >> 1.

> The optimal thing to do in this scenario is to onlyencode information about the data points which can yield the most significant improvement in data log-likelihood E<sub>q(z2|x2)</sub>[logp(x2|z2)].

For example, the dSprites dataset consists of generating factors `position`, `rotation`, `scale` and `shape`. The model can increase the likelihood the most by choosing `position` over other factors under such a constraint.

>  Intuitively, when optimizing a pixel-wise decoder log likelihood, information about position will result in the most gains compared to information about any of the other factors of variation in the data, since the likelihood will vanish if reconstructed position is off by just a few pixels. Continuing this intuitive picture, we can imagine that if the capacity of the information bottleneck were gradually increased, the model would continue to utilize those extra bits for an increasingly precise encoding of position, until some point of diminishing returns is reached for position information, where a larger improvement can be obtained by encoding and reconstructing another factor of variation in the dataset, such as sprite scale.

> A smooth representation of the new factor will allow an optimal packing of the posteriors in the new latent dimension, without affecting the other latent dimensions. We note that this pressure alone would not discourage the representational axes from rotating relative to the factors.  However, given the differing contributions each factor makes to the reconstruction log-likelihood, the model will try to allocate appropriately differing average capacities to the encoding axes of each factor (e.g. by optimizing the posterior variances).But, the diagonal covariance of the posterior distribution restricts the model to doing this in different latent dimensions, giving us the second pressure, encouraging the latent dimensions to align with the factors.

### Improving disentangling in β-VAE with controlled capacity increase

With the above information available, we can start training aiming to keep `KL(q(z|x), p(z))` close  to `C=0` at the beginning. Then we gradually increase `C` so as to increase the capacity of the model to learn a more expressive representation. We stop increasing `C` when the output images are of high quality.

Thus our objective then looks something like:

### E<sub>qφ(z|x)</sub>[log<sub>pθ</sub>(x|z)]−γ*|D<sub>KL</sub>(q<sub>φ</sub>(z|x)‖p(z))−C|

Note how β has been replaced with γ.

The code looks like ([source](https://github.com/1Konny/Beta-VAE/blob/master/solver.py#L164-L168)):

```python
if self.objective == 'H':
    beta_vae_loss = recon_loss + self.beta*total_kld
elif self.objective == 'B':
    C = torch.clamp(self.C_max/self.C_stop_iter*self.global_iter, 0, self.C_max.data[0])
    beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()
```
