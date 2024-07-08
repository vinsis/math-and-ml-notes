# Neural Redshift: Random Networks are not Random Functions

- [Arxiv PDF](https://arxiv.org/pdf/2403.02241)

"The success of deep learning is thus not a product primarily of GD, nor is it universal to all architectures. This paper propose an explanation compatible with all above observations. It builds on the growing evidence that NNs benefit from their parametrization and the structure of their weight space."

## Contributions
1. NNs are biased to implement functions of a particular level of complexity (not necessarily low) determined
by the architecture.
2. This preferred complexity is observable in networks with random weights from an uninformed prior.
3. Generalization is enabled by popular components like ReLUs setting this bias to a low complexity that often aligns with the target function.

> ... the parameter space of NNs is biased towards functions of low frequency, one of the measures of complexity used in this work.

That is why it is called Neural _Redshift_. They examine various architectures with random weights. They use three measures of complexity:
1. decompositions in Fourier series (low frequency)
2. in the bases of orthogonal polynomials (low order)
3. compressibility with an approximation of Kolmogorov complexity (compressibility)
They are collectively referred to as _simplicity_.

> We show that the simplicity bias is not universal but depends on common components (ReLUs, residual connections, layer normalizations). ReLU networks also have the unique property of maintaining their simplicity bias for any depth and weight magnitudes. It suggests that the historical importance of ReLUs in the development of deep learning goes beyond the common narrative about vanishing gradients.

### Analyzing random networks
- Weights and biases are uniformly sampled (not so important)
- The model maps to a function $R^2 -> R$. A 2D grid of points is used as input. This allows visualization of the function as a grayscale image.

- ReLU-like activations (GELU, Swish, SELU [16]) are also biased towards low complexity. Unlike ReLUs, close examination in Appendix F shows that increasing depth or weight magnitudes slightly increases the complexity.
- Others activations (TanH, Gaussian, sine) show completely different behaviour. Depth and weight magnitude cause a dramatic increase in complexity. Unsurprisingly, these activations are only used in special applications [58] with careful initializations [68]. Networks with these activations have no fixed preferred complexity independent of the weights’ or activations’ magnitudes.

![](https://github.com/vinsis/math-and-ml-notes/blob/30d38f7320ade8e4860d9166d2e0b157c0a7636b/images/redshift_table1.png)

- Width has no impact on complexity, perhaps surprisingly. Additional neurons change the capacity of a model (what can be represented after training) but they do not affect its inductive biases.
- __Layer normalization__: We place layer normalizations before each activation. Layer normalization has the significant effect of removing variations in complexity with the weights’ magnitude for all activations (Figure 5). The weights can now vary (e.g. during training) without directly affecting the preferred complexity of the architecture.
- Residual connections: This has the dramatic effect of forcing the preferred complexity to some of the lowest levels for all activations regardless of depth.
- __Multiplicative interactions__: They refer to multiplications of internal representations with one another [39] as in attention layers, highway networks, dynamic convolutions, etc. We place them in our MLPs as gating operations, such that each hidden layer corresponds to: x← ϕ(Wx+b) ⊙ σ(W\′x+b\′)
where `σ(·)` is the logistic function. This creates a clear increase in complexity dependent on depth and weight magnitude, even when ReLU is used.

![](https://github.com/vinsis/math-and-ml-notes/blob/fa0163f771c352a665fcd3c4f3cb3607eb5d7451/images/redshift_fig5.png)

- Unbiased model: This is build by creating a uniform bias over frequencies. The inverse Fourier transform is a weighted sum of sine waves, so this architecture can be implemented as a one-layer MLP with sine activations and fixed input weights representing each one Fourier component. This architecture behaves very differently from standard MLPs (Figure 4). With random weights, its Fourier spectrum is uniform, which gives a high complexity for any weight magnitude (depth is fixed). Functions implemented by this architecture look like white noise.

## Indictive biases in trained models
There is a strong correlation between the complexity at initialization (i.e. with random weights as examined in the previous section) and in the trained model. We will also see that unusual architectures with a bias towards high complexity can improve generalization on tasks where the standard “simplicity bias” is suboptimal.

### Learning complex functions

## Experimental setup
The input to our task is a vector of integers $x ∈ [0, N-1]^d$ and output is $ \Sigma{x_i} \le (M/2)\text{mod M} $. "We consider three versions with N = 16 and M ={10, 7, 4} that correspond to increasingly higher frequencies in the target function". ReLU MLP solves only the low-frequency version of this task.

> We then train MLPs with other
activations (TanH, Gaussian, sine) whose preferred complexity is sensitive to the activations’ magnitude. We also introduce a constant multiplicative prefactor before each activation function to modulate this bias without changing the weights’ magnitude, which could introduce optimization side effects. Some of these models succeed in learning all versions of the task when the prefactor is correctly tuned. For higher-frequency versions, the prefactor needs to be larger to shift the bias towards higher complexity.

## Impact on shortcut learning
### Experimental setup
> We consider a regression task similar to Colored-MNIST. Inputs are images of handwritten digits juxtaposed with a uniform band of colored pixels that simulate spurious features. The labels in the training data are values in [0, 1] proportional to the digit value as well as to the color intensity. Therefore, a model can attain high training accuracy by relying either on the simple linear relation with the color, or the more complex recognition of the digits (the target task). To measure the reliance of a model on color or digit, we use two test sets where either the color or digit is correlated with the label while the other is randomized.

> We see in Figure 9 that the LZ complexity at initialization increases with prefactor values for TanH, Gaussian, and sine activations. Most interestingly, the accuracy on the digit and color also varies with the prefactor. The color is learned more easily with small prefactors (corresponding to a low complexity at initialization) while the digit is learned more easily at an intermediate value (corresponding to medium complexity at initialization). The best performance on the digit is reached at a sweet spot that we explain as the hypothesized “best match” between the complexity of the target function, and that preferred by the architecture. With larger prefactors, i.e. beyond this sweet spot, the accuracy on the digit decreases, and even more so with sine activations for which the complexity also increases more rapidly, further supporting the proposed explanation.