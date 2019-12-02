## On mutual information maximization for representation learning

### Key idea
This paper challenged the (implicit) assumption that maximizing mutual information between encoder outputs leads to better representations. According to the paper, good representations are obtained when the encoder __discards__ some information (mostly noise). In other words, the encoder is non-invertible.

### Setup
* Take an image from MNIST
* From each image, create two inputs: one from upper half and one from lower half
* Maximize the MI between the encoder outputs of the above two inputs

Note that exact MI is hard to estimate. But there are many estimates of MI. Some estimates have a _tight bound_ meaning they are more accurate than those with a _loose bound_.

### Types of encoders used
* First they used encoders which could be invertible and non-invertible.

> We first consider encoders which are bijective by design. Even though the true MI is maximized for any choice of model parameters, the representation quality (measured by downstream linear classification accuracy) improves during training. Furthermore, there exist invertible encoders for which the representation quality is worse than using raw pixels, despite also maximizing MI.

They did this by using adversarial training. In other words, the encoder tries to come up with representations such that:
* the MI between representations is high
* the linear separability of representations is low

The fact that it is possible to successfully train such an encoder shows that high MI doesn't necessary mean high linear separability.

> We next consider encoders that can model both invertible and non-invertible functions. When the encoder can be non-invertible, but is initialized to be invertible, I<sub>EST</sub> still biases the encoders to be very ill-conditioned and hard to invert.

### Bias towards hard to invert encoders

The authors wanted to measure how _non-invertible_ the encoders got during training. They used a metric called `condition number` to measure the level of non-invertibility. The higher this number, the harder it is to invert the encoder.

#### Condition number
Condition number is the ratio σ<sub>largest</sub>/σ<sub>smallest</sub> where

σ<sub>largest</sub> and σ<sub>smallest</sub> are the largest and smallest singular values of Jacobian of `g(x)` where `g()` is the function represented by the encoder and `x` is the input.

Actually they computed the log of condition number:<br />
log(σ<sub>largest</sub>) - log(σ<sub>smallest</sub>)

#### What if the data occupied only a subspace of the entire space?
In this case, the Jacobian matrix would be singular (and condition number would be really really big). However the encoder might still be able to invert the output if the transformation of the subspace was non-singular. To deal with this problem, they added the same noise vector to `x1` and `x2` to ensure that the inputs spanned the entire space.

Below are some snippets of code taken from the [official implementation](https://github.com/google-research/google-research/blob/master/mutual_information_representation_learning/mirl.ipynb):

```python
from tensorflow.python.ops.parallel_for import gradients
x_1, x_2, _ = processed_train_data(data_dimensions, batch_size)

# to make sure x_1 and x_2 were not limited to a subspace
if noise_std > 0.0:
  assert x_1.shape == x_2.shape, "X1 and X2 shapes must agree to add noise!"
  noise = noise_std * tf.random.normal(x_1.shape)
  x_1 += noise
  x_2 += noise

code_1, code_2 = g1(x_1), g2(x_2)
if compute_jacobian:
    jacobian = gradients.batch_jacobian(code_1, x_1, use_pfor=False)
    singular_values = tf.linalg.svd(jacobian, compute_uv=False)

...
...
...

for run_number, results in enumerate(results_all_runs):
      stacked_singular_values = np.stack(results.singular_values)
      sorted_singular_values = np.sort(stacked_singular_values, axis=-1)
      log_condition_numbers = np.log(sorted_singular_values[..., -1]) \
                              - np.log(sorted_singular_values[..., 0])
      condition_numbers_runs.append(log_condition_numbers)
```

Here's what they found:

> Moreover, even though `g1` is initialized very close to the identity function (which maximizes the true MI), the condition number of its Jacobian evaluated at inputs randomly sampled from the data-distribution steadily deteriorates over time, suggesting that in practice (i.e. numerically)inverting the model becomes increasingly hard.

### Critics

Critics are basically functions (neural networks) used to predict whether or not two representations (vectors) come from the same image. They compared three critic architectures:
`bilinear`, `separable` and `MLP`. Below are some implementations of critics:

```python
class InnerProdCritic(tf.keras.Model):
  def call(self, x, y):
    return tf.matmul(x, y, transpose_b=True)

class BilinearCritic(tf.keras.Model):
  def __init__(self, feature_dim=100, **kwargs):
    super(BilinearCritic, self).__init__(**kwargs)
    self._W = tfkl.Dense(feature_dim, use_bias=False)

  def call(self, x, y):
    return tf.matmul(x, self._W(y), transpose_b=True)

class ConcatCritic(tf.keras.Model):
  def __init__(self, hidden_dim=200, layers=1, activation='relu', **kwargs):
    super(ConcatCritic, self).__init__(**kwargs)
    # output is scalar score
    self._f = MLP([hidden_dim for _ in range(layers)]+[1], False, {"activation": "relu"})

  def call(self, x, y):
    batch_size = tf.shape(x)[0]
    # Tile all possible combinations of x and y
    x_tiled = tf.tile(x[None, :],  (batch_size, 1, 1))
    y_tiled = tf.tile(y[:, None],  (1, batch_size, 1))
    # xy is [batch_size * batch_size, x_dim + y_dim]
    xy_pairs = tf.reshape(tf.concat((x_tiled, y_tiled), axis=2),
                          [batch_size * batch_size, -1])
    # Compute scores for each x_i, y_j pair.
    scores = self._f(xy_pairs)
    return tf.transpose(tf.reshape(scores, [batch_size, batch_size]))


class SeparableCritic(tf.keras.Model):
  def __init__(self, hidden_dim=100, output_dim=100, layers=1,
               activation='relu', **kwargs):
    super(SeparableCritic, self).__init__(**kwargs)
    self._f_x = MLP([hidden_dim for _ in range(layers)] + [output_dim], False, {"activation": activation})
    self._f_y = MLP([hidden_dim for _ in range(layers)] + [output_dim], False, {"activation": activation})

  def call(self, x, y):
    x_mapped = self._f_x(x)
    y_mapped = self._f_y(y)
    return tf.matmul(x_mapped, y_mapped, transpose_b=True)
```

Here's what they found:
> It can be seen that for both lower bounds, representations trained with the MLP critic barely outperform the baseline on pixel space, whereas the same lower bounds with bilinear and separable critics clearly lead to a higher accuracy than the baseline.

### Connection to deep metric learning and triplet losses

After decoupling representation quality and MI maximization, the authors made a connection between representation quality and triplet losses.

#### The metric learning view
> Given sets of triplets, namely an anchor point `x`, a positive instance `y`, and a negative instance `z`, the goal is to learn a representation `g(x)` such that the distances between `g(x)` and `g(y)` is smaller than the distance between `g(x)` and `g(z)`, for each triplet.

They make this association in two ways:
a) mathematically formulating the critic objective function and drawing parallels with the triplet loss function

and

b)emphasizing the importance of negative sampling. I didn't spend too much time trying to understand it so will not provide a gist here.
