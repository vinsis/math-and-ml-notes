## Data Efficient Image Recognition with Constrastive Predictive Coding

### The basic idea
Take two different overlapping patches from a single image x<sub>1</sub> and x<sub>2</sub>. Since they came from the same image and are close to each other (since they overlap), they are related. We use a neural network (f<sub>θ</sub>) to find a representation for each of these patches, say z<sub>1</sub> and z<sub>2</sub>. Since the patches are related, z<sub>1</sub> and z<sub>2</sub> should also be related. In other words, z<sub>1</sub> should be able to _predict_ z<sub>2</sub>.

### But what do we mean by a vector z<sub>1</sub> predicting another vector z<sub>2</sub>?

Let's say we take random patches from another image, x<sub>3</sub>, x<sub>4</sub> ... x<sub>10</sub>. We calculate z<sub>3</sub> = f<sub>θ</sub>(x<sub>3</sub>) ... z<sub>10</sub> = f<sub>θ</sub>(x<sub>10</sub>). Since z<sub>1</sub> is related to z<sub>2</sub> but not to z<sub>i</sub> for i > 2, it should be able to pick out z<sub>2</sub> from a set of vectors z<sub>i</sub> for i > 1.

### But what do we mean by a vector picking out another particular vector from a set of vectors?
It is a two step process:
Step 1: A vector at time step t, v<sub>t</sub>, defines a _context_ c<sub>t</sub>. This can be done by passing v<sub>t</sub> to an autoregressive model g<sub>ar</sub>. Sometimes not just the vector at last time step but all vectors from time 0 to time t are used to define a _context_.

(v<sub>0</sub>, v<sub>1</sub>, ..., v<sub>t</sub>) → [ g<sub>ar</sub> ] → c<sub>t</sub>

g<sub>ar</sub> could be a GRU, LSTM or CNN.

Step 2: The context vector at time t, c<sub>t</sub> can predict encoded vectors k steps ahead of time, z<sub>t+k</sub> where k>0. This is done by a simple linear transformation of c<sub>t</sub>. We use a separate linear transformation W<sub>k</sub> for predict
z<sub>t+k</sub>.

In other words, W<sub>1</sub> is used to predict c<sub>t+1</sub>, W<sub>2</sub> is used to predict c<sub>t+2</sub>, W<sub>3</sub> is used to predict c<sub>t+3</sub> and so on.

c<sub>t</sub> → [ W<sub>1</sub> ] → z<sub>t+1</sub>

c<sub>t</sub> → [ W<sub>2</sub> ] → z<sub>t+2</sub>

c<sub>t</sub> → [ W<sub>3</sub> ] → z<sub>t+3</sub>
<br>...<br>
c<sub>t</sub> → [ W<sub>k</sub> ] → z<sub>t+k</sub>

### How do we measure the accuracy of prediction?
Simple: a dot product. If z<sub>t+k</sub> came from the same image, the dot product should have a high value. If it came from a different image, it should have a low value. This can be turned into a loss by passing the dot product to sigmoid function and then calculating binary cross entropy loss.

### Code
The Keras implementation is lifted straight from [here](https://github.com/davidtellez/contrastive-predictive-coding/blob/master/train_model.py).

#### f<sub>θ</sub> (image patch x<sub>t</sub> → encoded vector z<sub>t</sub>)
It is a [simple CNN](https://github.com/davidtellez/contrastive-predictive-coding/blob/master/train_model.py#L14-L36):
```
def network_encoder(x, code_size):

    ''' Define the network mapping images to embeddings '''

    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=256, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding')(x)

    return x
```

#### g<sub>ar</sub>
It is a [simple GRU](https://github.com/davidtellez/contrastive-predictive-coding/blob/master/train_model.py#L39-L47). I have modified the docstring to make it clearer:
```
def network_autoregressive(x):

    ''' x is a iterable of vectors z1, z2, ... zt '''

    # x = keras.layers.GRU(units=256, return_sequences=True)(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GRU(units=256, return_sequences=False, name='ar_context')(x)

    return x
```

#### Implementation of W<sub>k</sub> transformations
Note that `k` is the number of time steps ahead in future the encoded vectors at which are predicted. We need to define a `Dense` (or `Linear` if you are coming from PyTorch) layer for each value to k. This is implemented [here](https://github.com/davidtellez/contrastive-predictive-coding/blob/master/train_model.py#L50-L63)

```
def network_prediction(context, code_size, predict_terms):

    ''' `predict_terms` is only used to determine the number of time steps ahead of time. '''

    outputs = []
    for i in range(predict_terms):
        outputs.append(keras.layers.Dense(units=code_size, activation="linear", name='z_t_{i}'.format(i=i))(context))

    if len(outputs) == 1:
        output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
    else:
        output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(outputs)

    return output
```

#### Taking the sigmoid of dot product
[Implementation here](https://github.com/davidtellez/contrastive-predictive-coding/blob/master/train_model.py#L66-L86):

```
class CPCLayer(keras.layers.Layer):

    ''' Computes dot product between true and predicted embedding vectors '''

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):

        # Compute dot product among vectors
        preds, y_encoded = inputs
        dot_product = K.mean(y_encoded * preds, axis=-1)
        dot_product = K.mean(dot_product, axis=-1, keepdims=True)  # average along the temporal dimension

        # Keras loss functions take probabilities
        dot_product_probs = K.sigmoid(dot_product)

        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)
```

All of the above functions are bundled into a single model [here](https://github.com/davidtellez/contrastive-predictive-coding/blob/master/train_model.py#L89).

---

## Data-Efficient Image Recognition with Contrastive Predictive Coding

This paper basically made improvements on the previous implementation of CPC.

> We revisit CPC in terms of its architecture and training methodology, and arrive at a new implementation  with  a  dramatically-improved  ability  to  linearly  separate  image  classes.

Here is how they made these improvements:

1. Increasing model capacity: `the  original  CPC  model  used  only the   first   3   stacks   of   a   ResNet-101 ... we converted the third residual stack of ResNet-101  to  use  46  blocks  with4096-dimensional feature maps and 512-dimensional bottleneck layers`.

2. Replacing batch normalization with layer normalization: `We hypothesize that batch normalization allows these models to find a trivial solution to CPC: it introduces  a  dependency  between  patches  (through  the  batch  statistics)  that  can  be  exploited  to bypass the constraints on the receptive field.   Nevertheless we find that we can reclaim much of batch normalization’s training efficiency using layer normalization`.

3. Predicting not just top to bottom from from all directions: `we repeatedly predict the patch using context frombelow,  the right and the left,  resulting in up to four times as many prediction tasks.`

4. Augmenting image patches better: `The originalCPC model spatially jitters individual patches independently. We further this logic by adopting the ‘color dropping’ method of [14],  which randomly drops two of the three color channels in each patch, and find it to deliver systematic gains (+3% accuracy).  We therefore continued by adding a fixed, generic augmentation scheme using the primitives from Cubuk et al. [10] (e.g. shearing, rotation, etc), as well as random elastic deformations and color transforms [11] (+4.5% accuracy).`

There is also some material on data-efficiency but I am going to skip it.
