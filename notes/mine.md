### Key idea

Gradient descent/ascent can be used to minimize/maximize any (differentiable) expression. If an expression E cannot be estimated directly then 
a) find a lower or upper bound for that expression, 
b) parameterize it 
c) maximize or minimize it

The idea behind variational autoencoder is the same. 

### What is the inequality

We have two inequalities:
1) Donsker Vardhan representation:<br>
D<sub>KL</sub>(P||Q) =   sup<sub>T:Ω→R</sub>E<sub>P</sub>[T]−log(E<sub>Q</sub>[e<sup>T</sup>])

2) f-divergence representation:<br>
D<sub>KL</sub>(P||Q) =   sup<sub>T:Ω→R</sub>E<sub>P</sub>[T]E<sub>Q</sub>[e<sup>T-1</sup>])

Note that the first inequality is stricter/tighter.

### Actual implementation

> ... the idea is to choose F to be the family of functions T<sub>θ</sub>:X×Z →R parametrized by a deep neural network with parameters θ∈Θ. We call this network the statistics network. We exploit the bound: <br>
> I(X;Z)≥I<sub>Θ</sub>(X,Z)<br>
>
> where I<sub>Θ</sub>(X,Z) is the _neural information measure_ defined as
>
> I<sub>Θ</sub>(X,Z) = sup<sub>θ∈Θ</sub>E<sub>P(x,z)</sub>[T<sub>Θ</sub>] - log( E<sub>P(x)P(z)</sub>[e<sup>TΘ</sup>] )<br>

#### Code

Lifted straight from [here](https://github.com/sungyubkim/MINE-Mutual-Information-Neural-Estimation-/blob/master/MINE.ipynb):

```python
def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = torch.exp(mine_net(marginal))
    mi_lb = torch.mean(t) - torch.log(torch.mean(et))
    return mi_lb, t, et
```

Here `mine_net` can be any neural network T<sub>Θ</sub>:

```python
class Mine(nn.Module):
    def __init__(self, input_size=2, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output


mine_net = Mine()
```

Pretty straight forward right? It is important to note how `joint` and `marginal` distributions are calculated.

> The expectations in Eqn. 10 are estimated using empirical samples from P(X,Z) and P(X)⊗P(Z) or by shuffling the samples from the joint distribution along the batch axis. 

When you look at samples `x`, you want the `y` samples to be independent since `y` has been marginalized out. This is what random shuffling achieves. Here is how it is implemented:

```python
def sample_batch(data, batch_size=100, sample_mode='joint'):
	'''
	data is the entire dataset
	'''
    if sample_mode == 'joint':
        indices = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        batch = data[indices]
    else:
        indices1 = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        indices2 = np.random.choice(range(data.shape[0]), size=batch_size, replace=False)
        # this step makes the two columns independent by shuffling
        batch = np.concatenate([data[indices1][:,0].reshape(-1,1), data[indices2][:,1].reshape(-1,1)], axis=1)
    return batch

```

### Correcting the bias from the stochastic gradients

Since the expectations are taken over the batch, they are biased.

> Fortunately, the bias can be reduced by replacing the estimate in the denominator by an exponential moving average. For small learning rates, this improved MINE gradient estimator can be made to have arbitrarily small bias. We found in our experiments that this improves all-around performance of MINE.

Biased loss:
```python
loss = torch.mean(t) - torch.log(torch.mean(et))
loss = loss * -1
```

Unbiasing the loss:
```python
#ma_et is initialized to 1.0
ma_et = (1-ma_rate)*ma_et + ma_rate*torch.mean(et)

loss = torch.mean(t) - torch.mean(et) / ma_et.mean().detach()
loss = loss * -1
```

### Other notes

* MINE is [strongly consistent](https://en.wikipedia.org/wiki/Consistent_estimator).

* MINE captures equitability: It is invariant to deterministic non-linear functions (as it should be).

> An important property of mutual information between random variables with relationship `Y=f(X) +σ*eps`, where `f` is a deterministic non-linear transformation and `eps` is random noise, is that it is invariant to the deterministic non-linear transformation, but should only depend on the amount of noise,`σ*eps`. 

In the experiments, `X∼U(−1,1)` and `Y=f(X) +σ*eps`,  where `f(x)∈ {x,x3,sin(x)}` and `eps∼N(0,I)`.

