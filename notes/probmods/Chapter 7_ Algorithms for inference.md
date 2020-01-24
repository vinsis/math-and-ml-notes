## Chapter 7: Algorithms for inference

### Markov Chain Monte Carlo (MCMC)
The idea is to find a Markov Chain whose stationary distribution is the same as the conditional distrbution we want to estimate. Eg. we want to estimate a geometric distribution:

```javascript
var p = .7

var geometric = function(p){
	return ((flip(p) == true) ? 1 : (1 + geometric(p)))
}

var post = Infer({method: 'MCMC', samples: 25000, lag: 10, model: function(){
	var mygeom = geometric(p);
	condition(mygeom>2)
	return(mygeom)
	}
})

viz.table(post)
```

The distribution of the above variable is the same as the stationary distribution of the Markov Chain shown here:

```javascript
var p = 0.7

var transition = function(state){
	return (state == 3 ? sample(Categorical({vs: [3, 4], ps: [(1 - 0.5 * (1 - p)), (0.5 * (1 - p))]})) :
						    sample(Categorical({vs: [(state - 1), state, (state + 1)], ps: [0.5, (0.5 - 0.5 * (1 - p)), (0.5 * (1 - p))]})))
}

var chain = function(state, n){
	return (n == 0 ? state : chain(transition(state), n-1))
}

var samples = repeat(5000, function() {chain(3, 250)})
viz.table(samples)
```

> As we have already seen, each successive sample from a Markov chain is highly correlated with the prior state.
> 

This can take long if the initial state was not bad. However in the long run these local correlations disappear. But we will get a large collection of samples. In order to get samples which are not strongly connected to each other, we can every `n`th sample.

> WebPPL provides an option for MCMC called `'lag'`.
> 
> Fortunately, it turns out that for any given (condition) distribution we might want to sample from, there is at least one Markov chain with a matching stationary distribution.
> 

---

### Metropolis-Hastings
> To create the necessary transition function, we first create a proposal distribution, `q(x→x′)`. A common option for continuous state spaces is to sample a new state from a multivariate Gaussian centered on the current state.
> 

Once you have defined `q` and `p(x)`, the target distribution, calculate the ratio `A / B` where:
- `A = p(x′) * q(x′ → x)` &
- `B = p(x) * q(x → x′)`

If `A/B` is bigger than 1, round it down to 1. Flip a coin with a probability of `A/B` of showing heads.

If heads shows up, transition from `x` to `x′`. Otherwise, stay in the current state `x`.

#### Balance condition and detailed balance condition
Balance condition is achieved when a Markov chain reaches a stationary state:
> p(x′) = ∑<sub>x</sub>p(x) π(x → x′) where π is the transition distribution.
>
 
A stronger condition is the detailed balance condition:
> p(x) π(x → x′) = p(x′) π(x′ → x)
> 
It is stronger in the sense that detailed balance condition implies balance condition.

It can be shown that MH algorithm gives a transition distribution π(x → x′) that satisfies detailed balance equation.

Let's look at an example:

```javascript
var p = 0.7

//the target distribution (not normalized):
//prob = 0 if x condition is violated, otherwise proportional to geometric distribution

// we want to sample from this distribution
// it is not easy to see what this distribution looks like just by looking at the formula below
// so how do we sample from it? This is where MH algorithm helps us
var target_dist = function(x){
  return (x < 3 ? 0 : (p * Math.pow((1-p),(x-1))))
}

// the proposal function and distribution,
// here we're equally likely to propose x+1 or x-1.

// this function decides where to go in the next state
var proposal_fn = function(x){
  return (flip() ? x - 1 : x + 1)
}
var proposal_dist = function (x1, x2){
  return 0.5
}

// the MH recipe:
var accept = function (x1, x2){
  let p = Math.min(1, (target_dist(x2) * proposal_dist(x2, x1)) / (target_dist(x1) * proposal_dist(x1,x2)))
  return flip(p)
}
var transition = function(x){
  // decide where to go in the next step
  let proposed_x = proposal_fn(x)
  // decide whether to go or not
  return (accept(x, proposed_x) ? proposed_x : x)
}

//the MCMC loop:
var mcmc = function(state, iterations){
  return ((iterations == 1) ? [state] : mcmc(transition(state), iterations-1).concat(state))
}

var chain = mcmc(3, 10000) // mcmc for conditioned geometric
```

---

### Hamiltonian Monte Carlo

Sometimes the MHMC will get stuck if it cannot find states which have a high probability (in other words the ratio `A/B` is really small.) For example if we want to find ten numbers randomly sampled from `uniform(0,1)` and we use a Gaussian distribution for transition probabilities it will likely get stuck:

```javascript
var constrainedSumModel = function() {
  var xs = repeat(10, function() {
    return uniform(0, 1);
  });
  observe(Gaussian({mu: 5, sigma: 0.005}), sum(xs));
  return map(bin, xs);
};
```

The acceptance ratio in this case is 1-2%.Hamiltonian Monte Carlo solves this problem by calcuating `the gradient of the posterior with respect to the random choices made by the program`. The book does not explain very well how it works.

It also does not go into detail how particle filter works.

---

### Variational Inference

In contrast to non-parametric methods mentioned above, VI is a parametric method. __Mean-field variational inference__ tries to find an optimized set of parameters by `approximating the posterior with a product of independent distributions (one for each random choice in the program).`

```javascript
var trueMu = 3.5
var trueSigma = 0.8

var data = repeat(100, function() { return gaussian(trueMu, trueSigma)})

var gaussianModel = function() {
  var mu = gaussian(0, 20)
  var sigma = Math.exp(gaussian(0, 1)) // ensure sigma > 0
  map(function(d) {
    observe(Gaussian({mu: mu, sigma: sigma}), d)
  }, data)
  return {mu: mu, sigma: sigma}
};
```

> By default, it takes the given arguments of random choices in the program (in this case, the arguments `(0, 20)` and `(0, 1)` to the two gaussian random choices used as priors) and replaces with them with free parameters which it then optimizes to bring the resulting distribution as close as possible to the true posterior... __the mean-field approximation necessarily fails to capture correlation between variables.__
> 

For example it fails to capture the correlation between `reflectance` and `illumination`.

```javascript
var observedLuminance = 3;
                            
var model = function() {
  var reflectance = gaussian({mu: 1, sigma: 1})
  var illumination = gaussian({mu: 3, sigma: 1})
  var luminance = reflectance * illumination
  observe(Gaussian({mu: luminance, sigma: 1}), observedLuminance)
  return {reflectance: reflectance, illumination: illumination}
}
```

