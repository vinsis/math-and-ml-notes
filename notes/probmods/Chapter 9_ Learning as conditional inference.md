## Chapter 9: Learning as conditional inference

### Learning and the rate of learning

Let's say you see a series of heads when a coin is tossed. Your beliefs about the bias of the coin depend on two items:
1. How likely it is to see a biased coin?
2. How much data have you seen?

One can measure the rate of learning (when the inferred belief of a learner comes close to the actual fact that the coin is biased).

```javascript
var fairnessPosterior = function(observedData) {
  return Infer({method: 'enumerate'}, function() {
    var fair = flip(0.999)
    var coin = Bernoulli({p: fair ? 0.5 : 0.95})
    var obsFn = function(datum){observe(coin, datum == 'h')}
    mapData({data: observedData}, obsFn)
    return fair
  })
}
```

> If we set `fairPrior` to be 0.5, equal for the two alternative hypotheses, just 5 heads in a row are sufficient to favor the trick coin by a large margin. If `fairPrior` is 99 in 100, 10 heads in a row are sufficient. We have to increase `fairPrior` quite a lot, however, before 15 heads in a row is no longer sufficient evidence for a trick coin: even at `fairPrior` = 0.9999, 15 heads without a single tail still weighs in favor of the trick coin. This is because the evidence in favor of a trick coin accumulates exponentially as the data set increases in size; each successive h flip increases the evidence by nearly a factor of 2.
> 

---

### Independent and Exchangeable Sequences
Coin flips are i.i.d. This can be seen by conditioning the next flip result on the previous result.

Similarly the below program samples i.i.d:

```javascript
var words = ['chef', 'omelet', 'soup', 'eat', 'work', 'bake', 'stop']
var probs = [0.0032, 0.4863, 0.0789, 0.0675, 0.1974, 0.1387, 0.0277]
var thunk = function() {return categorical({ps: probs, vs: words})};
```

However the below function does not:
```javascript
var words = ['chef', 'omelet', 'soup', 'eat', 'work', 'bake', 'stop']
var probs = (flip() ?
             [0.0032, 0.4863, 0.0789, 0.0675, 0.1974, 0.1387, 0.0277] :
             [0.3699, 0.1296, 0.0278, 0.4131, 0.0239, 0.0159, 0.0194])
var thunk = function() {return categorical({ps: probs, vs: words})};
```

This is because `learning about the first word tells us something about the probs, which in turn tells us about the second word.`

The samples are not i.i.d but are `exchangeable`: `the probability of a sequence of values remains the same if permuted into any order`. 

`de Finetti’s theorem` says that, under certain technical conditions, any exchangeable sequence can be represented as follows, for some latentPrior distribution and observation function `f`:

```javascript
var latent = sample(latentPrior)
var thunk = function() {return f(latent)}
var sequence = repeat(2,thunk)
```

### Polya's urn

> Imagine an urn that contains some number of white and black balls. On each step we draw a random ball from the urn, note its color, and return it to the urn along with another ball of that color.
> 

It can be shown that the distribution of samples is exchangeable: `bbw`, `bwb`, `wbb` have the same probability; `bww`, `wbw`, `wwb` as well.

> Because the distribution is exchangeable, we know that there must be an alterative representation in terms of a latent quantity followed by independent samples. The de Finetti representation of this model is:
> 

```javascript
var urn_deFinetti = function(urn, numsamples) {
  var numWhite = sum(map(function(b){return b=='w'},urn))
  var numBlack = urn.length - numWhite
  var latentPrior = Beta({a: numWhite, b: numBlack})
  var latent = sample(latentPrior)
  return repeat(numsamples, function() {return flip(latent) ? 'b' : 'w'}).join("")
}

var urnDist = Infer({method: 'forward', samples: 10000},
                    function(){return urn_deFinetti(['b', 'w'],3)})

viz(urnDist)
```

> We sample a shared latent parameter – in this case, a sample from a Beta distribution – generating the sequence samples independently given this parameter.
> 

---

### Ideal learners

A common pattern often used in building models is:
```javascript
Infer({...}, function() {
  var hypothesis = sample(prior)
  // obsFn will usually contain an observe function
  var obsFn = function(datum){...uses hypothesis...}
  mapData({data: observedData}, obsFn)
  return hypothesis
});
```

---

### Learning a continuous parameter

Here we see how a proper definition of priors is important to mimic how humans learn. 

- When a coin shows 7/10 heads, most humans will still believe it to be fair. But if we used `uniform(0,1)` as prior, we get 0.7 as MLE. Not the same as humans.
- We can replace `uniform(0,1)` with `beta(10,10)`. But then even when the coin shows 100/100 heads, most humans will believe the coin always shows heads. But the model shows the bias to be 0.9 (instead of 1).
- This can be remedied by having a small bias for a fair coin in the prior:

```javascript
var weightPosterior = function(observedData){
  return Infer({method: 'MCMC', burn:1000, samples: 10000}, function() {
    var isFair = flip(0.999)
    var realWeight = isFair ? 0.5 : uniform({a:0, b:1})
    var coin = Bernoulli({p: realWeight})
    var obsFn = function(datum){observe(coin, datum=='h')}
    mapData({data: observedData}, obsFn)
    return realWeight
  })
}
```

> This model stubbornly believes the coin is fair until around 10 successive heads have been observed. After that, it rapidly concludes that the coin can only come up heads. The shape of this learning trajectory is much closer to what we would expect for humans.
> 

### Another example: estimating causal power

An effect E can occur due to a cause C or a background effect. We want to find out, `from observed evidence about the co-occurrence of events, attempt to infer the causal structure relating them.`

```javascript
var observedData = [{C:true, E:true}, {C:true, E:true}, {C:false, E:false}, {C:true, E:true}]

var causalPowerPost = Infer({method: 'MCMC', samples: 10000}, function() {
  // Causal power of C to cause E
  var cp = uniform(0, 1)

  // Background probability of E
  var b = uniform(0, 1)

  var obsFn = function(datum) {
    // The noisy causal relation to get E given C
    var E = (datum.C && flip(cp)) || flip(b)
    condition( E == datum.E)
  }

  mapData({data: observedData}, obsFn)

  return {causal_power: cp}
});

viz(causalPowerPost);
```