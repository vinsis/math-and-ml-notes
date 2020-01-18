## Chapter 3: Conditioning

> Much of cognition can be understood in terms of conditional inference. In its most basic form, causal attribution is conditional inference: given some observed effects, what were the likely causes? Predictions are conditional inferences in the opposite direction: given that I have observed some cause, what are its likely effects?

Inference can be done in various ways. The most basic way is rejection sampling:

```javascript
var model = function () {
    var A = flip()
    var B = flip()
    var C = flip()
    var D = A + B + C
    condition(D >= 2)
    return A
}
var dist = Infer({method: 'rejection', samples: 100}, model)
viz(dist)
```

Another way is to enumerate all possibilities and use Bayes theorem.

> In the case of a WebPPL `Infer` statement with a `condition`, `A = a` will be the “event” that the return value is `a` while `B = b` will be the event that the value passed to condition is `true`. Because each of these is a regular (unconditional) probability, they and their ratio can often be computed exactly using the rules of probability. In WebPPL the inference method 'enumerate' attempts to do this calculation (by first enumerating all the possible executions of the model):

```javascript
var model = function () {
    var A = flip()
    var B = flip()
    var C = flip()
    var D = A + B + C
    condition(D >= 2)
    return A
}
var dist = Infer({method: 'enumerate'}, model)
viz(dist)
```

### Other ways to implement `Infer`
> Much of the difficulty of implementing the WebPPL language (or probabilistic models in general) is in finding useful ways to do conditional inference—to implement `Infer`.

### Conditions and observations
You can add complex propositions to `condition` without assigning a variable to them:

```javascript
var dist = Infer(
  function () {
    var A = flip()
    var B = flip()
    var C = flip()
    condition(A + B + C >= 2)
    return A
});
viz(dist)
```

> Using `condition` allows the flexibility to build complex random expressions like this as needed, making assumptions that are phrased as complex propositions, rather than simple observations. Hence the effective number of queries we can construct for most programs will not merely be a large number but countably infinite, much like the sentences in a natural language.
> 

This will run forever:
```javascript
var model = function(){
  var trueX = sample(Gaussian({mu: 0, sigma: 1}))
  var obsX = sample(Gaussian({mu: trueX, sigma: 0.1}))
  condition(obsX == 0.2)
  return trueX
}
viz(Infer({method: 'rejection', samples:1000}, model))
```

Instead of `condition`ing, it is better to `observe`:
```javascript
var model = function(){
  var trueX = sample(Gaussian({mu: 0, sigma: 1}))
  observe(Gaussian({mu: trueX, sigma: 0.1}), 0.2)
  return trueX
}
viz(Infer({method: 'rejection', samples:1000, maxScore: 2}, model))
```

`observe` does the same thing as shown below but in a much more efficient manner:

```javascript
var x = sample(distribution);
condition(x === value);
return x;
```

> In particular, it’s essential to use `observe` to condition on the value drawn from a _continuous_ distribution.

### Factors

> In WebPPL, `condition` and `observe` are actually special cases of a more general operator: `factor`. Whereas `condition` is like making an assumption that must be `true`, then `factor` is like making a _soft_ assumption that is merely preferred to be `true`.
> 

For example if we use `observe`, `A` will always be `true`:

```javascript
var dist = Infer(
  function () {
    var A = flip(0.001)
    condition(A)
    return A
});
viz(dist)
```

But if we use `factor`, we can _tweak_ how often we want `A` to be true:

```javascript
var dist = Infer(
  function () {
    var A = flip(0.01)
    factor(A?10:0)
    return A
});
viz(dist)
```

`factor(A?10:0)` gives a much higher preference to `true` than `factor(A?5:0)` for example.

`factor(A?x:y)` technically means:

`P(A=true) = e^x / (e^x + e^y)`

### Reasoning about Tug of War

An interesting question posed here is:
> For instance, how likely is it that Bob is strong, given that he’s been on a series of winning teams?
> 

Some points to reflect on:
* If team [Bob, randomly chosen player] almost always defeats [Tom, Y], is Bob strong or is [Tom, Y] weak? Do these beliefs change as the number of matches between them goes up?
* Does the likelihood of Bob being strong go up if the team he is in defeats _different_ teams?


