## Chapter 4: Causal and Statistical Dependence

### Causal Dependence
> expression A depends on expression B if it is __ever__ necessary to evaluate B in order to evaluate A
> 

What about an expression like:
```
A = C ? B + 2 : 5
```

Does `A` depend on `B`? Answer is `only in certain contexts`.

Note that `A`, `B` and `C` are evaluations of a function. This incorporates another level of subtlety:

`a specific evaluation of A might depend on a specific evaluation of B`

> However, note that if a specific evaluation of A depends on a specific evaluation of B, then any other specific evaluation of A will depend on some specific evaluation of B. Why?
> 

My interpretation is that if `A=5` depends on `B=3`, `A != 5` will depend on some value `x` in `B=x`. This implies causation as a mapping from domain of `B` (the cause) to the domain of `A` (the effect).

### Detecting Dependence Through Intervention

The idea is pretty straight-forward:
> If we manipulate A, does B tend to change?
> 

Note how `var A` is given a value directly.

>  If setting A to different values in this way changes the distribution of values of B, then B causally depends on A.

```javascript
var BdoA = function(Aval) {
  return Infer({method: 'enumerate'}, function() {
    var C = flip()
    var A = Aval //we directly set A to the target value
    var B = A ? flip(.1) : flip(.4)
    return {B: B}
  })
}

viz(BdoA(true))
viz(BdoA(false))
```

Another example:

```javascript
var cold = flip(0.02)

var cough = (cold && flip(0.5)) || (lungDisease && flip(0.5)) || flip(0.001)
```

You can set `cold = true (or false)` manually and see if it changes the distribution of `cough` (it does). But if you set `cough = true (or false)`, it does not change the distribution of `cold`.

> treating the symptoms of a disease directly doesn’t cure the disease (taking cough medicine doesn’t make your cold go away), but treating the disease does relieve the symptoms.
> 

### Statistical Dependence

It simply means
> learning information about A tells us something about B, and vice versa.
> 

> causal dependencies give rise to statistical dependencies
> 

A simple example:
```javascript
var BcondA = function(Aval) {
  return Infer({method: 'enumerate'}, function() {
    var C = flip()
    var A = flip()
    var B = A ? flip(.1) : flip(.4)
    condition(A == Aval) //condition on new information about A
    return {B: B}
  })
}

viz(BcondA(true))
viz(BcondA(false))
```

> Because the two distributions on `B` (when we have different information about `A`) are different, we can conclude that B statistically depends on `A`.
> 

Two variables can be statistically dependent even though they is no causal dependence between them. For example, if `A` and `B` are leaves of an inverted V graphical model (`Ʌ`), where the root is `C` then `A` and `B` are not causally related but statistically related.