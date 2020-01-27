## Chapter 12: Occam's Razor

Humans choose the least _complex_ hypothesis that _fits_ the data well.

- How is complexity measured? How is fitness measured?
If fitness is semantically measured and complexity is syntactically measured (eg description length of the hypothesis in some representation language, or a count of the number of free parameters used to specify the hypothesis), the two are incommensurable.

In Bayesian models both complexity and fitness are measured semantically. Complexity is measured by _flexibility_: the ability to generate a more diverse set of observations.

### Key Idea: The Law of Conservation of Belief
Since all probabilities should add up to 1, a complex model spreads its probabilities over a larger number of possibilities whereas a simple model will have high probabilities for a smaller set of events. Hence:

`P(simple hypothesis | event) > P(complex hypothesis | event)`

---

### The Size Principle
> Of hypotheses which generate data uniformly, the one with smallest extension that is still consistent with the data is the most probable.
> 

Consider the two possible hypotheses:
```javascript
Categorical({vs: ['a', 'b', 'c', 'd', 'e', 'f'], ps: [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]}) :
Categorical({vs: ['a', 'b', 'c'], ps: [1/3, 1/3, 1/3]}));
```

Let say we observe a sample to be `a`. Which hypothesis is more likely? The smaller one! In fact the smaller hypothesis is twice as likely as the first one.

```javascript
var fullData = ['a', 'b', 'a', 'b', 'b', 'a', 'b']
```

As we observe more data, how does our learning look like? Take a look here:

https://github.com/vinsis/math-and-ml-notes/blob/master/images/size_principle.svg

#### Example 2:

Now consider these two hypotheses:
```javascript
Categorical({vs: ['a', 'b', 'c', 'd'], ps: [0.375, 0.375, 0.125, 0.125]}) :
Categorical({vs: ['a', 'b', 'c', 'd'], ps: [0.25, 0.25, 0.25, 0.25]}))
```

And our observed data is:
```javascript
var observedData = ['a', 'b', 'a', 'b', 'c', 'd', 'b', 'b']
```

> The Bayesian Occam’s razor says that all else being equal the hypothesis that assigns the highest likelihood to the data will dominate the posterior. Because of the law of conservation of belief, assigning higher likelihood to the observed data requires assigning lower likelihood to other possible data.
> 

Hence the observed data is much more likely to have come from the first hypothesis.

```javascript
var hypothesisToDist = function(hypothesis) {
  return (hypothesis == 'A' ?
          Categorical({vs: ['a', 'b', 'c', 'd'], ps: [0.375, 0.375, 0.125, 0.125]}) :
          Categorical({vs: ['a', 'b', 'c', 'd'], ps: [0.25, 0.25, 0.25, 0.25]}))
}

var observedData = ['a', 'b', 'a', 'b', 'c', 'd', 'b', 'b']

var posterior = Infer({method: 'enumerate'}, function(){
  var hypothesis = flip() ? 'A' : 'B'
  mapData({data: observedData}, function(d){observe(hypothesisToDist(hypothesis),d)})
  return hypothesis
})

viz(posterior)
```

---

### Example: The Rectangle Game

Given a set of points `[(x,y)]` uniformly sampled from a rectangle, which rectangle is most likely?

By the same argument as above, the tightest fitting rectangle.