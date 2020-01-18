## Chapter 5: Conditional dependence

Two forms of dependence are explored in detail:

a) __Screening off__: The graphical model looks like `• ← • → •` or `• → • → •`. It is called so because if the variable(s) in the middle node are observed, the corner variables become independent. 

> Screening off is a purely statistical phenomenon. For example, consider the the causal chain model, where A directly causes C, which in turn directly causes B. Here, when we observe C – the event that mediates an indirect causal relation between A and B – A and B are still causally dependent in our model of the world: it is just our beliefs about the states of A and B that become uncorrelated. There is also an analogous causal phenomenon. If we can actually manipulate or intervene on the causal system, and set the value of C to some known value, then A and B become both statistically and causally independent (by intervening on C, we break the causal link between A and C).

b) __Explaning away__: The graphical model looks like `• → • ← •`. If the bottom variable is observed, previously independent variables (the two roots at the top) become dependent. 

> The most typical pattern of explaining away we see in causal reasoning is a kind of anti-correlation: the probabilities of two possible causes for the same effect increase when the effect is observed, but they are conditionally anti-correlated, so that observing additional evidence in favor of one cause should lower our degree of belief in the other cause. (This pattern is where the term explaining away comes from.)
> 

### Non-monotonic Reasoning

> In formal logic, a theory is said to be monotonic if adding an assumption (or formula) to the theory never reduces the set of conclusions that can be drawn. 
> For instance, if I tell you that Tweety is a bird, you conclude that he can fly; if I now tell you that Tweety is an ostrich you retract the conclusion that he can fly.
> 

> Another way to think about monotonicity is by considering the trajectory of our belief in a specific proposition, as we gain additional relevant information.
> 

#### Traditional logic
> there are only three states of belief: true, false, and unknown. As we learn more about the world, maintaining logical consistency requires that our belief in any proposition only move from unknown to true or false. __That is our “confidence” in any conclusion only increases.__
> 

#### Probabilistic approach
> We can think of confidence as a measure of how far our beliefs are from a uniform distribution. Our confidence in a proposition can both increase and decrease. 
> 

### Example: Trait Attribution

If a student fails in an example, it could be due to two reasons:
1. Non-personal reason: the exam was not fair. It was too difficult.
2. Person reason: the student did not do their homework.

```javascript
var examPosterior = Infer({method: 'enumerate'}, function() {
  var examFair = flip(.8)
  var doesHomework = flip(.8)
  var pass = flip(examFair ?
                  (doesHomework ? 0.9 : 0.4) :
                  (doesHomework ? 0.6 : 0.2))
  condition(!pass)
  return {doesHomework: doesHomework, examFair: examFair}
})

viz.marginals(examPosterior)
viz(examPosterior)
```

> whether a student does homework has a greater influence on passing the test than whether the exam is fair. This in turns means that when inferring the cause of a failed exam, the model tends to attribute it to the person property (not doing homework) over the situation property (exam being unfair). This asymmetry is an example of the fundamental attribution bias ([Ross, 1977](https://scholar.google.com/scholar?q=%22The%20intuitive%20psychologist%20and%20his%20shortcomings%3A%20Distortions%20in%20the%20attribution%20process%22)): we tend to attribute outcomes to personal traits rather than situations.
> 

The above example can be modified to gather evidence from several students and several exams:

```javascript
var examPosterior = Infer({method: 'enumerate'}, function() {
  var examFair = mem(function(exam){return flip(0.8)})
  var doesHomework = mem(function(student){return flip(0.8)})

  var pass = function(student, exam) {
    return flip(examFair(exam) ?
                (doesHomework(student) ? 0.9 : 0.4) :
                (doesHomework(student) ? 0.6 : 0.2))
  };

  condition(!pass('bill', 'exam1'))

  return {doesHomework: doesHomework('bill'), examFair: examFair('exam1')}
})

viz.marginals(examPosterior)
viz(examPosterior)
```

### Visual Perception of Surface Color

Explaining away can be used to explain an optical illusion: `the checker shadow illusion`. In the figure below, squares A and B are the same shade of gray.

![Source: https://probmods.org](https://probmods.org/assets/img/Checkershadow_illusion_small.png)

The key idea is that the expression:
`var luminance = reflectance * illumination`
induces a graphical model like so:
`reflectance → luminance ← illumination`.

> The visual system has to determine what proportion of the `luminance` is due to `reflectance` and what proportion is due to the `illumination` of the scene.

When the cylinder is present, we perceive the `illumination` to be low. Hence the observed `luminance` is `explained away` by a higher perceived `reflectance`.

> The presence of the cylinder is providing evidence that the illumination of square B is actually less than that of square A (because it is expected to cast a shadow). Thus we perceive square B as having higher reflectance since its luminance is identical to square A and we believe there is less light hitting it.