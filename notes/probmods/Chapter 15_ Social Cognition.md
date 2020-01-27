## Chapter 15: Social Cognition

### Prelude: Two ways to look at the same problem

> Imagine a factory where the widget-maker makes a stream of widgets, and the widget-tester removes the faulty ones. You don’t know what tolerance the widget tester is set to, and wish to infer it.
> 

#### Way 1 to create `n` widgets:
Sample a widget from a distribution of widgets. Condition on the widget passing the test.
- If it passes the test, create `n-1` widgets recursively
- If it fails, create `n` widgets recursively

#### Way 2 to create `n` widgets:
Sample `n` widgets at the same time from a distribution of widgets. Condition on _all_ widgets passing the test.

Way 1:
```javascript
var makeWidgetSeq = function(numWidgets, threshold) {
  if(numWidgets == 0) {
    return [];
  } else {
    var widget = sample(widgetMachine);
    return (widget > threshold ? 
            [widget].concat(makeWidgetSeq(numWidgets - 1, threshold)) : 
            makeWidgetSeq(numWidgets, threshold));
  }
}

var widgetDist = Infer({method: 'rejection', samples: 300}, function() {
  var threshold = sample(thresholdPrior);
  var goodWidgetSeq = makeWidgetSeq(3, threshold);
  condition(_.isEqual([.6, .7, .8], goodWidgetSeq))
  return [threshold].join("");
})
```

Way 2:
```javascript
var makeGoodWidgetSeq = function(numWidgets, threshold) {
  return Infer({method: 'enumerate'}, function() {
    var widgets = repeat(numWidgets, function() {return sample(widgetMachine)});
    condition(all(function(widget) {return widget > threshold}, widgets));
    return widgets;
  })
}
```

> #### Rather than thinking about the details inside the widget tester, we are now abstracting to represent that the machine correctly chooses a good widget
> 

We don't know the behavior of widget testing machine. So we think of testing at an abstract level and infer to maximize what we want.

---

### Social Cognition

> An agent tends to choose actions that she expects to lead to outcomes that satisfy her goals.
> 

Let's say Sally wants to buy a cookie from a __deterministic__ vending machine. This is how it works:

```javascript
var vendingMachine = function(state, action) {
  return (action == 'a' ? 'bagel' :
          action == 'b' ? 'cookie' :
          'nothing');
}
```

Here is how actions are chosen:
```javascript
var chooseAction = function(goalSatisfied, transition, state) {
  return Infer(..., function() {
    var action = sample(actionPrior)
    condition(goalSatisfied(transition(state, action)))
    return action;
  })
}
```

where `transition` is the output of taking `action` in `state`.

She is clearly always press `b` to get the cookie. Now let's say the vending machine is not realistic:

```javascript
var vendingMachine = function(state, action) {
  return (action == 'a' ? categorical({vs: ['bagel', 'cookie'], ps: [.9, .1]}) :
          action == 'b' ? categorical({vs: ['bagel', 'cookie'], ps: [.1, .9]}) :
          'nothing');
}
```

We see Sally still presses `b` most of the time (but not every time).

> Technically, this method of making a choices is not optimal, but rather it is soft-max optimal (also known as following the “Boltzmann policy”).
> 

Here is how we would represent the whole thing:

```javascript
///fold:
var actionPrior = Categorical({vs: ['a', 'b'], ps: [.5, .5]})
var haveCookie = function(obj) {return obj == 'cookie'};
///
var vendingMachine = function(_, action) {
  return (action == 'a' ? categorical({vs: ['bagel', 'cookie'], ps: [.9, .1]}) :
          action == 'b' ? categorical({vs: ['bagel', 'cookie'], ps: [.1, .9]}) :
          'nothing');
}

var chooseAction = function(goalSatisfied, transition, state) {
  return Infer({method: 'enumerate'}, function() {
    var action = sample(actionPrior)
    condition(goalSatisfied(transition(_, action)))
    return action;
  })
}

viz.auto(chooseAction(haveCookie, vendingMachine, '_'));
```

---

### Goal Inference
Let's say we don't know what Sally wants but we observe her pressing `b`. How can we infer what she wants?

Here we don't know what the goal is. So we our `goalSatisfied` becomes probabilistic instead of deterministic:

```javascript
var goal = categorical({vs: ['bagel', 'cookie'], ps: [.5, .5]})
var goalSatisfied = function(outcome) {return outcome == goal};
```

We randomly sample `goalSatisfied` and then for that goal, infer the best action. 

We draw an inference on `goal` by observing that the chosen action was `b`:

```javascript
var goalPosterior = Infer({method: 'enumerate'}, function() {
  var goal = categorical({vs: ['bagel', 'cookie'], ps: [.5, .5]})
  var goalSatisfied = function(outcome) {return outcome == goal};
  var actionDist = chooseAction(goalSatisfied, vendingMachine, 'state')
  factor(actionDist.score('b'));
  return goal;
})
```

Note how we are doing inference inside an inference here.

Now let's say the button `b` gives any of the two options equally probably:

```javascript
var vendingMachine = function(state, action) {
  return (action == 'a' ? categorical({vs: ['bagel', 'cookie'], ps: [.9, .1]}) :
          action == 'b' ? categorical({vs: ['bagel', 'cookie'], ps: [.5, .5]}) :
          'nothing');
}
```

> Despite the fact that button b is equally likely to result in either bagel or cookie, we have inferred that Sally probably wants a cookie. This is a result of the inference implicitly taking into account the counterfactual alternatives: if Sally had wanted a bagel, she would have likely pressed button a. 
> 

---

### Preferences

Let's say we observe Sally pressing `b` several times. We don't know what she wants but we do know she has some preference. In this case this is how we define the `goal`:

```javascript
var preference = uniform(0, 1);
var goalPrior = function() {return flip(preference) ? 'bagel' : 'cookie'};
var makeGoal = function(food) {return function(outcome) {return outcome == food}};
```

... and this is how we condition:

```javascript
var goalPosterior = Infer({method: 'MCMC', samples: 20000}, function() {
  var preference = uniform(0, 1);
  var goalPrior = function() {return flip(preference) ? 'bagel' : 'cookie'};
  var makeGoal = function(food) {return function(outcome) {return outcome == food}};
  condition((sample(chooseAction(makeGoal(goalPrior()), vendingMachine, 'state')) == 'b') &&
            (sample(chooseAction(makeGoal(goalPrior()), vendingMachine, 'state')) == 'b') &&
            (sample(chooseAction(makeGoal(goalPrior()), vendingMachine, 'state')) == 'b'));
  return goalPrior();
})
```

---


### Epistemic States

When we defined the vending machine like so:
```javascript
var vendingMachine = function(state, action) {
  return (action == 'a' ? categorical({vs: ['bagel', 'cookie'], ps: [.9, .1]}) :
          action == 'b' ? categorical({vs: ['bagel', 'cookie'], ps: [.5, .5]}) :
          'nothing');
}
```

we made the assumption that we knew how the vending machine worked. What if don't know how it works? Then we can replace `ps: [.9, .1]` and `ps: [.5, .5]` by a distribution:

```javascript
var makeVendingMachine = function(aEffects, bEffects) {
  return function(state, action) {
    return (action == 'a' ? categorical({vs: ['bagel', 'cookie'], ps: aEffects}) :
            action == 'b' ? categorical({vs: ['bagel', 'cookie'], ps: bEffects}) :
            'nothing');
  }
};

var aEffects = dirichlet(Vector([1,1]))
var bEffects = dirichlet(Vector([1,1]))
var vendingMachine = makeVendingMachine(aEffects, bEffects);
```

Now if we assume that Sally knows how it works, she does not need to `Infer` it. Thus

> We can capture this by placing uncertainty on the vending machine, inside the overall query but “outside” of Sally’s inference:
> 
```javascript
///fold:
var actionPrior = Categorical({vs: ['a', 'b'], ps: [.5, .5]})

var chooseAction = function(goalSatisfied, transition, state) {
  return Infer({method: 'enumerate'}, function() {
    var action = sample(actionPrior)
    condition(goalSatisfied(transition(state, action)))
    return action;
  })
}
///
var makeVendingMachine = function(aEffects, bEffects) {
  return function(state, action) {
    return (action == 'a' ? categorical({vs: ['bagel', 'cookie'], ps: aEffects}) :
            action == 'b' ? categorical({vs: ['bagel', 'cookie'], ps: bEffects}) :
            'nothing');
  }
};

var goalPosterior = Infer({method: 'MCMC', samples: 50000}, function() {
  var aEffects = dirichlet(Vector([1,1]))
  var bEffects = dirichlet(Vector([1,1]));

  var vendingMachine = makeVendingMachine(aEffects, bEffects);
  
  var goal = categorical({vs: ['bagel', 'cookie'], ps: [.5, .5]})
  var goalSatisfied = function(outcome) {return outcome == goal};
  
  condition(goal == 'cookie' &&
            sample(chooseAction(goalSatisfied, vendingMachine, 'state')) == 'b');
  return T.get(bEffects, 1);
})

print("probability of action 'b' giving cookie")
viz.auto(goalPosterior);
```

Observe the below line:

```javascript
condition(goal == 'cookie' &&
            sample(chooseAction(goalSatisfied, vendingMachine, 'state')) == 'b');
  return T.get(bEffects, 1);
```

We are basically asking:
> Assuming Sally knows how the machine works and she wants a cookie and she is seen pressing `b`,  what is the probability of action 'b' giving cookie?
> 

> Now imagine a vending machine that has only one button, but it can be pressed many times. We don’t know what the machine will do in response to a given button sequence. We do know that pressing more buttons is less a priori likely.
> 
```javascript
var actionPrior = function() {
  return categorical({vs: ['a', 'aa', 'aaa'], ps:[0.7, 0.2, 0.1] })
}
```

The vending machine is now defined as:
```javascript
  var buttonsToOutcomeProbs = {'a': T.toScalars(dirichlet(ones([2,1]))),
                               'aa': T.toScalars(dirichlet(ones([2,1]))),
                               'aaa': T.toScalars(dirichlet(ones([2,1])))}
  var vendingMachine = function(state, action) {
    return categorical({vs: ['bagel', 'cookie'], ps: buttonsToOutcomeProbs[action]})
  }
```

We first condition on Sally pressing `a` to get a `cookie`:
```javascript
condition(goal == 'cookie' && chosenAction == 'a')
```

Then we compare it with Sally pressing `aa` to get a `cookie`:
```javascript
condition(goal == 'cookie' && chosenAction == 'aa')
```

>  Why can we draw much stronger inferences about the machine when Sally chooses to press the button twice? When Sally does press the button twice, she could have done the “easier” (or rather, a priori more likely) action of pressing the button just once. Since she doesn’t, a single press must have been unlikely to result in a cookie. This is an example of the principle of efficiency—all other things being equal, an agent will take the actions that require least effort (and hence, when an agent expends more effort all other things must not be equal).
>  

> In these examples we have seen two important assumptions combining to allow us to infer something about the world from the indirect evidence of an agents actions. The first assumption is the principle of rational action, the second is an assumption of knowledgeability—we assumed that Sally knows how the machine works, though we don’t. Thus inference about inference, can be a powerful way to learn what others already know, by observing their actions.
> 

### Joint inference about beliefs and desires
> Suppose we condition on two observations: that Sally presses the button twice, and that this results in a cookie. Then, assuming that she knows how the machine works, we jointly infer that she wanted a cookie, that pressing the button twice is likely to give a cookie, and that pressing the button once is unlikely to give a cookie.
> 
```javascript
var actionPrior = function() {
  return categorical({vs: ['a', 'aa', 'aaa'], ps:[0.7, 0.2, 0.1] })
}

var goalPosterior = Infer({method: 'rejection', samples: 5000}, function() {
  var buttonsToOutcomeProbs = {'a': T.toScalars(dirichlet(ones([2,1]))),
                               'aa': T.toScalars(dirichlet(ones([2,1]))),
                               'aaa': T.toScalars(dirichlet(ones([2,1])))}
  
  var vendingMachine = function(state, action) {
    return categorical({vs: ['bagel', 'cookie'], ps: buttonsToOutcomeProbs[action]})
  }
  
  var goal = categorical({vs: ['bagel', 'cookie'], ps: [.5, .5]})
  var goalSatisfied = function(outcome) {return outcome == goal}
  var chosenAction = sample(chooseAction(goalSatisfied, vendingMachine, 'state'))
  // we saw Sally press `aa`
  condition(chosenAction == 'aa')
  // we saw a cookie came out when Sally pressed `aa`
  condition(vendingMachine('state', 'aa') == 'cookie')

  return {goal: goal, 
          once: buttonsToOutcomeProbs['a'][1],
          twice: buttonsToOutcomeProbs['aa'][1]};
})

```

Probability of cookie given `a` was pushed:
![](https://github.com/vinsis/math-and-ml-notes/raw/master/images/cookie_given_a.jpg)

Probability of cookie given `a` was pushed:
![](https://github.com/vinsis/math-and-ml-notes/raw/master/images/cookie_given_aa.jpg)

> Notice the U-shaped distribution for the effect of pressing the button just once.
> 
How do we explain this?

Note that the probability that she wanted a cookie is 0.65. Thus there is a 0.35 chance that she did not want a cookie. But we saw her press `aa`. Thus it is likely that pressing `aa` gives a bagel. Thus pressing `a` gives a cookie.

> This very complex (and hard to describe!) inference comes naturally from joint inference of goals and knowledge.
> 

---

### Communication and Language

__Key idea__
We have two entities communicating to each other. First entity tries to infer how second entity thinks. Second entity tries to infer how first entity thinks. They keep on updating their beliefs. 

However this is an infinite loop. To prevent this we find a way to get out of it after a certain depth.

Say we have two dice with the probabilities as shown:

```javascript
var dieToProbs = function(die) {
  return (die == 'A' ? [0, .2, .8] :
          die == 'B' ? [.1, .3, .6] :
          'uhoh')
}
```

> On each round the “teacher” pulls a die from a bag of weighted dice, and has to communicate to the “learner” which die it is by showing them faces of the die. Both players are familiar with the dice and their weights.
> 
Teacher has a prior over sides. Student has a prior over dice.
```javascript
var sidePrior = Categorical({vs: ['red', 'green', 'blue'], ps: [1/3, 1/3, 1/3]})
var diePrior = Categorical({vs: ['A', 'B'], ps: [1/2, 1/2]})
```

A simple roll of a die:
```javascript
var roll = function(die) {return categorical({vs: ['red', 'green', 'blue'], ps: dieToProbs(die)})}
```

This is how teacher and student communicate and infer about each other:
```javascript
var teacher = function(die, depth) {
  return Infer({method: 'enumerate'}, function() {
    var side = sample(sidePrior);
    condition(sample(learner(side, depth)) == die)
    return side
  })
}

var learner = function(side, depth) {
  return Infer({method: 'enumerate'}, function() {
    var die = sample(diePrior);
    condition(depth == 0 ? 
              side == roll(die) :
              side == sample(teacher(die, depth - 1)))
    return die
  })
}
```

> assume that there are two dice, A and B, which each have three sides (red, green, blue) that have weights like so:
> 
![](https://probmods.org/assets/img/pedagogy-pic.png)

Now let's say the learner is shown a `green` side.

- If the depth is 0, the learner will infer it came from `B` since it has a higher chance of showing green.
- If the depth is increased, the learner will infer it came from `A`. Why? `because “if the teacher had meant to communicate B, they would have shown the red side because that can never come from A.”`

### ToDo: Read [this paper](https://langcog.stanford.edu/papers/SGF-perspectives2012.pdf) to get some more examples of this kind of learning.