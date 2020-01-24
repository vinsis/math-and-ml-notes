## Chapter 10: Learning with a language of thought

#### Key idea
How can we create complex hypotheses and representation spaces? Simply by using stochastic recursion. When the recursion is going to end is not deterministic, it is probabilistic. 

Here is how we can create an infinite amount of mathematical expressions:
```javascript
var randomConstant = function() {
  return uniformDraw(_.range(10))
}

var randomCombination = function(f,g) {
  var op = uniformDraw(['+','-','*','/','^']);
  return '('+f+op+g+')'
}

// sample an arithmetic expression
var randomArithmeticExpression = function() {
  flip() ? 
    randomCombination(randomArithmeticExpression(), randomArithmeticExpression()) : 
    randomConstant()
}

randomArithmeticExpression()
```

But more complex strings are less likely due to the way the function is defined.

### Inferring an arithmetic expression
We can do so by conditioning:
```javascript
  var e = randomArithmeticExpression();
  var s = prettify(e);
  var f = runify(e);
  condition(f(1) == 3 && f(2) == 4);

  return {s: s};
```

### Grammar based induction
> What is the general principle in the two above examples? We can think of it as the following recipe: we build hypotheses by stochastically choosing between primitives and combination operations, this specifies an infinite “language of thought”; each expression in this language in turn specifies the likelihood of observations. Formally, the stochastic combination process specifies a probabilistic grammar; which yields terms compositionally interpreted into a likelihood over data.
> 

---

### ToDo: Example: Rational Rules
I didn't quite get how this works. Required reading: [A Rational Analysis of Rule-based Concept Learning](https://onlinelibrary.wiley.com/doi/epdf/10.1080/03640210701802071).

[Online, non-pdf version](https://onlinelibrary.wiley.com/doi/full/10.1080/03640210701802071)