## Chapter 13: Learning (deep) continuous functions

It's the same idea of updating prior beliefs, just applied to neural networks.

### Key idea
The parameters of a neural network comes from one or more Gaussian distributions. Given some data we can update the priors to come up with neural nets that fit the data better.

```javascript
var dm = 10 //size of hidden layer

var makeFn = function(M1,M2,B1){
  return function(x){
    return T.toScalars(
      // M2 * sigm(x * M1 + B1):
      T.dot(M2,T.sigmoid(T.add(T.mul(M1,x),B1)))
    )[0]}
}

var observedData = [{"x":-4,"y":69.76636938284166},{"x":-3,"y":36.63586217969598},{"x":-2,"y":19.95244368751754},{"x":-1,"y":4.819485497724985},{"x":0,"y":4.027631414787425},{"x":1,"y":3.755022418210824},{"x":2,"y":6.557548104903805},{"x":3,"y":23.922485493795072},{"x":4,"y":50.69924692420815}]

var inferOptions = {method: 'optimize', samples: 100, steps: 3000, optMethod: {adam: {stepSize: 0.1}}}

var post = Infer(inferOptions,
  function() {  
    var M1 = sample(DiagCovGaussian({mu: zeros([dm, 1]), sigma: ones([dm,1])}))
    var B1 = sample(DiagCovGaussian({mu: zeros([dm, 1]), sigma: ones([dm,1])}))
    var M2 = sample(DiagCovGaussian({mu: zeros([1, dm]), sigma: ones([1,dm])}))
    
    var f = makeFn(M1,M2,B1)
    
    var obsFn = function(datum){
      observe(Gaussian({mu: f(datum.x), sigma: 0.1}), datum.y)
    }
    mapData({data: observedData}, obsFn)

    return {M1: M1, M2: M2, B1: B1}
  }
)

print("observed data:")
viz.scatter(observedData)

var postFnSample = function(){
  var p = sample(post)
  return makeFn(p.M1,p.M2,p.B1) 
}
```

Notice two things here:
1. How the parameters for the network are sampled from a `DiagCovGaussian` (multivariate Gaussian distribution).
2. How we `observe` the data: `observe(Gaussian({mu: f(datum.x), sigma: 0.1}), datum.y)`

The second step is key to updating the parameters. __A non-Bayesian way of updating parameters requires a loss function to backpropagate on. In a Bayesian way, we are trying to increase the likelihood of getting `y` from a Gaussian centered at the output of the neural net.__


__Note__: As the width of hidden layer goes to infinity, the network approaches a Gaussian process.

> Infinitely “wide” neural nets yield a model where `f(x)` is Gaussian distributed for each `x`, and further (it turns out) the covariance among different `x`s is also Gaussian.

---

### Deep generative models

> Many interesting problems are unsupervised: we get a bunch of examples and want to understand them by capturing their distribution.
> 
Notice how this works:

```javascript
var hd = 10
var ld = 2
var outSig = Vector([0.1, 0.1])

var post = Infer(inferOptions,
  function() {  
    var M1 = sample(DiagCovGaussian({mu: zeros([hd,ld]), sigma: ones([hd,ld])}), {
      guide: function() {return Delta({v: param({dims: [hd, ld]})})}})
    var B1 = sample(DiagCovGaussian({mu: zeros([hd, 1]), sigma: ones([hd,1])}), {
      guide: function() {return Delta({v: param({dims: [hd, 1]})})}})
    var M2 = sample(DiagCovGaussian({mu: zeros([2,hd]), sigma: ones([2,hd])}), {
      guide: function() {return Delta({v: param({dims: [2,hd]})})}})
    
    var f = makeFn(M1,M2,B1)
    var sampleXY = function(){return f(sample(DiagCovGaussian({mu: zeros([ld, 1]), sigma: ones([ld,1])})))}

    var means = repeat(observedData.length, sampleXY)
    var obsFn = function(datum,i){
      observe(DiagCovGaussian({mu: means[i], sigma: outSig}), Vector([datum.x, datum.y]))
    }
    mapData({data: observedData}, obsFn)

    return {means: means, 
            pp: repeat(100, sampleXY)}
  }
)
```

Note:
1. The output is not a scalar anymore; it is a vector of length two.
2. We use `sampleXY` defined as `var sampleXY = function(){return f(sample(DiagCovGaussian({mu: zeros([ld, 1]), sigma: ones([ld,1])})))}` to sample `means`.
3. We observe `means` to be as close to our observed data as possible: `observe(DiagCovGaussian({mu: means[i], sigma: outSig}), Vector([datum.x, datum.y]))`

---

### Minibatches and amortized inference

> Minibatches: the idea is that randomly sub-sampling the data on each step can give us a good enough approximation to the whole data set.
> 
But if split the data into smaller batches, we need to be sure the latent variable used to sample `means` improves over time. 

### ToDo: Read about [Amortized Inference in Probabilistic Reasoning](https://web.stanford.edu/~ngoodman/papers/amortized_inference.pdf)