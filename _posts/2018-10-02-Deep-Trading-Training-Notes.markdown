---
title: Deep Trading Training Notes
layout: post
categories: jonlogger
math: true
comments: true
author: Jon Liu
date: 2018-10-02 23:51:13
meta: "Springfield"
tags: deep-learning tensorflow trading
---

> This post talks on CNN

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

## From weighted average to convolution

Convolution is a specialized kind of linear operation.

> **DeepLearningBook:** Convolutional Neural Networks are simply neural networks that use convolution in place of general matrix multiplication in at least one of their layers.

$$
s(t) = \int x(a)w(t-a)da = \sum_{a=-\infty}^{\infty} x(a)w(t-a) = (x*w)(t).
$$


## Pain of regularization

When designing the architecture of a deep learning model, one may always want to place many regularization layers intentionally or not. Too much regularization can cause the network to underfit badly. When building the infrastructure, try to reduce the usage of regularization such as dropout, batch norm, weight/bias $L^2$ regularization first. Overfit helps you better in this phase.

## Turn off all bells and whistles at the very first begining.


## Metrics

1. Number 1
```python
print("Hello World")
```

2. Number 2
```ruby
puts 'Hello World'
```

3. Number 3
```c
printf("Hello World");
```

Accuracy of class 0/1/2: Take label 2 (up) as an example. The sum of the number correctly predicted up and non-up over all passed samples

$$
\textbf{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}
$$

TensorFlow provides `tf.metrics.precision_at_k` function which calculates the precision for class with `class_id` by considering only the entries in the batch for which `class_id` is in the top k hightest predictions among the passed samples in the mini-batch.

```python
precision_at_1_label0 = tf.metrics.precision_at_k(labels, logits, 1,
                                                  class_id=0,
                                                  name='precision_at_1_label0')
```

When labeling the target change, it's ideal to make the partioned classes have close number of samples. But this will not be always the case. Some methods can be proposed to deal with the imbalance training data.

- Use last days' data to normalize today's item by item. Before normalizing, one can apply some transform (log) on the origin data to get a proper scale.

- Set the weights of classes contribute to the final loss function. Say, we have fewer samples for classes 0/2 classes than class 1 in a multi-classificaion problem: $n_0, n_2 \ll n_1$, by setting the weights of the minorities a little bit higher we can imporve the accuracy w.r.t. the minorities. Usually this can be set as the probability distribution normalized from the inverse of the empirical frenquencies. According to my findings, while set the weights distribution properly imporves the overall accuracy dramatically, give the minority class too much weight will lead to a low precision, after all the model will try to predict more evaluated data as its label.

- Start with fewer days' TFRecords samples (10 days) and set the evaluate data as same as the training data. Build up the tower, overfit on it and gradually add more data. As I commit, underfit is more horriable than its counter when I have nothing. Find the minimum batch size with which you can tolerate the training time. The batch size which makes optimal use of the GPU parallelism might not be the best when it comes to accuracy as at some point a larger batch size will require training the network for more epochs to achieve the same level of accuracy. Don't be scared to start with a very small batch size such as 16, 8, or even 1.

(NN not working)[http://theorangeduck.com/page/neural-network-not-working]

- Smaller batch size. Using too large a batch size can have a negative effect on the accuracy of your network during training since it reduces the stochasticity of the gradient descent.

- During some training periods I find the eval output may often stucked in one class. The expressive power of the model could be limited to capture the target due to the regularization terms: redudent dropout layers or batchnorm layers and a large l2 term of the loss function are the most often excuses. If that's not the case, consider either set more hidden units of the existing layers or add more extra layers.


## Debugging in TensorFlow

### Utilizing TensorBoard to Visualizing the Training

During traing phase, it's advised to have a look at the varying of trainable variables (weights, updates and activations) along traing iterations. The experience rule claims that good looking basis should be in a magnitude of $10^{-3}$.

Be on the lookout for layer activations with a mean much larger than 0. Try batch norm layer or `tf.nn.elu`.

Check layer updates, they should have a Gaussian distribution.
```python
def function():
    print('Yes')
```
### Basic ways
- Explicitly fetch and print: `tf.Session().run()`;
- TensorBoard: Scalar, histogram and image: `tf.summary.FileWriter()`;
```python
[tf.summary.histogram(v.name, v) for v in tf.trainable_variables()]
```
- Print op's side effect: `tf.Print`
```python
output = tf.Print(output, [tensor1, tensor2])
```
- Assertation: `tf.Assert()`

### Advanced ways:
- Embed and interpose any python codelet in the computation graph

```python
tf.py_func(func, inp, Tout, stateful=True, name=None)
```
Given a python function func, which takes numpy arrays as its inputs and returns numpy arrays as its outputs, the function is wrapped as an operation.

```python
def my_func(x):
    # x will be a numpy array with the contents of the placeholder below
    return np.sinh(x)
inp = tf.placeholder(tf.float32, [...])
y = py_func(my_func, [inp], [tf.float32])
```

```python
def multilayer_perceptron(x):
    fc1 = tf.layers.dense(x, 256, activation_fn=tf.nn.relu, scope='fc1')
    fc2 = tf.layers.dense(fc1, 256, activation_fn=tf.nn.relu, scope='fc2')
    out = tf.layers.dense(fc2, 10, activation_fn=None, scope='out')
    def _debug_print_func(fc1_val, fc2_val):
        tf.logging.info('FC1 : {}, FC2 : {}'.format(fc1_val.shape, fc2_val.shape))
        tf.logging.info('min, max of FC2 = {}, {}'.format(fc2_val.min(), fc2_val.max()))
        return False
    debug_print_op = tf.py_func(_debug_print_func, [fc1, fc2], [tf.bool])
    with tf.control_dependencies(debug_print_op):
        out = tf.identity(out, name='out')
    return out
```

```python
def multilayer_perceptron(x):
    fc1 = tf.layers.dense(x, 256, activation_fn=tf.nn.relu, scope='fc1')
    fc2 = tf.layers.dense(fc1, 256, activation_fn=tf.nn.relu, scope='fc2')
    out = tf.layers.dense(fc2, 10, activation_fn=None, scope='out')
    def _debug_func(x_val, fc1_val, fc2_val, out_val):
        if (out_val == 0.0).any():
            import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
        from IPython import embed; embed()  # XXX DEBUG
        return False
    debug_op = tf.py_func(_debug_func, [x, fc1, fc2, out], [tf.bool])
    with tf.control_dependencies(debug_op):
        out = tf.identity(out, name='out')
    return out
```
- A step-by-step debugger
- The TensorFlow debugger: `tfdbg`



### 1$\times$1 Conv

When we deal with series inputs that are 2-dimensional the 1$\times$1 Conv will degenerate to pointwise scaling. The effect changes when the data feeded is 3-dimensional volumes of shape `(batch, num_steps, num_features)`, teh filters are going to extend through the full depth of the depth (the data channel) dimension. Working on LOB, however, simply apply conv2d as what we usually do upon academic image dataset could be wrong -- consider the information contained in each price level: some research points out that the best ask/bid levels contribute most to the price discovery and contribution of all other levels is considerably less, estimated at as little as $20\%$. Without deep learning method what practioners do most frequently is to somehow take weighted average to summarise the expect of the info contained in deeper levels. This is the average along the so-called data channel dimension, `$1\times1$ Conv` from our arsenal can be helpful.

During usage one can treat 1x1 convolution with unit filter as a neuron that receives a `data_channel` length array locates at some indicies of the input tensor and optionally follows it with a non-linearity. For example, if the input is `[32 x 32 x 3]` then doing 1x1 convolutions would effectively be doing 3-dimensional dot products (since the input depth is 3 channels). 1x1 convolutions could be used to reduce the data channel dimension and imporve the compution efficiency.

One more thing, be careful about the batch normalization layer after the $1\times1$ Conv.


### conv1d

When dealing with a LOB image input tensor of shape `(batch, width=num_steps, channels=num_features)`, imagine the situation where we use a small 2d kernel (4d in its tensor rank) to convolves out the local pattern. We would necessarily apply the same parameters to every leavel of a LOB and thus the L1 data shares the same parameters with the deeper level. This is counter-intuitive -- we are willing to overight the importance of the L1 data and depress the deepers. Following such sight, one can apply a large kernel. Ideally, we set the hight of the kernel $ker_h = num_features$, which means the kernel spans all of the channel dimension. For a 1 filter conv1d, this outputs a batch of single lines. $F$ filters stack the 1d results to form a batch of 2d matrices.

### Dilated convolutions
Under this settings, when doing convolution it’s possible to have filters that have spaces between each cell, called dilation. It allows us to use fewer parameters to get a closer effect.

### Inception module

Technical traders use technical indicators to setup their rules of trading. Most indicators embed a rolling window concept. One may use moving averages (MA) with different window settings to get serval smoother time-series and gain insight on the momentum through comparsion.  In practice, it is a daunting task to set the right decay weights. Instead, we can use Inception Modules and the weights are then learned during back-propagation.

### Data normalization
- simple log transform
- z-score, while using the mean and standard deviation of the previous trading day's data to normalize the current day's data.
- Normalize inside rollowing window. Each price level normalized by mid/micro price
- Stride inside rolling window. Equivalent to bar based input data for some stride $> 0$.

### Evaluation metrics

### Benchmark model

- Simply use dense layer with linear activation leads to a linear model

### Canned model

- DNNClassifier. Feature column.

```python
def input_fn():
    ...
    features = {
        'ap': [100, 111, 97],
        'bp': [98, 107, 96],
        'aq': [12, 2, 9],
        'bq': [3, 8, 5],
    }
    labels = [0, 2, 1]
    return features, labels
```

match feature names from input_fn; bridge input and model
Utilize [tf.feature_column](https://www.tensorflow.org/guide/feature_columns "TF feature_column doc") to bridge the input and model.

```python
feature_columns = [
    tf.feature_column.numeric_column('ap', shape=[], dtype=tf.float32),
    tf.feature_column.numeric_column('bp', shape=[], dtype=tf.float32),
    tf.feature_column.numeric_column('aq', shape=[], dtype=tf.float32),
    tf.feature_column.numeric_column('bq', shape=[], dtype=tf.float32),
]

```

```python
estimator = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[10, 12],
    n_classes=3,
    model_dir=paginator.previous_page_path)

```

In most micro projects we create feature columns from a `pandas.DataFrame` object, the pandas library provides several types checking methods for help [](https://pandas.pydata.org/pandas-docs/stable/api.html#data-types-related-functionality):
```python
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

df = ...
feature_columns = []

for col in df.columns:
    if is_string_dtype(df[col]):
        feature_columns.append(tf.feature_column.categorical_column_with_hash_bucket(
            col, hash_bucket_size=len(df[col].unique())))
    elif is_numeric_dtype(df[col]):
        feature_columns.append(tf.feature_column.numeric_column(col))
```


```python
def get_normalized_params(df, features):

    def _z_score(column):
        mean = df[column].mean()
        std = df[column].std()
        return {'mean': mean, 'std': std}

    normalized_params = {}
    [normalized_params[column] = _z_score(column) for column in features]
    return normalized_params
```

### Labeling the data
Some papers use a smoothing method to label the target. They use target equals to forward mean minus past mean then clip the target into several classes by some threshold(s). The traing statistics get more pretty but this could not be a *tradable* signal. The time $t$ prediction contains the past info too much: while the absloutely value of threshold is not large enough, we could only predict the target will not decrease if the past mean has already grown large enough.

<script src="https://gist.github.com/SePTimO7/8b08eb17842924c05d592a2179494050.js?file=numbas.py"></script>

### Directional bets optimal place order formulas
\newcommand{\mypm}{$\phi$}

With exponential utility function $\phi(s,q,x) = -\exp(-\gamma(x+qs-\eta q^2))$ and the best level order flow intensity $\lambda^{\mypm} = Ae^{-k\delta^{\mypm}}$. If mid-price $S_t$ is an arithmetic Brownian motion with drift:
$$ d S_t = b d t + \sigma d W_t,$$
which implies that
$$E_{t,s}[S_T] = s + b(T-t).$$
Then the optimal control gives:
$$\delta_*^{\mypm} = \frac{1}{\gamma} \log(1 + \frac{\gamma}{k}) + \eta + \frac{1}{2}\gamma\sigma^2(T-t) \mypm (b(T-t) - q[2\eta + \gamma\sigma^2(T-t)]).$$
If $S_t$ is an O-U process
$$\di S_t = a(\mu - S_t) \di t + \sigma \di W_t,$$
then
$$\Expect_{t,s}[S_T] = se^{-a(T-t)} + \mu(1-e^{-a(T-t)}).$$
Then the optimal control gives:
$$\delta_*^{\mypm} = \frac{1}{\gamma} \log(1 + \frac{\gamma}{k}) + \eta + \frac{1}{4a}\gamma\sigma^2(1-e^{-2a(T-t)}) \mypm ((\mu - s)(1-e^{-a(T-t)}) - q[2\eta + \frac{1}{2a}\gamma\sigma^2(1-e^{-2a(T-t)})]).$$
For more details refer to

Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

> This quote will change your life. It will reveal the secrets of the universe, and all the wonders of humanity. Don't misuse it.

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Fusce bibendum neque eget nunc mattis eu sollicitudin enim tincidunt.

## ML techniques on trading data

### Resampling data
making data into buckets to switch the representation into another form, with less info showed but more efficient.

- Number ticks
- Transaction ticks
- Time ticks
- Trading volume ticks
- Dollar ticks
- Self defined imbalance ticks

### Labelling data

### Meta: ML on ML tricks


### Denoising
Market microstructure is full of noise. By denoising we fetch the informative part with respect to the targets.


### Imbalancing data

Vestibulum lacus tortor, ultricies id dignissim ac, bibendum in velit. Proin convallis mi ac felis pharetra aliquam. Curabitur dignissim accumsan rutrum.

```html
<html>
  <head>
  </head>
  <body>
    <p>Hello, World!</p>
  </body>
</html>
```


In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris.

#### You might want a sub-subheading (h4)

In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris.

In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris.

#### But it's probably overkill (h4)

In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris.

### Oh hai, an unordered list!!

In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris.

- First item, yo
- Second item, dawg
- Third item, what what?!
- Fourth item, fo sheezy my neezy

### Oh hai, an ordered list!!

In arcu magna, aliquet vel pretium et, molestie et arcu. Mauris lobortis nulla et felis ullamcorper bibendum. Phasellus et hendrerit mauris.

1. First item, yo
2. Second item, dawg
3. Third item, what what?!
4. Fourth item, fo sheezy my neezy



## Headings are cool! (h2)

Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc. Praesent varius interdum vehicula. Aenean risus libero, placerat at vestibulum eget, ultricies eu enim. Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

Praesent nulla tortor, malesuada adipiscing adipiscing sollicitudin, adipiscing eget est.

Proin eget nibh a massa vestibulum pretium. Suspendisse eu nisl a ante aliquet bibendum quis a nunc.

### Tables

Title 1               | Title 2               | Title 3               | Title 4
--------------------- | --------------------- | --------------------- | ---------------------
lorem                 | lorem ipsum           | lorem ipsum dolor     | lorem ipsum dolor sit
lorem ipsum dolor sit | lorem ipsum dolor sit | lorem ipsum dolor sit | lorem ipsum dolor sit
lorem ipsum dolor sit | lorem ipsum dolor sit | lorem ipsum dolor sit | lorem ipsum dolor sit
lorem ipsum dolor sit | lorem ipsum dolor sit | lorem ipsum dolor sit | lorem ipsum dolor sit


Title 1 | Title 2 | Title 3 | Title 4
--- | --- | --- | ---
lorem | lorem ipsum | lorem ipsum dolor | lorem ipsum dolor sit
lorem ipsum dolor sit amet | lorem ipsum dolor sit amet consectetur | lorem ipsum dolor sit amet | lorem ipsum dolor sit
lorem ipsum dolor | lorem ipsum | lorem | lorem ipsum
lorem ipsum dolor | lorem ipsum dolor sit | lorem ipsum dolor sit amet | lorem ipsum dolor sit amet consectetur

## Prices

In the standard economics paradigm, it is the intersection of supply and demand curved for a particular good. But what is it in the economy that coordinates the desires of demanders and suppliers so that a price emerges and trade occurs?


##

Some where between where the raw data is fed and the classification label is output, the model serves as a complex feature extractor.

Feed the data through the network to obtain the outputs of the intermediate layers. Define loss function wrt content and style seperately. The loss is defined as some distance in terms of representations.

Set the input data as trainable variables during the training session, the model with fixed internal nodes and edges will shape the feed data to reduce the loss function.

Consider whether the strategy is entering and exiting passively, by posting bids and offers, or aggressively, by crossing the spread to sell at the bid and buy at the offer. Usually passive entries and exits is used for big volume.

Make a much more conservative assumption that your limit order will only get filled when the market moves through them.

For a futures strategy, crossing the spread to enter or exit a trade more than a handful of times (or missing several limit order entries or exits) will quickly eviscerate the profitability of the system.  A HFT system in equities, by contrast, will typically prove more robust, because of the smaller tick size.

## Win Rate v.s. Risk : Reward rates
- Pick the market correctly more than half the time (positive win rate);
- win more than you lose on each trade (positive risk to reward ratio)
