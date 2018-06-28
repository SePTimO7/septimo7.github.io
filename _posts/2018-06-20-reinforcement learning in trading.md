---
layout: post
comment: falseS
title:  "Reinforcement Learning in Trading"
date:   2018-06-20 22:12:41 +0800
categories: jekyll update
math: true
tags: reinforcement-learning
---

> This post explains the application of reinforcement learning in trading.

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

Different from `predicting forward price by Deep Learning`, RL enable a trader with the new eyesight. I shall start from the basic building blocks.

## Target
The target is of course to make a profit, or in a better expression, to optimize some function who is affected by both profit and volatility. One need to first recognize the basic metrics used in her taste.

- Net PnL (Net Profit and Loss)
- NPT (Net Profit per Trade/Contract)
- Sharpe Ratio
- MD (Maximum Drowdown)
- ...

## Agent
Agent is your kid. Put in an environment, an agent grows up with information feed and interacts with the env. It chooses its own actions, and fetches feedback in the near or far future. With the sequence of info, it becomes compatiable with the env and (maybe) become the king.

## Actions
The agent has been declaring that this `Env` park is her stage and she can do with it what she pleases. But usually it's not such a good idea to extend the boundry of actions set. For the most naive situation, one may define the set as \\(\mathscr{A} = \\{\mathrm{long, short, hold}\\}\\). In a more detailed setting, `long` corresponds to sending a market order (MO) to buy some/unit underlying assets. Naturally, limit order(LO), FAK, FOK, .etc can also be used by the agent.

`tensorflow.static_rnn` gives the graph of unrolled RNN, w.r.t. the time dimension. Tensorboard will show a stacked architecture with `sequence_length` of `rnn_cell`s. Thus during the feed phase, every batch is required to have the same `sequence_length`, a.k.a. `time_steps`.

`tensorflow.dynamic_rnn` does not unroll RNN. It uses `tf.while_loop` and other control flow ops to generate a graph to execute the loop. For the dynamic version, `sequence_length` means loop numbers and batches can have variable `sequence_length`. 

#### NN for regression

When applied to regression problem, the NN's output layer has only one node, the output of the node is the same as the input of the node. That is, linear activation function: \\(f(x) = x\\).

A function that takes the input signal and generates an output signal, but takes into account the threshold, is called an *activation function*. 

We work through each layer of our network calculating the outputs for each neuron. All of the outputs from one layer become inputs to the neurons on the next layer. This process is called *forward propagation*.

The weights act as bridges. We use the weights to propagate signals forward from the input to the output layers in a neural network. We use the weights to also propagate error backwards from the output back into the network to update our weights. This is called *back propagation*.

> **Hint:** You'll need the derivative of the output activation function ($f(x) = x$) for the backpropagation implementation.

> **Other Variations of GAN**: There are many variations of GANs in different contexts or designed for different tasks. For example, for semi-supervised learning, one idea is to update the discriminator to output real class labels, $$1, \dots, K-1$$, as well as one fake class label $$K$$. The generator model aims to trick the discriminator to output a classification label smaller than $$K$$.


**Tensorfor Implementation**: [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)

{: class="info"}
| Symbol        | Meaning           | Notes  |
| ------------- | ------------- | ------------- |
| $$p_{z}$$ | Data distribution over noise input $$z$$ | Usually, just uniform. |
| $$p_{g}$$ | The generator's distribution over data $$x$$ | |
| $$p_{r}$$| Data distribution over real sample $$x$$ | |

![KL and JS divergence]({{ '/assets/images/favicon.ico' | relative_url }})
{: style="width: 640px;" class="center"}
*Fig. 1. Tony-Ironman.*

Jekyll also offers powerful support for code snippets:

~~~~
This is a 
piece of code 
in a block
~~~~

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

{% highlight python %}
from tensorflow.python.training.moving_averages import assign_moving_average

def batch_norm(x, train, eps=1e-05, decay=0.9, affine=True, name=None):
    with tf.variable_scope(name, default_name='BatchNorm2d'):
        params_shape = tf.shape(x)[-1:]
        moving_mean = tf.get_variable('mean', params_shape,
                                      initializer=tf.zeros_initializer,
                                      trainable=False)
        moving_variance = tf.get_variable('variance', params_shape,
                                          initializer=tf.ones_initializer,
                                          trainable=False)

        def mean_var_with_update():
            mean, variance = tf.nn.moments(x, tf.shape(x)[:-1], name='moments')
            with tf.control_dependencies([assign_moving_average(moving_mean, mean, decay),
                                          assign_moving_average(moving_variance, variance, decay)]):
                return tf.identity(mean), tf.identity(variance)
        mean, variance = tf.cond(train, mean_var_with_update, lambda: (moving_mean, moving_variance))
        if affine:
            beta = tf.get_variable('beta', params_shape,
                                   initializer=tf.zeros_initializer)
            gamma = tf.get_variable('gamma', params_shape,
                                    initializer=tf.ones_initializer)
            x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
        else:
            x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
        return x
{% endhighlight %}


{% highlight bash %}
gem install rake && bundle install
{% endhighlight%}

\\( 1/x^{2} \\)

$$\mathrm{B}$$

$$e^{i\pi} + 1 = 0$$

$$
    \begin{matrix}
    1 & x & x^2 \\
    1 & y & y^2 \\
    1 & z & z^2 \\
    \end{matrix}
$$

$$
  \begin{pmatrix}
    a & b\\
    c & d\\
  \hline
    1 & 0\\
    0 & 1
  \end{pmatrix}
$$

$\bigl( \begin{smallmatrix} a & b \\ c & d \end{smallmatrix} \bigr)$

$$
\begin{align}
\sqrt{37} & = \sqrt{\frac{73^2-1}{12^2}} \\
 & = \sqrt{\frac{73^2}{12^2}\cdot\frac{73^2-1}{73^2}} \\ 
 & = \sqrt{\frac{73^2}{12^2}}\sqrt{\frac{73^2-1}{73^2}} \\
 & = \frac{73}{12}\sqrt{1 - \frac{1}{73^2}} \\ 
 & \approx \frac{73}{12}\left(1 - \frac{1}{2\cdot73^2}\right)
\end{align}
$$

$$
f(n) =
\begin{cases}
n/2,  & \text{if $n$ is even} \\
3n+1, & \text{if $n$ is odd}
\end{cases}
$$

{% raw %}
$$a^2 + b^2 = c^2$$ --> note that all equations between these tags will not need escaping! 
{% endraw %}

$$
\begin{array}{c|lcr}
n & \text{Left} & \text{Center} & \text{Right} \\
\hline
1 & 0.24 & 1 & 125 \\
2 & -1 & 189 & -8 \\
3 & -20 & 2000 & 1+10i
\end{array}
$$

Compare $\displaystyle \lim_{t \to 0} \int_t^1 f(t)\, dt$
versus $\lim_{t \to 0} \int_t^1 f(t)\, dt$.

$$
\begin{array}{ll}
\text{maximize}  & c^T x \\
\text{subject to}& d^T x = \alpha \\
&0 \le x \le 1.
\end{array}
$$

$$
\begin{alignat}{5}
  \max \quad        & z = &   x_1  & + & 12 x_2  &   &       &         && \\
  \mbox{s.t.} \quad &     & 13 x_1 & + & x_2     & + & 12x_3 & \geq 5  && \tag{constraint 1} \\
                    &     & x_1    &   &         & + & x_3   & \leq 16 && \tag{constraint 2} \\
                    &     & 15 x_1 & + & 201 x_2 &   &       & =    14 && \tag{constraint 3} \\
                    &     & \rlap{x_i \ge 0, i = 1, 2, 3}
\end{alignat}
$$

$$|x|, ||v|| \quad\longrightarrow\quad \lvert x\rvert, \lVert v\rVert$$

$$
\underset{j=1}{\overset{\infty}{\LARGE\mathrm K}}\frac{a_j}{b_j}=\cfrac{a_1}{b_1+\cfrac{a_2}{b_2+\cfrac{a_3}{b_3+\ddots}}}
$$

~~~
from pylab import *
from matplotlib.patches import Polygon

def func(x):
    return (x-3)*(x-5)*(x-7)+85

ax = subplot(111)

a, b = 2, 9 # integral area
x = arange(0, 10, 0.01)
y = func(x)
plot(x, y, linewidth=1)

# make the shaded region
ix = arange(a, b, 0.01)
iy = func(ix)
verts = [(a,0)] + list(zip(ix,iy)) + [(b,0)]
poly = Polygon(verts, facecolor='0.8', edgecolor='k')
ax.add_patch(poly)

# text for the intergal
text(0.5 * (a + b), 30, r"$\int_a^b f(x)\mathrm{d}x$", horizontalalignment='center', fontsize=20)

# axis text
axis([0,10, 0, 180])
figtext(0.9, 0.05, 'x')
figtext(0.1, 0.9, 'y')
ax.set_xticks((a,b))
ax.set_xticklabels(('a','b'))
ax.set_yticks([])
show()
~~~

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
