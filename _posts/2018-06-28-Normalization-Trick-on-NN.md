---
layout: post
comment: false
title:  "Normalization Trick on NN"
date:   2018-06-28 19:50:00 +0800
math: true
tags: batch-normalization recurrent-batch-normalization layer-normalization
---

> This post explains the most used normalization trick on neural network.

<!--more-->

{: class="table-of-content"}
* TOC
{:toc}

[Batch normalization](https://arxiv.org/pdf/1502.03167.pdf) gives one possible solution if one looks for better optimization for their on hand NN model.

**Internal Covariate Shift**: the change in the distribution of network activations due to the change in network parameters during training.

{: class="info"}
| **Input**: values of $$x$$ over a mini-batch: $$\mathcal{B} = \{x_{1...m}\}$$, Parameters to be learned: $$\gamma, \beta$$  |
| --------------------------------------- |
| **Output**: $$\{y_i = \mathrm{BN}_{\gamma, \beta}(x_i)\}$$ |
| $$p_{g}$$  The generator's distribution over data $$x$$ | |
| $$p_{r}$$ Data distribution over real sample $$x$$ | |

![Batch normalization algo]({{ '/assets/images/batch_norm_algo.png' | relative_url }})
{: style="width: 480px;" class="center"}
*Fig. 1. Batch Normalizing Transform, applied to activation $$x$$ over a mini-batch*

> "Note that simply normalizing each input of a layer may change what the layer can represent. For instance, normalizing the inputs of a sigmoid would constrain them to the **linear regime of the nonlinearity**. To address this, we make sure that the transformation inserted in the network can represent the identity transform."