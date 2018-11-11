---
layout: post
comment: false
title:  "Implied Volatility"
date:   2018-06-20 22:12:41 +0800
categories: jekyll update
math: true
tags: options implied-vol
---
> This post explains the application of reinforcement learning in trading.

<!--more-->

> **Impvol:** Wrong number that when plugged into the wrong equation gives the right price

{: class="table-of-content"}
* TOC
{:toc}



## Facts
- There is no single impvol for any given underlying. An impvol scalar is constrained to a tuple $(K, \tau)$.
- From put-call parity, C(S, K, \tau) has the same impvol as P(S, K, \tau).
- The implied volatility curve is ploted as $\sigma_imp = f(T)$.
- The implied curve is convex, and the lowest part of the curve is either ATM or slightly above it.
- Options with less time to expiration have steeper and more convex impvol curves as a function of strike.

## Calibration

### Target Function
- MSE of the price differences in currency units
$$ \min_p \frac{1}{N} \sum_{n=1}^N (C^*_n - C_n^{model}(p))^2
$$

### No Arbitrage Curve
