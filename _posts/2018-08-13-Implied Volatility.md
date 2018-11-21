---
layout: post
comment: false
title: Implied Volatility
date: 2018-06-20 22:12:41 +0800
categories: jonlogger
math: true
comments: true
author: Jon Liu
tags: options implied-vol arbitrage-free hedging-errors
---
> A well-known market practice in the options market is to invert option pricing model and using market prices of highly liquid European vanilla calls/puts to get a so called **implied volatility (IV)**: a wrong number that when plugged into the wrong equation gives the right price. This post marks down some reviews.

<!--more-->

> TODO list:
- [x] @mentions, #refs, [links](#Reference), **formatting**, and <del>tags</del> supported
- [x] review of implied volatility curve
- [x] several key technique details
- [ ] this is an incomplete ite

{: class="table-of-content"}
* TOC
{:toc}



## Facts
- There is no single impvol for any given underlying. An impvol scalar is constrained to a parameter tuple $(K, \tau)$.
- From put-call parity, $C(S, K, \tau)$ has the same impvol as $P(S, K, \tau)$.
- The implied volatility curve is ploted as $\sigma_{imp} = f(T)$.
- The impvol curve is convex, and the lowest part of the curve is either ATM or slightly above it.
- Options with less time to expiration have steeper and more convex impvol curves as a function of strike.

## Calibration

Market's incompleteness leads to multiple risk-neutral probability (equivalent martingale measure) and to multiple price for derivative assets consistent with the absence of arbitrage principle. A trader has to perform calibration to the market, based on what to choose among the possiable risk-neutral measures. According to [[2]](#ref2), calibration yields the *market-consistant* risk-neutral probability measure in the sense that

- liquidly traded plain vanilla options are priced correctly and
- other (exotic) derivatives are priced such that prices are both consistent with the absence of arbitrage principle and indeed unique.

### Target Function
Generally 3 kinds of objective function is avaliable:
- MSE of the price differences in currency units
$$ \min_p \frac{1}{N} \sum_{n=1}^N (C^*_n - C_n^{model}(p))^2$$;

- MSE of the relative price differences
$$ \min_p \frac{1}{N} \sum_{n=1}^N (1 - C_n^{model}(p)/C^*_n)^2$$;

- MSE of the implied volatility differences
$$ \min_p \frac{1}{N} \sum_{n=1}^N (\sigma^*_n - \sigma_n^{model}(p))^2$$;

where $p$ denotes the vectorize model parameter.

[[1]](#ref1) stress that the choice of an objective function for calibration purposes should take into account the specific objective itself. Heding and picing activities should favor different ones respectively.

Weighting terms, such as vega, bid-ask spread or impvol of the respective option, could be used to avoid the biases. E.g. the objective function

$$ \min_p \frac{1}{N} \sum_{n=1}^N \Big((\sigma^*_n - \sigma_n^{model}(p)) \frac{\partial C_n^{BSM}}{\partial \sigma_n^*}\Big)^2$$

basically speaks applying less weight to impvol differences for short-term far ITM or OTM options, which should lead to a more preferable fit with the main objective being hedging. This is based from the idea that vega in general increases with closeness to the ATM strike level and with longer maturities.

Regularization term might also become necessary when the origin target function has multiple local minima which makes it hard to identify the global minimum. E.g, add a $w * L(p)$ term to the objective function given the vectorize parameter $p$ and the penalty function $L(p) = \|\|p-p_0\|\|$ wrt some norm.

### No Arbitrage Curve


## Reference
<ol>
    <li id="ref1">Christoffersen, Peter and Heston, Steven L. and Jacobs, Kris, Option Valuation with Conditional Skewness (July 15, 2003). EFA 2004 Maastricht Meetings Paper No. 2964. Available at SSRN: https://ssrn.com/abstract=557079 or http://dx.doi.org/10.2139/ssrn.55707</li>
    <li id="ref2">Hilpisch, Yves. Derivatives Analytics with Python: Data Analysis, Models, Simulation, Calibration and Hedging. John Wiley & Sons, 2015.</li>
    <li id="ref3">Same as <a href="#ref2">2</a></li>
</ol>
