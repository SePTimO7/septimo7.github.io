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
- [x] @mentions, #refs, [links](#reference), **formatting**, and <del>tags</del> supported
- [x] review of implied volatility curve
- [x] several key technique details
- [ ] this is an incomplete ite

{: class="table-of-content"}
* TOC
{:toc}

![fig1]({{ '/assets/images/impvol.png' | relative_url }})
{: style="width: 400px;" class="center"}
* Fig. 1. Understand implied vol. *


## Facts
- There is no single IV for any given underlying. At some timestamp $$t$$, an IV scalar is constrained to a parameter tuple $$(K, \tau)$$. Different derivative contracts on the same underlying have different IVs as a function of their own supply and demand dynamics.
- From put-call parity, $$C(S, K, \tau)$$ and $$P(S, K, \tau)$$ share a common IV. But in practice this may be not the case, even puts and calls give slightly different IVs. Whether it is increased demand (more buyers) or increased scarcity (fewer sellers), the result is the same: Higher prices for put options.
- The following relationship exists: IV rises when markets decline; IV falls when markets rally. This is because the idea of a falling market tends to (often, but not always) encourage (frighten?) people to buy puts -- or at least stop selling them.
- The impvol curve is convex, and the lowest part of the curve is either ATM or slightly above it.
- Options with less time to expiration have steeper and more convex impvol curves as a function of strike.
- Calculate IVs of all optons on the same expiry date and then buy one with the lowest vol and sell one with the highest vol with the belief that prices move in such a way that IVs become more or less comparable and the trader makes a profit from it.
- Combine options positions for the same expiry in the portfolios, then deduce the overall delta and hedge it.

## Explanation

## Calibration

Market's incompleteness leads to multiple risk-neutral probability (equivalent martingale measure) and to multiple price for derivative assets consistent with the absence of arbitrage principle. A trader has to perform calibration to the market, based on what to choose among the possiable risk-neutral measures. According to [[2]](#ref2), calibration yields the *market-consistant* risk-neutral probability measure in the sense that

- liquidly traded plain vanilla options are priced correctly and
- other (exotic) derivatives are priced such that prices are both consistent with the absence of arbitrage principle and indeed unique.

### Target Function
Generally 3 kinds of objective function is avaliable:
- MSE of the price differences in currency units:
$$ \min_p \frac{1}{N} \sum_{n=1}^N (C^*_n - C_n^{model}(p))^2$$;

- MSE of the relative price differences:
$$ \min_p \frac{1}{N} \sum_{n=1}^N (1 - C_n^{model}(p)/C^*_n)^2$$;

- MSE of the implied volatility differences:
$$ \min_p \frac{1}{N} \sum_{n=1}^N (\sigma^*_n - \sigma_n^{model}(p))^2$$;

where $$p$$ denotes the vectorize model parameter.

[[1]](#ref1) stress that the choice of an objective function for calibration purposes should take into account the specific objective itself. Heding and picing activities should favor different ones respectively.

Weighting terms, such as vega, bid-ask spread or impvol of the respective option, could be used to avoid the biases. E.g. the objective function

$$ \min_p \frac{1}{N} \sum_{n=1}^N \Big((\sigma^*_n - \sigma_n^{model}(p)) \frac{\partial C_n^{BSM}}{\partial \sigma_n^*}\Big)^2$$

basically speaks applying less weight to impvol differences for short-term far ITM or OTM options, which should lead to a more preferable fit with the main objective being hedging. This is based from the idea that vega in general increases with closeness to the ATM strike level and with longer maturities.

Regularization term might also become necessary when the origin target function has multiple local minima which makes it hard to identify the global minimum. E.g, add a $$w * L(p)$$ term to the objective function given the vectorize parameter $$p$$ and the penalty function $$L(p) = \|p-p_0\|$$ wrt some norm.

### No Arbitrage Curve

- Volatility skew/smile: The IV varies with different strike prices $$K$$ when fix maturity $$\tau$$ and of course the same underlying: $$\sigma_{imp} = \sigma_{imp}(K, \tau_0)$$. The term *volatility skew* refers to the fact that IV is noticeably higher for OTM options with strike prices below the underlying asset's price. And IV is noticeably lower for OTM options that are struck above the underlying asset price.

## IVS Regimes
Question: what is the spot $$S$$-dependence of $$\sigma_{imp}(S,K,\tau)$$ given its observed strike $$K$$-dependence?

### The Sticky Rule
The sticky (wrt some $$x$$) rule states that the current volatility skew is invariant to $$X$$ as the spot moves over short times.

- Stick to strike: $$\frac{\partial \sigma_{imp}}{\partial S}(S, K_0, \tau) \vert_{\delta S_t} \equiv 0$$. Apparently, the yield IV has no dependence on $$S$$, and therefore such a rule keeps $$\Delta = \Delta_{BS}$$. If we formally apply taylor expansion to $$\sigma_{imp}(K, \tau)$$ wrt $$K$$ at $$\Sigma_0 \equiv \sigma_{imp}(S_0,T-t_0)$$:

    $$
    \sigma_{imp}(K, \tau) - \Sigma_0 = \frac{\partial \sigma_{imp}}{\partial K}(K-S_0) + \frac{1}{2}\frac{\partial \sigma_{imp}^2}{\partial^2 K}(K-S_0)^2 + \cdots,
    $$

    keep the linear term only and let $$K \rightarrow S_0$$ (i.e. near the money), we deduce that:

    $$
    \sigma_{imp}(K, \tau) = \Sigma_0 + \frac{\partial \sigma_{imp}}{\partial K} \Big\vert_{K=S_0}(K - S_0) := \Sigma_0 - b(K - S_0),
    $$

    where b is assumed to be a time-insensitive consant because of our assumption of stickiness. The new most liquid option - ATM IV $$\sigma_{imp}^{atm}(K=S, \tau-\delta t) = \Sigma_0 - b(S - S_0)$$ decreases as the spot price increases.

- Stick to delta ([as a measure of moneyness](https://en.wikipedia.org/wiki/Moneyness#Black%E2%80%93Scholes_formula_auxiliary_variables)): Decrease one degree of freedom by introducing monyness $$m := K/S$$, we get $$\sigma_{imp} = \sigma_{imp}(m, \tau)$$, $$\frac{\partial \sigma_{imp}}{\partial S} (m_0, \tau) \vert_{\delta S_t} \equiv 0$$. Similiarly, we now have:

    $$\begin{aligned}
    \sigma_{imp}(K/S, \tau) &= \Sigma_0 + \frac{\partial \sigma_{imp}}{\partial m} \Big\vert_{m=1}(K/S - 1) := \Sigma_0 - \tilde{b}(K/S - 1)S_0 \\
    &\approx \Sigma_0 - \tilde{b}(K - S), \quad \text{when } K, S \rightarrow S_0.
    \end{aligned}$$

    Apparently, the yield IV has no dependence on $$S - K$$ (all share a same level of IV), and therefore it keeps ATM IV $$\sigma_{imp}^{atm}(m=1, \tau)$$ because $$S - K = 0$$. In a negatively skewed market (downward sloping for $$\sigma_{imp}(K)$$) the IV for an option of given strike $$K$$ increases with spot level $$S$$, thus the rule leads to a $$\Delta$$ larger than the $$\Delta_{BS}$$ for an option with the same, remaining constant BS volatility. In fact, when a pricing model is parameterized as $$V = V(S, \sigma_{imp}(S))$$, a derivative's total directional exposure:

    $$\begin{aligned}
    \frac{d V}{d S} &= \frac{\partial V}{\partial S} +  \frac{\partial V}{\partial \sigma} \frac{\partial \sigma}{\partial S} \\
    &= \Delta_{BS} + \mathcal{V} \frac{\partial \sigma}{\partial S} > \Delta_{BS}.
    \end{aligned}$$

    A man may argue that the volatility empirically appears to be negatively correlated with changes in the underlying. This points to the positively skewed market.

- Stick to [[5] Derman-Kani implied binomial tree](#ref5). The implied tree model allows the detailed numerical extraction of future local volatilities $$\sigma_{loc}(S_t, \tau)$$ and implied volatilities $$\sigma_{imp}(S_t, K, \tau)$$ from current implied volatilities. Comparerd to [Cox, Ross & Rubinstein (CRR) tree](https://en.wikipedia.org/wiki/Binomial_options_pricing_model), the implied binomial tree instead of prescribing $$\sigma$$, it enforces the amount of proportional jumps ($$u = 1/d$$) in asset price so that consistency with market observed call and put prices are observed (and the same with current skew). The jump ratio in the stock price tree reflects the level of volatility at the time level and stock price level.

    Roughly speaking, the sticky implied tree rule claims that the unknown dependence of volatility on the underlying, $$\frac{\partial \sigma}{\partial S}$$, can be approximated by the slope of the volatility smile, $$\frac{\partial \sigma}{\partial K}$$. The implied volatility is linearly approximated as a local functional form $$\sigma_{imp} = \sigma(S + K, \tau)$$:

    $$
    \sigma_{imp}(S + K, \tau) = \Sigma_0 + \frac{\partial \sigma_{imp}}{\partial (S + K)} \Big\vert_{K=S=S_0}(S + K - 2S_0) := \Sigma_0 - 2\hat{b}(S - S_0),
    $$

     And the option's model delta:

    $$
    \Delta_{SAD} = \Delta_{BS} + \mathcal{V}_{BS} \frac{\partial \sigma}{\partial K}.
    $$

    The delta given by equation above is refered to as the *smile-adjusted delta*. Within such an assumption, as $$S$$ changes by one unit, there is a parallel shift of $$\frac{\partial \sigma}{\partial S} = \frac{\partial \sigma}{\partial K}$$ units in the volatility smile. If the current smile is downward sloping $$\frac{\partial \sigma}{\partial K} \le 0$$, the approximation assumes that the volatility smile is shifted downwards as the price of the underlying stock increases, and thus, all fixed-strike volatilities decrease by the amount defined by the slope of the current smile and at-the-money volatility decreases twice as much.

    More analysis on this topic can be found at [[4]](#ref4) and [[derman1996local](#derman1996local)].



## Reference
<ol>
    <li id="ref1">Christoffersen, Peter and Heston, Steven L. and Jacobs, Kris, Option Valuation with Conditional Skewness (July 15, 2003). EFA 2004 Maastricht Meetings Paper No. 2964. Available at SSRN: https://ssrn.com/abstract=557079 or http://dx.doi.org/10.2139/ssrn.55707</li>
    <li id="ref2">Hilpisch, Yves. Derivatives Analytics with Python: Data Analysis, Models, Simulation, Calibration and Hedging. John Wiley & Sons, 2015.</li>
    <li id="ref3">Gatheral, Jim. The volatility surface: a practitioner's guide. Vol. 357. John Wiley & Sons, 2011.</li>
    <li id="ref4">Emanuel Derman. Leature 9: Patterns of Volatility Change. Avaliable at http://www.emanuelderman.com/media/smile-lecture9.pdf</li>
    <li id="ref5">Derman, Emanuel, Iraj Kani, Deniz Ergener, and Indrajit Bardhan. "Quantitative Strategies Research Notes." The Volatility Smile and Its Implied Tree (1995).</li>
    <li id="derman1996local">Derman, Emanuel, Iraj Kani, and Joseph Z. Zou. "The local volatility surface: Unlocking the information in index option prices." Financial analysts journal (1996): 25-36.</li>
</ol>
