---
title: Handle High Frequency Data
layout: post
math: true
comments: true
author: Jon Liu
tags: hft tick-data
---

> This post states several features of high frequency trading data and shows hosts of methods to play with it.

<!--more-->

HFT data, commonly called tick data, are records of market activity. Two levels of data format complexity is grouped into: Level I data includes the best bid/ask price/size, and last trade price and size, where avaliable. Level II data includes all changes to the order book, including new limit order arrivals and cnacellations at prices away from the market price. etc. [LOBSTER](https://lobsterdata.com/info/DataStructure.php) provides 7 types of events causing an update of the limit order book in the requested price range.

Tick owns multiple meaning in this field. A tick is a measure of the minimum change unit in the price of a security. A tick can also refer to the change in the price of a security from trade to trade: uptick and downtick.

MOs contribute to HFT data in the field of 'last trade' info. They are executed immediately at the best avaliable bid/ask prices when placed. A large MO may need to be matched with one or several best quotes, generating several 'last trade' (represented as 'tp' and 'tq' but without buyer or seller initiated identifiers info for most small tick size tradable assets) data points.

Last trade price can differ from the bid and ask. The differences can arise when a customer posts a favorable limit order that is immediately matched by the broker without broadcasting the customer’s quote.

Most LOs and MOs are placed in so called "lot sizes": the minimal number of shares
that can be exchanged with one trade. For example, while commodity futures may enforce a 50 share minimum requirement for orders, a lot can be as low as one share on common equity exchanges.

In order to arrival at the right trading decisions and formulate trading strategies which are implemented with very low latencies, one has to know high frequency data's features well. Many of them has already been found in the literature.

- HF data is irregularly spaced in time (between observations). There is a lot literature on dealing with unevenly spaced data. [1] is a good point to get start

- Market microstructure noise leads to observed deviation from the base price. It also makes high frequency magnitude estimator very unstable. The well-known *bid-ask bounce* occurs when the price keeps jumping between the best bid and ask. With eyes  on such behavior, the micro-volatility could be calculated into a high level even if the price stays within the spread.

- Fat tail distribution. Extreme events happened much more often than a normal distribution could state. There is higher probability of big losses or profit in the real world markets.

- Volatility clustering and long memory in absolute values of returns.

## Parsing cumbersome Level II information

## Distributions

## Order type
- FAS (Fill and Save): An limit order that fills as much as possiable. The resting quantity of the limit order becomes passive.
- FOK (Fill or Kill): Fills order completely or cancels the order immediately. Partially fill is not acceptable.
- FAK (Fill and Kill): Fills as much as possiable. The resting quantity is cancelled immediately.
- Iceberg (Order with Hidden Size): Discloses equal quantities to the market based upon a number or percentage of the order. The resting quantity of the limit order becomes passive.

## Market making

The market maker’s bids are said to be **hit** by market sell orders (hit the bid), and the market maker’s ask limit orders are said to be **lifted** by incoming market buy orders (lift the ask). The market order traders are known as **liquidity takers**.

Two kinds of risk a market maker could face:
- Inventory risk. Besides the common sense, it also includes the opportunity costs reflecting the gains the market maker misses while waiting for execution of his limit orders.
- Risk of adverse selection. This loss comes from informational difference between the makers and takers.

Too little inventory may be insufficient to generate a profit; too much inventory makes the trader risk inability to quickly liquidate his position and face a certain loss.

Trading frequency has been shown to be key to market makers' profitablity. Market makers’ profit is directly tied to the frequency with which their orders are executed. Denser round-trip trades usually lead to a better performance. The higher the number of times the market maker can “flip” his capital, the higher the cumulative spread the market-maker can capture, and the lower the risk the market maker bears waiting for execution of his orders.

Super-fast technology allows market makers to continually cancel and resubmit their limit orders to ensure their spot on the top of the book, high execution rates, and high profitability. Slower technology can still deliver profitable market-making strategies via larger offsets away from the going market price.

Some market making ideas:
- Fixed-offset.
- Volatility-offset. Make the offset a fucntion of volatiliity: in high vol conditions, LOs far away from the market are likely to be hit, generating higher premium for market makers; in low vol counterpart, LOs may need to be placed closer to the market to be executed. *Realized variance* $$O_t = \frac{1}{T}\sum_{\tau=t-1}^{t-T} (P_{\tau} - P_{\tau - 1})^2$$ or *volatility* releated estimator could be helpful, and [here](https://realized.oxford-man.ox.ac.uk/documentation/estimators) is a potential list. Besides, the book [Volatility Trading](https://www.amazon.com/Volatility-Trading-CD-ROM-Euan-Sinclair/dp/0470181990) also provides several volatility estimator with parts of them implemented [here](https://github.com/jasonstrimpel/volatility-trading).

- Order-arrival rate offset. LOs compete with other LOs, both existing and those submitted in the future. Furthermore, all LOs execute against future market orders. The MO arrival rate, therefore, is an important determinant of market-making profitability. Releated models include exponentially-distributed inter-arrival times (OK, Possion in fact), Hawkes process, etc. The core concept here is the arrival rate of MO $\mu$; with $1/\mu$ representing the average time between two sequential MO arrivals, which makes it easier to calibrate.  When the model is used to determine levels of aggressiveness of limit buy orders, $\mu$ is the rate of arrival of market sell orders.

- Differentiation between trending and mean-reverting markets. In a mean-reverting msrket, the price bounces up and down within a range, reducing the risk of adverse selection and making such market conditions perfect for market making. In trending markets, however, a market maker needs to reduce the size of LOs on the wrong side of the markets, hedge the exposure, or exit the markets altogether. Directional analysis need to be performed to determine the hidden regime. Hurst exponent is a good benchmark to test.

- Resistance and support levels. When the level II data are not available, the shape of the order book can still be estimated using techniques that hail from technical analysis, like support and resistance levels and moving averages. Computing support and resistance helps pinpoint liquidity peaks without parsing cumbersome level II information. Say, we accumulate 1 min bars $$\{B_i\}_{i=1}^t$$, $$p_t=\{p_t^i\}_{i=1}^{t_N}$$ reprs the price sequence in the $t$-th bar. One can update the support/resistance levels:

$$
SL_{t+1} = \min(p_t) + (\min(p_t) - \min(p_{t-1}))
$$

$$
RL_{t+1} = \max(p_t) + (\max(p_t) - \max(p_{t-1}))
$$

- Order aggressiveness. Aggressiveness refers to the percentage of orders that are submitted at market prices, as opposed to limit prices. If a trader executes immediately instead of waiting for a more favorable price, the trader may convey information about his beliefs about where the market is going.

## Reference
[1] Eckner, Andreas. "A framework for the analysis of unevenly spaced time series data." Preprint. Available at: http://www.eckner.com/papers/unevenly_spaced_time_series_analysis (2012).
