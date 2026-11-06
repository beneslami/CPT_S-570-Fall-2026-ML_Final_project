## Overall working framework

This is the formula that express the average profit per trade:
$$\text{expectancy}_{\text{net}} = (1+\alpha)(p - p_0) - \frac{c}{SL}$$

Proof:

1- Measure everything in units of risk. Call the amount you lose when the stop-loss is hit 1R. If the stop-loss is $50 away from the entry, then 1R = $50. 

2- Measure the win in the same units. The take-profit is TP away from the entry, so a win earns $\frac{TP}{SL}\ \text{R} \;=\; \alpha\ \text{R}$ where $\alpha = \frac{TP}{SL}$ is the reward-to-risk ratio. TP = $100 and SL = $50 gives a win of 2R.

3- Average over many trades. If a fraction $p$ of trades win and th rest ($1-p$) lose, we have: $\text{expectancy} = \underbrace{p \times \alpha}_{\text{average gain}} \;-\; \underbrace{(1-p) \times 1}_{\text{average loss}}$. It answers: on average, how many R do I make per trade? With +0.1R per trade and $50 risked each time, 100 trades make about 100 × 0.1 × $50 = $500.

4- The break-even win rate: It means that, on average, with what win rate and with what $\alpha$, I become break-even (zero profit and zero loss)?
$$p \cdot \alpha - (1-p) = 0 \quad\Longrightarrow\quad p_{\text{break-even}} = \frac{1}{1 + \alpha}$$

|TP : SL |	$\alpha$	| Win rate needed to break even|
|:----:|:----:|:----:|
|1 : 3	|0.33	| 75%|
|1 : 2	| 0.5	|66.7%|
|1 : 1	| 1	| 50% |
|2 : 1	| 2	|33.3%|
|3 : 1	| 3	|25%|

Accuracy has no meaning without $\alpha$. 70% is a loss at 1:3 and excellent at 1:1.

5- We take $P_0 = \frac{1}{1+\alpha}$ as the breakeven probability. The above table is our benchmark. We rewrite $[p(1 + \alpha) - 1]$ by substituting -1 with $P_0(1 -\alpha)$. We reconstruct below formula:
$$(1 + \alpha)\,\big(p - p_0\big)$$

* $(p - p_0)$ is our edge: how much our model’s win rate beats a trade with no information.
* $(1 + \alpha)$ is a multiplier: how much each point of edge is worth.

6- costs. Spread and fees are paid on every trade. In R units, a cost $c$ per trade is $c/SL$. A tight stop-loss makes costs bigger in R. That’s why very short-timeframe setups (1m, 5m) often lose to costs, even with an edge. it's fixed and constant per broker. Each broker defines its own costs. 

It is important to note that, low win rate means long runs of losses. Based on the inverse geometric distribution, the probability of occurrence of n consecutive loss follows this formula:
$$
\frac{ln(n)}{ln(\frac{1}{1-p})}
$$
For example, if $P=0.35$ with $\alpha = 2$, it might be possible we experience 11 losses in a row. On the other hand, $P=0.75$ with $\alpha = 0.33$, might produce 3 losses in a row. Both have the same expectancy, but the 2:1 setup is much harder to sit through and needs a larger account buffer.


### steps we should take

1- calculate the cost per trade. $c$ is everything you pay for one complete trade (enter + exit), in price units. For example, Coinbase takes 0.1% of the price for the whole round trip trade on BTC. At BTC’s median price over the last 12 months ($77,275), that’s about $77 per trade.

2- Define SL. We have two rules to define SL:

1) beyond the noise. If the stop is closer than normal candle-to-candle wiggle, random noise stops you out. Set it in ATRs, typically ≥ 1 ATR of your timeframe. ATR means Average True Range. For example BTC shows ATR of 476 dollars in 1h timeframe. This would lead to $\frac{c}{SL} = \frac{77}{476} = 0.16$ per trade. It means that, on 1h, costs alone eat 16\% of each trade’s risk.

2) big enough that costs are small in R. $c/SL$ is the share of each trade’s risk eaten by costs. Keep it below about 0.05–0.10, so SL should be at least 10× the cost.

3- $\alpha$. We select a range of $\alpha$: 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5. Then calculate TP based on that.  Note that, a higher $\alpha$ means a lower win rate and longer losing streaks, but each point of edge is woth more. 

4- Calculate $P_0$ based on $\alpha$ to form a table of break-even scenarios. 

5- $P$ must come from the same TP and SL. To use the formula, train and test the model on triple-barrier labels with your chosen SL and $\alpha$, and measure $p$ on those trades.

Triple-barrier label is a way of labelling each candle by what would have happened to a trade opened at that candle, instead of by the next candle’s color. At the close of candle t you imagine opening a trade, and draw three barriers around the price path that follows:

1) Upper barrier: take-profit, entry + TP
2) Lower barrier: stop-loss, entry − SL
3) Vertical barrier: a time limit of H candles. If neither price barrier is hit by then, the trade is closed anyway.

The label is whichever barrier the price touches first. That’s literally “did this trade win, lose, or time out?” The barriers are set from the volatility at the moment of entry, for example:
$SL = 1 \times ATR(t)$ and $TP = \alpha \times SL$.

6- We train our model and calculate $P$. As we already defined SL and $\alpha$, we put the numbers in the formula:
$$\text{expectancy}_{\text{net}} = (1+\alpha)(p - p_0) - \frac{c}{SL}$$
and the sign tells you whether it’s profitable: above 0 makes money, below 0 loses. The size tells you how much per trade, in units of R. There’s one condition, and it’s what “$p$ must come from the same TP and SL” means: the 60% has to be measured in exactly the way the formula assumes.