# Financial Metrics: Formulas, Explanations, and Interpretations

## Basic Return Calculations

1. **Simple Return**
   - Formula: $$R = \frac{P_t - P_{t-1}}{P_{t-1}}$$
   - Where $P_t$ is the price at time $t$, and $P_{t-1}$ is the price at time $t-1$
   - Explanation: Measures the percentage change in price over a single period.

2. **Log Return**
   - Formula: $$r = \ln(\frac{P_t}{P_{t-1}})$$
   - Explanation: Measures the continuously compounded return, useful for multi-period analyses.

## Performance and Risk Metrics

1. **Annualized Return**
   - Formula: $$(1 + \text{Total Return})^{(1/n)} - 1$$
   - Where $n$ is the number of years
   - Explanation: Measures the average yearly return of a strategy, accounting for compounding.
   - Good values: Generally, higher is better. A value above the risk-free rate plus a risk premium is desirable.

2. **Maximum Drawdown**
   - Formula: $$\frac{\text{Trough Value} - \text{Peak Value}}{\text{Peak Value}}$$
   - Explanation: Shows the largest percentage drop from a peak to a trough in a portfolio's value.
   - Good values: Lower is better. Values below 20-30% are often considered acceptable, but this can vary by strategy and risk tolerance.

3. **Calmar Ratio**
   - Formula: $$\frac{\text{Annualized Return}}{\text{Maximum Drawdown}}$$
   - Explanation: Compares the annualized return to the maximum drawdown, indicating return per unit of downside risk.
   - Good values: Higher is better. A ratio above 1 is generally considered good, with top performers often achieving ratios of 3 or higher.

4. **Sharpe Ratio**
   - Formula: $$\frac{R_p - R_f}{\sigma_p}$$
   - Where $R_p$ is portfolio return, $R_f$ is risk-free rate, $\sigma_p$ is portfolio standard deviation
   - Explanation: Evaluates risk-adjusted performance by relating excess return to volatility.
   - Good values: Higher is better. A ratio above 1 is considered acceptable, above 2 is very good, and above 3 is excellent.

5. **Sortino Ratio**
   - Formula: $$\frac{R_p - R_f}{\sigma_d}$$
   - Where $\sigma_d$ is downside deviation
   - Explanation: Similar to Sharpe Ratio but focuses only on downside volatility.
   - Good values: Higher is better. Interpretation is similar to the Sharpe ratio, but values tend to be higher due to considering only downside risk.

6. **Treynor Ratio**
   - Formula: $$\frac{R_p - R_f}{\beta}$$
   - Where $\beta$ is portfolio beta
   - Explanation: Assesses risk-adjusted returns relative to systematic risk (beta).
   - Good values: Higher is better. Should be compared to the market's Treynor ratio for context.

7. **Information Ratio**
   - Formula: $$\frac{R_p - R_b}{\text{Tracking Error}}$$
   - Where $R_b$ is benchmark return
   - Explanation: Evaluates risk-adjusted return of a portfolio against a benchmark.
   - Good values: Higher is better. Values above 0.5 are good, above 0.75 are very good, and above 1 are excellent.

8. **Beta**
   - Formula: $$\frac{\text{Cov}(R_p, R_m)}{\text{Var}(R_m)}$$
   - Where $R_m$ is market return
   - Explanation: Measures the portfolio's sensitivity to market movements.
   - Interpretation: A beta of 1 indicates market-like volatility, <1 indicates lower volatility, and >1 indicates higher volatility than the market.

9. **Alpha**
   - Formula: $$R_p - [R_f + \beta(R_m - R_f)]$$
   - Explanation: Represents excess return after adjusting for market-related risk.
   - Good values: Positive alpha is desirable, indicating outperformance relative to the risk taken.

10. **R-Squared**
    - Formula: $$(\text{Correlation coefficient between portfolio and benchmark})^2$$
    - Explanation: Indicates how closely portfolio performance matches benchmark performance.
    - Interpretation: Ranges from 0 to 1. Higher values indicate closer correlation with the benchmark.

11. **Skewness**
    - Formula: $$E[(\frac{X - \mu}{\sigma})^3]$$
    - Where $X$ is the return, $\mu$ is mean, $\sigma$ is standard deviation
    - Explanation: Measures asymmetry of return distribution.
    - Interpretation: Positive skew is generally preferable, indicating more extreme positive returns than negative.

12. **Kurtosis**
    - Formula: $$E[(\frac{X - \mu}{\sigma})^4]$$
    - Explanation: Indicates "tailedness" of return distribution and potential for extreme outcomes.
    - Interpretation: Higher kurtosis indicates more frequent extreme outcomes. Normal distribution has a kurtosis of 3.

13. **Omega Ratio**
    - Formula: $$\frac{E[\max(R - \tau, 0)]}{E[\max(\tau - R, 0)]}$$
    - Where $\tau$ is threshold return
    - Explanation: Compares likelihood of returns above a threshold to likelihood of returns below it.
    - Good values: Higher is better. A ratio above 1 indicates more potential for gains than losses relative to the threshold.

14. **Downside Deviation**
    - Formula: $$\sqrt{\frac{\sum(\min(R - \tau, 0))^2}{n}}$$
    - Explanation: Measures volatility of negative returns, used in Sortino Ratio calculation.
    - Interpretation: Lower values indicate less downside risk.

15. **Value at Risk (VaR)**
    - Formula: $$\text{VaR} = \mu - (z \cdot \sigma \cdot \sqrt{t})$$
    - Where $\mu$ is expected return, $z$ is z-score for confidence level, $\sigma$ is standard deviation, $t$ is time horizon
    - Explanation: Predicts the maximum loss likely to occur over a specified time period at a given confidence interval.
    - Interpretation: Lower absolute values are better, indicating less potential for extreme losses.

16. **Conditional Value at Risk (CVaR) or Expected Shortfall**
    - Formula: $$\text{CVaR} = E[X | X > \text{VaR}]$$
    - Where $X$ represents the loss
    - Explanation: Gives the expected loss given that a loss is beyond the VaR threshold.
    - Interpretation: Lower values are better, indicating smaller expected losses in worst-case scenarios.

17. **Profit Factor**
    - Formula: $$\frac{\text{Gross Profit}}{\text{Gross Loss}}$$
    - Explanation: The ratio of gross profit to gross loss, indicating overall profitability of a trading strategy.
    - Good values: Above 1 is profitable, with higher values being better. A value of 2 means twice as much profit as loss.

18. **Recovery Factor**
    - Formula: $$\frac{\text{Net Profit}}{\text{Maximum Drawdown}}$$
    - Explanation: Compares the net profit to the maximum drawdown, showing how well the strategy recovers from losses.
    - Good values: Higher is better. A value above 1 indicates that profits exceed the worst drawdown.

19. **Ulcer Index**
    - Formula: $$\sqrt{\frac{\sum D_i^2}{n}}$$
    - Where $D_i$ is the drawdown from previous peak, $n$ is number of periods
    - Explanation: Measures the depth and duration of drawdowns in price.
    - Interpretation: Lower values are better, indicating less severe and prolonged drawdowns.

20. **Win Rate**
    - Formula: $$\frac{\text{Number of Winning Trades}}{\text{Total Number of Trades}}$$
    - Explanation: The percentage of trades that are profitable.
    - Good values: Higher is generally better, but should be considered alongside average win/loss size. Above 50% is often considered good.

21. **Reward-to-Risk Ratio**
    - Formula: $$\frac{\text{Average Winning Trade}}{\text{Average Losing Trade}}$$
    - Explanation: Compares the average winning trade to the average losing trade.
    - Good values: Higher is better. A ratio above 1 indicates larger average wins than losses.

22. **Expectancy**
    - Formula: $$(\text{Win Rate} \cdot \text{Average Win}) - (\text{Loss Rate} \cdot \text{Average Loss})$$
    - Explanation: Gives the average amount you can expect to win (or lose) per trade.
    - Good values: Positive values indicate a profitable system, with higher values being better.

23. **Trade Duration**
    - Formula: $$\frac{\sum(\text{Trade End Time} - \text{Trade Start Time})}{\text{Number of Trades}}$$
    - Explanation: Average time a position is held.
    - Interpretation: Optimal duration depends on the trading strategy. Consistency is often more important than the absolute value.

24. **Number of Trades**
    - Formula: Count of total trades executed
    - Explanation: Indicates how frequently the strategy trades.
    - Interpretation: Depends on the strategy. More trades can provide more data points but may also incur higher transaction costs.

25. **Turnover Rate**
    - Formula: $$\frac{\min(\text{Asset Sales}, \text{Asset Purchases})}{\text{Average Portfolio Value}}$$
    - Explanation: Measures how frequently assets within a portfolio are bought and sold.
    - Interpretation: Lower values indicate a more passive strategy, while higher values indicate more active trading. Optimal levels depend on the strategy and associated costs.