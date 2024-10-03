# Financial Metrics: Formulas, Explanations, and Interpretations

## Basic Return Calculations

1. **Simple Return**
   - Formula: $$R = \frac{P_t}{P_{t-1}} - 1$$
   - Where $P_t$ is the price at time $t$, and $P_{t-1}$ is the price at time $t-1$
   - Explanation: Measures the percentage change in price over a single period.

2. **Log Return**
   - Formula: $$r = \ln(\frac{P_t}{P_{t-1}})$$
   - Explanation: Measures the continuously compounded return, useful for multi-period analyses.

## Performance and Risk Metrics

1. **Annualized Return - Measures the average yearly return of a strategy, accounting for compounding.**
   - Formula: $$Annualized\; Return = (1 + \text{Total Return})^{(1/n)} - 1$$
   - Where $n$ is the number of years

2. **Maximum Drawdown**
   - Formula: $$Max \; Drawdown = \frac{\text{Trough Value} - \text{Peak Value}}{\text{Peak Value}}$$
   - Explanation: Shows the largest percentage drop from a peak to a trough in a portfolio's value.
     - Lower is better. 
     - Values below 20-30% are often considered acceptable, but this can vary by strategy and risk tolerance.

3. **Calmar Ratio - Compares the annualized return to the maximum drawdown, indicating return per unit of downside risk.**
   - Formula: $$Calmar\; Ratio = \frac{\text{Annualized Return}}{\text{Maximum Drawdown}}$$
   - Interpretation 
     - Higher is better. 
     - A ratio above 1 is generally considered good, with top performers often achieving ratios of 3 or higher.

4. **Sharpe Ratio - Evaluates risk-adjusted performance by relating excess return to volatility.**
   - Formula: $$ Sharpe\; Ratio =  \frac{R_p - R_f}{\sigma_p}$$
   - Where $R_p$ is portfolio return, $R_f$ is risk-free rate, $\sigma_p$ is portfolio standard deviation
   - Interpretation
     - Higher is better.
     - A ratio above 1 is considered acceptable, above 2 is very good, and above 3 is excellent.

5. **Sortino Ratio - Evaluates risk-adjusted performance by relating excess return to downside volatility.**
   - Formula: $$Sortino\; Ratio = \frac{R_p - R_f}{\sigma_d}$$
   - Where $\sigma_d$ is downside deviation
   - Interpretation
      - Higher is better. 
      - Values tend to be higher due to considering only downside risk.

6. **Treynor Ratio - Assesses risk-adjusted returns relative to systematic risk (beta).**
   - Formula: $$Treynor\; Ratio = \frac{R_p - R_f}{\beta}$$
   - Where $\beta$ is portfolio beta
   - Interpretation
     - Higher is better. Should be compared to the market's Treynor ratio for context.

7. **Information Ratio - Evaluates risk-adjusted return of a portfolio against a benchmark.**
   Formula: $$\text{IR} = \frac{R_p - R_b}{\text{Tracking Error (TE)}}, \text{TE} = \sqrt{\frac{\sum_{i=1}^n (R_{p,i} - R_{b,i})^2}{n-1}}$$
    
    - Where $R_p$ is portfolio return and $R_b$ is benchmark return
    - Where $R_{p,i}$ is the portfolio return in period $i$, $R_{b,i}$ is the benchmark return in period $i$, and $n$ is the number of periods
   - The tracking error represents the standard deviation of the difference between portfolio and benchmark returns.
   - Interpretation
     - Higher is better. 
     - Values above 0.5 are good, above 0.75 are very good, and above 1 are excellent.

8. **Beta$(\beta)$ - Measures the portfolio's sensitivity to benchmark movements.**
   - Formula: $$\beta = \frac{\text{Cov}(R_p, R_b)}{\text{Var}(R_b)}$$
   - Where $R_b$ is benchmark return
   - Interpretation: 
     -  1 indicates benchmark-like volatility, 
     - <1 indicates lower volatility, and 
     - \>1 indicates higher volatility than the market.

9. **Alpha$(\alpha)$ - Represents excess return after adjusting for market-related risk.**
   - Formula: $$ \alpha = R_p - [R_f + \beta(R_m - R_f)]$$
   - Intepretation
     - Positive alpha is desirable, indicating outperformance relative to the risk taken.

10. **R-Squared$(R^2)$- Indicates how closely portfolio performance matches benchmark performance.**
    - Formula: $$R^2 = (\text{Correlation coefficient between portfolio and benchmark})^2$$
    - Interpretation: 
      - Ranges from 0 to 1. 
      - Higher values indicate closer correlation with the benchmark.

11. **Skewness - Measures asymmetry of return distribution.**
    - Formula: $$E[(\frac{X - \mu}{\sigma})^3]$$
    - Where $X$ is the return, $\mu$ is mean, $\sigma$ is standard deviation
    - Interpretation: 
      - Positive skew is generally preferable, indicating more extreme positive returns than negative.

12. **Kurtosis - Indicates "tailedness" of return distribution and potential for extreme outcomes.**
    - Formula: $$E[(\frac{X - \mu}{\sigma})^4]$$
    - Interpretation: 
      - Higher kurtosis indicates more frequent extreme outcomes. 
      - Normal distribution has a kurtosis of 3.

13. **Omega Ratio - Compares likelihood of returns above a threshold to likelihood of returns below it.**
    - Formula: $$Omega\; Ratio, \Omega = \frac{E[\max(R - \tau, 0)]}{E[\max(\tau - R, 0)]}$$
    - Where $\tau$ is threshold return
    - $(R - τ)$ represents the excess return above the threshold
    - $(τ - R)$ represents the shortfall below the threshold
    - Interpretation
      - Higher is better. 
      - A ratio above 1 indicates more potential for gains than losses relative to the threshold.

14. **Downside Deviation - Measures volatility of negative returns, used in Sortino Ratio calculation.**
    - Formula: $$Downside \; Deviation = \sqrt{\frac{\sum(\min(R - \tau, 0))^2}{n}}$$
    - $\min(R - τ, 0)$ captures only the returns that fall below the threshold
    - Interpretation: 
      - Lower values indicate less downside risk.

15. **Value at Risk (VaR) - Predicts the maximum loss likely to occur over a specified time period at a given confidence interval.**
    - Formula: $$\text{VaR} = \mu - (z \cdot \sigma \cdot \sqrt{t})$$
    - Where $\mu$ is expected return, $z$ is z-score for confidence level, $\sigma$ is standard deviation, $t$ is time horizon
    - Interpretation: 
      - Lower absolute values are better, indicating less potential for extreme losses.

16. **Conditional Value at Risk (CVaR) or Expected Shortfall - Gives the expected loss given that a loss is beyond the VaR threshold.**
    - Formula: $$\text{CVaR} = E[X | X > \text{VaR}]$$
    - Where $X$ represents the loss
    - Interpretation: 
      - Lower values are better, indicating smaller expected losses in worst-case scenarios.

17. **Profit Factor - The ratio of gross profit to gross loss, indicating overall profitability of a trading strategy.**
    - Formula: $$Profit \; Factor=\frac{\text{Gross Profit}}{\text{Gross Loss}}$$
    - Good values: 
      - Above 1 is profitable, with higher values being better. 
      - A value of 2 means twice as much profit as loss.

18. **Recovery Factor - Compares the net profit to the maximum drawdown, showing how well the strategy recovers from losses.**
    - Formula: $$Recovery\;Factor = \frac{\text{Net Profit}}{\text{Maximum Drawdown}}$$
    - Interpretation
      - Higher is better. A value above 1 indicates that profits exceed the worst drawdown.

19. **Ulcer Index - Measures the depth and duration of drawdowns in price.**
    - Formula: $$Ulcer \; Index = \sqrt{\frac{\sum D_i^2}{n}}$$
    - Where $D_i$ is the drawdown from previous peak, $n$ is number of periods
    - Interpretation: 
      - Lower values are better, indicating less severe and prolonged drawdowns.
### Trade Metrics
1.  **Win Rate - The percentage of trades that are profitable.**
2.  **Reward-to-Risk Ratio - Compares the average winning trade to the average losing trade.**
    - Formula: $$Risk-to-Reward\;Ratio = \frac{\text{Average Winning Trade}}{\text{Average Losing Trade}}$$
    - Interpretation 
      - Higher is better. A ratio above 1 indicates larger average wins than losses.

3.  **Expectancy - Gives the average amount you can expect to win (or lose) per trade.**
    - Formula: $$Expectancy = (\text{Win Rate} \cdot \text{Average Win}) - (\text{Loss Rate} \cdot \text{Average Loss})$$
    - Interpretation
      - Positive values indicate a profitable system, with higher values being better.

4.  **Trade Duration - Average time a position is held.**
5.  **Number of Trades - Indicates how frequently the strategy trades.**
6.  **Turnover Rate - Measures how frequently assets within a portfolio are bought and sold.**
    - Formula: $$ Turnover\; Rate = \frac{\min(\text{Asset Sales}, \text{Asset Purchases})}{\text{Average Portfolio Value}}$$
    - Interpretation: 
      - Lower values indicate a more passive strategy, while 
      - Higher values indicate more active trading. 
# Financial Ratios: Interpretation Guide

| Higher Values Are Better | Lower Values Are Better |
|--------------------------|--------------------------|
| Annualized Return | Maximum Drawdown |
| Calmar Ratio | Downside Deviation |
| Sharpe Ratio | Value at Risk (VaR) |
| Sortino Ratio | Conditional Value at Risk (CVaR) |
| Treynor Ratio | Ulcer Index |
| Information Ratio | |
| Alpha | |
| Omega Ratio | |
| Profit Factor | |
| Recovery Factor | |
| Win Rate | |
| Reward-to-Risk Ratio | |
| Expectancy | |

## Notes:
1. **R-Squared**: Higher or lower isn't necessarily better. It depends on the investment strategy. A higher R-squared indicates closer correlation with the benchmark.

2. **Beta**: Neither higher nor lower is inherently better. It depends on the investor's risk tolerance and market expectations.
   - Beta > 1: More volatile than the market
   - Beta < 1: Less volatile than the market
   - Beta = 1: Same volatility as the market

3. **Skewness**: Positive skew is generally preferred, indicating more extreme positive returns than negative.

4. **Kurtosis**: Neither higher nor lower is inherently better. Higher kurtosis indicates more frequent extreme outcomes, which could be positive or negative.

5. **Trade Duration**: Optimal duration depends on the specific trading strategy. Consistency is often more important than absolute value.

6. **Number of Trades**: Optimal number depends on the strategy. More trades provide more data points but may incur higher transaction costs.

7. **Turnover Rate**: Optimal level depends on the strategy and associated costs. Lower values indicate a more passive strategy, while higher values indicate more active trading.